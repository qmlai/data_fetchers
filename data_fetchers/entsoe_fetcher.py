from lxml import etree
from entsoe import EntsoeRawClient
from collections.abc import Callable
from pandas.tseries.offsets import DateOffset
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import DataFetcherBase
from .constants import (ENSTOE_BASE_DATA_DIR, MAX_RETRIES, NEIGHBOURS, CURVE_SCHEMA, 
                      ENTSOE_CODES_TO_ZONES, ENTSOE_ZONES_TO_CODES, WRITE_RETRY_SECONDS, FREQUENCIES)

import io
import zipfile
import numpy as np
import pandas as pd
import pyarrow as pa

class EntsoeDataFetcher(DataFetcherBase):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client  = EntsoeRawClient(api_key=api_key)

    #-------------------------------------------------------------
    #                       XML Parsers
    #-------------------------------------------------------------

    def parse_day_ahead_prices(self, xml_string: str) -> pd.DataFrame:
        """
        Parses ENTSO-E day ahead prices XML into a DataFrame compliant with CURVE_SCHEMA.
        """
        root = etree.fromstring(xml_string.encode("utf-8"))
        ns = {"ns": root.nsmap[None]}  # namespace

        markets, neighbours = [], []
        timestamps, values, resolutions, units = [], [], [], []
        years, months, days = [], [], []

        for ts in root.xpath(".//ns:TimeSeries", namespaces=ns):
            in_domain = ts.find(".//ns:in_Domain.mRID", namespaces=ns).text
            out_domain = ts.find(".//ns:out_Domain.mRID", namespaces=ns).text
            price_unit = ts.find(".//ns:price_Measure_Unit.name", namespaces=ns).text

            in_zone  = ENTSOE_CODES_TO_ZONES[in_domain]
            out_zone = ENTSOE_CODES_TO_ZONES[out_domain]

            for period in ts.xpath(".//ns:Period", namespaces=ns):
                start_str = period.find(".//ns:timeInterval/ns:start", namespaces=ns).text
                resolution = period.find(".//ns:resolution", namespaces=ns).text
                freq = FREQUENCIES.get(resolution, "h")
                start = pd.Timestamp(start_str, tz="UTC")
                points = period.findall(".//ns:Point", namespaces=ns)
                times = pd.date_range(start, periods=len(points), freq=freq, tz="UTC")

                for t, p in zip(times, points):
                    value = float(p.findtext(".//ns:price.amount", namespaces=ns))

                    markets.append(in_zone)
                    neighbours.append(out_zone)
                    timestamps.append(t) 
                    resolutions.append(resolution)
                    units.append(price_unit)
                    values.append(value) 
                    years.append(t.year)
                    months.append(t.month)
                    days.append(t.day)

        if not timestamps:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})
        
        n = len(timestamps)

        return pa.table({
                    "market": markets,
                    "curve_type": np.full(n, "day_ahead_price", dtype=object),
                    "neighbour": neighbours,
                    "production_type": np.full(n, "N/A", dtype=object),
                    "timestamp": timestamps,
                    "value": values,
                    "resolution": resolutions,
                    "unit": units,
                    "source": np.full(n, "ENTSOE", dtype=object),
                    "year": years,
                    "month": months,
                    "day": days,
                    }, schema=CURVE_SCHEMA)

    def parse_balancing_prices_from_zip(self, zip_bytes: bytes, market: str = "activated_balancing") -> pa.Table:
        """
        Parse ENTSO-E imbalance/balancing prices from a ZIP (returned by query_imbalance_prices).
        Returns a PyArrow table compatible with CURVE_SCHEMA.
        """
        if not zip_bytes:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        tables = []

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    xml_content = f.read()
                    # parse each XML inside the ZIP
                    t = self.parse_balancing_prices(xml_content, market)
                    tables.append(t)

        if not tables:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        return pa.concat_tables(tables)

    def parse_balancing_prices(self, xml_bytes: bytes | str, market: str) -> pa.Table:
        """
        Parse a single ENTSO-E balancing price XML into a PyArrow table compliant with CURVE_SCHEMA.
        """
        if isinstance(xml_bytes, bytes):
            text = xml_bytes.decode("utf-8")
        elif isinstance(xml_bytes, str):
            text = xml_bytes
        else:
            raise ValueError(f"xml_bytes must be str or bytes, got {type(xml_bytes)}")

        if not text.strip().startswith("<"):
            # non-XML content, e.g., error HTML
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        root = etree.fromstring(text.encode("utf-8"))
        ns = {"ns": root.nsmap[None]}  # namespace

        timestamps, values, resolutions, units = [], [], [], []
        neighbours, years, months, days = [], [], [], []

        for ts in root.xpath(".//ns:TimeSeries", namespaces=ns):
            direction_el = ts.find(".//ns:flowDirection.direction", namespaces=ns)
            direction = direction_el.text if direction_el is not None else "N/A"

            price_unit_el = ts.find(".//ns:price_Measure_Unit.name", namespaces=ns)
            price_unit = price_unit_el.text if price_unit_el is not None else "N/A"

            for period in ts.xpath(".//ns:Period", namespaces=ns):
                start_str = period.findtext(".//ns:timeInterval/ns:start", namespaces=ns)
                resolution = period.findtext(".//ns:resolution", namespaces=ns)
                freq = FREQUENCIES.get(resolution, "h")
                start = pd.Timestamp(start_str, tz="UTC")
                points = period.findall(".//ns:Point", namespaces=ns)
                times = pd.date_range(start, periods=len(points), freq=freq, tz="UTC")

                for t, p in zip(times, points):
                    price_el = p.find(".//ns:activation_Price.amount", namespaces=ns)
                    price = float(price_el.text) if price_el is not None else float("nan")

                    timestamps.append(t)
                    values.append(price)
                    resolutions.append(resolution)
                    units.append(price_unit)
                    neighbours.append(direction)
                    years.append(t.year)
                    months.append(t.month)
                    days.append(t.day)

        if not timestamps:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        n = len(timestamps)

        return pa.table({
            "market": np.full(n, market, dtype=object),
            "curve_type": np.full(n, "activated_balancing_price", dtype=object),
            "neighbour": neighbours,
            "production_type": np.full(n, "N/A", dtype=object),
            "timestamp": timestamps,
            "value": values,
            "resolution": resolutions,
            "unit": units,
            "source": np.full(n, "ENTSOE", dtype=object),
            "year": years,
            "month": months,
            "day": days,
        }, schema=CURVE_SCHEMA)

    def parse_load_forecast(self, xml_string: str) -> pa.Table:
        """
        Parses ENTSO-E load/forecast XML into a PyArrow table compliant with CURVE_SCHEMA.
        """
        # Parse XML
        root = etree.fromstring(xml_string.encode("utf-8") if isinstance(xml_string, str) else xml_string)
        ns = {"ns": root.nsmap[None]}  # namespace

        timestamps, values, resolutions, units = [], [], [], []
        markets, years, months, days = [], [], [], []

        for ts in root.xpath(".//ns:TimeSeries", namespaces=ns):
            out_domain_el = ts.find(".//ns:outBiddingZone_Domain.mRID", namespaces=ns)
            unit_el = ts.find(".//ns:quantity_Measure_Unit.name", namespaces=ns)

            if out_domain_el is None or unit_el is None:
                continue

            out_zone = ENTSOE_CODES_TO_ZONES[out_domain_el.text]
            unit = unit_el.text

            for period in ts.xpath(".//ns:Period", namespaces=ns):
                start_str = period.findtext(".//ns:timeInterval/ns:start", namespaces=ns)
                resolution = period.findtext(".//ns:resolution", namespaces=ns)
                freq = FREQUENCIES.get(resolution, "h")
                start = pd.Timestamp(start_str)
                points = period.findall(".//ns:Point", namespaces=ns)
                times = pd.date_range(start, periods=len(points), freq=freq)

                for t, p in zip(times, points):
                    quantity = float(p.findtext(".//ns:quantity", namespaces=ns))

                    timestamps.append(t)
                    values.append(quantity)
                    resolutions.append(resolution)
                    units.append(unit)
                    markets.append(out_zone)
                    years.append(t.year)
                    months.append(t.month)
                    days.append(t.day)

        if not timestamps:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        n = len(timestamps)

        return pa.table({
            "market": markets,
            "curve_type": np.full(n, "load_forecast", dtype=object),
            "neighbour": np.full(n, "N/A", dtype=object),
            "production_type": np.full(n, "N/A", dtype=object),
            "timestamp": timestamps,
            "value": values,
            "resolution": resolutions,
            "unit": units,
            "source": np.full(n, "ENTSOE", dtype=object),
            "year": years,
            "month": months,
            "day": days,
        }, schema=CURVE_SCHEMA)

    def parse_generation_forecast(self, xml_string: str) -> pa.Table:
        """
        Parses ENTSO-E generation forecast XML into a PyArrow table compliant with CURVE_SCHEMA.
        """
        root = etree.fromstring(xml_string.encode("utf-8") if isinstance(xml_string, str) else xml_string)
        ns = {"ns": root.nsmap[None]}

        timestamps, values, resolutions, units = [], [], [], []
        markets, years, months, days, prod_types = [], [], [], [], []

        for ts in root.xpath(".//ns:TimeSeries", namespaces=ns):
            in_domain_el = ts.find(".//ns:inBiddingZone_Domain.mRID", namespaces=ns)
            if in_domain_el is None:
                continue
            in_zone = ENTSOE_CODES_TO_ZONES[in_domain_el.text]
            quantity_unit = ts.findtext(".//ns:quantity_Measure_Unit.name", namespaces=ns)
            business_type = ts.findtext(".//ns:businessType", namespaces=ns)

            for period in ts.xpath(".//ns:Period", namespaces=ns):
                start_str = period.findtext(".//ns:timeInterval/ns:start", namespaces=ns)
                resolution = period.findtext(".//ns:resolution", namespaces=ns)
                freq = FREQUENCIES.get(resolution, "h")
                start = pd.Timestamp(start_str, tz="UTC")

                points = period.findall(".//ns:Point", namespaces=ns)
                times = pd.date_range(start, periods=len(points), freq=freq, tz="UTC")

                for t, p in zip(times, points):
                    quantity = float(p.findtext(".//ns:quantity", namespaces=ns))
                    timestamps.append(t)
                    values.append(quantity)
                    resolutions.append(resolution)
                    units.append(quantity_unit)
                    markets.append(in_zone)
                    prod_types.append(business_type)
                    years.append(t.year)
                    months.append(t.month)
                    days.append(t.day)

        if not timestamps:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        n = len(timestamps)
        return pa.table({
            "market": markets,
            "curve_type": np.full(n, "generation_forecast", dtype=object),
            "neighbour": np.full(n, "N/A", dtype=object),
            "production_type": prod_types,
            "timestamp": timestamps,
            "value": values,
            "resolution": resolutions,
            "unit": units,
            "source": np.full(n, "ENTSOE", dtype=object),
            "year": years,
            "month": months,
            "day": days,
        }, schema=CURVE_SCHEMA)

    def parse_wind_solar_forecast(self, xml_string: str) -> pa.Table:
        root = etree.fromstring(xml_string.encode("utf-8") if isinstance(xml_string, str) else xml_string)
        ns = {"ns": root.nsmap[None]}  

        timestamps, values, resolutions, units = [], [], [], []
        markets, years, months, days, prod_types = [], [], [], [], []

        for ts in root.xpath(".//ns:TimeSeries", namespaces=ns):
            in_domain = ENTSOE_CODES_TO_ZONES[ts.findtext(".//ns:inBiddingZone_Domain.mRID", namespaces=ns)]
            quantity_unit = ts.findtext(".//ns:quantity_Measure_Unit.name", namespaces=ns)
            psr_type = ts.findtext(".//ns:MktPSRType/ns:psrType", namespaces=ns) or "N/A"

            for period in ts.xpath(".//ns:Period", namespaces=ns):
                start_str = period.findtext(".//ns:timeInterval/ns:start", namespaces=ns)
                resolution = period.findtext(".//ns:resolution", namespaces=ns)
                freq = FREQUENCIES.get(resolution, "h")
                start = pd.Timestamp(start_str, tz="UTC")

                points = period.findall(".//ns:Point", namespaces=ns)
                times = pd.date_range(start, periods=len(points), freq=freq, tz="UTC")

                for t, p in zip(times, points):
                    quantity = float(p.findtext(".//ns:quantity", namespaces=ns))
                    timestamps.append(t)
                    values.append(quantity)
                    resolutions.append(resolution)
                    units.append(quantity_unit)
                    markets.append(in_domain)
                    prod_types.append(psr_type)
                    years.append(t.year)
                    months.append(t.month)
                    days.append(t.day)

        if not timestamps:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        n = len(timestamps)
        return pa.table({
            "market": markets,
            "curve_type": np.full(n, "intraday_wind_solar_forecast", dtype=object),
            "neighbour": np.full(n, "N/A", dtype=object),
            "production_type": prod_types,
            "timestamp": timestamps,
            "value": values,
            "resolution": resolutions,
            "unit": units,
            "source": np.full(n, "ENTSOE", dtype=object),
            "year": years,
            "month": months,
            "day": days,
        }, schema=CURVE_SCHEMA)


    def parse_crossborder_flows(self, xml_string: str) -> pa.Table:
        root = etree.fromstring(xml_string.encode("utf-8") if isinstance(xml_string, str) else xml_string)
        ns = {"ns": root.nsmap[None]}  

        timestamps, values, resolutions, units = [], [], [], []
        markets, neighbours, years, months, days, prod_types = [], [], [], [], [], []

        for ts in root.xpath(".//ns:TimeSeries", namespaces=ns):
            from_zone = ENTSOE_CODES_TO_ZONES[ts.findtext(".//ns:in_Domain.mRID", namespaces=ns)]
            to_zone = ENTSOE_CODES_TO_ZONES[ts.findtext(".//ns:out_Domain.mRID", namespaces=ns)]
            quantity_unit = ts.findtext(".//ns:quantity_Measure_Unit.name", namespaces=ns)
            business_type = ts.findtext(".//ns:businessType", namespaces=ns)

            for period in ts.xpath(".//ns:Period", namespaces=ns):
                start_str = period.findtext(".//ns:timeInterval/ns:start", namespaces=ns)
                resolution = period.findtext(".//ns:resolution", namespaces=ns)
                freq = FREQUENCIES.get(resolution, "h")
                start = pd.Timestamp(start_str)

                points = period.findall(".//ns:Point", namespaces=ns)
                times = pd.date_range(start, periods=len(points), freq=freq)

                for t, p in zip(times, points):
                    flow = float(p.findtext(".//ns:quantity", namespaces=ns))
                    timestamps.append(t)
                    values.append(flow)
                    resolutions.append(resolution)
                    units.append(quantity_unit)
                    markets.append(from_zone)
                    neighbours.append(to_zone)
                    prod_types.append(business_type)
                    years.append(t.year)
                    months.append(t.month)
                    days.append(t.day)

        if not timestamps:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        n = len(timestamps)
        return pa.table({
            "market": markets,
            "curve_type": np.full(n, "flow", dtype=object),
            "neighbour": neighbours,
            "production_type": prod_types,
            "timestamp": timestamps,
            "value": values,
            "resolution": resolutions,
            "unit": units,
            "source": np.full(n, "ENTSOE", dtype=object),
            "year": years,
            "month": months,
            "day": days,
        }, schema=CURVE_SCHEMA)

    #-------------------------------------------------------------
    #                       Fetchers
    #-------------------------------------------------------------

    def _with_retries(self, fn: Callable, *args, max_retries: int, retry_seconds: int, **kwargs):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    time_sleep = retry_seconds * attempt
                    print(f"[WARN] attempt {attempt} failed, sleeping {time_sleep}s: {e}")
                    import time

                    time.sleep(time_sleep)
                else:
                    raise
        raise last_exc
    
    def _fetch_xml(self, fn: Callable, *args, **kwargs) -> str:
        return self._with_retries(fn, *args, **kwargs)

    def fetch_day_ahead_prices(self, zone: str, start: pd.Timestamp, end: pd.Timestamp):
        try:
            xml = self._fetch_xml(self.client.query_day_ahead_prices, zone, start, end, max_retries=MAX_RETRIES, retry_seconds=WRITE_RETRY_SECONDS)
            t = self.parse_day_ahead_prices(xml)
            self.store_table_idempotent(t, market=zone, schema=CURVE_SCHEMA, base=ENSTOE_BASE_DATA_DIR)
        except Exception as e:
            print(f"[WARN] DA prices {zone} {start}-{end} failed: {e}")
        
    def fetch_balancing_prices(self, zone: str, start: pd.Timestamp, end: pd.Timestamp):
        try:
            xml = self._fetch_xml(self.client.query_imbalance_prices, zone, start, end, max_retries=MAX_RETRIES, retry_seconds=WRITE_RETRY_SECONDS)
            t = self.parse_balancing_prices_from_zip(xml, zone)
            self.store_table_idempotent(t, market=zone, schema=CURVE_SCHEMA, base=ENSTOE_BASE_DATA_DIR)
        except Exception as e:
            print(f"[WARN] Balancing prices {zone} {start}-{end} failed: {e}")

    def fetch_load_forecast(self, zone: str, start: pd.Timestamp, end: pd.Timestamp):
        try:
            xml = self._fetch_xml(self.client.query_load_forecast, zone, start, end, max_retries=MAX_RETRIES, retry_seconds=WRITE_RETRY_SECONDS)
            t = self.parse_load_forecast(xml)
            self.store_table_idempotent(t, market=zone, schema=CURVE_SCHEMA, base=ENSTOE_BASE_DATA_DIR)
        except Exception as e:
            print(f"[WARN] load forecast {zone} {start}-{end} failed: {e}")
        
    def fetch_generation_forecast(self, zone: str, start: pd.Timestamp, end: pd.Timestamp):
        try:
            xml = self._fetch_xml(self.client.query_generation_forecast, zone, start, end, max_retries=MAX_RETRIES, retry_seconds=WRITE_RETRY_SECONDS)
            t = self.parse_generation_forecast(xml)
            self.store_table_idempotent(t, market=zone, schema=CURVE_SCHEMA, base=ENSTOE_BASE_DATA_DIR)
        except Exception as e:
            print(f"[WARN] generation forecast {zone} {start}-{end} failed: {e}")

    def fetch_flows(self, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pa.Table:
        from_code = ENTSOE_ZONES_TO_CODES.get(market)
        flows_list = []
        for neighbour in NEIGHBOURS.get(market):
            to_code = ENTSOE_ZONES_TO_CODES.get(neighbour)
            try:
                xml = self.client.query_crossborder_flows(
                    country_code_from=from_code,
                    country_code_to=to_code,
                    start=start,
                    end=end
                )
                df = self.parse_crossborder_flows(xml)
                flows_list.append(df)

            except Exception as e:
                print(f"[WARN] flow from {market} to {neighbour} for {start}-{end} failed or unavailable: {e}")

        if flows_list:
            return pa.concat_tables(flows_list, ignore_index=True)
        
        return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

    def fetch_wind_solar_forecast(self, zone: str, start: pd.Timestamp, end: pd.Timestamp):
        try:
            xml = self._fetch_xml(self.client.query_wind_and_solar_forecast, zone, start, end, max_retries=MAX_RETRIES, retry_seconds=WRITE_RETRY_SECONDS)
            t = self.parse_wind_solar_forecast(xml)
            self.store_table_idempotent(t, market=zone, schema=CURVE_SCHEMA, base=ENSTOE_BASE_DATA_DIR) # TODO: split solar and wind
        except Exception as e:
            print(f"[WARN] wind/solar forecast {zone} {start}-{end} failed: {e}")

    def fetch_all_curves(self, market: str, start: pd.Timestamp, end: pd.Timestamp):
        self.fetch_day_ahead_prices(market, start, end)
        self.fetch_balancing_prices(market, start, end)
        self.fetch_load_forecast(market, start, end)
        self.fetch_generation_forecast(market, start, end)
        self.fetch_wind_solar_forecast(market, start, end)
        self.fetch_flows(market, start, end)

        return None

    def split_date_range(self, start: pd.Timestamp, end: pd.Timestamp, months: int = 3) -> list:
        """
        Helper: split date range into <=3 month chunks
        """
        periods = []
        current_start = start
        while current_start < end:
            current_end = min(current_start + DateOffset(months=months) - pd.Timedelta(seconds=1), end)
            periods.append((current_start, current_end))
            current_start = current_end + pd.Timedelta(seconds=1)
        return periods

    def fetch_market_chunk(self, market: str, start: pd.Timestamp, end: pd.Timestamp, fetch_fn: Callable = fetch_all_curves):
        """
        Wrapper to fetch a single market/time chunk
        """
        try:
            fetch_fn(self, market, start, end)
            print(f"[INFO] Completed {market} {start.date()} to {end.date()}")
        except Exception as e:
            print(f"[WARN] Failed {market} {start}-{end}: {e}")

    def generate_fetch_tasks(self, markets: list[str], start: pd.Timestamp, end: pd.Timestamp, fetch_fn: Callable = fetch_all_curves) -> list:
        """
        Generate tasks for all markets and all 3-month chunks
        """
        tasks = []
        for market in markets:
            for start_chunk, end_chunk in self.split_date_range(start, end, months=3):
                tasks.append((self.fetch_market_chunk, market, start_chunk, end_chunk, fetch_fn))
        return tasks

    def fetch_data_concurrent(self, markets: list[str], start: pd.Timestamp, end: pd.Timestamp, max_workers: int = 6, fetch_fn: Callable = fetch_all_curves):
        """
        Concurrent executor for tasks
        """
        tasks = self.generate_fetch_tasks(markets, start, end, fetch_fn=fetch_fn)
        print(f"[INFO] Total tasks: {len(tasks)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(func, *args): (args[0], args[1], args[2]) for func, *args in tasks}
            for future in as_completed(future_to_task):
                market, task_start, task_end = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[WARN] Task {market} {task_start}-{task_end} failed: {e}")

    