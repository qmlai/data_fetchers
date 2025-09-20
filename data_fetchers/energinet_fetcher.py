import json
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import DataFetcherBase
from .constants import (ENERGINET_BASE_DATA_DIR, CURVE_SCHEMA, 
                        BALANCING_CURVE_SCHEMA, REGULATING_CUTOFF_DATE)

class EnerginetDataFetcher(DataFetcherBase):
    BASE_URL = "https://api.energidataservice.dk/dataset/"

    def __init__(self, cache: bool = True):
        self.cache = cache

    def _cache_path(self, name: str, start: str, end: str, filter_obj: Optional[Dict[str, Any]]) -> Path:
        key = f"{name}_{start}_{end}_{json.dumps(filter_obj, sort_keys=True)}.parquet"
        return ENERGINET_BASE_DATA_DIR / key.replace("/", "_")

    def _fetch_dataset(self, name: str, start: str, end: str, filter_obj: Optional[Dict[str, Any]] = None) -> dict:
        params = {"start": start, "end": end}
        if filter_obj:
            params["filter"] = json.dumps(filter_obj)

        response = requests.get(f"{self.BASE_URL}{name}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def _to_da_table(self, data: List[dict], curve_type: str, market: str, unit: str = "MW", resolution: str = "PT60M") -> pa.Table:
        if not data:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        table = pa.Table.from_pylist(data)

        num_rows = len(table)
        table = table.append_column("market", pa.array([market] * num_rows))
        table = table.append_column("curve_type", pa.array([curve_type] * num_rows))
        table = table.append_column("resolution", pa.array([resolution] * num_rows))
        table = table.append_column("source", pa.array(["ENERGINET"] * num_rows))
        table = table.append_column("neighbour", pa.array(["N/A"] * num_rows))
        table = table.append_column("production_type", pa.array(["N/A"] * num_rows))
        table = table.append_column("unit", pa.array([unit] * num_rows))
        
        ts = pc.cast(pc.strptime(table["HourUTC"], format="%Y-%m-%dT%H:%M:%S", unit="ns"),
             pa.timestamp("ns", tz="UTC"))
        
        table = table.append_column("timestamp", ts)        
        table = table.append_column("year", pc.year(ts))
        table = table.append_column("month", pc.month(ts))
        table = table.append_column("day", pc.day(ts))

        table = table.rename_columns({
            "SpotPriceEUR": "value"
        })

        table = table.drop_columns(["HourUTC", "HourDK", "PriceArea", "SpotPriceDKK"])

        return table.select([f.name for f in CURVE_SCHEMA])

    def _to_bm_table(self, data: List[dict], curve_type: str, market: str, resolution: str = "PT15M") -> pa.Table:
        if not data:
            return pa.Table.from_pydict({c.name: [] for c in BALANCING_CURVE_SCHEMA})

        table = pa.Table.from_pylist(data)

        num_rows = len(table)
        table = table.append_column("market", pa.array([market] * num_rows))
        table = table.append_column("curve_type", pa.array([curve_type] * num_rows))
        table = table.append_column("resolution", pa.array([resolution] * num_rows))
        table = table.append_column("source", pa.array(["ENERGINET"] * num_rows))
        
        ts = pc.cast(pc.strptime(table["TimeUTC"], format="%Y-%m-%dT%H:%M:%S", unit="ns"),
             pa.timestamp("ns", tz="UTC"))
        
        table = table.append_column("timestamp", ts)        
        table = table.append_column("year", pc.year(ts))
        table = table.append_column("month", pc.month(ts))
        table = table.append_column("day", pc.day(ts))

        table = table.rename_columns({
            "ImbalancePriceEUR": "value",
            "BalancingDemand": "balancing_demand",
            "DominatingDirection": "direction",
            "aFRRUpMW": "aFRR_up",
            "aFRRVWAUpEUR": "aFRR_up_vwa",
            "aFRRDownMW": "aFRR_down",
            "aFRRVWADownEUR": "aFRR_down_vwa",
            "mFRRMarginalPriceUpEUR": "mFRR_marginal_price_up",
            "mFRRMarginalPriceDownEUR": "mFRR_marginal_price_down"
        })

        table = table.drop_columns(["TimeUTC", "TimeDK", "PriceArea", 
                                    "ImbalancePriceDKK", "SpotPriceEUR", 
                                    "aFRRVWAUpDKK", "aFRRVWADownDKK", "mFRRMarginalPriceDownDKK", 
                                    "mFRRMarginalPriceUpDKK"])

        return table.select([f.name for f in BALANCING_CURVE_SCHEMA])

    def _to_bm_table_old(self, data: List[dict], curve_type: str, market: str, resolution: str = "PT1H") -> pa.Table:
        if not data:
            return pa.Table.from_pydict({c.name: [] for c in BALANCING_CURVE_SCHEMA})

        table = pa.Table.from_pylist(data)
        
        num_rows = len(table)
        table = table.append_column("market", pa.array([market] * num_rows))
        table = table.append_column("curve_type", pa.array([curve_type] * num_rows))
        table = table.append_column("resolution", pa.array([resolution] * num_rows))
        table = table.append_column("source", pa.array(["ENERGINET"] * num_rows))
        
        ts = pc.cast(pc.strptime(table["HourUTC"], format="%Y-%m-%dT%H:%M:%S", unit="ns"),
             pa.timestamp("ns", tz="UTC"))
        
        table = table.append_column("timestamp", ts)        
        table = table.append_column("year", pc.year(ts))
        table = table.append_column("month", pc.month(ts))
        table = table.append_column("day", pc.day(ts))
        table = table.append_column("balancing_demand", pa.array([0.0] * num_rows))
        table = table.append_column("direction", pa.array([0] * num_rows))
        table = table.append_column("aFRR_up", pa.array([0.0] * num_rows))
        table = table.append_column("aFRR_up_vwa", pa.array([0.0] * num_rows))
        table = table.append_column("aFRR_down", pa.array([0.0] * num_rows))
        table = table.append_column("aFRR_down_vwa", pa.array([0.0] * num_rows))
        table = table.append_column("mFRR_marginal_price_up", pa.array([0.0] * num_rows))
        table = table.append_column("mFRR_marginal_price_down", pa.array([0.0] * num_rows))

        table = table.rename_columns({
            "ImbalancePriceEUR": "value",
        })

        table = table.drop_columns(["HourUTC", "HourDK", "PriceArea", 
                                    "mFRRUpActBal", "mFRRDownActBal", 
                                    "mFRRUpActSpec", "mFRRDownActSpec", 
                                    "ImbalanceMWh",
                                    "ImbalancePriceDKK", "BalancingPowerPriceUpEUR", 
                                    "BalancingPowerPriceUpDKK", "BalancingPowerPriceDownEUR", 
                                    "BalancingPowerPriceDownDKK"])

        return table.select([f.name for f in BALANCING_CURVE_SCHEMA])

    # --------------------------------------------------------------
    #                           Fetchers
    # --------------------------------------------------------------

    def fetch_day_ahead_prices(self, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pa.Table:
        try:
            data = self._fetch_dataset("Elspotprices", start.date().strftime("%Y-%m-%d"), 
                                       end.date().strftime("%Y-%m-%d"), 
                                       {"PriceArea": [market]})
            
            d = data.get("records", [])
            t = self._to_da_table(d, "day_ahead_price", market, unit="EUR/MWh")
            self.store_table_idempotent(t, market=market, schema=CURVE_SCHEMA, base=ENERGINET_BASE_DATA_DIR)
  
        except Exception as e:
            print(f"[WARN] Day Ahead prices {market} {start}-{end} failed: {e}")

    def fetch_balancing_prices(self, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pa.Table:
        # Split the interval into old/new datasets based on cutoff
        old_start, old_end = start, min(end, REGULATING_CUTOFF_DATE)
        new_start, new_end = max(start, REGULATING_CUTOFF_DATE), end

        # Fetch old dataset if needed
        tables = []
        if old_start < REGULATING_CUTOFF_DATE:
            try:
                data_old = self._fetch_dataset("RegulatingBalancePowerdata",
                                               old_start.date().strftime("%Y-%m-%d"), 
                                               old_end.date().strftime("%Y-%m-%d"),
                                               {"PriceArea": [market]})
                
                d_old = data_old.get("records", [])
                t_old = self._to_bm_table_old(d_old, "imbalance_price", market)
                tables.append(t_old)
            except Exception as e:
                print(f"[WARN] RegulatingBalancePowerdata {market} {old_start}-{old_end} failed: {e}")

        # Fetch new dataset if needed
        if new_start < end:
            try:
                data_new = self._fetch_dataset("ImbalancePrice",
                                               new_start.date().strftime("%Y-%m-%d"), 
                                               new_end.date().strftime("%Y-%m-%d"),
                                               {"PriceArea": [market]})
                
                d_new = data_new.get("records", [])
                t_new = self._to_bm_table(d_new, "imbalance_price", market)
                tables.append(t_new)
            except Exception as e:
                print(f"[WARN] ImbalancePrice {market} {new_start}-{new_end} failed: {e}")

        # Concatenate tables if both exist
        if not tables:
            return pa.Table.from_pydict({c.name: [] for c in BALANCING_CURVE_SCHEMA})

        final_table = pa.concat_tables(tables)

        # Store idempotently
        self.store_table_idempotent(final_table, market=market,
                                    schema=BALANCING_CURVE_SCHEMA,
                                    base=ENERGINET_BASE_DATA_DIR)

        return final_table
    
    def fetch_all_curves(self, market: str, start: pd.Timestamp, end: pd.Timestamp):
        self.fetch_day_ahead_prices(market, start, end)
        self.fetch_balancing_prices(market, start, end)
        
        return None
