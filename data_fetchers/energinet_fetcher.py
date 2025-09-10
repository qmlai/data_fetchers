import json
import requests
import pandas as pd
import pyarrow as pa

from pathlib import Path
from typing import Optional, Dict, Any

from .base import DataFetcherBase
from .constants import ENERGINET_BASE_DATA_DIR, CURVE_SCHEMA, BALANCING_CURVE_SCHEMA

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

    def _to_da_table(self, data: dict, curve_type: str, market: str, unit: str = "MW", resolution: str = "PT60M") -> pa.Table:
        if data:
            return pa.Table.from_pydict({c.name: [] for c in CURVE_SCHEMA})

        data["market"] = market
        data["curve_type"] = curve_type
        data["neighbour"] = "N/A"
        data["production_type"] = "N/A"

        data["timestamp"] = pd.to_datetime(data["HourUTC"], utc=True)
        data["value"] = data["SpotPriceEUR"]

        data["resolution"] = resolution
        data["unit"] = unit
        data["source"] = "ENERGINET"
        data["year"] = data["timestamp"].dt.year.astype("int32")
        data["month"] = data["timestamp"].dt.month.astype("int32")
        data["day"] = data["timestamp"].dt.day.astype("int32")

        return pa.Table.from_pydict(data, schema=CURVE_SCHEMA, preserve_index=False)

    def _to_bm_table(self, data: dict, curve_type: str, market: str, resolution: str = "PT15M") -> pa.Table:
        if data:
            return pa.Table.from_pydict({c.name: [] for c in BALANCING_CURVE_SCHEMA})

        data["market"] = market
        data["curve_type"] = curve_type
        data["resolution"] = resolution

        data["timestamp"] = pd.to_datetime(data["TimeUTC"], utc=True)
        data["value"] = data["SpotPriceEUR"]
        data["balancing_demand"] = data["BalancingDemand"]
        data["direction"] = data["DominatingDirection"]

        data["aFRR_up"] = data["aFRRUpMW"]
        data["aFRR_up_vwa"] = data["aFRRVWAUpEUR"]
        data["aFRR_down"] = data["aFRRDownMW"]
        data["aFRR_down_vwa"] = data["aFRRVWADownEUR"]

        data["mFRR_marginal_price_up"] = data["mFRRMarginalPriceUpEUR"]
        data["mFRR_marginal_price_down"] = data["mFRRMarginalPriceDownEUR"]
        
        data["source"] = "ENERGINET"
        data["year"] = data["timestamp"].dt.year.astype("int32")
        data["month"] = data["timestamp"].dt.month.astype("int32")
        data["day"] = data["timestamp"].dt.day.astype("int32")

        return pa.Table.from_pydict(data, schema=BALANCING_CURVE_SCHEMA, preserve_index=False)

    # --------------------------------------------------------------
    #                           Fetchers
    # --------------------------------------------------------------

    def fetch_day_ahead_prices(self, market: str, start: str, end: str) -> pa.Table:
        try:
            data = self._fetch_dataset("Elspotprices", start, end, {"PriceArea": [market]})

            d = data.get("records", [])
            t = self._to_da_table(d, "day_ahead_price", market, unit="EUR/MWh")
            self.store_table_idempotent(t, market=market, schema=CURVE_SCHEMA, base=ENERGINET_BASE_DATA_DIR)
  
        except Exception as e:
            print(f"[WARN] Balancing prices {market} {start}-{end} failed: {e}")

    def fetch_balancing_prices(self, market: str, start: str, end: str) -> pa.Table:
        try:
            data = self._fetch_dataset("ImbalancePrice", start, end, {"PriceArea": [market]})

            d = data.get("records", [])
            t = self._to_bm_table(d, "imbalance_price", market)
            self.store_table_idempotent(t, market=market, schema=BALANCING_CURVE_SCHEMA, base=ENERGINET_BASE_DATA_DIR)
            
        except Exception as e:
            print(f"[WARN] Balancing prices {market} {start}-{end} failed: {e}")
    
    def fetch_all_curves(self, market: str, start: pd.Timestamp, end: pd.Timestamp):
        self.fetch_day_ahead_prices(market, start, end)
        self.fetch_balancing_prices(market, start, end)
        
        return None
