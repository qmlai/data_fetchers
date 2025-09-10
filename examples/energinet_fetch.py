import pandas as pd

from data_fetchers import EnerginetDataFetcher, ENERGINET_BASE_DATA_DIR

if __name__ == "__main__":

    fetcher = EnerginetDataFetcher()
    
    markets = "DK1"
    start   = pd.Timestamp(2020, 7, 30, tz="UTC").date().strftime("%Y-%m-%d")
    end     = pd.Timestamp(2025, 8, 30, tz="UTC").date().strftime("%Y-%m-%d")
    
    fetcher.fetch_all_curves(market=markets, start=start, end=end)
    fetcher.load_curves(base_dir=ENERGINET_BASE_DATA_DIR, market="DK_1")