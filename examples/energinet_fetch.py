import pandas as pd

from data_fetchers import EnerginetDataFetcher, ENERGINET_BASE_DATA_DIR

if __name__ == "__main__":

    fetcher = EnerginetDataFetcher()
    
    zone  = "DK2"
    start = pd.Timestamp(2020, 1, 1, tz="UTC")
    end   = pd.Timestamp(2025, 8, 31, tz="UTC")

    fetcher.fetch_all_curves(market=zone, start=start, end=end)
    df = fetcher.load_curves(base_dir=ENERGINET_BASE_DATA_DIR, market="DK2")
    print(df.head())
