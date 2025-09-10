import os
import pandas as pd

from dotenv import load_dotenv
from data_fetchers import EntsoeDataFetcher, ENSTOE_BASE_DATA_DIR

if __name__ == "__main__":
    load_dotenv()
    ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
    
    fetcher = EntsoeDataFetcher(api_key=ENTSOE_API_KEY)
    
    markets = ["DK_1"]
    start   = pd.Timestamp(2025, 8, 30, tz="UTC")
    end     = pd.Timestamp(2025, 8, 31, tz="UTC")
    
    fetcher.fetch_data_concurrent(markets, start, end, max_workers=6)
    df = fetcher.load_curves(base_dir=ENSTOE_BASE_DATA_DIR, market="DK_1")

    