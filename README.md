## data_fetchers
**data_fetchers** is a Python utility for downloading, parsing, and storing ENTSO-E and ENERGINET market data (day-ahead prices, balancing, load forecasts, generation forecasts, wind/solar, and cross-border flows).
It stores results as partitioned Parquet datasets (market/curve_type/year/month/day) for efficient downstream analysis.

## Features
- Fetch data from ENTSO-E and ENERGINET APIs with retries & concurrent execution.
- Parse XML/ZIP responses into normalized PyArrow tables compliant with a shared schema (CURVE_SCHEMA).
- Store data idempotently (deduplicated, partitioned by Hive-style directory structure).
- Load filtered datasets efficiently via pyarrow.dataset.

## Supported curves:
- Day-ahead prices
- Balancing (imbalance) prices
- Load forecasts
- Generation forecasts
- Intraday wind/solar forecasts
- Cross-border flows

## Requirements
- Python ≥ 3.10
- Poetry for dependency management
- ENTSO-E API key (free registration required)

## Installation
Clone the repository and install dependencies with Poetry:
```bash
git clone https://github.com/qmlai/data-fetchers.git
cd data-fetchers
```

```bash
poetry install
```

## Configuration
Create a .env file in the project root:

```
ENTSOE_API_KEY=your_api_key_here
```

## Example Usage
```python
import os
import pandas as pd
from dotenv import load_dotenv
from data_fetchers import EntsoeDataFetcher, BASE_DATA_DIR

if __name__ == "__main__":
    # Load API key
    load_dotenv()
    ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

    # Initialize fetcher
    fetcher = EntsoeDataFetcher(api_key=ENTSOE_API_KEY)

    # Example: Fetch day-ahead and other curves for DK_1 on Aug 30–31, 2025
    markets = ["DK_1"]
    start   = pd.Timestamp(2025, 8, 30, tz="UTC")
    end     = pd.Timestamp(2025, 8, 31, tz="UTC")

    fetcher.fetch_data_concurrent(markets, start, end, max_workers=6)

    # Load back stored parquet dataset
    df = fetcher.load_curves(BASE_DATA_DIR, market="DK_1")
    print(df.head())
```

## Data Storage Layout
Data is written to a Hive-partitioned dataset:

```
BASE_DATA_DIR/
  market=DK_1/
    curve_type=day_ahead_price/
      year=2025/month=08/day=30/data.parquet
      year=2025/month=08/day=31/data.parquet
    curve_type=load_forecast/
    curve_type=generation_forecast/
    ...
```

Each partition is deduplicated by (market, curve_type, production_type, neighbour, timestamp).

## Running in Poetry
Run scripts inside the Poetry environment:

```bash
poetry run python examples/entsoe_fetch.py
```

Or start a shell inside the virtual environment:

```bash
poetry shell
```

```bash
python examples/entsoe_fetch.py
```
