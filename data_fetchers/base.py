import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

from pathlib import Path
from functools import reduce

class DataFetcherBase:
    def __init__(self):
        pass

    def _partition_dir(self, base: Path, market: str, curve_type: str, year: int, month: int, day: int) -> Path:
        return base / f"market={market}" / f"curve_type={curve_type}" / f"year={year:04d}" / f"month={month:02d}" / f"day={day:02d}"

    def ensure_dir(self, p: Path):
        p.mkdir(parents=True, exist_ok=True)

    def store_table_idempotent(self, table: pa.Table, market: str, schema: pa.schema, base: Path):
        """
        Write a pyarrow Table into the dataset partitioned by market/curve_type/year/month/day.
        If the partition already exists, read existing data, merge, deduplicate and overwrite.
        """
        if table.num_rows == 0:
            return

        # Convert to pandas for easy groupby / dedupe (per-day tables are small)
        df = table.to_pandas()
        
        # ensure timestamp tz-aware
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # ensure necessary columns exist
        table_cols = schema.names
        for col in table_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column {col} in table")

        # group by partitions
        grp_cols = ["market", "curve_type", "year", "month", "day"]
        for (market, curve_type, year, month, day), g in df.groupby(grp_cols):
            part_dir = self._partition_dir(base, market, curve_type, int(year), int(month), int(day))
            self.ensure_dir(part_dir)
            out_file = part_dir / "data.parquet"

            if out_file.exists():
                try:
                    existing = pd.read_parquet(out_file)
                    # normalize timestamp
                    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True) if existing["timestamp"].dt.tz is None else pd.to_datetime(existing["timestamp"], utc=True)
                    combined = pd.concat([existing, g], ignore_index=True)
                except Exception:
                    combined = g.copy()
            else:
                combined = g.copy()

            subset_cols = ["market", "curve_type", "timestamp"]
            if "production_type" in table_cols:
                subset_cols.append("production_type")
            elif "neighbour" in table_cols:
                subset_cols.append("neighbour")

            # dedupe on (market, curve_type, production_type, neighbour, timestamp)
            combined = combined.drop_duplicates(subset=subset_cols, keep="last")
            combined = combined.sort_values(by="timestamp")

            # write combined back (overwrite)
            # remove older files in partition dir
            for f in part_dir.glob("*.parquet"):
                try:
                    f.unlink()
                except Exception:
                    pass

            out_table = pa.Table.from_pandas(combined, schema=schema, preserve_index=False)
            pq.write_table(out_table, out_file)

    def load_curves(self, base_dir: Path, *, market: str | None = None, curve_type: str | None = None, production_type: str | None = None, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        dataset = ds.dataset(base_dir, format="parquet", partitioning="hive")
        filters = []

        table_cols = dataset.schema.names
        
        if market:
            filters.append(pc.field("market") == market)
        if curve_type:
            filters.append(pc.field("curve_type") == curve_type)
            
        if production_type and "production_type" in table_cols:
            filters.append(pc.field("production_type") == production_type)
        if start:
            start_ts = pd.to_datetime(start, utc=True)
            filters.append(pc.field("timestamp") >= start_ts)
        if end:
            end_ts = pd.to_datetime(end, utc=True)
            filters.append(pc.field("timestamp") <= end_ts)

        if filters:
            filter_expr = reduce(lambda a, b: a & b, filters)
            table = dataset.to_table(filter=filter_expr)
        else:
            table = dataset.to_table()

        return table.to_pandas()

