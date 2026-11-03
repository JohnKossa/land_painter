"""Input files: cheap inspection (schema, row count) and column reads.

Supports Parquet/GeoParquet, CSV/TSV, and OGR formats (GeoPackage, shapefile,
GeoJSON, FlatGeobuf). Nothing here assumes particular column names: callers
read only the columns the user mapped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pyogrio

PARQUET = {".parquet", ".geoparquet"}
DELIMITED = {".csv": ",", ".txt": ",", ".tsv": "\t"}
OGR = {".gpkg", ".shp", ".geojson", ".json", ".fgb"}
SUPPORTED = PARQUET | set(DELIMITED) | OGR
FILE_FILTER = "Data files (" + " ".join(f"*{s}" for s in sorted(SUPPORTED)) + ");;All files (*)"

# Column kinds, derived from storage type only.
NUMBER, INTEGER, TEXT, BOOLEAN, DATE, BINARY, OTHER = (
    "number", "integer", "text", "boolean", "date", "binary", "other")
NUMERIC = (NUMBER, INTEGER)


def kind_of(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean:
        return BOOLEAN
    if dtype.is_integer():
        return INTEGER
    if dtype.is_float() or dtype.is_decimal():
        return NUMBER
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
        return TEXT
    if dtype.is_temporal() and dtype != pl.Time and dtype != pl.Duration:
        return DATE
    if dtype == pl.Binary:
        return BINARY
    return OTHER


@dataclass
class SourceInfo:
    path: Path
    n_rows: int
    columns: dict[str, str]  # name -> kind, excluding geometry columns
    geometry_column: str | None

    @property
    def name(self) -> str:
        return self.path.name


def _hidden(column: str) -> bool:
    return column.startswith("__index_level_")  # pandas index artifacts


def inspect(path: str | Path) -> SourceInfo:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PARQUET:
        meta = pq.read_metadata(path)
        geo = json.loads((meta.metadata or {}).get(b"geo", b"{}") or b"{}")
        geometry_columns = set(geo.get("columns", {}))
        schema = pl.read_parquet_schema(path)
        columns = {c: kind_of(t) for c, t in schema.items() if c not in geometry_columns and not _hidden(c)}
        return SourceInfo(path, meta.num_rows, columns, geo.get("primary_column") if geometry_columns else None)
    if suffix in DELIMITED:
        scan = pl.scan_csv(path, separator=DELIMITED[suffix], infer_schema_length=10_000)
        columns = {c: kind_of(t) for c, t in scan.collect_schema().items()}
        return SourceInfo(path, scan.select(pl.len()).collect().item(), columns, None)
    if suffix in OGR:
        info = pyogrio.read_info(path)
        sample = pl.from_pandas(pyogrio.read_dataframe(path, read_geometry=False, max_features=1))
        columns = {c: kind_of(t) for c, t in sample.schema.items()}
        return SourceInfo(path, int(info["features"]), columns, "geometry" if info.get("geometry_type") else None)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def read_columns(path: str | Path, columns: list[str]) -> pl.DataFrame:
    """Read only the given (non-geometry) columns."""
    path = Path(path)
    columns = list(dict.fromkeys(columns))
    suffix = path.suffix.lower()
    if suffix in PARQUET:
        return pl.read_parquet(path, columns=columns)
    if suffix in DELIMITED:
        return pl.read_csv(path, columns=columns, separator=DELIMITED[suffix], infer_schema_length=10_000)
    if suffix in OGR:
        return pl.from_pandas(pyogrio.read_dataframe(path, columns=columns, read_geometry=False))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def scan(path: str | Path) -> pl.LazyFrame:
    """Lazy access to all columns, for whole-file profiling."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PARQUET:
        return pl.scan_parquet(path)
    if suffix in DELIMITED:
        return pl.scan_csv(path, separator=DELIMITED[suffix], infer_schema_length=10_000)
    return pl.from_pandas(pyogrio.read_dataframe(path, read_geometry=False)).lazy()
