"""Reading mapped columns and reviewing them before a session is built.

``prepare`` reads just the mapped columns, converts them to the types their
roles need, aligns parcel-level fields to the parcel file's keys, and reports
key-matching problems. ``numeric_stats``/``clean_numeric`` flag and (only if
the user chooses) clean suspect values. Nothing is fixed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .config import (
    CATEGORICAL,
    CONTINUOUS,
    DEPRECIATED_COST,
    FILL_MEDIAN,
    FILL_VALUE,
    FILL_ZERO,
    FLAG,
    NEIGHBORHOOD,
    REPLACEMENT_COST,
    ROLL_BUILDING,
    ROLL_LAND,
    UNUSED,
    WHEN,
    Cleaning,
    ColumnRef,
    ImportConfig,
    unit_field,
)
from .sources import BOOLEAN, DATE, NUMERIC, TEXT, SourceInfo, read_columns, scan

ERROR, WARNING, INFO = "error", "warning", "info"

# Values commonly used as "no data" markers in numeric columns.
PLACEHOLDERS = (-2147483648.0, -2147483647.0, 2147483647.0, 4294967295.0,
                -9999.0, -99999.0, -999999.0, 99999999.0, 999999999.0, 9999999999.0)
OUTLIER_K = 3.0  # IQR multiplier, on log10 of positive values

DATE_FORMATS = {
    "%Y-%m-%d": "2024-01-31",
    "%Y-%m-%d %H:%M:%S": "2024-01-31 13:45:00",
    "%Y-%m-%dT%H:%M:%S": "2024-01-31T13:45:00",
    "%m/%d/%Y": "01/31/2024 (month first)",
    "%d/%m/%Y": "31/01/2024 (day first)",
    "%m/%d/%Y %H:%M": "01/31/2024 13:45 (month first)",
    "%d/%m/%Y %H:%M": "31/01/2024 13:45 (day first)",
    "%m/%d/%y": "01/31/24 (month first)",
    "%d/%m/%y": "31/01/24 (day first)",
    "%Y%m%d": "20240131",
}


@dataclass
class Issue:
    severity: str
    message: str


class DataError(Exception):
    """The data cannot be used as configured (e.g. duplicate parcel keys)."""


# --- Conversion -------------------------------------------------------------------


def to_number(s: pl.Series) -> tuple[pl.Series, int]:
    """Float64 version of ``s`` and the count of values that could not be converted."""
    if s.dtype.is_numeric() or s.dtype == pl.Boolean:
        out = s.cast(pl.Float64)
    else:
        text = s.cast(pl.String).str.strip_chars().str.replace_all(r"[,$]", "")
        out = text.cast(pl.Float64, strict=False)
    out = out.fill_nan(None)
    return out.alias(s.name), int((out.is_null() & s.is_not_null()).sum())


@dataclass
class DateParse:
    values: pl.Series  # Datetime
    unparseable: int
    format: str | None  # format used; None for columns already stored as dates
    alternatives: list[str]  # other formats that parse equally well but give different dates


def _parse_dates(text: pl.Series, fmt: str) -> pl.Series:
    try:
        return text.str.to_datetime(fmt, strict=False, time_unit="us")
    except pl.exceptions.PolarsError:
        return pl.Series(text.name, [None] * len(text), dtype=pl.Datetime("us"))


def parse_dates(s: pl.Series, fmt: str | None = None) -> DateParse:
    """Convert ``s`` to datetimes using ``fmt``, or the best-fitting known format.

    When detecting, formats that parse the same number of values but disagree
    (e.g. 01/02/2024 as month-first or day-first) are reported as
    ``alternatives`` so the user can choose.
    """
    if s.dtype.is_temporal():
        return DateParse(s.cast(pl.Datetime("us")).alias(s.name), 0, None, [])
    text = s.cast(pl.String).str.strip_chars()
    present = int(text.is_not_null().sum())
    if fmt is not None:
        out = _parse_dates(text, fmt)
        return DateParse(out.alias(s.name), present - int(out.is_not_null().sum()), fmt, [])
    results = {f: _parse_dates(text, f) for f in DATE_FORMATS}
    lost = {f: present - int(out.is_not_null().sum()) for f, out in results.items()}
    best = min(lost, key=lost.get)
    alternatives = [f for f in results if f != best and lost[f] == lost[best] and not results[f].equals(results[best])]
    return DateParse(results[best].alias(s.name), lost[best], best, alternatives)


def to_date(s: pl.Series, fmt: str | None = None) -> tuple[pl.Series, int]:
    """Datetime version of ``s`` and the count of values that could not be converted."""
    parsed = parse_dates(s, fmt)
    return parsed.values, parsed.unparseable


def as_text(s: pl.Series) -> pl.Series:
    return s.cast(pl.String)


def vacant_flags(s: pl.Series, vacant_values: list[str]) -> pl.Series:
    """True where ``s`` (as text, case-insensitive) is one of ``vacant_values``; null where ``s`` is null."""
    text = as_text(s).str.to_lowercase()
    wanted = [v.lower() for v in vacant_values]
    return pl.select(pl.when(text.is_null()).then(None).otherwise(text.is_in(wanted))).to_series().alias(s.name)


def distinct_values(s: pl.Series, limit: int = 50) -> list[str]:
    """Most common values of ``s`` as text, most frequent first."""
    counts = as_text(s).drop_nulls().value_counts(sort=True)
    return counts[:, 0].head(limit).to_list()


# --- Numeric review ------------------------------------------------------------------


@dataclass
class NumericStats:
    count: int
    missing: int
    unparseable: int
    zero: int
    negative: int  # excludes placeholders
    placeholder: int
    outlier: int
    low_fence: float | None
    high_fence: float | None
    minimum: float | None
    median: float | None
    maximum: float | None


def _masks(values: pl.Series, low: float | None, high: float | None) -> dict[str, pl.Series]:
    placeholder = values.is_in(list(PLACEHOLDERS)).fill_null(False)
    positive = (values > 0).fill_null(False) & ~placeholder
    outlier = pl.Series([False] * len(values))
    if low is not None:
        outlier = positive & ((values < low) | (values > high)).fill_null(False)
    return {
        "zero": (values == 0).fill_null(False),
        "negative": (values < 0).fill_null(False) & ~placeholder,
        "placeholder": placeholder,
        "outlier": outlier,
    }


def _fences(values: pl.Series) -> tuple[float | None, float | None]:
    positive = values.filter((values > 0) & ~values.is_in(list(PLACEHOLDERS))).to_numpy()
    if positive.size < 4:
        return None, None
    logs = np.log10(positive)
    q1, q3 = np.percentile(logs, [25, 75])
    iqr = q3 - q1
    return float(10 ** (q1 - OUTLIER_K * iqr)), float(10 ** (q3 + OUTLIER_K * iqr))


def numeric_stats(values: pl.Series, unparseable: int = 0) -> NumericStats:
    low, high = _fences(values)
    masks = _masks(values, low, high)
    valid = values.filter(values.is_not_null() & ~masks["placeholder"])
    return NumericStats(
        count=len(values),
        missing=int(values.is_null().sum()),
        unparseable=unparseable,
        zero=int(masks["zero"].sum()),
        negative=int(masks["negative"].sum()),
        placeholder=int(masks["placeholder"].sum()),
        outlier=int(masks["outlier"].sum()),
        low_fence=low,
        high_fence=high,
        minimum=valid.min() if len(valid) else None,
        median=valid.median() if len(valid) else None,
        maximum=valid.max() if len(valid) else None,
    )


def clean_numeric(values: pl.Series, stats: NumericStats, cleaning: Cleaning) -> pl.Series:
    """Apply the user's choices: treat flagged values as missing, then fill missing."""
    masks = _masks(values, stats.low_fence, stats.high_fence)
    drop = pl.Series([False] * len(values))
    for rule in cleaning.missing:
        drop = drop | masks[rule]
    out = pl.select(pl.when(drop).then(None).otherwise(values)).to_series().alias(values.name)
    if cleaning.fill == FILL_ZERO:
        out = out.fill_null(0.0)
    elif cleaning.fill == FILL_MEDIAN and out.drop_nulls().len():
        out = out.fill_null(out.median())
    elif cleaning.fill == FILL_VALUE and cleaning.value is not None:
        out = out.fill_null(float(cleaning.value))
    return out


# --- Preparing mapped data ---------------------------------------------------------------


@dataclass
class Prepared:
    parcel_keys: pl.Series  # String, parcel file order (null keys dropped)
    parcels: pl.DataFrame  # "key" + one column per parcel-level field id, aligned to parcel_keys
    sales: pl.DataFrame | None  # key, price, date, vacant
    unparseable: dict[str, int] = field(default_factory=dict)  # field id -> count
    issues: list[Issue] = field(default_factory=list)
    date_format: str | None = None  # format used for text sale dates

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]


def parcel_fields(config: ImportConfig) -> dict[str, list[ColumnRef]]:
    """Parcel-level field id -> source columns (summed when there are several)."""
    fields: dict[str, list[ColumnRef]] = {}
    if config.replacement_cost:
        fields[REPLACEMENT_COST] = config.replacement_cost
    if config.depreciated_cost:
        fields[DEPRECIATED_COST] = config.depreciated_cost
    if config.roll_building_value:
        fields[ROLL_BUILDING] = config.roll_building_value
    if config.roll_land_value:
        fields[ROLL_LAND] = config.roll_land_value
    for name, ref in config.units.items():
        fields[unit_field(name)] = [ref]
    return fields


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _prepare_keys(config: ImportConfig, issues: list[Issue]) -> tuple[pl.Series, pl.Series]:
    """Parcel keys (nulls dropped) and their imploded set; flags missing and duplicate keys."""
    keys = as_text(read_columns(config.parcel_file, [config.parcel_key]).to_series())
    if n := int(keys.is_null().sum()):
        issues.append(Issue(WARNING, f"{_plural(n, 'parcel')} in the parcel file have no key and will be skipped."))
    keys = keys.drop_nulls()
    if n := int(keys.is_duplicated().sum()):
        issues.append(Issue(ERROR, f"{_plural(n, 'parcel row')} share a key with another row. "
                                   "Parcel keys must be unique; choose a different key column or fix the file."))
    return keys, keys.implode()


def _reconcile_secondary(frame: pl.DataFrame, path: str, keys: pl.Series, key_set: pl.Series,
                         issues: list[Issue]) -> pl.DataFrame:
    """Drop null keys and flag suspect joins (repeated keys, keys/parcels with no match) on a non-parcel file."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    frame = frame.drop_nulls("key")
    if n := frame.height - frame["key"].n_unique():
        issues.append(Issue(WARNING,
            f"{name}: {_plural(n, 'row')} repeat a key; the first row for each key is used."))
        frame = frame.unique("key", keep="first", maintain_order=True)
    if n := int((~frame["key"].is_in(key_set)).sum()):
        issues.append(Issue(WARNING,
            f"{name}: {_plural(n, 'key')} not found in the parcel file; those rows are ignored."))
    if n := int((~keys.is_in(frame["key"].implode())).sum()):
        issues.append(Issue(WARNING,
            f"{name}: {_plural(n, 'parcel')} have no row in this file; their values are missing."))
    return frame


def _join_parcel_files(config: ImportConfig, keys: pl.Series, key_set: pl.Series,
                       issues: list[Issue]) -> pl.DataFrame:
    """Join every mapped parcel-level column (plus the neighborhood column) onto the parcel keys.

    Columns are renamed to ``path::column`` so same-named columns from different files don't collide.
    """
    by_file: dict[str, set[str]] = {}
    for refs in parcel_fields(config).values():
        for ref in refs:
            by_file.setdefault(ref.file, set()).add(ref.column)
    if config.neighborhood:
        by_file.setdefault(config.neighborhood.file, set()).add(config.neighborhood.column)

    base = pl.DataFrame({"key": keys})
    for path, columns in by_file.items():
        key_column = config.key_for(path)
        frame = read_columns(path, [key_column, *sorted(columns)])
        frame = frame.rename({c: f"{path}::{c}" for c in columns}).with_columns(
            key=as_text(frame[key_column])).select("key", *[f"{path}::{c}" for c in columns])
        if path != config.parcel_file:
            frame = _reconcile_secondary(frame, path, keys, key_set, issues)
        base = base.join(frame, on="key", how="left", maintain_order="left")
    return base


def _assemble_fields(config: ImportConfig, base: pl.DataFrame, unparseable: dict[str, int]) -> pl.DataFrame:
    """Build the parcels frame: one numeric column per field id (summed when several), plus neighborhood."""
    columns = {"key": base["key"]}
    for field_id, refs in parcel_fields(config).items():
        parts = []
        lost = 0
        for ref in refs:
            values, n = to_number(base[f"{ref.file}::{ref.column}"])
            parts.append(values)
            lost += n
        unparseable[field_id] = lost
        if len(parts) == 1:
            columns[field_id] = parts[0].alias(field_id)
        else:  # sum, ignoring missing components unless all are missing
            frame = pl.DataFrame(parts)
            columns[field_id] = frame.select(
                pl.when(pl.all_horizontal(pl.all().is_null())).then(None)
                .otherwise(pl.sum_horizontal(pl.all())).alias(field_id)).to_series()
    if config.neighborhood:
        ref = config.neighborhood
        columns[NEIGHBORHOOD] = as_text(base[f"{ref.file}::{ref.column}"]).alias(NEIGHBORHOOD)
    return pl.DataFrame(columns)


def _prepare_sales(config: ImportConfig, key_set: pl.Series, unparseable: dict[str, int],
                   issues: list[Issue]) -> tuple[pl.DataFrame | None, str | None]:
    """The sales frame (key, price, date, vacant) and the date format used; None when no sales are mapped."""
    if not (config.sales_file and config.sales):
        return None, None
    m = config.sales
    raw = read_columns(config.sales_file, [m.key, m.price, m.date, m.vacant])
    price, unparseable["sale_price"] = to_number(raw[m.price])
    dates = parse_dates(raw[m.date], m.date_format)
    unparseable["sale_date"] = dates.unparseable
    if dates.alternatives:
        options = " or ".join(DATE_FORMATS[f] for f in [dates.format, *dates.alternatives])
        issues.append(Issue(WARNING, f"Sale dates are ambiguous (could be {options}). "
                                     "Check the date format on the Mapped fields tab."))
    sales = pl.DataFrame({
        "key": as_text(raw[m.key]),
        "price": price,
        "date": dates.values,
        "vacant": vacant_flags(raw[m.vacant], m.vacant_values),
    })
    if n := int(sales["key"].is_null().sum()):
        issues.append(Issue(WARNING, f"{_plural(n, 'sale')} have no parcel key; they are excluded."))
    if n := int((~sales["key"].is_in(key_set) & sales["key"].is_not_null()).sum()):
        issues.append(Issue(WARNING,
            f"{_plural(n, 'sale')} have keys not found in the parcel file; they are excluded."))
    if n := int(sales["vacant"].is_null().sum()):
        issues.append(Issue(WARNING,
            f"{_plural(n, 'sale')} have no vacant/improved value; they are excluded from metrics."))
    n_vacant = int(sales["vacant"].sum())
    issues.append(Issue(INFO, f"{_plural(n_vacant, 'vacant sale')} and "
                              f"{_plural(int((~sales['vacant']).sum()), 'improved sale')}."))
    return sales, dates.format


def prepare(config: ImportConfig) -> Prepared:
    issues: list[Issue] = []
    unparseable: dict[str, int] = {}
    keys, key_set = _prepare_keys(config, issues)
    base = _join_parcel_files(config, keys, key_set, issues)
    parcels = _assemble_fields(config, base, unparseable)
    sales, date_format = _prepare_sales(config, key_set, unparseable, issues)
    return Prepared(keys, parcels, sales, unparseable, issues, date_format)


# --- Characteristics -----------------------------------------------------------------


@dataclass
class ColumnProfile:
    ref: ColumnRef
    storage: str  # kind from sources
    distinct: int
    missing: int
    rows: int
    suggested: str  # characteristic kind
    review: str | None  # why the user should look at it, if ambiguous

    @property
    def config_key(self) -> str:
        return f"{self.ref.file}::{self.ref.column}"


def _suggest_kind(storage: str, distinct: int, missing: int, rows: int, numeric_rate: float) -> tuple[str, str | None]:
    if distinct == 0:
        return UNUSED, "No values"
    if storage == BOOLEAN:
        return FLAG, None
    if storage == DATE:
        return WHEN, None
    if storage in NUMERIC:
        if distinct <= 15:
            return CONTINUOUS, f"Only {distinct} distinct values; may be category codes"
        return CONTINUOUS, None
    if storage == TEXT:
        if numeric_rate >= 0.98:
            return CATEGORICAL, f"Text, but {numeric_rate:.0%} of values are numbers"
        if distinct > max(1000, 0.5 * (rows - missing)):
            return UNUSED, f"{distinct:,} distinct values; likely names or addresses"
        return CATEGORICAL, None
    return UNUSED, None


def profile_characteristics(config: ImportConfig, infos: dict[str, SourceInfo]) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    for path in config.parcel_level_files():
        info = infos[path]
        key = config.key_for(path)
        columns = [c for c, kind in info.columns.items() if c != key and kind in (*NUMERIC, TEXT, BOOLEAN, DATE)]
        if not columns:
            continue
        lazy = scan(path).select(columns)
        stats = lazy.select(
            [pl.col(c).n_unique().alias(f"u::{c}") for c in columns]
            + [pl.col(c).null_count().alias(f"n::{c}") for c in columns]
        ).collect().row(0, named=True)
        text_columns = [c for c in columns if info.columns[c] == TEXT]
        rates = {}
        if text_columns:
            sample = lazy.select(text_columns).head(20_000).collect()
            for c in text_columns:
                text = as_text(sample[c]).drop_nulls()
                parsed = text.str.replace_all(r"[,$]", "").cast(pl.Float64, strict=False)
                rates[c] = float(parsed.is_not_null().mean()) if len(text) else 0.0
        for c in columns:
            missing = stats[f"n::{c}"]
            distinct = stats[f"u::{c}"] - (1 if missing else 0)
            kind, review = _suggest_kind(info.columns[c], distinct, missing, info.n_rows, rates.get(c, 0.0))
            profiles.append(ColumnProfile(ColumnRef(path, c), info.columns[c], distinct, missing,
                                          info.n_rows, kind, review))
    return profiles
