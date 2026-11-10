"""The loaded, cleaned dataset a session works on."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..land_table import LOT
from ..parcels import ParcelLayer, load_parcels
from .config import CATEGORICAL, CONTINUOUS, FLAG, SALE_PRICE, WHEN, Cleaning, ImportConfig, unit_field
from .review import (
    WARNING,
    DataError,
    Issue,
    Prepared,
    as_text,
    clean_numeric,
    distinct_values,
    numeric_stats,
    parcel_fields,
    prepare,
    to_number,
)
from .sources import read_columns

# Characteristic kinds a land table adjustment can filter on.
FILTER_KINDS = (CONTINUOUS, CATEGORICAL, FLAG)
# Characteristic kinds worth showing in the per-parcel detail view (everything but "unused").
DETAIL_KINDS = (CONTINUOUS, CATEGORICAL, FLAG, WHEN)


@dataclass
class Dataset:
    config: ImportConfig
    layer: ParcelLayer
    parcels: pl.DataFrame  # row i is layer parcel i: "key" + cleaned parcel-level fields
    sales: pl.DataFrame | None  # sales matched to parcels: "parcel" (index), key, price, date, vacant
    issues: list[Issue]
    _characteristics: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False, compare=False)
    _float_columns: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False, compare=False)
    _ones: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def neighborhood_codes(self) -> pl.Series | None:
        return self.parcels["neighborhood"] if "neighborhood" in self.parcels.columns else None

    def invalidate_caches(self) -> None:
        """Drop every cached column array.

        The caches below assume the per-parcel source columns never change after
        construction. Both ways a Dataset is built (``Session.new`` and
        ``Session.open``) make a fresh instance, so that holds today. A future
        feature that augments an *existing* dataset in place — adding columns to
        ``parcels`` or remapping ``config`` rather than rebuilding — must call
        this, or ``float_column``/``unit_values``/``characteristic`` would keep
        returning stale arrays. Add any new column cache here too.
        """
        self._characteristics.clear()
        self._float_columns.clear()
        self._ones = None

    # --- Parcel columns ---------------------------------------------------------------
    # Cached numeric arrays are shared, aligned to parcel order. Treat them as
    # read-only; copy before mutating (as ``compare`` does). See
    # ``invalidate_caches`` before mutating ``parcels`` in place.

    def float_column(self, name: str) -> np.ndarray:
        """A parcel column as a float array (NaN where missing), cached by column name."""
        if name not in self._float_columns:
            self._float_columns[name] = self.parcels[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
        return self._float_columns[name]

    # --- Units of measure -------------------------------------------------------------

    def unit_names(self) -> list[str]:
        """LOT first, then the mapped units in the order they were configured."""
        return [LOT, *self.config.units]

    def unit_values(self, name: str) -> np.ndarray:
        """Quantity of unit ``name`` for every parcel (float, NaN where missing)."""
        if name == LOT:  # every parcel is exactly 1 lot
            if self._ones is None:
                self._ones = np.ones(self.parcels.height)
            return self._ones
        return self.float_column(unit_field(name))

    # --- Characteristics ----------------------------------------------------------------
    # Keyed "path::column" as in ImportConfig.characteristic_kinds. Loaded from the
    # source file on first use and cached, aligned to parcel order.

    def _column(self, key: str) -> pl.Series:
        path, column = key.split("::", 1)
        key_column = self.config.key_for(path)
        raw = read_columns(path, [key_column] if column == key_column else [key_column, column])
        frame = pl.DataFrame({"key": as_text(raw[key_column]), "value": raw[column]})
        frame = frame.drop_nulls("key").unique("key", keep="first", maintain_order=True)
        return self.parcels.select("key").join(frame, on="key", how="left", maintain_order="left")["value"]

    def characteristic(self, key: str) -> np.ndarray:
        """Values of a characteristic for every parcel.

        Continuous characteristics are float arrays (NaN where missing); all
        others are object arrays of text (None where missing), matching what
        ``distinct_values`` offers and what ``Condition.mask`` compares against.
        """
        if key not in self._characteristics:
            values = self._column(key)
            if self.config.characteristic_kinds.get(key) == CONTINUOUS:
                if not values.dtype.is_numeric():
                    values, _ = to_number(values)
                array = values.cast(pl.Float64).fill_null(np.nan).to_numpy()
            else:
                array = np.array(as_text(values).to_list(), dtype=object)
            self._characteristics[key] = array
        return self._characteristics[key]

    def _labeled_fields(self, kinds: tuple[str, ...]) -> list[tuple[str, str, str]]:
        """(key, label, kind) for each characteristic of one of ``kinds``, sorted by label."""
        files = set(self.config.parcel_level_files())
        fields = [(key, key.split("::", 1)[1], kind) for key, kind in self.config.characteristic_kinds.items()
                  if kind in kinds and key.split("::", 1)[0] in files]
        counts: dict[str, int] = {}
        for _, label, _ in fields:
            counts[label] = counts.get(label, 0) + 1
        named = []
        for key, label, kind in fields:
            if counts[label] > 1:  # same column name in several files: say which file
                label = f"{label} ({key.split('::', 1)[0].replace(chr(92), '/').rsplit('/', 1)[-1]})"
            named.append((key, label, kind))
        return sorted(named, key=lambda f: f[1].lower())

    def filter_fields(self) -> list[tuple[str, str, str]]:
        """(key, label, kind) for each characteristic an adjustment can filter on, sorted by label."""
        return self._labeled_fields(FILTER_KINDS)

    def detail_fields(self) -> list[tuple[str, str, str]]:
        """(key, label, kind) for each characteristic shown in the per-parcel detail view."""
        return self._labeled_fields(DETAIL_KINDS)

    def distinct_values(self, key: str, limit: int = 10_000) -> list[str]:
        """Values of a (non-continuous) characteristic as text, most common first."""
        return distinct_values(pl.Series(self.characteristic(key), dtype=pl.String), limit)


def _clean(values: pl.Series, cleaning: Cleaning | None, unparseable: int) -> pl.Series:
    if cleaning is None or (not cleaning.missing and cleaning.fill == "none"):
        return values
    return clean_numeric(values, numeric_stats(values, unparseable), cleaning)


def build_dataset(config: ImportConfig, prepared: Prepared | None = None) -> Dataset:
    prepared = prepared or prepare(config)
    if prepared.blocking:
        raise DataError("\n".join(i.message for i in prepared.blocking))
    issues = list(prepared.issues)

    layer = load_parcels(config.parcel_file, config.parcel_key)
    if n := len(prepared.parcel_keys) - layer.n_parcels:
        issues.append(Issue(WARNING, f"{n:,} parcels have missing or invalid geometry and are left out."))

    parcels = prepared.parcels
    for field_id in parcel_fields(config):
        parcels = parcels.with_columns(
            _clean(parcels[field_id], config.cleaning.get(field_id), prepared.unparseable.get(field_id, 0)))
    order = pl.DataFrame({"key": layer.keys, "parcel": np.arange(layer.n_parcels, dtype=np.uint32)})
    parcels = order.join(parcels, on="key", how="left", maintain_order="left").drop("parcel")

    sales = prepared.sales
    if sales is not None:
        sales = sales.with_columns(
            _clean(sales["price"], config.cleaning.get(SALE_PRICE), prepared.unparseable.get(SALE_PRICE, 0)))
        sales = order.join(sales, on="key", how="inner", maintain_order="right")

    return Dataset(config, layer, parcels, sales, issues)
