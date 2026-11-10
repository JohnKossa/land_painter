"""Export: parcels tagged with neighborhoods and land values, plus the land tables.

Three files side by side:

* ``<name>``: one row per parcel, in the chosen format. As Parquet it is written
  as **GeoParquet** (a parcel-geometry column in WGS84), so it opens directly in
  QGIS and other GIS tools; as CSV it is attributes only (CSV has no native
  geometry).
* ``<name>_land_tables.csv``: one row per (live) neighborhood.
* ``<name>_adjustments.csv``: one row per land table adjustment.

The two reference tables are always CSV: they have no geometry, and CSV opens
anywhere (e.g. LibreOffice). Only applied land tables are exported.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import shapely

from .land_table import IN, components
from .neighborhoods import NeighborhoodModel
from .parcels import WGS84


def neighborhood_paths(model: NeighborhoodModel) -> list[str]:
    """Full path of every neighborhood, e.g. "Everything / Northside / Lakefront"."""
    paths: list[str] = []
    for n in model.neighborhoods:  # parents always come before their children
        paths.append(n.name if n.parent is None else f"{paths[n.parent]} / {n.name}")
    return paths


def parcel_table(session) -> pl.DataFrame:
    model, dataset = session.neighborhoods, session.dataset
    n = session.layer.n_parcels
    units = np.full(n, np.nan)
    rate = np.full(n, np.nan)
    base = np.full(n, np.nan)
    multiplier = np.full(n, np.nan)
    addition = np.full(n, np.nan)
    unit_name = np.full(n, None, dtype=object)
    for neighborhood, parcels in session.parcel_groups():
        table = model.neighborhoods[neighborhood].table
        if table is None:
            continue
        u = dataset.unit_values(table.unit)[parcels]
        b, m, a = components(table, u, dataset.characteristic, parcels)
        units[parcels], base[parcels], multiplier[parcels], addition[parcels] = u, b, m, a
        rate[parcels] = table.rate.rate
        unit_name[parcels] = table.unit

    names = np.array([nb.name for nb in model.neighborhoods], dtype=object)
    paths = np.array(neighborhood_paths(model), dtype=object)
    null = lambda a: pl.Series(a).fill_nan(None)  # noqa: E731
    return pl.DataFrame({
        "key": session.layer.keys,
        "neighborhood": names[model.assignment],
        "neighborhood_path": paths[model.assignment],
        "unit_of_measure": pl.Series(unit_name, dtype=pl.String),
        "units": null(units),
        "rate": null(rate),
        "base_value": null(base),
        "multiplier": null(multiplier),
        "addition": null(addition),
        "land_value": null(session.land_values.copy()),
    })


def _field_name(key: str) -> tuple[str, str]:
    path, column = key.split("::", 1)
    return column, Path(path).name


def land_table_tables(session) -> tuple[pl.DataFrame, pl.DataFrame]:
    model = session.neighborhoods
    paths = neighborhood_paths(model)
    counts = np.bincount(model.assignment, minlength=len(model.neighborhoods))
    tables, adjustments = [], []
    for i in model.live():
        nb = model.neighborhoods[i]
        t = nb.table
        tables.append({
            "neighborhood": nb.name,
            "neighborhood_path": paths[i],
            "parent": model.neighborhoods[nb.parent].name if nb.parent is not None else None,
            "parcels": int(counts[i]),
            "unit_of_measure": t.unit if t else None,
            "rate_model": t.rate.type if t else None,
            "rate": t.rate.rate if t else None,
            "adjustments": len(t.adjustments) if t else 0,
        })
        for order, adj in enumerate(t.adjustments if t else (), start=1):
            column, source = _field_name(adj.condition.field)
            c = adj.condition
            adjustments.append({
                "neighborhood": nb.name,
                "neighborhood_path": paths[i],
                "order": order,
                "name": adj.name or None,
                "field": column,
                "field_source": source,
                "operator": c.op,
                "value": None if c.op == IN else c.value,
                "values": "|".join(c.values) if c.op == IN else None,
                "condition": c.describe(column),
                "effect": adj.kind,
                "amount": adj.amount,
            })
    table_schema = {"neighborhood": pl.String, "neighborhood_path": pl.String, "parent": pl.String,
                    "parcels": pl.Int64, "unit_of_measure": pl.String, "rate_model": pl.String,
                    "rate": pl.Float64, "adjustments": pl.Int64}
    adjustment_schema = {"neighborhood": pl.String, "neighborhood_path": pl.String, "order": pl.Int64,
                         "name": pl.String, "field": pl.String, "field_source": pl.String, "operator": pl.String,
                         "value": pl.Float64, "values": pl.String, "condition": pl.String, "effect": pl.String,
                         "amount": pl.Float64}
    return (pl.DataFrame(tables, schema=table_schema, orient="row"),
            pl.DataFrame(adjustments, schema=adjustment_schema, orient="row"))


def parcel_geometry(layer) -> np.ndarray:
    """One geometry per parcel (MultiPolygon, WGS84), aligned to ``layer.keys``.

    The layer stores geometry as individual parts with ``part_parcel`` mapping each
    back to its parcel; this regroups the parts into one geometry per parcel.
    """
    return shapely.multipolygons(layer.parts, indices=layer.part_parcel)


def _write_parcel_geoparquet(parcels: pl.DataFrame, layer, target: Path) -> None:
    """Write the parcel table as GeoParquet so it renders directly in GIS tools."""
    gdf = gpd.GeoDataFrame(parcels.to_pandas(), geometry=parcel_geometry(layer), crs=WGS84)
    gdf.to_parquet(target)


def output_paths(path: str | Path) -> list[Path]:
    """The three files an export to ``path`` writes.

    The parcel file keeps ``path``'s extension; the land-table and adjustment
    reference tables are always CSV (no geometry, and CSV opens everywhere).
    """
    path = Path(path)
    return [path, path.with_name(f"{path.stem}_land_tables.csv"),
            path.with_name(f"{path.stem}_adjustments.csv")]


def _write_csv(frame: pl.DataFrame, target: Path) -> None:
    # Quote text so readers keep keys like "00123" as text rather than numbers.
    frame.write_csv(target, quote_style="non_numeric")


def export(session, path: str | Path) -> list[Path]:
    """Write the parcels (in ``path``'s format) plus the land-table and adjustment CSVs."""
    parcel_path, land_tables_path, adjustments_path = targets = output_paths(path)
    parcels = parcel_table(session)
    land_tables, adjustments = land_table_tables(session)
    if parcel_path.suffix.lower() == ".csv":
        _write_csv(parcels, parcel_path)
    else:
        _write_parcel_geoparquet(parcels, session.layer, parcel_path)  # GeoParquet: includes geometry
    _write_csv(land_tables, land_tables_path)
    _write_csv(adjustments, adjustments_path)
    return targets

