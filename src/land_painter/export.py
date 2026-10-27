"""Export: parcels tagged with neighborhoods and land values, plus the land tables.

Three tables, written as Parquet or CSV side by side:

* ``<name>``: one row per parcel.
* ``<name>_land_tables``: one row per (live) neighborhood.
* ``<name>_adjustments``: one row per land table adjustment.

Only applied land tables are exported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .land_table import IN, components
from .neighborhoods import NeighborhoodModel

PARQUET, CSV = "parquet", "csv"


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


def output_paths(path: str | Path) -> list[Path]:
    """The three files an export to ``path`` writes."""
    path = Path(path)
    return [path, path.with_name(f"{path.stem}_land_tables{path.suffix}"),
            path.with_name(f"{path.stem}_adjustments{path.suffix}")]


def export(session, path: str | Path) -> list[Path]:
    """Write parcels, land tables and adjustments; the format follows ``path``'s extension."""
    targets = output_paths(path)
    fmt = CSV if targets[0].suffix.lower() == ".csv" else PARQUET
    frames = [parcel_table(session), *land_table_tables(session)]
    for frame, target in zip(frames, targets):
        if fmt == CSV:
            # Quote text so readers keep keys like "00123" as text rather than numbers.
            frame.write_csv(target, quote_style="non_numeric")
        else:
            frame.write_parquet(target)
    return targets

