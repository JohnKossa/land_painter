"""Parcel geometry loading and spatial queries.

Geometry is loaded once, reprojected to WGS84 (lng/lat) for the map, and
flattened into the binary buffers deck.gl consumes. Multipolygon parcels are
exploded into parts; ``part_parcel`` maps each rendered part back to its parcel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

WGS84 = "EPSG:4326"


@dataclass
class ParcelLayer:
    keys: np.ndarray  # parcel key per parcel (object/str)
    part_parcel: np.ndarray  # uint32, parcel index for each rendered part
    positions: np.ndarray  # float64, flat [x0, y0, x1, y1, ...] for all ring vertices
    starts: np.ndarray  # uint32, vertex start index per part, with a trailing total
    parts: np.ndarray  # shapely Polygon per part, for spatial queries
    tree: shapely.STRtree

    @property
    def n_parcels(self) -> int:
        return len(self.keys)

    @property
    def n_parts(self) -> int:
        return len(self.part_parcel)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xy = self.positions.reshape(-1, 2)
        return (*xy.min(axis=0).tolist(), *xy.max(axis=0).tolist())

    def query_polygon(self, coords) -> np.ndarray:
        """Return sorted indices of parcels any of whose parts intersect the polygon."""
        polygon = shapely.make_valid(shapely.Polygon(coords))
        hits = self.tree.query(polygon, predicate="intersects")
        return np.unique(self.part_parcel[hits])


def read_parcels(path: str | Path, key_column: str = "key") -> gpd.GeoDataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path, engine="pyogrio")
    if key_column not in gdf.columns:
        raise ValueError(f"Parcel file has no {key_column!r} column (columns: {list(gdf.columns)})")
    return gdf


def build_layer(gdf: gpd.GeoDataFrame, key_column: str = "key") -> ParcelLayer:
    if gdf.crs is not None and not gdf.crs.equals(WGS84):
        gdf = gdf.to_crs(WGS84)

    geoms = np.asarray(gdf.geometry.array)
    keep = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    keep &= np.isin(shapely.get_type_id(geoms), (3, 6))  # Polygon, MultiPolygon
    gdf, geoms = gdf[keep], geoms[keep]

    parts, part_parcel = shapely.get_parts(geoms, return_index=True)
    parts = shapely.orient_polygons(parts, exterior_cw=False)  # deck.gl expects CCW

    # TODO: holes are dropped; render them via deck.gl holeIndices.
    rings = shapely.get_exterior_ring(parts)
    coords, ring_index = shapely.get_coordinates(rings, return_index=True)
    counts = np.bincount(ring_index, minlength=len(parts))
    starts = np.zeros(len(parts) + 1, dtype=np.uint32)
    np.cumsum(counts, out=starts[1:])

    return ParcelLayer(
        keys=gdf[key_column].to_numpy(),
        part_parcel=part_parcel.astype(np.uint32),
        positions=np.ascontiguousarray(coords, dtype=np.float64).ravel(),
        starts=starts,
        parts=parts,
        tree=shapely.STRtree(parts),
    )


def load_parcels(path: str | Path, key_column: str = "key") -> ParcelLayer:
    return build_layer(read_parcels(path, key_column), key_column)
