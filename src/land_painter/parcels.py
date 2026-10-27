"""Parcel geometry loading and spatial queries.

Geometry is loaded once, reprojected to WGS84 (lng/lat) for the map, and
flattened into the binary buffers deck.gl consumes. Multipolygon parcels are
exploded into parts; ``part_parcel`` maps each rendered part back to its parcel.
Parts are triangulated here (holes included) so the map does not have to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import mapbox_earcut
import numpy as np
import shapely

from .data.sources import PARQUET, inspect

WGS84 = "EPSG:4326"


@dataclass
class ParcelLayer:
    keys: np.ndarray  # parcel key per parcel (str)
    part_parcel: np.ndarray  # uint32, parcel index for each rendered part
    positions: np.ndarray  # float64, flat [x0, y0, x1, y1, ...] for all ring vertices
    starts: np.ndarray  # uint32, vertex start index per part, with a trailing total
    ring_starts: np.ndarray  # uint32, vertex start index per ring (exteriors and holes), with a trailing total
    indices: np.ndarray  # uint32, triangle vertex indices into positions
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

    def parcel_rings(self, parcel: int) -> list[tuple[int, int]]:
        """Vertex ranges [start, end) of every ring (all parts, holes included) of a parcel."""
        first = np.searchsorted(self.part_parcel, parcel, side="left")  # parts are grouped by parcel
        last = np.searchsorted(self.part_parcel, parcel, side="right")
        v0, v1 = self.starts[first], self.starts[last]
        r0, r1 = np.searchsorted(self.ring_starts, [v0, v1])
        return [(int(self.ring_starts[r]), int(self.ring_starts[r + 1])) for r in range(r0, r1)]

    def parcel_points(self, parcels: np.ndarray) -> np.ndarray:
        """A point inside each parcel (in its first part), as an (n, 2) lng/lat array."""
        first = np.searchsorted(self.part_parcel, parcels, side="left")
        return shapely.get_coordinates(shapely.point_on_surface(self.parts[first]))

    def query_polygon(self, coords) -> np.ndarray:
        """Return sorted indices of parcels any of whose parts intersect the polygon."""
        polygon = shapely.make_valid(shapely.Polygon(coords))
        hits = self.tree.query(polygon, predicate="intersects")
        return np.unique(self.part_parcel[hits])


def read_parcels(path: str | Path, key_column: str) -> gpd.GeoDataFrame:
    """Read only the key and geometry columns."""
    path = Path(path)
    if path.suffix.lower() in PARQUET:
        info = inspect(path)
        if info.geometry_column is None:
            raise ValueError(f"{path.name} has no geometry column")
        gdf = gpd.read_parquet(path, columns=[key_column, info.geometry_column])
    else:
        gdf = gpd.read_file(path, columns=[key_column], engine="pyogrio")
    if key_column not in gdf.columns:
        raise ValueError(f"{path.name} has no {key_column!r} column")
    return gdf


def triangulate(positions: np.ndarray, starts: np.ndarray, ring_starts: np.ndarray, part_first_ring: np.ndarray) -> np.ndarray:
    xy = positions.reshape(-1, 2)
    out = []
    for p in range(len(starts) - 1):
        s, e = int(starts[p]), int(starts[p + 1])
        r0, r1 = int(part_first_ring[p]), int(part_first_ring[p + 1])
        ring_ends = (ring_starts[r0 + 1:r1 + 1] - s).astype(np.uint32)
        out.append(mapbox_earcut.triangulate_float64(xy[s:e], ring_ends) + s)
    return np.concatenate(out).astype(np.uint32) if out else np.zeros(0, np.uint32)


def build_layer(gdf: gpd.GeoDataFrame, key_column: str) -> ParcelLayer:
    if gdf.crs is not None and not gdf.crs.equals(WGS84):
        gdf = gdf.to_crs(WGS84)

    geoms = np.asarray(gdf.geometry.array)
    keep = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    keep &= np.isin(shapely.get_type_id(geoms), (3, 6))  # Polygon, MultiPolygon
    keep &= gdf[key_column].notna().to_numpy()
    gdf, geoms = gdf[keep], geoms[keep]

    parts, part_parcel = shapely.get_parts(geoms, return_index=True)
    parts = shapely.orient_polygons(parts, exterior_cw=False)  # CCW exteriors, CW holes

    rings, ring_part = shapely.get_rings(parts, return_index=True)  # exterior first, then holes
    coords, ring_index = shapely.get_coordinates(rings, return_index=True)
    ring_starts = np.zeros(len(rings) + 1, dtype=np.uint32)
    np.cumsum(np.bincount(ring_index, minlength=len(rings)), out=ring_starts[1:])
    part_first_ring = np.searchsorted(ring_part, np.arange(len(parts) + 1)).astype(np.int64)
    starts = ring_starts[part_first_ring].astype(np.uint32)
    positions = np.ascontiguousarray(coords, dtype=np.float64).ravel()

    return ParcelLayer(
        keys=gdf[key_column].astype(str).to_numpy(),
        part_parcel=part_parcel.astype(np.uint32),
        positions=positions,
        starts=starts,
        ring_starts=ring_starts,
        indices=triangulate(positions, starts, ring_starts, part_first_ring),
        parts=parts,
        tree=shapely.STRtree(parts),
    )


def load_parcels(path: str | Path, key_column: str) -> ParcelLayer:
    return build_layer(read_parcels(path, key_column), key_column)
