"""Generate a synthetic parcel GeoParquet for performance testing.

Parcels are a jittered grid of 8-vertex polygons with shared edges, so
neighbors genuinely touch, laid out in UTM 17N near Fort Myers, FL.

    python scripts/make_synthetic_parcels.py data/synthetic_400k.parquet --count 400000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

ORIGIN = (415_000.0, 2_943_000.0)  # UTM 17N, roughly Fort Myers
CELL = 30.0  # metres


def make_parcels(count: int, seed: int = 0) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    jitter = CELL * 0.2

    def grid(ny, nx, ox, oy):
        y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
        pts = np.stack([(x + ox) * CELL, (y + oy) * CELL], axis=-1)
        return pts + rng.uniform(-jitter, jitter, pts.shape)

    corners = grid(rows + 1, cols + 1, 0.0, 0.0)
    h_mid = grid(rows + 1, cols, 0.5, 0.0)  # midpoints of horizontal edges
    v_mid = grid(rows, cols + 1, 0.0, 0.5)  # midpoints of vertical edges

    r, c = np.divmod(np.arange(count), cols)
    ring = np.stack([
        corners[r, c], h_mid[r, c], corners[r, c + 1], v_mid[r, c + 1],
        corners[r + 1, c + 1], h_mid[r + 1, c], corners[r + 1, c], v_mid[r, c],
        corners[r, c],
    ], axis=1) + ORIGIN

    geometry = shapely.polygons(ring)
    gdf = gpd.GeoDataFrame(
        {"key": np.char.add("P", np.arange(count).astype(str).astype(np.str_))},
        geometry=geometry,
        crs="EPSG:32617",
    )
    centroids = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf["longitude"] = centroids.x
    gdf["latitude"] = centroids.y
    return gdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_parcels(args.count, args.seed).to_parquet(args.output)
    print(f"Wrote {args.count:,} parcels to {args.output}")


if __name__ == "__main__":
    main()
