"""Owns the map view: the bulk-data buffers it serves and the small signals it emits.

This keeps the main window's control flow free of map plumbing — version counters,
JSON/byte packing, and ``bridge.*.emit`` calls. The window calls intent methods
(``refresh_assignment``, ``focus_parcels``, …); this class handles the mechanics.

Static parcel geometry is served once; everything else updates attributes. Sale
points and colours depend on the sales pane, so the window supplies those
producers via ``provide_sales`` rather than this class reaching into the UI.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import shapely

from ..map import MapView
from ..session import Session


class MapController:
    def __init__(self, map_view: MapView, session: Session, basemap_urls: Callable[[], list[str]]):
        self.map = map_view
        self.session = session
        self.layer = session.layer
        self._basemap_urls = basemap_urls
        self._assignment_version = 0
        self._sales_version = 0
        self._focus_version = 0
        self._focus_parts = b""  # per-part focus mask (empty = no focus)

        handler = map_view.scheme_handler
        handler.provide("meta", self._meta)
        handler.provide("positions", lambda: self.layer.positions.tobytes())
        handler.provide("starts", lambda: self.layer.starts.tobytes())
        handler.provide("rings", lambda: self.layer.ring_starts.tobytes())
        handler.provide("indices", lambda: self.layer.indices.tobytes())
        handler.provide("assignment", self._part_assignment)
        handler.provide("focus", lambda: self._focus_parts)

    def provide_sales(self, points: Callable[[], bytes], colors: Callable[[], bytes]) -> None:
        """Register the sale-point producers (they depend on the sales pane)."""
        self.map.scheme_handler.provide("sale_points", points)
        self.map.scheme_handler.provide("sale_colors", colors)

    # --- Bulk-data producers (served over app://data) ----------------------------------

    def _meta(self) -> bytes:
        return json.dumps({
            "n_parcels": self.layer.n_parcels,
            "bounds": self.layer.bounds,
            "palette": self.session.neighborhoods.palette(),
            "basemap": self._basemap_urls(),
        }).encode()

    def _part_assignment(self) -> bytes:
        return self.session.neighborhoods.assignment[self.layer.part_parcel].tobytes()

    # --- Attribute updates and camera (small messages over the bridge) -----------------

    def refresh_assignment(self) -> None:
        """Tell the map to re-fetch neighborhood assignment and recolor."""
        self._assignment_version += 1
        self.map.bridge.assignmentChanged.emit(
            self._assignment_version, json.dumps(self.session.neighborhoods.palette()))

    def refresh_sales(self) -> None:
        """Tell the map to re-fetch the sale points and colors."""
        self._sales_version += 1
        self.map.bridge.salesChanged.emit(self._sales_version)

    def select(self, neighborhood: int) -> None:
        self.map.bridge.selectionChanged.emit(neighborhood)

    def highlight_rings(self, rings_json: str) -> None:
        self.map.bridge.highlightRings.emit(rings_json)

    def fly_to(self, longitude: float, latitude: float, zoom: float) -> None:
        self.map.bridge.flyTo.emit(longitude, latitude, zoom)

    def focus_parcels(self, parcels: np.ndarray | None) -> None:
        """Make ``parcels`` stand out on the map and fade the rest; None or empty clears it."""
        if parcels is None or not len(parcels):
            self._focus_parts = b""
        else:
            mask = np.zeros(self.layer.n_parcels, dtype=bool)
            mask[parcels] = True
            self._focus_parts = mask[self.layer.part_parcel].astype(np.uint8).tobytes()
        self._focus_version += 1
        self.map.bridge.focusChanged.emit(self._focus_version)

    def zoom_to_parcels(self, parcels: np.ndarray | None) -> None:
        if parcels is None or not len(parcels):
            return
        mask = np.zeros(self.layer.n_parcels, dtype=bool)
        mask[parcels] = True
        bounds = shapely.bounds(self.layer.parts[mask[self.layer.part_parcel]])
        self.map.bridge.fitBounds.emit(float(bounds[:, 0].min()), float(bounds[:, 1].min()),
                                       float(bounds[:, 2].max()), float(bounds[:, 3].max()))
