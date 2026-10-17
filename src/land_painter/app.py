"""Main window: wires parcel data and the neighborhood model to the map."""

from __future__ import annotations

import json
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from .map import MapView
from .neighborhoods import ROOT, NeighborhoodModel
from .parcels import ParcelLayer

RATIO_BANNER = "Land ratio study  ·  Median: {median}  ·  COD: {cod}  ·  PRD: {prd}  ·  PRB: {prb}"


class MainWindow(QMainWindow):
    timing = Signal(str, float)  # every timing, Python- or JS-side, in ms

    def __init__(self, layer: ParcelLayer):
        super().__init__()
        self.setWindowTitle("Land Painter")
        self.resize(1400, 900)

        self.layer = layer
        self.model = NeighborhoodModel(layer.n_parcels)
        self.selected = ROOT
        self._version = 0

        self.banner = QLabel(RATIO_BANNER.format(median="Undefined", cod="Undefined", prd="Undefined", prb="Undefined"))
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet("padding: 4px; color: #ccc; background: #2b2d30;")
        self.map = MapView()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.banner)
        layout.addWidget(self.map, 1)
        self.setCentralWidget(central)

        handler = self.map.scheme_handler
        handler.provide("meta", self._meta)
        handler.provide("positions", lambda: layer.positions.tobytes())
        handler.provide("starts", lambda: layer.starts.tobytes())
        handler.provide("assignment", self._part_assignment)

        bridge = self.map.bridge
        bridge.partClicked.connect(self._on_part_clicked)
        bridge.lassoDrawn.connect(self.lasso)
        bridge.timingReported.connect(self.timing)
        self.timing.connect(self._show_timing)

        self.map.start()

    def _meta(self) -> bytes:
        return json.dumps({
            "n_parcels": self.layer.n_parcels,
            "bounds": self.layer.bounds,
            "palette": self.model.palette(),
        }).encode()

    def _part_assignment(self) -> bytes:
        return self.model.assignment[self.layer.part_parcel].tobytes()

    def _on_part_clicked(self, part: int) -> None:
        self.selected = int(self.model.assignment[self.layer.part_parcel[part]])
        self.statusBar().showMessage(f"Selected {self.model.neighborhoods[self.selected].name}", 5000)
        self.map.bridge.selectionChanged.emit(self.selected)

    def lasso(self, coords) -> int | None:
        """Create a child of the selected neighborhood from lassoed parcels."""
        t0 = time.perf_counter()
        parcels = self.layer.query_polygon(coords)
        child = self.model.create_child(self.selected, parcels)
        self.timing.emit("lasso_query", (time.perf_counter() - t0) * 1000)
        if child is None:
            self.statusBar().showMessage("Lasso contained no parcels from the selected neighborhood", 5000)
            return None
        self._version += 1
        self.map.bridge.assignmentChanged.emit(self._version, json.dumps(self.model.palette()))
        return child

    def _show_timing(self, name: str, ms: float) -> None:
        self.statusBar().showMessage(f"{name}: {ms:,.0f} ms", 8000)
