"""Map widget: a QWebEngineView hosting deck.gl, plus the QWebChannel bridge."""

from __future__ import annotations

import json

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView

from .scheme import SCHEME, AppSchemeHandler


class MapBridge(QObject):
    """Small messages between Python and the map. Bulk data goes over app://data."""

    # Python -> JS
    assignmentChanged = Signal(int, str)  # version, neighborhood palette JSON
    selectionChanged = Signal(int)  # selected neighborhood index
    salesChanged = Signal(int)  # version; the map re-fetches sale points and colors
    highlightRings = Signal(str)  # JSON list of [start, end) vertex ranges to outline; "[]" clears
    flyTo = Signal(float, float, float)  # longitude, latitude, zoom

    # JS -> Python (re-emitted from slots so Python code can connect normally)
    ready = Signal()
    partClicked = Signal(int)
    saleClicked = Signal(int)  # index into the currently shown sale points
    lassoDrawn = Signal(object, str)  # list of [lng, lat], mode ("create" or "reassign")
    timingReported = Signal(str, float)

    @Slot()
    def jsReady(self):
        self.ready.emit()

    @Slot(int)
    def clickPart(self, part: int):
        self.partClicked.emit(part)

    @Slot(int)
    def clickSale(self, index: int):
        self.saleClicked.emit(index)

    @Slot(str, str)
    def lasso(self, coords_json: str, mode: str):
        self.lassoDrawn.emit(json.loads(coords_json), mode)

    @Slot(str, float)
    def reportTiming(self, name: str, ms: float):
        self.timingReported.emit(name, ms)

    @Slot(str)
    def log(self, message: str):
        print(f"[map] {message}", flush=True)


class MapView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = MapBridge(self)
        self.scheme_handler = AppSchemeHandler(self)
        self.page().profile().installUrlSchemeHandler(SCHEME, self.scheme_handler)
        self._channel = QWebChannel(self.page())
        self._channel.registerObject("bridge", self.bridge)
        self.page().setWebChannel(self._channel)

    def start(self) -> None:
        self.load(QUrl("app://map/index.html"))
