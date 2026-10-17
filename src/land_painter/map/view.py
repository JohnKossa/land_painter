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

    # JS -> Python (re-emitted from slots so Python code can connect normally)
    ready = Signal()
    partClicked = Signal(int)
    lassoDrawn = Signal(object)  # list of [lng, lat]
    timingReported = Signal(str, float)

    @Slot()
    def jsReady(self):
        self.ready.emit()

    @Slot(int)
    def clickPart(self, part: int):
        self.partClicked.emit(part)

    @Slot(str)
    def lasso(self, coords_json: str):
        self.lassoDrawn.emit(json.loads(coords_json))

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
