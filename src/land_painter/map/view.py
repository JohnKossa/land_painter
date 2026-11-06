"""Map widget: a QWebEngineView hosting deck.gl, plus the QWebChannel bridge."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView

from .scheme import SCHEME, AppSchemeHandler

# Cap on the on-disk basemap tile cache. Tiles are small; this holds plenty.
HTTP_CACHE_MAX_BYTES = 256 * 1024 * 1024  # 256 MB


class MapBridge(QObject):
    """Small messages between Python and the map. Bulk data goes over app://data."""

    # Python -> JS
    assignmentChanged = Signal(int, str)  # version, neighborhood palette JSON
    selectionChanged = Signal(int)  # selected neighborhood index
    salesChanged = Signal(int)  # version; the map re-fetches sale points and colors
    highlightRings = Signal(str)  # JSON list of [start, end) vertex ranges to outline; "[]" clears
    flyTo = Signal(float, float, float)  # longitude, latitude, zoom
    focusChanged = Signal(int)  # version; the map re-fetches the per-part focus mask (empty = no focus)
    fitBounds = Signal(float, float, float, float)  # min lng, min lat, max lng, max lat

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

        # A named profile is persistent (the default profile is off-the-record and
        # caches only in memory, so every launch would re-download every basemap
        # tile). A disk HTTP cache lets Chromium serve already-seen tiles straight
        # from disk between runs, keeping us within the shared Carto key's budget.
        self.profile = QWebEngineProfile("land_painter", self)
        cache_dir = Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation)) / "web"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.profile.setCachePath(str(cache_dir))
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
        self.profile.setHttpCacheMaximumSize(HTTP_CACHE_MAX_BYTES)
        # Identify the app to basemap tile servers, as their usage policies ask.
        self.profile.setHttpUserAgent(f"{self.profile.httpUserAgent()} LandPainter/0.1")

        self.scheme_handler = AppSchemeHandler(self)
        self.profile.installUrlSchemeHandler(SCHEME, self.scheme_handler)

        self.bridge = MapBridge(self)
        self._page = QWebEnginePage(self.profile, self)
        self.setPage(self._page)
        self._channel = QWebChannel(self._page)
        self._channel.registerObject("bridge", self.bridge)
        self._page.setWebChannel(self._channel)

    def start(self) -> None:
        self.load(QUrl("app://map/index.html"))
