"""The ``app://`` URL scheme: serves bundled map assets and binary data buffers.

``app://map/<path>`` serves files from the assets directory. ``app://data/<name>``
serves bytes from registered providers, which is how bulk geometry and per-part
attributes reach the map without going through QWebChannel's JSON encoding.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QFile, QIODevice
from PySide6.QtWebEngineCore import (
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)

SCHEME = b"app"
ASSETS_DIR = Path(__file__).parent / "assets"
QWEBCHANNEL_JS = ":/qtwebchannel/qwebchannel.js"


def register_scheme() -> None:
    """Register the scheme. Must be called before the QApplication is created."""
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.CorsEnabled
        | QWebEngineUrlScheme.Flag.FetchApiAllowed
    )
    QWebEngineUrlScheme.registerScheme(scheme)


class AppSchemeHandler(QWebEngineUrlSchemeHandler):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: dict[str, Callable[[], bytes]] = {}

    def provide(self, name: str, provider: Callable[[], bytes]) -> None:
        self._data[name] = provider

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:
        url = job.requestUrl()
        host, path = url.host(), url.path().lstrip("/")
        if host == "map":
            result = self._asset(path)
        elif host == "data" and path in self._data:
            result = (self._data[path](), "application/octet-stream")
        else:
            result = None
        if result is None:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        body, mime = result
        buffer = QBuffer(job)
        buffer.setData(QByteArray(body))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(mime.encode(), buffer)

    def _asset(self, path: str) -> tuple[bytes, str] | None:
        if path == "qwebchannel.js":
            f = QFile(QWEBCHANNEL_JS)
            if not f.open(QIODevice.OpenModeFlag.ReadOnly):
                return None
            return bytes(f.readAll().data()), "text/javascript"
        target = (ASSETS_DIR / path).resolve()
        if not target.is_relative_to(ASSETS_DIR.resolve()) or not target.is_file():
            return None
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            mime = "text/javascript"
        return target.read_bytes(), mime
