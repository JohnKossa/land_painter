"""Running slow work (file reads, geometry loading) off the UI thread."""

from __future__ import annotations

import threading
from collections.abc import Callable

from PySide6.QtCore import QEventLoop, QObject, Qt, Signal
from PySide6.QtWidgets import QProgressDialog, QWidget


class _Relay(QObject):
    finished = Signal(object, object)  # result, exception


def run_with_progress[T](parent: QWidget | None, label: str, work: Callable[[], T]) -> T:
    """Run ``work`` on a worker thread behind a modal busy dialog; return its result.

    Exceptions raised by ``work`` are re-raised here, on the UI thread.
    """
    dialog = QProgressDialog(label, None, 0, 0, parent)
    dialog.setWindowTitle("Land Painter")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(300)
    dialog.setCancelButton(None)

    relay = _Relay()
    loop = QEventLoop()
    outcome: list = []

    def finished(result, error):
        outcome.extend([result, error])
        loop.quit()

    relay.finished.connect(finished, Qt.ConnectionType.QueuedConnection)

    def target():
        try:
            relay.finished.emit(work(), None)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the UI thread
            relay.finished.emit(None, exc)

    threading.Thread(target=target, daemon=True).start()
    loop.exec()
    dialog.close()
    result, error = outcome
    if error is not None:
        raise error
    return result
