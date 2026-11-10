"""Session file commands for the main window: new, open, save, export, and the
close-with-unsaved-changes prompt.

These are mixed into ``MainWindow`` rather than kept as free functions because
they are window behaviors — they drive file dialogs, read the current session and
land panel, and refresh the window chrome. Splitting them out keeps app.py about
the map and neighborhoods. The session-lifecycle helpers they call
(``new_session``/``open_session``/``show_window``/``_windows``) live in app.py and
are imported lazily here to avoid an import cycle.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..session import SESSION_SUFFIX, Session
from .tasks import run_with_progress

SESSION_FILTER = f"Land Painter session (*{SESSION_SUFFIX})"


class SessionFilesMixin:
    """File/open/save/export commands. Expects the host to be a ``MainWindow``
    (it uses ``self.session``, ``self.land_panel``, ``self._refresh_chrome`` …)."""

    def _new_session(self) -> None:
        from ..app import new_session

        if self._can_discard():
            self._replace_with(new_session(self))

    def _open_session(self) -> None:
        from ..app import open_session

        if not self._can_discard():
            return
        path = QFileDialog.getOpenFileName(self, "Open session", "", SESSION_FILTER)[0]
        if path:
            self._replace_with(open_session(path, self))

    def _can_discard(self) -> bool:
        if self.session.path is not None or self.session.journal.length == 0:
            return True
        answer = QMessageBox.question(
            self, "Unsaved session", "This session has never been saved. Discard it?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _replace_with(self, session: Session | None) -> None:
        from ..app import show_window

        if session is None:
            return
        show_window(session)
        self._discard_on_close = True
        self.close()

    def _save(self) -> None:
        if self.session.path is None:
            self._save_as()
        else:
            self.statusBar().showMessage("Changes are saved automatically once a session has a file", 5000)

    def _save_as(self) -> bool:
        path = QFileDialog.getSaveFileName(self, "Save session", "", SESSION_FILTER)[0]
        if not path:
            return False
        if not path.lower().endswith(SESSION_SUFFIX):
            path += SESSION_SUFFIX
        self.session.save_as(path)
        self._refresh_chrome()
        self.statusBar().showMessage(f"Saved to {path}", 5000)
        return True

    def _export(self) -> None:
        from ..export import export, output_paths

        if self.land_panel.dirty:
            answer = QMessageBox.question(
                self, "Unapplied land table changes",
                "The open land table has changes that have not been applied. The export uses applied land tables "
                "only. Export anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        default = (self.session.path.with_suffix("") if self.session.path else Path("land_painter_export"))
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export parcels and land tables", f"{default}_export.parquet",
            "Parquet (*.parquet);;CSV (*.csv)")
        if not path:
            return
        suffix = ".csv" if chosen.startswith("CSV") else ".parquet"
        if Path(path).suffix.lower() not in (".csv", ".parquet"):
            path += suffix
        companions = [p for p in output_paths(path)[1:] if p.exists()]
        if companions:
            names = "".join(f"<li>{p.name}</li>" for p in companions)
            answer = QMessageBox.question(self, "Replace files?", f"These files will also be replaced:<ul>{names}</ul>",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            written = run_with_progress(self, "Exporting…", lambda: export(self.session, path))
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        files = "".join(f"<li>{p.name}</li>" for p in written)
        QMessageBox.information(self, "Export complete", f"Wrote to {written[0].parent}:<ul>{files}</ul>")

    def closeEvent(self, event) -> None:
        from ..app import _windows

        if (self.session.path is None and self.session.journal.length > 0
                and not getattr(self, "_discard_on_close", False)):
            answer = QMessageBox.question(
                self, "Unsaved session", "Save this session before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel or (
                    answer == QMessageBox.StandardButton.Save and not self._save_as()):
                event.ignore()
                return
        if self in _windows:
            _windows.remove(self)
        event.accept()
