"""Main window: wires the session to the map and menus."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCursor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QVBoxLayout, QWidget,
)
import numpy as np
import shapely

from .data.dataset import build_dataset
from .data.review import DataError
from .data.sources import FILE_FILTER
from .map import MapView
from .neighborhoods import ROOT
from .sales import DISQUALIFIED, KNOWN_GOOD, UNFLAGGED
from .session import SESSION_SUFFIX, Session, SessionError, read_config, relocate_inputs
from .ui.tasks import run_with_progress

SALE_COLORS = {UNFLAGGED: (240, 240, 240, 255), KNOWN_GOOD: (152, 195, 121, 255), DISQUALIFIED: (224, 108, 117, 255)}
SESSION_FILTER = f"Land Painter session (*{SESSION_SUFFIX})"

# TEMPORARY online basemap (see SPEC.md lines 11, 33 — the shipping app uses a
# local, offline basemap). Carto "Positron" raster tiles, keyed by the API key
# in CARTO_API_KEY_ENV so the shared key stays out of source. No key -> no URLs
# -> the map draws no basemap. {z}/{x}/{y} are left for deck.gl to substitute.
CARTO_API_KEY_ENV = "LAND_PAINTER_CARTO_KEY"


def _basemap_urls() -> list[str]:
    key = os.environ.get(CARTO_API_KEY_ENV, "").strip()
    if not key:
        return []
    return [f"https://{sub}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png?key={key}"
            for sub in ("a", "b", "c", "d")]

_windows: list[MainWindow] = []  # keep top-level windows alive


def show_window(session: Session) -> MainWindow:
    window = MainWindow(session)
    _windows.append(window)
    window.show()
    return window


def new_session(parent=None, files: list[str] | None = None) -> Session | None:
    """Run the upload wizard and build a session from it (or open a session, if the user chose that)."""
    from .ui.upload import UploadWizard

    wizard = UploadWizard(files, parent)
    accepted = wizard.exec()
    if wizard.session_path:
        return open_session(wizard.session_path, parent)
    if not accepted:
        return None
    config, prepared = wizard.config, wizard.prepared()
    try:
        return run_with_progress(parent, "Loading parcels and building the session…",
                                 lambda: Session.new(build_dataset(config, prepared)))
    except (DataError, ValueError, OSError) as exc:
        QMessageBox.critical(parent, "Could not load data", str(exc))
        return None


def open_session(path: str, parent=None) -> Session | None:
    """Open a session file, asking the user to locate any input files that have moved."""
    def locate(missing: Path) -> Path | None:
        QMessageBox.information(parent, "Input file not found",
                                f"Could not find {missing}.\nPlease locate it (it may have been moved or renamed).")
        chosen = QFileDialog.getOpenFileName(parent, f"Locate {missing.name}", "", FILE_FILTER)[0]
        return Path(chosen) if chosen else None

    try:
        config = read_config(path)
        relocate_inputs(config, path, locate)
        return run_with_progress(parent, "Opening session…", lambda: Session.open(path, config))
    except (SessionError, DataError, ValueError, OSError) as exc:
        QMessageBox.critical(parent, "Could not open session", str(exc))
        return None


class MainWindow(QMainWindow):
    timing = Signal(str, float)  # every timing, Python- or JS-side, in ms

    def __init__(self, session: Session):
        super().__init__()
        self.resize(1400, 900)

        self.session = session
        self.layer = session.layer
        self.selected = ROOT
        self._version = 0
        self._focus_parts = b""  # per-part focus mask for the map (empty = no focus)
        self._focus_version = 0

        self.banner = QLabel()
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet("padding: 4px; color: #ccc; background: #2b2d30;")
        self.banner.setToolTip("Highlighted when outside IAAO ranges: median 0.90–1.10, COD ≤ 25 (vacant land), "
                               "PRD 0.98–1.03, PRB −0.05–0.05")
        self.map = MapView()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.banner)
        layout.addWidget(self.map, 1)
        self.setCentralWidget(central)
        self._build_menus()
        self.sales_pane = None
        self._sales_version = 0
        if session.dataset.sales is not None:
            self._build_sales_pane()
        self._build_neighborhood_popup()
        self._build_compare_pane()
        self._refresh_metrics()

        handler = self.map.scheme_handler
        handler.provide("meta", self._meta)
        handler.provide("positions", lambda: self.layer.positions.tobytes())
        handler.provide("starts", lambda: self.layer.starts.tobytes())
        handler.provide("rings", lambda: self.layer.ring_starts.tobytes())
        handler.provide("indices", lambda: self.layer.indices.tobytes())
        handler.provide("assignment", self._part_assignment)
        handler.provide("sale_points", self._sale_points)
        handler.provide("sale_colors", self._sale_colors)
        handler.provide("focus", lambda: self._focus_parts)

        bridge = self.map.bridge
        bridge.partClicked.connect(self._on_part_clicked)
        bridge.saleClicked.connect(self._on_sale_point_clicked)
        bridge.lassoDrawn.connect(self.lasso)
        bridge.timingReported.connect(self.timing)
        self.timing.connect(self._show_timing)

        self._refresh_chrome()
        self._show_data_summary()
        self.map.start()
        self.statusBar().showMessage("Click a neighborhood to open it · shift+drag: new sub-neighborhood · "
                                     "ctrl+shift+drag: move parcels into the selected neighborhood", 15000)

    # --- Menus -------------------------------------------------------------------

    def _build_menus(self) -> None:
        def action(menu, text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut is not None:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        file_menu = self.menuBar().addMenu("&File")
        action(file_menu, "&New Session…", self._new_session, QKeySequence.StandardKey.New)
        action(file_menu, "&Open Session…", self._open_session, QKeySequence.StandardKey.Open)
        action(file_menu, "&Save Session", self._save, QKeySequence.StandardKey.Save)
        action(file_menu, "Save Session &As…", self._save_as, QKeySequence.StandardKey.SaveAs)
        file_menu.addSeparator()
        action(file_menu, "&Export…", self._export, QKeySequence("Ctrl+E"))
        file_menu.addSeparator()
        action(file_menu, "E&xit", QApplication.closeAllWindows, QKeySequence.StandardKey.Quit)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = action(edit_menu, "Undo", self.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = action(edit_menu, "Redo", self.redo, QKeySequence.StandardKey.Redo)

        tools_menu = self.menuBar().addMenu("&Tools")
        action(tools_menu, "&Land/Building Split Check (current roll)…", self._split_check)

    def _split_check(self) -> None:
        from .ui.split_check import SplitCheckDialog

        if getattr(self, "session", None) is None:
            return
        SplitCheckDialog(self.session, self).exec()

    def _build_sales_pane(self) -> None:
        from .ui.sales_pane import SalesPane, SalesTableModel

        sales = self.session.dataset.sales
        self._sale_parcels = sales["parcel"].to_numpy()
        self._sale_xy = self.layer.parcel_points(self._sale_parcels)
        parcels = self.session.dataset.parcels
        units = {c.removeprefix("unit:"): parcels[c].to_numpy()[self._sale_parcels]
                 for c in parcels.columns if c.startswith("unit:")}
        model = SalesTableModel(sales, self.session.sale_flags, self.session.land_values, units,
                                self._neighborhood_names)
        self.sales_pane = pane = SalesPane(model)
        pane.flagRequested.connect(self._flag_sales)
        pane.saleSelected.connect(self._highlight_sale)
        pane.saleActivated.connect(self._zoom_to_sale)
        pane.visibleChanged.connect(self._sales_changed)
        pane.show_on_map.toggled.connect(self._sales_changed)

        dock = QDockWidget("Sales", self)
        dock.setObjectName("sales")
        dock.setWidget(pane)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [620], Qt.Orientation.Horizontal)
        self._view_menu().addAction(dock.toggleViewAction())
        self.sales_dock = dock
        self._update_sales_neighborhood()

    def _build_compare_pane(self) -> None:
        from .ui.compare_pane import ComparePane

        self.compare_pane = ComparePane(self.session)
        self.compare_pane.map_selection = self.selected
        self.compare_pane.focusRequested.connect(self._focus_parcels)
        self.compare_pane.zoomRequested.connect(self._zoom_to_parcels)
        dock = QDockWidget("Compare", self)
        dock.setObjectName("compare")
        dock.setWidget(self.compare_pane)
        sales_dock = getattr(self, "sales_dock", None)
        if sales_dock is not None:
            self.tabifyDockWidget(sales_dock, dock)  # a tab beside Sales
            sales_dock.raise_()  # Sales stays the tab shown first
        else:
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self.resizeDocks([dock], [620], Qt.Orientation.Horizontal)
        self._view_menu().addAction(dock.toggleViewAction())

    def _build_neighborhood_popup(self) -> None:
        from .ui.land_table_panel import LandTablePanel
        from .ui.metrics_panel import MetricsPanel
        from .ui.neighborhood_popup import NeighborhoodPopup

        self.land_panel = LandTablePanel(self.session)
        self.land_panel.neighborhoodChosen.connect(self._select)
        self.land_panel.applyRequested.connect(self._apply_land_table)
        self.land_panel.applyToChildrenRequested.connect(self._apply_table_to_children)
        self.metrics_panel = MetricsPanel()
        self.popup = NeighborhoodPopup(self.session, self.land_panel, self.metrics_panel)
        self.popup.renameRequested.connect(self._rename_neighborhood)
        self.popup.deleteRequested.connect(self._delete_neighborhood)
        self.popup.splitRequested.connect(self._split_neighborhood)
        self.land_panel.convertRequested.connect(self._convert_adjustment)
        self.popup.show_neighborhood(self.selected)

        # A floating dock: it pops up beside the map, and can be docked if preferred.
        self.popup_dock = dock = QDockWidget("Neighborhood", self)
        dock.setObjectName("neighborhood")
        dock.setWidget(self.popup)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        dock.setFloating(True)
        dock.resize(1020, 760)
        dock.hide()
        view = self._view_menu()
        view.addAction(dock.toggleViewAction())
        self.popup_on_click = QAction("Open Neighborhood on Click", self)
        self.popup_on_click.setCheckable(True)
        self.popup_on_click.setChecked(True)
        view.addAction(self.popup_on_click)

    def _open_popup(self) -> None:
        dock = self.popup_dock
        if dock.isFloating() and not dock.isVisible():
            # Beside the cursor, kept on the screen.
            screen = self.screen().availableGeometry()
            pos = QCursor.pos()
            x = pos.x() + 30 if pos.x() + 30 + dock.width() < screen.right() else pos.x() - 30 - dock.width()
            y = min(max(screen.top(), pos.y() - dock.height() // 3), screen.bottom() - dock.height())
            dock.move(max(screen.left(), x), y)
        dock.show()
        dock.raise_()

    def _rename_neighborhood(self, index: int) -> None:
        current = self.session.neighborhoods.neighborhoods[index].name
        name, ok = QInputDialog.getText(self, "Rename neighborhood", "Name:", text=current)
        if ok and self.session.rename_neighborhood(index, name):
            self._assignment_changed()

    def _delete_neighborhood(self, index: int) -> None:
        from .ui.neighborhood_popup import DeleteDialog

        dialog = DeleteDialog(self.session, index, self)
        if dialog.exec():
            recipient = dialog.chosen()
            self.session.delete_neighborhood(index, recipient)
            self.selected = recipient
            self._assignment_changed()
            self._select(recipient)

    def _split_neighborhood(self, index: int) -> None:
        from .ui.land_table_panel import SplitDialog

        if self.land_panel.dirty and self.land_panel.neighborhood == index:
            QMessageBox.information(self, "Unapplied changes",
                                    "Apply or revert your land table changes first: the new neighborhood "
                                    "starts with a copy of this neighborhood's applied land table.")
            return
        dialog = SplitDialog(self.session, index, self)
        if not dialog.exec():
            return
        child = self.session.split_neighborhood(index, dialog.condition_value(), dialog.name_value())
        self._after_split(child)

    def _convert_adjustment(self, neighborhood: int, position: int) -> None:
        from .land_table import MULTIPLY, FlatRate
        from .ui.land_table_panel import SNAPSHOT_NOTE

        model = self.session.neighborhoods
        parent = model.neighborhoods[neighborhood]
        table = parent.table
        if table is None or not 0 <= position < len(table.adjustments):
            return
        adjustment = table.adjustments[position]
        n = len(self.session.matching_parcels(neighborhood, adjustment.condition))
        where = self.session.describe_condition(adjustment.condition)
        if adjustment.kind == MULTIPLY and isinstance(table.rate, FlatRate):
            rate = table.rate.rate
            table_text = (f"Its land table copies {parent.name}'s, with the {adjustment.describe()} adjustment folded "
                          f"into the rate (${rate:,.4g} → ${rate * adjustment.amount:,.4g}).")
        else:
            table_text = (f"Its land table copies {parent.name}'s, keeping the {adjustment.describe()} adjustment "
                          "(it applies to every parcel there).")
        message = (f"Moves the {n:,} parcels directly in {parent.name} where {where} into a new child "
                   f"neighborhood. {table_text} The adjustment is removed from {parent.name}. "
                   f"Land values don't change.\n\n{SNAPSHOT_NOTE}\n\nName:")
        name, ok = QInputDialog.getText(self, "Convert to child neighborhood", message,
                                        text=adjustment.name or where)
        if not ok:
            return
        child = self.session.convert_adjustment(neighborhood, position, name)
        self._after_split(child)

    def _after_split(self, child: int | None) -> None:
        if child is None:
            self.statusBar().showMessage("No parcels matched, so no neighborhood was created", 5000)
            return
        n = int((self.session.neighborhoods.assignment == child).sum())
        name = self.session.neighborhoods.neighborhoods[child].name
        self.statusBar().showMessage(f"Created {name} with {n:,} parcels", 5000)
        self._assignment_changed()

    def _refresh_metrics(self) -> None:
        """Recompute the banner (county-wide) and the selected neighborhood's metrics."""
        from .metrics import compute
        from .spatial import residual_autocorrelation
        from .ui.metrics_panel import banner_html

        if not hasattr(self, "metrics_panel"):
            return
        model = self.session.neighborhoods
        everything = np.ones(self.layer.n_parcels, dtype=bool)
        self.banner.setText(banner_html(compute(self.session, everything).ratio))
        table = model.neighborhoods[self.selected].table
        unit = table.unit if table else None
        mask = model.assignment == self.selected
        metrics = compute(self.session, mask, unit)
        spatial = residual_autocorrelation(self.session, mask)  # selected neighborhood only: not cheap county-wide
        self.metrics_panel.show_metrics(model.neighborhoods[self.selected].name, metrics, unit, spatial)

    def _view_menu(self):
        if not hasattr(self, "_view"):
            self._view = self.menuBar().addMenu("&View")
        return self._view

    def _apply_land_table(self, neighborhood: int, table) -> None:
        if self.session.set_land_table(neighborhood, table):
            self._values_changed()

    def _apply_table_to_children(self, neighborhood: int) -> None:
        n = self.session.apply_table_to_children(neighborhood)
        self.statusBar().showMessage(f"Land table applied to {n:,} child neighborhoods" if n
                                     else "All child neighborhoods already have this land table", 5000)
        if n:
            self._values_changed()

    def _values_changed(self) -> None:
        self.land_panel.reload()
        if self.sales_pane is not None:
            self.sales_pane.model.refresh()
        self._compare_data_changed()
        self._refresh_metrics()
        self._refresh_chrome()

    def _focus_parcels(self, parcels) -> None:
        """Make ``parcels`` stand out on the map and fade the rest; None or empty clears the focus."""
        if parcels is None or not len(parcels):
            self._focus_parts = b""
        else:
            mask = np.zeros(self.layer.n_parcels, dtype=bool)
            mask[parcels] = True
            self._focus_parts = mask[self.layer.part_parcel].astype(np.uint8).tobytes()
        self._focus_version += 1
        self.map.bridge.focusChanged.emit(self._focus_version)

    def _zoom_to_parcels(self, parcels) -> None:
        if parcels is None or not len(parcels):
            return
        mask = np.zeros(self.layer.n_parcels, dtype=bool)
        mask[parcels] = True
        bounds = shapely.bounds(self.layer.parts[mask[self.layer.part_parcel]])
        self.map.bridge.fitBounds.emit(float(bounds[:, 0].min()), float(bounds[:, 1].min()),
                                       float(bounds[:, 2].max()), float(bounds[:, 3].max()))

    def _compare_data_changed(self) -> None:
        """Land values, neighborhoods or sale flags changed (the Compare tab recomputes only while shown)."""
        if hasattr(self, "compare_pane"):
            self.compare_pane.data_changed()

    def _refresh_chrome(self) -> None:
        path = self.session.path
        self.setWindowTitle(f"Land Painter — {path.name if path else 'Unsaved session'}")
        undo, redo = self.session.undo_label(), self.session.redo_label()
        self.undo_action.setText(f"Undo {undo}" if undo else "Undo")
        self.undo_action.setEnabled(undo is not None)
        self.redo_action.setText(f"Redo {redo}" if redo else "Redo")
        self.redo_action.setEnabled(redo is not None)

    def _show_data_summary(self) -> None:
        dataset = self.session.dataset
        parts = [f"{self.layer.n_parcels:,} parcels"]
        if dataset.sales is not None:
            vacant = int(dataset.sales["vacant"].sum())
            parts.append(f"{dataset.sales.height:,} sales ({vacant:,} vacant)")
        parts.append(f"{len(self.session.neighborhoods.neighborhoods):,} neighborhoods")
        self.statusBar().showMessage(" · ".join(parts), 15000)

    def _new_session(self) -> None:
        if self._can_discard():
            self._replace_with(new_session(self))

    def _open_session(self) -> None:
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
        from .export import export, output_paths

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

    # --- Map data ----------------------------------------------------------------

    def _meta(self) -> bytes:
        return json.dumps({
            "n_parcels": self.layer.n_parcels,
            "bounds": self.layer.bounds,
            "palette": self.session.neighborhoods.palette(),
            "basemap": _basemap_urls(),
        }).encode()

    def _part_assignment(self) -> bytes:
        return self.session.neighborhoods.assignment[self.layer.part_parcel].tobytes()

    def _assignment_changed(self) -> None:
        self._version += 1
        self.map.bridge.assignmentChanged.emit(self._version, json.dumps(self.session.neighborhoods.palette()))
        if self.sales_pane is not None:
            self.sales_pane.model.refresh_neighborhoods()
            self._update_sales_neighborhood()
        if hasattr(self, "land_panel"):
            self.land_panel.reload()
            self.popup.show_neighborhood(self.selected)
        self._compare_data_changed()
        self._refresh_metrics()
        self._refresh_chrome()

    def _neighborhood_names(self, parcels: np.ndarray) -> np.ndarray:
        names = np.array([n.name for n in self.session.neighborhoods.neighborhoods], dtype=object)
        return names[self.session.neighborhoods.assignment[parcels]]

    # --- Sales -------------------------------------------------------------------------

    def _sale_points(self) -> bytes:
        if self.sales_pane is None or not self.sales_pane.show_on_map.isChecked():
            return b""
        visible = self.sales_pane.visible_sales()
        points = np.zeros((len(visible), 3), dtype=np.float64)
        points[:, :2] = self._sale_xy[visible]
        return points.tobytes()

    def _sale_colors(self) -> bytes:
        if self.sales_pane is None or not self.sales_pane.show_on_map.isChecked():
            return b""
        lut = np.array([SALE_COLORS[f] for f in (UNFLAGGED, KNOWN_GOOD, DISQUALIFIED)], dtype=np.uint8)
        return lut[self.session.sale_flags[self.sales_pane.visible_sales()]].tobytes()

    def _sales_changed(self) -> None:
        self._sales_version += 1
        self.map.bridge.salesChanged.emit(self._sales_version)

    def _flag_sales(self, sales: np.ndarray, flag: int) -> None:
        if self.session.set_sale_flags(sales, flag):
            self.sales_pane.flags_changed()
            self._sales_changed()
            self._compare_data_changed()
            self._refresh_metrics()
            self._refresh_chrome()

    def _highlight_sale(self, sale: int) -> None:
        self.map.bridge.highlightRings.emit(json.dumps(self.layer.parcel_rings(int(self._sale_parcels[sale]))))

    def _zoom_to_sale(self, sale: int) -> None:
        lng, lat = self._sale_xy[sale]
        self.map.bridge.flyTo.emit(float(lng), float(lat), 17.0)

    def _on_sale_point_clicked(self, index: int) -> None:
        visible = self.sales_pane.visible_sales()
        if 0 <= index < len(visible):
            self.sales_pane.select_sale(int(visible[index]))

    def _update_sales_neighborhood(self) -> None:
        if self.sales_pane is None:
            return
        model = self.session.neighborhoods
        in_subtree = model.subtree_mask(self.selected)[model.assignment[self._sale_parcels]]
        self.sales_pane.set_neighborhood_mask(in_subtree)

    # --- Interaction -----------------------------------------------------------------

    def _on_part_clicked(self, part: int) -> None:
        parcel = int(self.layer.part_parcel[part])
        self._select(int(self.session.neighborhoods.assignment[parcel]))
        if self.popup_on_click.isChecked():
            self._open_popup()
        value = self.session.land_values[parcel]
        name = self.session.neighborhoods.neighborhoods[self.selected].name
        self.statusBar().showMessage(
            f"Parcel {self.layer.keys[parcel]} · {name} · land value "
            + ("none (no land table or missing units)" if np.isnan(value) else f"${value:,.0f}"), 10000)

    def _select(self, neighborhood: int) -> None:
        self.selected = neighborhood
        if hasattr(self, "compare_pane"):
            self.compare_pane.map_selection = neighborhood
        self.statusBar().showMessage(f"Selected {self.session.neighborhoods.neighborhoods[neighborhood].name}", 5000)
        self.map.bridge.selectionChanged.emit(neighborhood)
        self._update_sales_neighborhood()
        if hasattr(self, "land_panel"):
            self.land_panel.show_neighborhood(neighborhood)
            self.popup.show_neighborhood(neighborhood)
        self._refresh_metrics()

    def lasso(self, coords, mode: str = "create") -> int | None:
        """Shift+drag: create a child of the selected neighborhood from lassoed parcels.
        Ctrl+shift+drag (``mode="reassign"``): move lassoed parcels into the selected neighborhood."""
        if mode == "reassign":
            return self.reassign(coords)
        t0 = time.perf_counter()
        parcels = self.layer.query_polygon(coords)
        child = self.session.create_neighborhood(self.selected, parcels)
        self.timing.emit("lasso_query", (time.perf_counter() - t0) * 1000)
        if child is None:
            self.statusBar().showMessage("Lasso contained no parcels from the selected neighborhood", 5000)
            return None
        self._assignment_changed()
        return child

    def reassign(self, coords) -> int | None:
        model = self.session.neighborhoods
        moving, emptied = self.session.plan_reassign(self.selected, self.layer.query_polygon(coords))
        target = model.neighborhoods[self.selected].name
        if moving.size == 0:
            self.statusBar().showMessage(f"Lasso contained no parcels outside {target}", 5000)
            return None
        if emptied:
            names = "".join(f"<li>{model.neighborhoods[e].name}</li>" for e in emptied)
            answer = QMessageBox.question(
                self, "Neighborhoods will be deleted",
                f"Moving {moving.size:,} parcels into <b>{target}</b> leaves "
                f"{'this neighborhood' if len(emptied) == 1 else 'these neighborhoods'} empty:<ul>{names}</ul>"
                f"{'It' if len(emptied) == 1 else 'They'} will be deleted. Continue?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Ok:
                return None
        self.session.reassign(self.selected, moving)
        self._assignment_changed()
        self.statusBar().showMessage(f"Moved {moving.size:,} parcels into {target}"
                                     + (f"; deleted {len(emptied)} empty neighborhood(s)" if emptied else ""), 8000)
        return self.selected

    def undo(self) -> None:
        if self.session.undo() is not None:
            self._after_history_change()

    def redo(self) -> None:
        if self.session.redo() is not None:
            self._after_history_change()

    def _after_history_change(self) -> None:
        # The selected neighborhood may no longer exist after undo (or may be deleted after redo).
        model = self.session.neighborhoods
        if self.selected >= len(model.neighborhoods) or model.neighborhoods[self.selected].deleted:
            self._select(ROOT)
        self._assignment_changed()
        if self.sales_pane is not None:
            self.sales_pane.flags_changed()
            self._sales_changed()

    def _show_timing(self, name: str, ms: float) -> None:
        self.statusBar().showMessage(f"{name}: {ms:,.0f} ms", 8000)
