"""Main window: wires the session to the map and menus."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCursor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .data.dataset import build_dataset
from .data.review import DataError
from .data.sources import FILE_FILTER
from .map import MapView
from .neighborhoods import ROOT
from .sales import DISQUALIFIED, KNOWN_GOOD, UNFLAGGED
from .session import Session, SessionError, read_config, relocate_inputs
from .ui.map_controller import MapController, versioned
from .ui.session_files import SessionFilesMixin
from .ui.tasks import run_with_progress

SALE_COLORS = {UNFLAGGED: (240, 240, 240, 255), KNOWN_GOOD: (152, 195, 121, 255), DISQUALIFIED: (224, 108, 117, 255)}

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


class MainWindow(SessionFilesMixin, QMainWindow):
    timing = Signal(str, float)  # every timing, Python- or JS-side, in ms

    def __init__(self, session: Session):
        super().__init__()
        self.resize(1400, 900)

        self.session = session
        self.layer = session.layer
        self.selected = ROOT
        self._detail_target: tuple[int, int] | None = None  # (parcel, sale) shown in the details panel; sale -1 if none
        self._sale_view_version = 0  # bumped when the shown set of sales changes (not on flag-only changes)

        self.banner = QLabel()
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet("padding: 4px; color: #ccc; background: #2b2d30;")
        self.banner.setToolTip("Highlighted when outside IAAO ranges: median 0.90–1.10, COD ≤ 25 (vacant land), "
                               "PRD 0.98–1.03, PRB −0.05–0.05")
        self.map = MapView()
        self.map_controller = MapController(self.map, session, _basemap_urls)
        # Sale points depend only on the shown set, so a flag-only change (which bumps
        # colors) does not rebuild them; colors are cheap and always rebuilt.
        self.map_controller.provide_sales(
            versioned(self._sale_points, lambda: self._sale_view_version), self._sale_colors)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.banner)
        layout.addWidget(self.map, 1)
        self.setCentralWidget(central)
        self._build_menus()
        self.sales_pane = None
        if session.dataset.sales is not None:
            self._build_sales_pane()
        self._build_neighborhood_popup()
        self._build_compare_pane()
        self._build_parcel_details()
        self._refresh_banner()  # panel metrics are computed when the popup is first shown

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
        pane.saleSelected.connect(self._follow_sale_details)
        pane.saleActivated.connect(self._zoom_to_sale)
        pane.detailsRequested.connect(self._show_sale_details)
        pane.visibleChanged.connect(self._sales_view_changed)
        pane.show_on_map.toggled.connect(self._sales_view_changed)

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
        self.popup.mergeRequested.connect(self._merge_neighborhood)
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
        # The panel's metrics (and the spatial permutation test) are only computed
        # while the dock is shown; compute them when it opens. See _refresh_panel_metrics.
        dock.visibilityChanged.connect(self._popup_visibility)
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

    def _build_parcel_details(self) -> None:
        from .ui.parcel_details_panel import ParcelDetailsPanel

        self.parcel_details_panel = ParcelDetailsPanel()
        dock = self.parcel_details_dock = QDockWidget("Parcel details", self)
        dock.setObjectName("parcel_details")
        dock.setWidget(self.parcel_details_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        dock.setFloating(True)
        dock.resize(560, 680)
        dock.hide()
        # Fill in (or clear) the panel whenever it is shown.
        dock.visibilityChanged.connect(self._parcel_details_visibility)
        self._view_menu().addAction(dock.toggleViewAction())

    def _parcel_details_visibility(self, visible: bool) -> None:
        if visible and self._detail_target is not None:
            self._render_parcel_details(*self._detail_target)
        elif visible:
            self.parcel_details_panel.clear_view()

    def _sale_context(self, sale: int) -> str:
        model = self.sales_pane.model
        price, date, kind = model.price[sale], model.date_text[sale], model._type_text(sale)
        ratio = model.ratios()[sale]
        bits = [f"Price ${price:,.0f}" if not np.isnan(price) else "Price —"]
        if date:
            bits.append(date)
        bits.append(kind)
        bits.append(f"Ratio {ratio:.3f}" if np.isfinite(ratio) else "Ratio —")
        return "<b>Sale</b> · " + " · ".join(bits)

    def _render_parcel_details(self, parcel: int, sale: int = -1) -> None:
        from .parcel_details import parcel_details

        self._detail_target = (parcel, sale)
        details = parcel_details(self.session.dataset, self.session.land_values,
                                 self.session.neighborhoods.assignment, parcel)
        name = self.session.neighborhoods.neighborhoods[details.neighborhood].name
        self.parcel_details_panel.show_parcel(details, name, self._sale_context(sale) if sale >= 0 else None)

    def _show_parcel_details(self, parcel: int, sale: int = -1) -> None:
        self._render_parcel_details(parcel, sale)
        dock = self.parcel_details_dock
        dock.show()
        dock.raise_()

    def _show_sale_details(self, sale: int) -> None:
        self._show_parcel_details(int(self._sale_parcels[sale]), sale)

    def _follow_sale_details(self, sale: int) -> None:
        """Track the Sales list selection while the details dock is open."""
        if self.parcel_details_dock.isVisible():
            self._show_parcel_details(int(self._sale_parcels[sale]), sale)

    def _refresh_parcel_details(self) -> None:
        """Recompute the open details panel after values or neighborhoods change (without raising it)."""
        if self._detail_target is not None and self.parcel_details_dock.isVisible():
            self._render_parcel_details(*self._detail_target)

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

    def _merge_neighborhood(self, index: int) -> None:
        from .ui.neighborhood_popup import MergeDialog

        dialog = MergeDialog(self.session, index, self)
        if dialog.exec():
            destination = dialog.chosen()
            name = self.session.neighborhoods.neighborhoods[index].name
            self.session.merge_neighborhood(index, destination)
            self.selected = destination
            self._assignment_changed()
            self._select(destination)
            self.statusBar().showMessage(
                f"Merged {name} into {self.session.neighborhoods.neighborhoods[destination].name}", 5000)

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

    def _refresh_banner(self) -> None:
        """The always-visible county-wide ratio study.

        It depends on land values and sale flags, not on the selection, so it is
        refreshed on value/flag/assignment changes but not on a mere selection.
        """
        from .metrics import compute
        from .ui.metrics_panel import banner_html

        if not hasattr(self, "metrics_panel"):
            return
        everything = np.ones(self.layer.n_parcels, dtype=bool)
        self.banner.setText(banner_html(compute(self.session, everything).ratio))

    def _refresh_panel_metrics(self) -> None:
        """The selected neighborhood's metrics and spatial test, shown in the popup panel.

        Skipped while the popup is hidden (recomputed when it is shown, via
        _popup_visibility), since the spatial permutation test is expensive and
        runs on the UI thread. The popup is floating and hidden by default, so this
        keeps every map click, flag and edit from paying for metrics nothing shows.
        """
        from .metrics import compute
        from .spatial import residual_autocorrelation

        if not hasattr(self, "metrics_panel") or not self.popup_dock.isVisible():
            return
        model = self.session.neighborhoods
        table = model.neighborhoods[self.selected].table
        unit = table.unit if table else None
        mask = model.assignment == self.selected
        metrics = compute(self.session, mask, unit)
        spatial = residual_autocorrelation(self.session, mask)  # selected neighborhood only: not cheap county-wide
        self.metrics_panel.show_metrics(model.neighborhoods[self.selected].name, metrics, unit, spatial)

    def _popup_visibility(self, visible: bool) -> None:
        if visible:
            self._refresh_panel_metrics()

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
        self._refresh_banner()
        self._refresh_panel_metrics()
        self._refresh_parcel_details()
        self._refresh_chrome()

    def _focus_parcels(self, parcels) -> None:
        self.map_controller.focus_parcels(parcels)

    def _zoom_to_parcels(self, parcels) -> None:
        self.map_controller.zoom_to_parcels(parcels)

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

    def _assignment_changed(self) -> None:
        self.map_controller.refresh_assignment()
        if self.sales_pane is not None:
            self.sales_pane.model.refresh_neighborhoods()
            self._update_sales_neighborhood()
        if hasattr(self, "land_panel"):
            self.land_panel.reload()
            self.popup.show_neighborhood(self.selected)
        self._compare_data_changed()
        self._refresh_banner()
        self._refresh_panel_metrics()
        self._refresh_parcel_details()
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
        self.map_controller.refresh_sales()

    def _sales_view_changed(self, *_args) -> None:
        """The shown set of sales changed (filter, neighborhood scope, show-on-map),
        so the sale points — not just their colors — must be rebuilt."""
        self._sale_view_version += 1
        self._sales_changed()

    def _flag_sales(self, sales: np.ndarray, flag: int) -> None:
        if self.session.set_sale_flags(sales, flag):
            self.sales_pane.flags_changed()
            self._sales_changed()
            self._compare_data_changed()
            self._refresh_banner()
            self._refresh_panel_metrics()
            self._refresh_chrome()

    def _highlight_sale(self, sale: int) -> None:
        self.map_controller.highlight_rings(json.dumps(self.layer.parcel_rings(int(self._sale_parcels[sale]))))

    def _zoom_to_sale(self, sale: int) -> None:
        lng, lat = self._sale_xy[sale]
        self.map_controller.fly_to(float(lng), float(lat), 17.0)

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
        if self.parcel_details_dock.isVisible():
            self._render_parcel_details(parcel)
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
        self.map_controller.select(neighborhood)
        self._update_sales_neighborhood()
        if hasattr(self, "land_panel"):
            self.land_panel.show_neighborhood(neighborhood)
            self.popup.show_neighborhood(neighborhood)
        self._refresh_panel_metrics()  # selection doesn't change the county-wide banner

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
