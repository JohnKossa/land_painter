"""Sales pane: browse, filter and flag sales.

The table model keeps the visible rows as an index array into
``Dataset.sales`` and sorts/filters with numpy, so it stays fast with tens
of thousands of sales.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from PySide6.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QMenu, QPushButton,
    QTableView, QVBoxLayout, QWidget,
)

from ..format import money, number, ratio
from ..sales import DISQUALIFIED, FLAG_LABELS, KNOWN_GOOD, UNFLAGGED
from .colors import GREEN, GREY, RED

FLAG_COLORS = {UNFLAGGED: QColor(GREY), KNOWN_GOOD: QColor(GREEN), DISQUALIFIED: QColor(RED)}
TYPE_VACANT, TYPE_IMPROVED, TYPE_ALL = "vacant", "improved", "all"
FLAG_ANY = -1


@dataclass
class Column:
    header: str
    text: Callable[[int], str]  # sale index -> display text
    key: Callable[[], np.ndarray]  # sort key for every sale (read live, so it reflects current state)
    numeric: bool = False  # right-aligned


class SalesTableModel(QAbstractTableModel):
    def __init__(self, sales: pl.DataFrame, flags: np.ndarray, land_values: np.ndarray, units: dict[str, np.ndarray],
                 neighborhood_of: Callable[[np.ndarray], np.ndarray]):
        """``flags`` (per sale) and ``land_values`` (per parcel) are the session's arrays, read live.
        ``units`` maps unit name -> quantity per sale; ``neighborhood_of`` maps parcels -> names."""
        super().__init__()
        self.sale_flags = flags
        self.land_values = land_values
        self.parcel = sales["parcel"].to_numpy()
        self.keys = sales["key"].to_numpy()
        self.price = sales["price"].to_numpy().astype(float)  # NaN where missing
        self.dates = sales["date"].to_numpy()
        self.date_text = sales["date"].dt.strftime("%Y-%m-%d").fill_null("").to_numpy()
        vacant = sales["vacant"]
        self.vacant = vacant.fill_null(False).to_numpy()
        self.type_known = vacant.is_not_null().to_numpy()
        self._neighborhood_of = neighborhood_of
        self.neighborhood = neighborhood_of(self.parcel)
        # Full-length (per-sale) caches, indexed by absolute sale index so they
        # survive filtering/sorting; cleared by refresh() when land values change.
        self._sale_land_values: np.ndarray | None = None
        self._ratios: np.ndarray | None = None

        self.columns = [
            Column("Flag", lambda s: FLAG_LABELS[int(self.sale_flags[s])], lambda: self.sale_flags),
            Column("Key", lambda s: str(self.keys[s]), lambda: self.keys),
            Column("Date", lambda s: self.date_text[s], lambda: self.dates),
            Column("Price", lambda s: money(self.price[s], blank=""), lambda: self.price, True),
            Column("Land value", lambda s: money(self.sale_land_values()[s], blank=""), self.sale_land_values, True),
            Column("Ratio", lambda s: ratio(self.ratios()[s]), self.ratios, True),
            Column("Type", self._type_text, lambda: self.vacant),
            Column("Neighborhood", lambda s: self.neighborhood[s], lambda: self.neighborhood),
        ]
        for name, quantity in units.items():
            q = np.asarray(quantity, dtype=float)
            per_unit = np.where(self.vacant & (q > 0), self.price / np.where(q > 0, q, 1), np.nan)
            self.columns += [
                Column(name, lambda s, q=q: number(q[s]), lambda q=q: q, True),
                Column(f"$ / {name.lower()}", lambda s, p=per_unit: money(p[s], decimals=2, blank=""),
                       lambda p=per_unit: p, True),
            ]
        self.flag_column = 0
        self.neighborhood_column = next(i for i, c in enumerate(self.columns) if c.header == "Neighborhood")
        self.rows = np.arange(len(self.parcel))
        self._sort: tuple[int, Qt.SortOrder] | None = None

    def sale_land_values(self) -> np.ndarray:
        """Land value of each sale's parcel (cached; cleared by refresh)."""
        if self._sale_land_values is None:
            self._sale_land_values = self.land_values[self.parcel]
        return self._sale_land_values

    def ratios(self) -> np.ndarray:
        """Land value ÷ price per sale (cached; cleared by refresh)."""
        if self._ratios is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                self._ratios = self.sale_land_values() / self.price
        return self._ratios

    def _type_text(self, sale: int) -> str:
        return ("Vacant" if self.vacant[sale] else "Improved") if self.type_known[sale] else "Unknown"

    # Qt model interface ---------------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section].header
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        sale = self.rows[index.row()]
        column = self.columns[index.column()]
        flag = int(self.sale_flags[sale])
        if role == Qt.ItemDataRole.DisplayRole:
            return column.text(sale)
        if role == Qt.ItemDataRole.ForegroundRole:
            if index.column() == self.flag_column:
                return QBrush(FLAG_COLORS[flag])
            if flag == DISQUALIFIED:
                return QBrush(QColor("#808080"))
        if role == Qt.ItemDataRole.FontRole and flag == DISQUALIFIED and index.column() != self.flag_column:
            font = QFont()
            font.setStrikeOut(True)
            return font
        if role == Qt.ItemDataRole.TextAlignmentRole and column.numeric:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    # Filtering and sorting -------------------------------------------------------------------

    def set_mask(self, mask: np.ndarray) -> None:
        self.beginResetModel()
        self.rows = np.flatnonzero(mask)
        if self._sort is not None:
            self._apply_sort(*self._sort)
        self.endResetModel()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        self.layoutAboutToBeChanged.emit()
        self._sort = (column, order)
        self._apply_sort(column, order)
        self.layoutChanged.emit()

    def _apply_sort(self, column: int, order: Qt.SortOrder) -> None:
        values = self.columns[column].key()[self.rows]
        if values.dtype.kind == "O":
            values = values.astype(str)
        if values.dtype.kind == "f":  # missing values last either way
            missing = np.isnan(values)
            order_idx = np.argsort(np.where(missing, np.inf, values), kind="stable")
            if order == Qt.SortOrder.DescendingOrder:
                order_idx = order_idx[::-1]
                order_idx = np.concatenate([order_idx[~missing[order_idx]], order_idx[missing[order_idx]]])
        else:
            order_idx = np.argsort(values, kind="stable")
            if order == Qt.SortOrder.DescendingOrder:
                order_idx = order_idx[::-1]
        self.rows = self.rows[order_idx]

    # Updates ---------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Repaint every cell (after flags or land values change)."""
        self._sale_land_values = None  # land values may have changed; recompute lazily
        self._ratios = None
        if len(self.rows):
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, self.columnCount() - 1))

    refresh_flags = refresh

    def refresh_neighborhoods(self) -> None:
        self.neighborhood = self._neighborhood_of(self.parcel)
        self.refresh()


class SalesPane(QWidget):
    flagRequested = Signal(object, int)  # sale indices, flag
    saleSelected = Signal(int)  # sale index (the current row)
    saleActivated = Signal(int)  # sale index (double-click: zoom to it)
    detailsRequested = Signal(int)  # sale index (show its parcel's details)
    visibleChanged = Signal()  # filter changed: the set of visible sales is different

    def __init__(self, model: SalesTableModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._neighborhood_mask: np.ndarray | None = None  # per sale; None = no neighborhood filter

        self.type_filter = QComboBox()
        self.type_filter.addItem("Vacant sales", TYPE_VACANT)
        self.type_filter.addItem("Improved sales", TYPE_IMPROVED)
        self.type_filter.addItem("All sales", TYPE_ALL)
        self.flag_filter = QComboBox()
        self.flag_filter.addItem("Any flag", FLAG_ANY)
        for flag, label in FLAG_LABELS.items():
            self.flag_filter.addItem(label, flag)
        self.in_neighborhood = QCheckBox("Selected neighborhood only")
        self.show_on_map = QCheckBox("Show on map")
        self.show_on_map.setChecked(True)
        for widget in (self.type_filter, self.flag_filter):
            widget.currentIndexChanged.connect(self.apply_filters)
        self.in_neighborhood.toggled.connect(self.apply_filters)

        self.table = QTableView()
        self.table.setModel(model)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.selectionModel().currentRowChanged.connect(self._current_changed)
        self.table.doubleClicked.connect(lambda index: self.saleActivated.emit(int(model.rows[index.row()])))

        buttons = QHBoxLayout()
        self._flag_actions = []
        for flag, shortcut in ((KNOWN_GOOD, "G"), (UNFLAGGED, "U"), (DISQUALIFIED, "D")):
            button = QPushButton(f"{FLAG_LABELS[flag]} ({shortcut})")
            button.clicked.connect(lambda _=False, f=flag: self._flag_selected(f))
            buttons.addWidget(button)
            action = QAction(FLAG_LABELS[flag], self.table)
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            action.triggered.connect(lambda _=False, f=flag: self._flag_selected(f))
            self.table.addAction(action)
            self._flag_actions.append(action)

        self.summary = QLabel()
        filters = QHBoxLayout()
        filters.addWidget(self.type_filter)
        filters.addWidget(self.flag_filter)
        filters.addWidget(self.in_neighborhood)
        filters.addStretch()
        filters.addWidget(self.show_on_map)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(filters)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.summary)

        self.apply_filters()
        self.table.resizeColumnsToContents()

    # Filtering ------------------------------------------------------------------------------------

    def set_neighborhood_mask(self, mask: np.ndarray | None) -> None:
        """Per-sale mask of sales in the selected neighborhood (and its descendants)."""
        self._neighborhood_mask = mask
        if self.in_neighborhood.isChecked():
            self.apply_filters()
        else:
            self._update_summary()

    def apply_filters(self) -> None:
        m = self.model
        kind = self.type_filter.currentData()
        mask = np.ones(len(m.parcel), dtype=bool)
        if kind == TYPE_VACANT:
            mask &= m.vacant & m.type_known
        elif kind == TYPE_IMPROVED:
            mask &= ~m.vacant & m.type_known
        flag = self.flag_filter.currentData()
        if flag != FLAG_ANY:
            mask &= m.sale_flags == flag
        if self.in_neighborhood.isChecked() and self._neighborhood_mask is not None:
            mask &= self._neighborhood_mask
        m.set_mask(mask)
        self._update_summary()
        self.visibleChanged.emit()

    def visible_sales(self) -> np.ndarray:
        return self.model.rows

    def _update_summary(self) -> None:
        rows = self.model.rows
        flags = self.model.sale_flags[rows]
        self.summary.setText(f"{len(rows):,} shown · {int((flags == KNOWN_GOOD).sum()):,} known good · "
                             f"{int((flags == DISQUALIFIED).sum()):,} disqualified")

    # Flagging ---------------------------------------------------------------------------------------

    def selected_sales(self) -> np.ndarray:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return self.model.rows[rows] if rows else np.zeros(0, dtype=np.int64)

    def _flag_selected(self, flag: int) -> None:
        sales = self.selected_sales()
        if sales.size:
            self.flagRequested.emit(sales, flag)

    def current_sale(self) -> int:
        """The sale under the current row, or -1 if none."""
        current = self.table.selectionModel().currentIndex()
        return int(self.model.rows[current.row()]) if current.isValid() else -1

    def _context_menu(self, pos) -> None:
        sale = self.current_sale()
        if not self.selected_sales().size and sale < 0:
            return
        menu = QMenu(self)
        for action in self._flag_actions:
            menu.addAction(action)
        if sale >= 0:
            menu.addSeparator()
            menu.addAction("View parcel details", lambda: self.detailsRequested.emit(sale))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def flags_changed(self) -> None:
        """Call after flags change (flagging, undo, redo)."""
        if self.flag_filter.currentData() != FLAG_ANY:
            self.apply_filters()
        else:
            self.model.refresh_flags()
            self._update_summary()

    # Selection ----------------------------------------------------------------------------------------

    def _current_changed(self, current: QModelIndex, _previous) -> None:
        if current.isValid():
            self.saleSelected.emit(int(self.model.rows[current.row()]))

    def select_sale(self, sale: int) -> None:
        """Select a sale's row (e.g. after clicking it on the map), if it is visible."""
        where = np.flatnonzero(self.model.rows == sale)
        if not where.size:
            return
        index = self.model.index(int(where[0]), 0)
        self.table.selectionModel().setCurrentIndex(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        self.table.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtCenter)
