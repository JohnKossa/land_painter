"""Land table panel: view and edit the selected neighborhood's land table.

Edits are made to a draft, previewed live, and committed with Apply as one
undoable action (so typing a rate does not fill the undo history).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QCompleter, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QRadioButton, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..data.config import CONTINUOUS
from ..land_table import ADD, IN, LOT, MULTIPLY, OPS, Adjustment, Condition, FlatRate, LandTable, evaluate
from ..metrics import compute, per_unit_land_evidence
from ..session import Session
from .metrics_panel import ratio_html


def _money(value: float) -> str:
    if value is None or np.isnan(value):
        return "—"
    return f"−${-value:,.0f}" if value < 0 else f"${value:,.0f}"


def summarize(values: np.ndarray) -> str:
    valid = values[~np.isnan(values)]
    if not len(values):
        return "no parcels"
    if not len(valid):
        return f"{len(values):,} parcels, none valued"
    missing = len(values) - len(valid)
    negative = int((valid < 0).sum())
    text = f"median {_money(float(np.median(valid)))} · total {_money(float(valid.sum()))}"
    if missing:
        text += f" · {missing:,} without a value"
    if negative:
        text += f" · {negative:,} negative (check unit values for placeholders)"
    return text


# --- Neighborhood picker ------------------------------------------------------------------


class NeighborhoodPicker(QComboBox):
    """All neighborhoods, indented by depth; type to search."""

    chosen = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMaxVisibleItems(25)
        completer = QCompleter(self.model(), self)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.activated[str].connect(self._completed)
        self.setCompleter(completer)
        self.activated.connect(lambda i: self.chosen.emit(self.itemData(i)))
        self._signature = None

    def set_neighborhoods(self, session: Session, current: int) -> None:
        model = session.neighborhoods
        signature = [(n.name, n.parent, n.deleted) for n in model.neighborhoods]
        if signature != self._signature:
            self.blockSignals(True)
            self.clear()
            for index in self._tree_order(model):
                self.addItem("    " * model.depth(index) + model.neighborhoods[index].name, index)
            self._signature = signature
            self.blockSignals(False)
        self.set_current(current)

    @staticmethod
    def _tree_order(model) -> list[int]:
        children: dict[int, list[int]] = {}
        for i, n in enumerate(model.neighborhoods):
            if n.parent is not None and not n.deleted:
                children.setdefault(n.parent, []).append(i)
        order, stack = [], [0]
        while stack:
            index = stack.pop()
            order.append(index)
            kids = sorted(children.get(index, []), key=lambda i: model.neighborhoods[i].name.lower())
            stack.extend(reversed(kids))
        return order

    def set_current(self, index: int) -> None:
        self.blockSignals(True)
        self.setCurrentIndex(self.findData(index))
        self.blockSignals(False)

    def _completed(self, text: str) -> None:
        i = self.findText(text, Qt.MatchFlag.MatchExactly)
        if i >= 0:
            self.setCurrentIndex(i)
            self.chosen.emit(self.itemData(i))


# --- Adjustment dialog ------------------------------------------------------------------------


class AdjustmentDialog(QDialog):
    def __init__(self, session: Session, parcels: np.ndarray, adjustment: Adjustment | None = None,
                 base_table: LandTable | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adjustment")
        self.resize(520, 560)
        self._session = session
        self._parcels = parcels
        # The land table without this adjustment, used by Guess to read the
        # matched parcels' and neighbours' current land values.
        self._base_table = base_table
        self._fields = session.dataset.filter_fields()

        self.name = QLineEdit()
        self.name.setPlaceholderText("optional, e.g. \"Lake access\"")
        self.field = QComboBox()
        for key, label, kind in self._fields:
            self.field.addItem(label, (key, kind))
        self.op = QComboBox()
        self.op.addItems(OPS)
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e12, 1e12)
        self.value.setDecimals(4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter values…")
        self.values = QListWidget()
        self.multiply = QRadioButton("Multiply the unit-of-measure value by")
        self.add = QRadioButton("Add a fixed amount ($)")
        kind_group = QButtonGroup(self)
        kind_group.addButton(self.multiply)
        kind_group.addButton(self.add)
        self.multiply.setChecked(True)
        self.amount = QDoubleSpinBox()
        self.amount.setRange(-1e12, 1e12)
        self.amount.setDecimals(4)
        self.amount.setValue(1.0)
        self.guess = QPushButton("Guess")
        self.guess.setToolTip("Set the amount so the matched parcels' median land value\n"
                              "matches the median of the rest of the neighborhood.")
        self.guess.clicked.connect(self._guess)
        self.guess.setEnabled(base_table is not None)
        self.guess_info = QLabel()
        self.guess_info.setWordWrap(True)
        self.guess_info.setStyleSheet("color: #888")
        self.matches = QLabel()

        self.continuous = QWidget()
        row = QHBoxLayout(self.continuous)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.op)
        row.addWidget(self.value, 1)
        self.categorical = QWidget()
        col = QVBoxLayout(self.categorical)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.search)
        col.addWidget(self.values)
        self.condition = QStackedWidget()
        self.condition.addWidget(self.continuous)
        self.condition.addWidget(self.categorical)

        form = QFormLayout()
        form.addRow("Name:", self.name)
        form.addRow("Applies where:", self.field)
        form.addRow("", self.condition)
        form.addRow("", self.matches)
        form.addRow("Effect:", self.multiply)
        form.addRow("", self.add)
        amount_row = QHBoxLayout()
        amount_row.addWidget(self.amount, 1)
        amount_row.addWidget(self.guess)
        form.addRow("Amount:", amount_row)
        form.addRow("", self.guess_info)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        layout = QVBoxLayout(self)
        layout.addLayout(form, 1)
        layout.addWidget(buttons)

        self.field.currentIndexChanged.connect(self._field_changed)
        self.op.currentIndexChanged.connect(self._update_matches)
        self.value.valueChanged.connect(self._update_matches)
        self.values.itemChanged.connect(self._update_matches)
        self.search.textChanged.connect(self._filter_values)
        self.multiply.toggled.connect(self._kind_changed)

        if not self._fields:
            self.matches.setText("No characteristics are available for filters. Mark some as Continuous or "
                                 "Categorical on the review screen when creating a session.")
            self._ok.setEnabled(False)
            return
        if adjustment is not None:
            self._load(adjustment)
        else:
            self._field_changed()

    def _load(self, adjustment: Adjustment) -> None:
        c = adjustment.condition
        index = next((i for i, (key, _, _) in enumerate(self._fields) if key == c.field), 0)
        self.field.setCurrentIndex(index)
        self._field_changed()
        if c.op == IN:
            for i in range(self.values.count()):
                item = self.values.item(i)
                item.setCheckState(Qt.CheckState.Checked if item.text() in c.values else Qt.CheckState.Unchecked)
        else:
            self.op.setCurrentText(c.op)
            self.value.setValue(c.value)
        (self.multiply if adjustment.kind == MULTIPLY else self.add).setChecked(True)
        self.amount.setValue(adjustment.amount)
        self.name.setText(adjustment.name)
        self._update_matches()

    def _field_changed(self) -> None:
        key, kind = self.field.currentData()
        self.condition.setCurrentWidget(self.continuous if kind == CONTINUOUS else self.categorical)
        if kind != CONTINUOUS:
            self.values.blockSignals(True)
            self.values.clear()
            for value in self._session.dataset.distinct_values(key):
                item = QListWidgetItem(value)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.values.addItem(item)
            self.values.blockSignals(False)
            self._filter_values()
        self._update_matches()

    def _filter_values(self) -> None:
        text = self.search.text().lower()
        for i in range(self.values.count()):
            item = self.values.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def _kind_changed(self) -> None:
        if self.multiply.isChecked() and self.amount.value() == 0:
            self.amount.setValue(1.0)

    def condition_value(self) -> Condition:
        key, kind = self.field.currentData()
        if kind == CONTINUOUS:
            return Condition(key, self.op.currentText(), self.value.value())
        checked = tuple(self.values.item(i).text() for i in range(self.values.count())
                        if self.values.item(i).checkState() == Qt.CheckState.Checked)
        return Condition(key, IN, values=checked)

    def _update_matches(self) -> None:
        condition = self.condition_value()
        data = self._session.dataset.characteristic(condition.field)[self._parcels]
        n = int(condition.mask(data).sum())
        self.matches.setText(f"Matches {n:,} of {len(self._parcels):,} parcels in this neighborhood")
        self._ok.setEnabled(condition.op != IN or bool(condition.values))
        self.guess_info.clear()  # any earlier guess no longer matches this condition

    def _guess(self) -> None:
        """Suggest the amount from the data.

        Additive: a flat shift so the matched parcels' median land value equals
        the rest of the neighborhood's (exact, since additions are flat).

        Multiplicative: calibrate the matched parcels' per-unit table value to
        their market per-unit value (vacant price/units and improved land
        residuals), so the multiplier reflects proportional, size-normalised
        evidence rather than raw land values.
        """
        if self._base_table is None:
            return
        dataset = self._session.dataset
        condition = self.condition_value()
        if condition.op == IN and not condition.values:
            self.guess_info.setText("Pick at least one value to match first.")
            return
        units = dataset.unit_values(self._base_table.unit)[self._parcels]
        land = evaluate(self._base_table, units, dataset.characteristic, self._parcels)
        matched = condition.mask(dataset.characteristic(condition.field)[self._parcels])

        if self.multiply.isChecked():
            self._guess_multiplier(land, units, matched)
        else:
            self._guess_addition(land, matched)

    def _guess_addition(self, land: np.ndarray, matched: np.ndarray) -> None:
        matched_lv = land[matched][~np.isnan(land[matched])]
        rest_lv = land[~matched][~np.isnan(land[~matched])]
        if not len(matched_lv) or not len(rest_lv):
            self.guess_info.setText("Need parcels with a land value in both the matched group "
                                    "and the rest of the neighborhood.")
            return
        m_med, r_med = float(np.median(matched_lv)), float(np.median(rest_lv))
        self.amount.setValue(r_med - m_med)
        self.guess_info.setText(f"Aligns medians — matched {_money(m_med)} → rest {_money(r_med)} "
                                f"({len(matched_lv):,} vs {len(rest_lv):,} valued parcels)")

    def _guess_multiplier(self, land: np.ndarray, units: np.ndarray, matched: np.ndarray) -> None:
        # Matched parcels' current per-unit table value.
        table_ok = matched & ~np.isnan(land) & (units > 0)
        if not table_ok.any():
            self.guess_info.setText("No matched parcels with a land value and positive units.")
            return
        table_pu = float(np.median(land[table_ok] / units[table_ok]))
        if table_pu == 0:
            self.guess_info.setText("Matched parcels' per-unit table value is $0 — can't compute a multiplier.")
            return
        # Matched parcels' market per-unit value (vacant + improved residuals).
        mask = np.zeros(len(self._session.land_values), dtype=bool)
        mask[self._parcels[matched]] = True
        market = per_unit_land_evidence(self._session, mask, self._base_table.unit)
        market = market[np.isfinite(market)]
        if not len(market):
            self.guess_info.setText("No usable sales on the matched parcels to calibrate against.")
            return
        market_pu = float(np.median(market))
        if market_pu <= 0:
            self.guess_info.setText("Market evidence gives a non-positive per-unit value — can't form a multiplier.")
            return
        self.amount.setValue(market_pu / table_pu)
        self.guess_info.setText(f"Calibrates to market — table {_money(table_pu)}/unit → "
                                f"market {_money(market_pu)}/unit ({len(market):,} sale observations)")

    def _accept(self) -> None:
        self.accept()

    def result_adjustment(self) -> Adjustment:
        return Adjustment(self.condition_value(), MULTIPLY if self.multiply.isChecked() else ADD,
                          self.amount.value(), self.name.text().strip())


# --- Panel -------------------------------------------------------------------------------------


class LandTablePanel(QWidget):
    neighborhoodChosen = Signal(int)
    applyRequested = Signal(int, object)  # neighborhood, LandTable | None
    applyToChildrenRequested = Signal(int)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.neighborhood = 0
        self.draft: LandTable | None = None
        self._parcels = np.zeros(0, dtype=np.int64)

        self.picker = NeighborhoodPicker()
        self.picker.chosen.connect(self.neighborhoodChosen)
        self.info = QLabel()

        # No-table page
        empty = QWidget()
        e_layout = QVBoxLayout(empty)
        e_layout.addWidget(QLabel("This neighborhood has no land table."))
        create = QPushButton("Create land table")
        create.clicked.connect(self._create)
        e_layout.addWidget(create)
        e_layout.addStretch()

        # Editor page
        editor = QWidget()
        self.unit = QComboBox()
        for name in session.dataset.unit_names():
            self.unit.addItem(name)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0, 1e12)
        self.rate.setDecimals(4)
        self.rate.setPrefix("$ ")
        self.rate.setGroupSeparatorShown(True)
        self.per_unit = QLabel()
        rate_row = QHBoxLayout()
        rate_row.addWidget(self.rate, 1)
        rate_row.addWidget(self.per_unit)
        form = QFormLayout()
        form.addRow("Unit of measure:", self.unit)
        form.addRow("Rate model:", QLabel("Flat rate"))
        form.addRow("Rate:", rate_row)

        self.adjustments = QTableWidget(0, 3)
        self.adjustments.setHorizontalHeaderLabels(["Applies where", "Effect", "Parcels"])
        self.adjustments.horizontalHeader().setStretchLastSection(False)
        self.adjustments.horizontalHeader().setSectionResizeMode(0, self.adjustments.horizontalHeader().ResizeMode.Stretch)
        self.adjustments.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.adjustments.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.adjustments.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.adjustments.verticalHeader().setVisible(False)
        self.adjustments.doubleClicked.connect(self._edit_adjustment)
        add = QPushButton("Add…")
        add.clicked.connect(self._add_adjustment)
        edit = QPushButton("Edit…")
        edit.clicked.connect(self._edit_adjustment)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_adjustment)
        adj_buttons = QHBoxLayout()
        for b in (add, edit, remove):
            adj_buttons.addWidget(b)
        adj_buttons.addStretch()
        adj_box = QGroupBox("Adjustments (multipliers apply first, then additions)")
        a_layout = QVBoxLayout(adj_box)
        a_layout.addWidget(self.adjustments)
        a_layout.addLayout(adj_buttons)

        remove_table = QPushButton("Remove land table")
        remove_table.clicked.connect(self._remove_table)
        ed_layout = QVBoxLayout(editor)
        ed_layout.setContentsMargins(0, 0, 0, 0)
        ed_layout.addLayout(form)
        ed_layout.addWidget(adj_box, 1)
        ed_layout.addWidget(remove_table, 0, Qt.AlignmentFlag.AlignLeft)

        self.pages = QStackedWidget()
        self.pages.addWidget(empty)
        self.pages.addWidget(editor)

        self.applied_summary = QLabel()
        self.applied_summary.setWordWrap(True)
        self.draft_summary = QLabel()
        self.draft_summary.setWordWrap(True)
        self.draft_summary.setStyleSheet("color: #e5c07b")
        summary_box = QGroupBox("Land values in this neighborhood")
        s_layout = QFormLayout(summary_box)
        s_layout.addRow("Applied:", self.applied_summary)
        s_layout.addRow("Draft:", self.draft_summary)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self._apply)
        self.revert_button = QPushButton("Revert")
        self.revert_button.clicked.connect(self._revert)
        self.children_button = QPushButton()
        self.children_button.clicked.connect(lambda: self.applyToChildrenRequested.emit(self.neighborhood))
        actions = QHBoxLayout()
        actions.addWidget(self.apply_button)
        actions.addWidget(self.revert_button)
        actions.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.picker)
        layout.addWidget(self.info)
        layout.addWidget(self.pages, 1)
        layout.addWidget(summary_box)
        layout.addLayout(actions)
        layout.addWidget(self.children_button)

        self.unit.currentIndexChanged.connect(self._edited)
        self.rate.valueChanged.connect(self._edited)
        self.show_neighborhood(0)

    # --- State ---------------------------------------------------------------------------------

    @property
    def applied(self) -> LandTable | None:
        return self.session.neighborhoods.neighborhoods[self.neighborhood].table

    @property
    def dirty(self) -> bool:
        return self.draft != self.applied

    def show_neighborhood(self, index: int) -> None:
        """Switch to a neighborhood, offering to apply unapplied edits first."""
        if index != self.neighborhood and self.dirty and self.isVisible():
            name = self.session.neighborhoods.neighborhoods[self.neighborhood].name
            answer = QMessageBox.question(self, "Unapplied changes", f"Apply your land table changes to {name}?",
                                          QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Discard)
            if answer == QMessageBox.StandardButton.Apply:
                self._apply()
        self.neighborhood = index
        self.reload()

    def reload(self) -> None:
        """Re-read the applied table from the session (after apply, undo, redo, or reassignment)."""
        model = self.session.neighborhoods
        if self.neighborhood >= len(model.neighborhoods):
            self.neighborhood = 0
        self.picker.set_neighborhoods(self.session, self.neighborhood)
        self._parcels = np.flatnonzero(model.assignment == self.neighborhood)
        n_children = len(model.children(self.neighborhood))
        self.info.setText(f"{len(self._parcels):,} parcels directly in this neighborhood · "
                          f"{n_children:,} child neighborhood{'s' if n_children != 1 else ''}")
        self.children_button.setText("Apply this land table to the child neighborhood" if n_children == 1
                                     else f"Apply this land table to all {n_children:,} child neighborhoods")
        self.children_button.setVisible(n_children > 0)
        self.draft = self.applied
        self._show_draft()

    def _show_draft(self) -> None:
        table = self.draft
        self.pages.setCurrentIndex(0 if table is None else 1)
        if table is not None:
            for w in (self.unit, self.rate):
                w.blockSignals(True)
            self.unit.setCurrentText(table.unit)
            self.rate.setValue(table.rate.rate)
            for w in (self.unit, self.rate):
                w.blockSignals(False)
            self._show_adjustments()
        self._update_state()

    def _show_adjustments(self) -> None:
        table = self.draft
        labels = {key: label for key, label, _ in self.session.dataset.filter_fields()}
        self.adjustments.setRowCount(len(table.adjustments))
        for r, adj in enumerate(table.adjustments):
            field = labels.get(adj.condition.field, adj.condition.field.split("::", 1)[-1])
            where = adj.condition.describe(field)
            n = int(adj.condition.mask(self.session.dataset.characteristic(adj.condition.field)[self._parcels]).sum())
            for c, text in enumerate([f"{adj.name}: {where}" if adj.name else where, adj.describe(), f"{n:,}"]):
                item = QTableWidgetItem(text)
                item.setToolTip(where)
                self.adjustments.setItem(r, c, item)
        self.adjustments.resizeColumnToContents(1)
        self.adjustments.resizeColumnToContents(2)

    def _update_state(self) -> None:
        table = self.draft
        if table is not None:
            self.per_unit.setText("per lot" if table.unit == LOT else f"per unit of {table.unit.lower()}")
        applied = self.session.land_values[self._parcels]
        self.applied_summary.setText(summarize(applied))
        if self.dirty:
            parcels, preview = self.session.values_for(self.neighborhood, table)
            land = self.session.land_values.copy()
            land[parcels] = preview
            mask = np.zeros(len(land), dtype=bool)
            mask[parcels] = True
            study = compute(self.session, mask, table.unit if table else None, land).ratio
            self.draft_summary.setText(summarize(preview) + "<br>Ratio study: " + ratio_html(study))
        else:
            self.draft_summary.setText("(no unapplied changes)")
        self.apply_button.setEnabled(self.dirty)
        self.revert_button.setEnabled(self.dirty)
        self.children_button.setEnabled(not self.dirty)

    # --- Editing -------------------------------------------------------------------------------

    def _edited(self) -> None:
        if self.draft is None:
            return
        self.draft = LandTable(self.unit.currentText(), FlatRate(self.rate.value()), self.draft.adjustments)
        self._update_state()

    def _set_adjustments(self, adjustments: list[Adjustment]) -> None:
        self.draft = LandTable(self.draft.unit, self.draft.rate, tuple(adjustments))
        self._show_adjustments()
        self._update_state()

    def _create(self) -> None:
        units = self.session.dataset.unit_names()
        self.draft = LandTable(units[1] if len(units) > 1 else LOT, FlatRate(0.0))
        self._show_draft()

    def _remove_table(self) -> None:
        self.draft = None
        self._show_draft()

    def _add_adjustment(self) -> None:
        # The draft has no new adjustment yet, so it is the base for Guess.
        dialog = AdjustmentDialog(self.session, self._parcels, base_table=self.draft, parent=self)
        if dialog.exec():
            self._set_adjustments([*self.draft.adjustments, dialog.result_adjustment()])

    def _edit_adjustment(self) -> None:
        row = self.adjustments.currentRow()
        if row < 0:
            return
        # Guess compares against the table without the adjustment being edited.
        others = tuple(a for i, a in enumerate(self.draft.adjustments) if i != row)
        base = LandTable(self.draft.unit, self.draft.rate, others)
        dialog = AdjustmentDialog(self.session, self._parcels, self.draft.adjustments[row], base_table=base, parent=self)
        if dialog.exec():
            adjustments = list(self.draft.adjustments)
            adjustments[row] = dialog.result_adjustment()
            self._set_adjustments(adjustments)

    def _remove_adjustment(self) -> None:
        row = self.adjustments.currentRow()
        if row >= 0:
            adjustments = list(self.draft.adjustments)
            del adjustments[row]
            self._set_adjustments(adjustments)

    def _apply(self) -> None:
        if self.dirty:
            self.applyRequested.emit(self.neighborhood, self.draft)

    def _revert(self) -> None:
        self.draft = self.applied
        self._show_draft()
