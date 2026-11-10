"""Land/building split check dialog (tests 5a and 5b), run on demand from the Tools menu."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..format import money, span
from ..land_table import LOT
from ..session import Session
from ..split import LARGE, MEDIUM, MIN_GROUP, SplitCheck, Variance, split_check
from .colors import BAD, GOOD, MUTED

IMPORTED, APP = "imported", "app"
LEAKAGE_FLAG = 0.75  # building ω² at least this share of land ω² is highlighted

EXPLAIN = (
    "5a — Building ÷ location-free cost: for each improved parcel, roll building value ÷ depreciated cost. "
    "A building's cost shouldn't depend on where it is, so neighborhood should explain little of this ratio's "
    "variance. A large share means location value is being booked as building.\n\n"
    "5b — Leakage: the same measure for roll land value per unit, on the same parcels. If neighborhood "
    "explains the building ratio about as well as land (or better), location is split across both "
    "components, and land is under-booked where the building ratio runs high.\n\n"
    "η² is the share of variance between neighborhoods; it reads high with many small neighborhoods, so ω² "
    "(adjusted) and the ICC are the fairer numbers. Rough guide (Cohen): 0.01 small, 0.06 medium, 0.14 large.\n\n"
    f"Neighborhoods with fewer than {MIN_GROUP} improved parcels are left out.\n"
    "Only meaningful if the mapped depreciated cost is location-free (cost book only). If it already includes "
    "a neighborhood or market factor, ratios sit near 1 and nothing shows up.")


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


class SplitCheckDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Land/building split check (current roll)")
        self.resize(900, 720)
        self._session = session
        dataset = session.dataset

        self.group_by = QComboBox()
        if dataset.neighborhood_codes() is not None:
            self.group_by.addItem("Imported neighborhood codes (the roll's neighborhoods)", IMPORTED)
        self.group_by.addItem("Neighborhoods drawn in this session", APP)
        self.unit = QComboBox()
        names = dataset.unit_names()
        for name in [n for n in names if n != LOT] + [LOT]:
            self.unit.addItem(name)
        run = QPushButton("Run")
        run.setDefault(True)
        run.clicked.connect(self._run)
        options = QFormLayout()
        options.addRow("Group parcels by:", self.group_by)
        options.addRow("Land per:", self.unit)

        self.summary_table = QTableWidget(0, 6)
        self.summary_table.setHorizontalHeaderLabels(["Measure", "Parcels", "Neighborhoods", "η²", "ω²", "ICC"])
        self.summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.summary_table.setMaximumHeight(130)
        self.summary_table.setToolTip(EXPLAIN)
        self.reading = QLabel("Choose the grouping and unit, then Run.")
        self.reading.setWordWrap(True)
        self.reading.setTextFormat(Qt.TextFormat.RichText)
        help_label = QLabel("ⓘ What to look for (hover)")
        help_label.setToolTip(EXPLAIN)
        help_label.setStyleSheet(f"color: {MUTED}")

        self.groups = QTableWidget(0, 5)
        self.groups.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.groups.verticalHeader().setVisible(False)
        self.groups.setSortingEnabled(True)
        self.groups.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.groups.setToolTip("Neighborhoods with the highest building ratios are where location value is most "
                               "likely booked as building (and land most likely under-booked).")
        groups_box = QGroupBox("By neighborhood (highest building ratio first)")
        g_layout = QVBoxLayout(groups_box)
        g_layout.addWidget(self.groups)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        top = QHBoxLayout()
        top.addLayout(options, 1)
        top.addWidget(run, 0, Qt.AlignmentFlag.AlignBottom)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.summary_table)
        layout.addWidget(self.reading)
        layout.addWidget(help_label)
        layout.addWidget(groups_box, 1)
        layout.addWidget(buttons)

    def _groups(self) -> tuple[np.ndarray, dict[int, str]]:
        session = self._session
        if self.group_by.currentData() == IMPORTED:
            codes = session.dataset.neighborhood_codes()
            names = codes.drop_nulls().unique().sort().to_list()
            index = {name: i for i, name in enumerate(names)}
            ids = np.array([index.get(c, -1) if c is not None else -1 for c in codes.to_list()])
            return ids, dict(enumerate(names))
        model = session.neighborhoods
        return model.assignment.astype(np.int64), {i: n.name for i, n in enumerate(model.neighborhoods)}

    def _run(self) -> None:
        ids, names = self._groups()
        self._show(split_check(self._session.dataset, self._session.land_values, ids, names,
                               self.unit.currentText()))

    def _show(self, r: SplitCheck) -> None:
        unit = "lot" if r.unit == LOT else r.unit.lower()
        rows: list[tuple[str, Variance | None]] = [
            ("5a  Building ÷ depreciated cost (roll)", r.building),
            (f"5b  Land per {unit} (roll)", r.roll_land),
            (f"      Land per {unit} (this session's land tables)", r.app_land),
        ]
        self.summary_table.setRowCount(len(rows))
        for i, (name, v) in enumerate(rows):
            cells = [name] + (["—"] * 5 if v is None else
                              [f"{v.n:,}", f"{v.k:,}", _num(v.eta2), _num(v.omega2), _num(v.icc)])
            for c, text in enumerate(cells):
                self.summary_table.setItem(i, c, QTableWidgetItem(text))
        self.summary_table.resizeColumnsToContents()

        lines = []
        if r.problem:
            lines.append(span(r.problem, BAD))
        else:
            lines += self._reading(r, unit)
        self.reading.setText("<br>".join(lines))

        headers = ["Neighborhood", "Improved parcels", "Median building ÷ cost", f"Median roll land per {unit}",
                   f"Median session land per {unit}"]
        self.groups.setSortingEnabled(False)
        self.groups.setHorizontalHeaderLabels(headers)
        self.groups.setRowCount(len(r.groups))
        for i, g in enumerate(r.groups):
            values = [g.name, g.n, g.building_ratio, g.roll_land_per_unit, g.app_land_per_unit]
            texts = [g.name, f"{g.n:,}", _num(g.building_ratio), money(g.roll_land_per_unit, decimals=2),
                     money(g.app_land_per_unit, decimals=2)]
            for c, (value, text) in enumerate(zip(values, texts)):
                item = (QTableWidgetItem(text) if c == 0  # numbers sort numerically, missing last
                        else _NumericItem(text, -np.inf if value is None else float(value)))
                self.groups.setItem(i, c, item)
        self.groups.setSortingEnabled(True)

    @staticmethod
    def _reading(r: SplitCheck, unit: str) -> list[str]:
        out = [f"{r.n_improved:,} improved parcels with a roll building value and depreciated cost; "
               f"median building ÷ cost {_num(r.median_ratio)}."]
        if r.excluded_groups:
            out.append(f"{r.excluded_groups:,} neighborhoods with under {MIN_GROUP} such parcels left out "
                       f"({r.excluded_parcels:,} parcels).")
        b = r.building
        if b is None:
            out.append("The building ratio can't be decomposed (needs 2+ neighborhoods and some variation).")
            return out
        level = "large" if b.omega2 >= LARGE else "medium" if b.omega2 >= MEDIUM else "small"
        color = GOOD if b.omega2 < MEDIUM else BAD
        out.append(f"5a: neighborhood explains {span(f'{b.omega2 * 100:.0f}%', color)} "
                   f"(ω²; η² {b.eta2 * 100:.0f}%) of the building ratio's variance — a {level} effect"
                   + (", so location value appears to be booked as building." if b.omega2 >= MEDIUM else "."))
        if r.roll_land is None:
            out.append("5b: needs roll land value mapped (and positive units) to compare.")
        elif r.leakage is not None:
            color = BAD if r.leakage >= LEAKAGE_FLAG else GOOD
            out.append(f"5b: building ω² is {span(f'{r.leakage:.2f}×', color)} land's "
                       f"({b.omega2:.2f} vs {r.roll_land.omega2:.2f})"
                       + (" — location is split across land and building." if r.leakage >= LEAKAGE_FLAG
                          else " — location sits mainly in land."))
        if b.omega2 >= MEDIUM:
            out.append("The neighborhoods at the top of the table below are where land is most likely "
                       "under-booked.")
        return out


class _NumericItem(QTableWidgetItem):
    """A table cell that sorts by a number rather than its text."""

    def __init__(self, text: str, value: float):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)
