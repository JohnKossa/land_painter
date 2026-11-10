"""Land/building split check dialog (tests 5a and 5b), run on demand from the Tools menu."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..format import money, span
from ..land_table import LOT
from ..session import Session
from ..split import LARGE, MEDIUM, MIN_GROUP, SplitCheck, Variance, split_check
from .colors import BAD, GOOD, MUTED

IMPORTED, APP = "imported", "app"
LEAKAGE_FLAG = 0.75  # building ω² at least this share of land ω² is highlighted

EXPLAIN = (
    "What this checks: whether the current roll has loaded location value onto buildings instead of land.\n\n"
    "A building costs about the same to build wherever it stands, so its roll value as a share of its cost "
    "shouldn't depend much on the neighborhood. When it does — when whole neighborhoods run high — that location "
    "value has likely been booked as building, which means land is under-valued there.\n\n"
    "How to read it:\n"
    "  • The top table scores how strongly the neighborhood explains the building-to-cost ratio. Higher means "
    "more location value sitting in buildings; it's highlighted once the effect is more than small.\n"
    "  • Leakage compares that with how strongly the neighborhood explains land value per unit. If location "
    "explains buildings about as well as land (or better), the land/building split is off.\n"
    "  • The bottom table lists neighborhoods by building-to-cost ratio, highest first — where land is most "
    "likely under-booked.\n\n"
    "The numbers: η² is the share of the variation explained by neighborhood; ω² adjusts it to stay fair when "
    "there are many small neighborhoods, and the ICC is a related reliability measure. Rough guide: 0.01 small, "
    "0.06 moderate, 0.14 large.\n\n"
    "This only means something if the depreciated cost you mapped is location-free — straight from the cost "
    "manual, with no neighborhood or market factor baked in. If location is already in the cost, every ratio "
    f"sits near 1 and nothing shows up. Neighborhoods with fewer than {MIN_GROUP} improved parcels are left out.")


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
        self.groups.setToolTip("Neighborhoods with the highest building-to-cost ratios are where location value "
                               "is most likely booked as building — and land most likely under-assessed.")
        groups_box = QGroupBox("Neighborhoods — highest building-to-cost ratio first")
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
            ("Building ÷ cost ratio (roll)", r.building),
            (f"Land per {unit} (roll)", r.roll_land),
            (f"Land per {unit} (your land tables)", r.app_land),
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

        headers = ["Neighborhood", "Improved parcels", "Building ÷ cost (median)", f"Roll land per {unit} (median)",
                   f"Your land per {unit} (median)"]
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
        out = [f"{r.n_improved:,} improved parcels have both a roll building value and a depreciated cost "
               f"(the median building value is {_num(r.median_ratio)}× its cost)."]
        if r.excluded_groups:
            out.append(f"{r.excluded_groups:,} neighborhoods with fewer than {MIN_GROUP} such parcels were left out "
                       f"({r.excluded_parcels:,} parcels).")
        b = r.building
        if b is None:
            out.append("Not enough data to measure this — it needs at least 2 neighborhoods with some "
                       "variation in their building-to-cost ratios.")
            return out
        level = "large" if b.omega2 >= LARGE else "moderate" if b.omega2 >= MEDIUM else "small"
        color = GOOD if b.omega2 < MEDIUM else BAD
        verdict = (" Location value is likely being recorded as building, which under-assesses land in some "
                   "neighborhoods." if b.omega2 >= MEDIUM
                   else " Location barely shows up in building values, which is what you want.")
        out.append(f"Neighborhood explains {span(f'{b.omega2 * 100:.0f}%', color)} of the differences in "
                   f"building-to-cost ratios — a {level} effect (ω² {b.omega2:.2f}, η² {b.eta2:.2f})." + verdict)
        if r.roll_land is None:
            out.append("Map the roll's land value (with positive units) to also check it against the land side.")
        elif r.leakage is not None:
            color = BAD if r.leakage >= LEAKAGE_FLAG else GOOD
            out.append(f"Neighborhood explains building values {span(f'{r.leakage:.2f}×', color)} as strongly as it "
                       f"explains land values (ω² {b.omega2:.2f} vs {r.roll_land.omega2:.2f})."
                       + (" Location is split across both, so land is under-booked where building ratios run high."
                          if r.leakage >= LEAKAGE_FLAG else " Location sits mainly in land, as it should."))
        if b.omega2 >= MEDIUM:
            out.append("The neighborhoods at the top of the table below are where land is most likely "
                       "under-assessed.")
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
