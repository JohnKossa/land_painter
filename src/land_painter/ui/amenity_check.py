"""Check an adjustment's amenity against the market (land/building allocation, test 5c)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QRadioButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..allocation import AmenityCheck, Premium, amenity_allocation
from ..data.config import CONTINUOUS
from ..land_table import LOT, Adjustment, LandTable
from ..session import Session
from .metrics_panel import BAD, GOOD

Z95 = 1.96
EXPLAIN = (
    "Fits ln(sale price), ln(land value) and ln(depreciated cost) on the same improved sales, with the chosen "
    "controls and (across neighborhoods) neighborhood fixed effects. Each amenity coefficient is shown as a "
    "percent and in dollars at the median sale.\n\n"
    "What to look for:\n"
    "  • Building share: how much of the assessed premium sits in the building. A building's cost shouldn't "
    "depend on location, so a large share means the cost side is carrying location value the land table "
    "should hold.\n"
    "  • Land + building vs market: whether the assessed premium matches what buyers pay. A sign that differs "
    "from the market (e.g. a premium where buyers pay a discount) points to a wrong adjustment.\n"
    "  • A wide market range that spans 0 means the sales can't pin down the premium; don't tune to it.\n\n"
    "Land values include this neighborhood's draft land table. Controls enter as ln(x) when every value is "
    "positive, otherwise linearly.")


def _money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"−${-value:,.0f}" if value < 0 else f"${value:,.0f}"


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


class AmenityCheckDialog(QDialog):
    # Remembered between openings within a session (control keys, scope)
    remembered_controls: set[str] | None = None
    remembered_all: bool = True

    def __init__(self, session: Session, neighborhood: int, draft: LandTable, adjustment: Adjustment,
                 label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Check against market — {label}")
        self.resize(760, 640)
        self._session = session
        self._neighborhood = neighborhood
        self._draft = draft
        self._adjustment = adjustment

        heading = QLabel(f"Amenity: <b>{label}</b>")
        heading.setWordWrap(True)

        # Scope
        self.all_scope = QRadioButton("All neighborhoods, with neighborhood fixed effects (more sales)")
        self.one_scope = QRadioButton("This neighborhood only")
        scope_group = QButtonGroup(self)
        scope_group.addButton(self.all_scope)
        scope_group.addButton(self.one_scope)
        (self.all_scope if AmenityCheckDialog.remembered_all else self.one_scope).setChecked(True)
        scope_box = QGroupBox("Sales used")
        s_layout = QVBoxLayout(scope_box)
        s_layout.addWidget(self.all_scope)
        s_layout.addWidget(self.one_scope)

        # Controls: units (sizes) and continuous characteristics
        dataset = session.dataset
        self.controls = QListWidget()
        default = {f"unit:{draft.unit}"} if draft.unit != LOT else set()
        chosen = AmenityCheckDialog.remembered_controls if AmenityCheckDialog.remembered_controls is not None \
            else default
        options = [(f"unit:{name}", name) for name in dataset.unit_names() if name != LOT]
        options += [(f"char:{key}", label) for key, label, kind in dataset.filter_fields() if kind == CONTINUOUS]
        for key, text in options:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in chosen else Qt.CheckState.Unchecked)
            self.controls.addItem(item)
        self.controls.setMaximumHeight(130)
        controls_box = QGroupBox("Structural controls (e.g. living area, age, lot size)")
        c_layout = QVBoxLayout(controls_box)
        c_layout.addWidget(self.controls)
        if not options:
            c_layout.addWidget(QLabel("No units or continuous characteristics are available as controls."))

        run = QPushButton("Run")
        run.setDefault(True)
        run.clicked.connect(self._run)
        top = QHBoxLayout()
        top.addWidget(scope_box, 1)
        top.addWidget(controls_box, 1)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Premium", "95% range", "≈ $ at median", "Median value"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setToolTip(EXPLAIN)
        self.summary = QLabel("Choose the sales and controls, then Run.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setToolTip(EXPLAIN)
        help_label = QLabel("ⓘ What to look for (hover)")
        help_label.setToolTip(EXPLAIN)
        help_label.setStyleSheet("color: #888")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addLayout(top)
        layout.addWidget(run, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.summary)
        layout.addWidget(help_label)
        layout.addWidget(buttons)

    def _chosen_controls(self) -> list[tuple[str, str]]:
        items = (self.controls.item(i) for i in range(self.controls.count()))
        return [(item.data(Qt.ItemDataRole.UserRole), item.text()) for item in items
                if item.checkState() == Qt.CheckState.Checked]

    def _run(self) -> None:
        session, dataset = self._session, self._session.dataset
        chosen = self._chosen_controls()
        AmenityCheckDialog.remembered_controls = {key for key, _ in chosen}
        AmenityCheckDialog.remembered_all = self.all_scope.isChecked()

        controls = {}
        for key, text in chosen:
            kind, name = key.split(":", 1)
            values = dataset.unit_values(name) if kind == "unit" else dataset.characteristic(name)
            controls[text] = np.asarray(values, dtype=float)
        # Land values with this neighborhood's draft table in place
        parcels, preview = session.values_for(self._neighborhood, self._draft)
        land = session.land_values.copy()
        land[parcels] = preview
        if self.all_scope.isChecked():
            mask = np.ones(len(land), dtype=bool)
        else:
            mask = session.neighborhoods.assignment == self._neighborhood
        self._show(amenity_allocation(session, self._adjustment.condition, mask, controls, land))

    def _show(self, r: AmenityCheck) -> None:
        rows: list[tuple[str, Premium | None]] = [("Market (sale price)", r.market), ("Land (land tables)", r.land),
                                                   ("Building (depreciated cost)", r.building)]
        self.table.setRowCount(len(rows) + 1)
        for i, (name, prem) in enumerate(rows):
            if prem is None:
                cells = [name, "—", "—", "—", "—"]
            else:
                low, high = np.expm1(prem.beta - Z95 * prem.se), np.expm1(prem.beta + Z95 * prem.se)
                cells = [name, _pct(prem.percent), f"{_pct(low)} to {_pct(high)}", _money(prem.dollars),
                         _money(prem.median)]
            for c, text in enumerate(cells):
                self.table.setItem(i, c, QTableWidgetItem(text))
        assessed = r.assessed_dollars
        last = ["Land + building (assessed)", "", "", _money(assessed), ""]
        for c, text in enumerate(last):
            self.table.setItem(len(rows), c, QTableWidgetItem(text))

        scope = (f"{r.n_groups:,} neighborhoods (fixed effects)" if r.n_groups > 1 else "one neighborhood")
        controls = ", ".join(r.controls) if r.controls else "none"
        lines = [f"{r.n_sales:,} improved sales ({r.n_amenity:,} with the amenity) across {scope}. "
                 f"Controls: {controls}."]
        if r.problem:
            lines.append(f"<span style='color:{BAD}'>{r.problem}</span>")
        else:
            lines += self._reading(r)
        self.summary.setText("<br>".join(lines))

    @staticmethod
    def _reading(r: AmenityCheck) -> list[str]:
        out = []
        market = r.market
        if abs(market.t) < Z95:
            out.append(f"<span style='color:{BAD}'>The market premium isn't distinguishable from 0 with these "
                       "sales (its range spans 0) — treat the comparison below with caution.</span>")
        share = r.building_share
        if r.building is None:
            out.append("No depreciated cost is mapped, so the building side can't be checked.")
        elif share is None:
            out.append(f"<span style='color:{BAD}'>Land and building premiums have opposite signs "
                       f"(land {_money(r.land.dollars)}, building {_money(r.building.dollars)}).</span>")
        else:
            color = BAD if share > 0.5 else GOOD
            out.append(f"Building share of the assessed premium: <span style='color:{color}'>{share * 100:.0f}%"
                       f"</span>" + (" — location value is likely booked as building." if share > 0.5 else ""))
        assessed = r.assessed_dollars
        if assessed is not None and market.dollars != 0:
            same_sign = assessed * market.dollars > 0
            color = GOOD if same_sign and 0.75 <= assessed / market.dollars <= 1.25 else BAD
            text = (f"Assessed premium {_money(assessed)} vs market {_money(market.dollars)}: "
                    f"<span style='color:{color}'>")
            text += (f"{assessed / market.dollars * 100:.0f}% of the market premium</span>" if same_sign
                     else "opposite sign to the market</span> — the adjustment's direction may be wrong")
            out.append(text)
        return out
