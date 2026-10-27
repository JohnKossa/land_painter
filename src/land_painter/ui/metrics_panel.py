"""Metrics & evidence panel, and the ratio study banner."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QFormLayout, QGroupBox, QHeaderView, QLabel, QScrollArea, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..metrics import Metrics, Range, RatioStudy

UNDEFINED = "Undefined"
GOOD, BAD = "#98c379", "#e5c07b"

# Acceptable ranges, from the IAAO Standard on Ratio Studies (vacant land COD
# upper bound; PRD and PRB general guidance). Values outside are highlighted.
TARGETS = {
    "median": (0.90, 1.10),
    "cod": (0.0, 25.0),
    "prd": (0.98, 1.03),
    "prb": (-0.05, 0.05),
}


def fmt(value: float | None, digits: int = 2, percent: bool = False) -> str:
    if value is None:
        return UNDEFINED
    return f"{value * 100:.1f}%" if percent else f"{value:,.{digits}f}"


def money(value: float) -> str:
    return f"−${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def range_text(r: Range | None) -> str:
    if r is None:
        return UNDEFINED
    return f"{money(r.minimum)} ≤ {money(r.mean)} ≤ {money(r.maximum)}  (n={r.n:,})"


def colored(name: str, value: float | None, digits: int) -> str:
    text = fmt(value, digits)
    if value is None:
        return f"<span style='color:#888'>{text}</span>"
    low, high = TARGETS[name]
    return f"<span style='color:{GOOD if low <= value <= high else BAD}'>{text}</span>"


def ratio_html(study: RatioStudy) -> str:
    return (f"Median {colored('median', study.median, 3)} · COD {colored('cod', study.cod, 1)} · "
            f"PRD {colored('prd', study.prd, 3)} · PRB {colored('prb', study.prb, 3)}")


def banner_html(study: RatioStudy) -> str:
    return f"Land ratio study, all vacant sales (n={study.n:,}):  " + ratio_html(study)


class MetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold")
        self.scope = QLabel()
        self.scope.setWordWrap(True)

        self.ratio_n = QLabel()
        self.ratio = QLabel()
        self.ratio.setWordWrap(True)
        ratio_box = QGroupBox("Land ratio study (vacant sales)")
        form = QFormLayout(ratio_box)
        form.addRow("Sales:", self.ratio_n)
        form.addRow(self.ratio)

        self.spread = QLabel()
        self.alloc_table = QLabel()
        self.alloc_vacant = QLabel()
        self.alloc_sales = QLabel()
        self.over_price = QLabel()
        tests = QGroupBox("Tests")
        form = QFormLayout(tests)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("Spread:", self.spread)
        form.addRow(QLabel("Land allocation (share of total value that is land):"))
        form.addRow("  from land table:", self.alloc_table)
        form.addRow("  from vacant sales:", self.alloc_vacant)
        form.addRow("  from improved sales:", self.alloc_sales)
        form.addRow("Land value > sale price:", self.over_price)
        for label in (self.spread, self.alloc_table, self.alloc_vacant, self.alloc_sales, self.over_price):
            label.setWordWrap(True)

        self.evidence = QTableWidget(0, 4)
        self.evidence.setHorizontalHeaderLabels(["Unit", "Vacant sales: price per unit", "Cost residual per unit",
                                                 "COV (vacant)"])
        self.evidence.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.evidence.verticalHeader().setVisible(False)
        self.evidence.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.evidence.setToolTip("Ranges are min ≤ mean ≤ max. The cost residual (sale price − depreciated cost) is "
                                 "also the land value implied by the sale-based allocation ratio.")
        self.best = QLabel()
        evidence_box = QGroupBox("Evidence")
        e_layout = QVBoxLayout(evidence_box)
        e_layout.addWidget(self.evidence)
        e_layout.addWidget(self.best)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self.title)
        layout.addWidget(self.scope)
        layout.addWidget(ratio_box)
        layout.addWidget(tests)
        layout.addWidget(evidence_box, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # wrap text instead
        self.evidence.setMinimumHeight(130)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def show_metrics(self, name: str, m: Metrics, table_unit: str | None) -> None:
        self.title.setText(name)
        self.scope.setText(f"{m.n_parcels:,} parcels directly in this neighborhood · {m.n_vacant:,} vacant and "
                           f"{m.n_improved:,} improved sales (excluding disqualified)")
        self.ratio_n.setText(f"{m.ratio.n:,} with a land value" + ("" if table_unit else " (no land table)"))
        self.ratio.setText(ratio_html(m.ratio))

        unit = (m.spread_unit or "").lower()
        self.spread.setText(f"COD {fmt(m.spread_cod, 1)} of land value per {unit or 'unit'}")
        if not m.has_cost and m.allocation_vacant is None:
            for label in (self.alloc_table, self.alloc_vacant, self.alloc_sales):
                label.setText("Undefined (no depreciated cost mapped)")
        else:
            self.alloc_table.setText(f"{fmt(m.allocation_table, percent=True)} median of built parcels")
            self.alloc_vacant.setText(f"{fmt(m.allocation_vacant, percent=True)} median of built parcels "
                                      f"(mean vacant price per {unit})")
            self.alloc_sales.setText(f"{fmt(m.allocation_sales, percent=True)} median of improved sales")
        if m.over_price_checked:
            color = BAD if m.over_price else GOOD
            self.over_price.setText(f"<span style='color:{color}'>{m.over_price:,}</span> of "
                                    f"{m.over_price_checked:,} sales")
        else:
            self.over_price.setText(UNDEFINED + " (no valued sales)")

        self.evidence.setRowCount(len(m.evidence))
        for r, e in enumerate(m.evidence):
            label = e.unit + ("  ★" if e.unit == m.best_unit else "") + ("  (table)" if e.unit == table_unit else "")
            for c, text in enumerate([label, range_text(e.vacant), range_text(e.residual), fmt(e.cov, 1)]):
                self.evidence.setItem(r, c, QTableWidgetItem(text))
        self.best.setText(f"★ Best unit (lowest COV of vacant sale prices per unit): {m.best_unit or UNDEFINED}")
