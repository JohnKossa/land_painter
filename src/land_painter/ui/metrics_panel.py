"""Metrics & evidence panel, and the ratio study banner."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QFormLayout, QGroupBox, QHeaderView, QLabel, QScrollArea, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..format import money, percent, span
from ..metrics import CHD_MIN_PARCELS, CHD_TARGET, MIN_SALES, Metrics, Range, RatioStudy, RollComparison
from ..spatial import K, PERMUTATIONS, WINDOW_DAYS, Autocorrelation, SpatialTest
from .colors import BAD, GOOD, MUTED

UNDEFINED = "Undefined"
ROLL_COLUMN = 3  # evidence table column for the previous roll's land per unit

# Acceptable ranges, from the IAAO Standard on Ratio Studies (vacant land COD
# upper bound; PRD and PRB general guidance). Values outside are highlighted.
TARGETS = {
    "median": (0.90, 1.10),
    "cod": (0.0, 25.0),
    "prd": (0.98, 1.03),
    "prb": (-0.05, 0.05),
    "chd": (0.0, CHD_TARGET),  # rule of thumb, not an IAAO standard
}


def fmt(value: float | None, digits: int = 2, percent: bool = False) -> str:
    if value is None:
        return UNDEFINED
    return f"{value * 100:.1f}%" if percent else f"{value:,.{digits}f}"


def range_text(r: Range | None) -> str:
    if r is None:
        return UNDEFINED
    return (f"{money(r.minimum, decimals=2)} ≤ {money(r.mean, decimals=2)} ≤ "
            f"{money(r.maximum, decimals=2)}  (n={r.n:,})")


def stat_color(name: str, value: float | None) -> str:
    """GOOD inside the target range for ``name``, BAD outside, MUTED when undefined."""
    if value is None:
        return MUTED
    low, high = TARGETS[name]
    return GOOD if low <= value <= high else BAD


def colored(name: str, value: float | None, digits: int) -> str:
    return span(fmt(value, digits), stat_color(name, value))


# Ratio study table: (RatioStudy attribute, decimals); the attribute is also the TARGETS key
RATIO_STATS = [("median", 3), ("cod", 1), ("prd", 3), ("prb", 3)]
RATIO_COLUMNS = ["Land values", "Sales", "Median", "COD", "PRD", "PRB"]


SIGNIFICANCE = 0.05  # p-value below which spatial clustering is highlighted


def autocorrelation_html(a: Autocorrelation) -> str:
    if a.rho is None:
        return span(UNDEFINED, MUTED)
    color = BAD if a.p_value < SIGNIFICANCE else GOOD
    return (f"{span(f'ρ {a.rho:.2f}', color)} (p {a.p_value:.3f}, null {a.null_mean:.2f}, "
            f"n={a.n:,})")


def spatial_html(test: SpatialTest | None) -> str:
    if test is None or test.n_sales < 3:
        return UNDEFINED + " (needs 3+ vacant sales with a land value)"
    return (f"{autocorrelation_html(test.all_pairs)}<br>excluding neighbors sold within {WINDOW_DAYS} days: "
            f"{autocorrelation_html(test.time_controlled)}<br>"
            f"{test.n_sales:,} vacant sales, {test.k} nearest neighbors each")


def ratio_html(study: RatioStudy) -> str:
    return (f"Median {colored('median', study.median, 3)} · COD {colored('cod', study.cod, 1)} · "
            f"PRD {colored('prd', study.prd, 3)} · PRB {colored('prb', study.prb, 3)}")


def signed(value: float | None) -> str:
    return percent(value, signed=True, blank=UNDEFINED)


def roll_change_html(r: RollComparison) -> str:
    """Median change from the roll and its middle half, e.g. "+8.4% median (middle half +2.0% to +15.1%)"."""
    if r.median_change is None:
        return UNDEFINED + " (no parcels with both a roll and a new land value)"
    return f"{signed(r.median_change)} median (middle half {signed(r.q25)} to {signed(r.q75)})"


def banner_html(study: RatioStudy) -> str:
    return f"Land ratio study, all vacant sales (n={study.n:,}):  " + ratio_html(study)


class MetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold")
        self.scope = QLabel()
        self.scope.setWordWrap(True)

        # One row per source of land values (the land tables, and the previous roll when mapped),
        # on the same vacant sales, so they compare head to head.
        self.ratio_table = QTableWidget(0, len(RATIO_COLUMNS))
        self.ratio_table.setHorizontalHeaderLabels(RATIO_COLUMNS)
        self.ratio_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ratio_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ratio_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ratio_table.verticalHeader().setVisible(False)
        self.ratio_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.ratio_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ratio_table.setToolTip(
            "Land value ÷ sale price on this neighborhood's usable vacant sales. Highlighted when outside IAAO "
            "ranges: median 0.90–1.10, COD ≤ 25 (vacant land), PRD 0.98–1.03, PRB −0.05 to 0.05.\n"
            "With roll land value mapped, the previous roll is studied on the same sales, so the two rows compare "
            "head to head. Sales count only where that source has a land value.")
        self.ratio_note = QLabel()
        self.ratio_note.setStyleSheet(f"color: {MUTED}")
        ratio_box = QGroupBox("Land ratio study (vacant sales)")
        r_layout = QVBoxLayout(ratio_box)
        r_layout.addWidget(self.ratio_table)
        r_layout.addWidget(self.ratio_note)

        self.spread = QLabel()
        self.alloc_table = QLabel()
        self.alloc_vacant = QLabel()
        self.alloc_sales = QLabel()
        self.over_price = QLabel()
        self.spatial = QLabel()
        spatial_tip = (
            "Residual spatial autocorrelation: do over- and under-assessed vacant sales cluster in space inside "
            "this neighborhood?\n\n"
            f"What to look for: a high ρ with a low p (< {SIGNIFICANCE}, highlighted) means one part of the "
            "neighborhood is consistently over-assessed and another under-assessed. That usually points to:\n"
            "  • a neighborhood that is too large (two sub-markets) — consider splitting it, or\n"
            "  • a missing adjustment (water, a busy road, a view, a flood zone) — consider a multiplier.\n"
            "Rough guide when p is low: ρ under ~0.1 is negligible, 0.1–0.3 is worth a look on the map, over ~0.3 "
            "is strong. A high p means any clustering could be chance.\n"
            f"If ρ drops sharply once neighbors sold within {WINDOW_DAYS} days are excluded, suspect market timing "
            "(the time adjustment) rather than location.\n\n"
            "How it works: residual = ln(land value / price) − the neighborhood's median. ρ is the correlation of "
            f"each sale's residual with the mean residual of its {K} nearest sales (never on the same parcel). "
            f"p is one-sided, from {PERMUTATIONS} shuffles of the residuals; 'null' is the mean shuffled ρ.\n"
            "With fewer than ~30 sales the test has little power.")
        self.spatial.setToolTip(spatial_tip)
        spatial_label = QLabel("Spatial clustering of ratios ⓘ:")
        spatial_label.setToolTip(spatial_tip)
        tests = QGroupBox("Tests")
        form = QFormLayout(tests)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("CHD:", self.spread)
        self.spread.setToolTip(
            "Coefficient of horizontal dispersion: COD of land value per unit (the land table's unit) across the "
            "neighborhood's parcels, sold or not. Measures whether similar land is valued alike.\n"
            f"Rule of thumb: ≤ {CHD_TARGET:g}, with {CHD_MIN_PARCELS}+ parcels.")
        form.addRow(QLabel("Land allocation (share of total value that is land):"))
        form.addRow("  from land table:", self.alloc_table)
        form.addRow("  from vacant sales:", self.alloc_vacant)
        form.addRow("  from improved sales:", self.alloc_sales)
        form.addRow("Land value > sale price:", self.over_price)
        form.addRow(spatial_label, self.spatial)
        for label in (self.spread, self.alloc_table, self.alloc_vacant, self.alloc_sales, self.over_price,
                      self.spatial):
            label.setWordWrap(True)

        self.evidence = QTableWidget(0, 7)
        self.evidence.setHorizontalHeaderLabels(["Unit", "Vacant sales: price per unit", "Cost residual per unit",
                                                 "Previous roll: land per unit", "Spread (vacant)",
                                                 "Spread (improved)", "Score"])
        self.evidence.horizontalHeaderItem(ROLL_COLUMN).setToolTip(
            "The previous roll's land value per unit across this neighborhood's parcels. A reference for the "
            "rate, not market evidence: it doesn't count toward the score or the best unit.")
        self.evidence.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.evidence.verticalHeader().setVisible(False)
        self.evidence.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.evidence.setToolTip(
            "Ranges are min ≤ mean ≤ max. The cost residual (sale price − depreciated cost) is also the land value "
            "implied by the sale-based allocation ratio.\n\n"
            f"Spreads are the SD of log(value per unit) in percent (≈ COV), needing {MIN_SALES}+ sales; residuals "
            "≤ 0 are left out.\nScore combines both groups, weighting each sale by 1 / its group's noise (the "
            "group's lowest spread across units), around one shared mean. Lowest score is best.\n\n"
            "Previous roll (when mapped) is a reference, not evidence, and isn't scored.")
        self.best = QLabel()
        self.best.setWordWrap(True)
        evidence_box = QGroupBox("Evidence")
        e_layout = QVBoxLayout(evidence_box)
        e_layout.addWidget(self.evidence)
        e_layout.addWidget(self.best)

        # Previous roll (shown only when roll land value is mapped)
        self.roll_change = QLabel()
        self.roll_movers = QLabel()
        self.roll_coverage = QLabel()
        self.roll_box = QGroupBox("Change from previous roll")
        self.roll_box.setToolTip(
            "How far this neighborhood's land values moved from the previous roll's, parcel by parcel "
            "(new ÷ roll − 1).\nHow well each matches the market is in the ratio study above.")
        form = QFormLayout(self.roll_box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("Change from roll:", self.roll_change)
        form.addRow("Big movers:", self.roll_movers)
        form.addRow("Not compared:", self.roll_coverage)
        for label in (self.roll_change, self.roll_movers, self.roll_coverage):
            label.setWordWrap(True)
        self.roll_box.hide()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self.title)
        layout.addWidget(self.scope)
        layout.addWidget(ratio_box)
        layout.addWidget(tests)
        layout.addWidget(self.roll_box)
        layout.addWidget(evidence_box, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # wrap text instead
        self.evidence.setMinimumHeight(130)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def show_metrics(self, name: str, m: Metrics, table_unit: str | None, spatial: SpatialTest | None = None) -> None:
        self.title.setText(name)
        self.scope.setText(f"{m.n_parcels:,} parcels directly in this neighborhood · {m.n_vacant:,} vacant and "
                           f"{m.n_improved:,} improved sales (excluding disqualified)")
        rows = [("Land tables", m.ratio)]
        if m.roll is not None:
            rows.append(("Previous roll", m.roll.roll_ratio))
        self._show_ratio_rows(rows)
        self.ratio_note.setText("" if table_unit else "No land table in this neighborhood yet.")
        self.ratio_note.setVisible(not table_unit)

        unit = (m.spread_unit or "").lower()
        few = (f"  {span(f'(under {CHD_MIN_PARCELS}, unreliable)', BAD)}"
               if m.chd is not None and m.chd_n < CHD_MIN_PARCELS else "")
        self.spread.setText(f"{colored('chd', m.chd, 1)} of land value per {unit or 'unit'} "
                            f"({m.chd_n:,} parcels){few}")
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
            self.over_price.setText(f"{span(f'{m.over_price:,}', color)} of "
                                    f"{m.over_price_checked:,} sales")
        else:
            self.over_price.setText(UNDEFINED + " (no valued sales)")
        self.spatial.setText(spatial_html(spatial))
        self._show_roll(m)

        self.evidence.setRowCount(len(m.evidence))
        for r, e in enumerate(m.evidence):
            label = e.unit + ("  ★" if e.unit == m.best_unit else "") + ("  (table)" if e.unit == table_unit else "")
            improved = fmt(e.residual_spread, 1) + (f"  ({e.residuals_dropped:,} ≤ 0 left out)"
                                                    if e.residuals_dropped else "")
            for c, text in enumerate([label, range_text(e.vacant), range_text(e.residual), range_text(e.roll),
                                      fmt(e.vacant_spread, 1), improved, fmt(e.score, 1)]):
                self.evidence.setItem(r, c, QTableWidgetItem(text))
        self.evidence.setColumnHidden(ROLL_COLUMN, m.roll is None)
        w = m.weights
        if w.vacant_weight and w.improved_weight:
            basis = (f"vacant noise {fmt(w.vacant_noise, 1)}, improved noise {fmt(w.improved_noise, 1)}; "
                     f"each vacant sale counts {w.vacant_weight / w.improved_weight:.2f}× an improved sale")
        elif w.vacant_weight or w.improved_weight:
            basis = f"{'vacant' if w.vacant_weight else 'improved'} sales only (the other group has under {MIN_SALES})"
        else:
            basis = f"needs {MIN_SALES}+ vacant or improved sales"
        self.best.setText(f"★ Best unit (lowest precision-weighted spread): {m.best_unit or UNDEFINED} — {basis}")

    def _show_roll(self, m: Metrics) -> None:
        r = m.roll
        self.roll_box.setVisible(r is not None)
        if r is None:
            return
        self.roll_change.setText(roll_change_html(r))
        if r.n_compared:
            share = (r.n_up + r.n_down) / r.n_compared
            color = BAD if r.n_up or r.n_down else GOOD
            self.roll_movers.setText(
                f"{span(f'{r.n_up + r.n_down:,}', color)} of {r.n_compared:,} parcels "
                f"({share * 100:.0f}%) moved more than ±{r.threshold * 100:.0f}% · {r.n_up:,} up, {r.n_down:,} down")
        else:
            self.roll_movers.setText(UNDEFINED)
        parcels = "parcel" if r.n_no_roll == 1 else "parcels"
        self.roll_coverage.setText(f"{r.n_no_roll:,} {parcels} with no roll land value · "
                                   f"{r.n_no_value:,} with no new land value")

    def _show_ratio_rows(self, rows: list[tuple[str, RatioStudy]]) -> None:
        table = self.ratio_table
        table.setRowCount(len(rows))
        for r, (name, study) in enumerate(rows):
            cells = [(name, None), (f"{study.n:,}", None)]
            cells += [(fmt(getattr(study, stat), digits), stat) for stat, digits in RATIO_STATS]
            for c, (text, stat) in enumerate(cells):
                item = QTableWidgetItem(text)
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if stat is not None:
                    item.setForeground(QColor(stat_color(stat, getattr(study, stat))))
                table.setItem(r, c, item)
        # Fit the rows exactly: no scrolling, no empty space
        height = table.horizontalHeader().height() + 2 * table.frameWidth()
        height += sum(table.rowHeight(r) for r in range(len(rows)))
        table.setFixedHeight(height)
