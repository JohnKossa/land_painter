"""Compare tab: an attribute's distribution across chosen neighborhoods.

Small multiples (one histogram row per group) on a shared x-axis, shared bins
and a shared y-scale, so positions and heights compare directly. Bars are
shares of each group, so groups of different sizes compare. Identity is
carried by each row's label; color only separates parcels from sales in the
"sales vs. parcels" view (with a legend). Below the chart: a summary table and
a matrix of pairwise effect sizes.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QToolTip, QVBoxLayout, QWidget,
)

from ..compare import (
    BOTH, CLIFF_LEVELS, CRAMER_LEVELS, MAGNITUDES, MONEY, PARCELS, PERCENT, RATIO, SALES, Attribute, Comparison,
    attributes, bin_parcels, compare, group_parcels, has_descendants, rest_of_parent_selection, selection_for,
)
from ..session import Session

# Validated reference palette (dataviz skill): categorical slots 1–2, and the blue sequential ramp.
SERIES = {"light": {PARCELS: "#2a78d6", SALES: "#eb6834"}, "dark": {PARCELS: "#3987e5", SALES: "#d95926"}}
# Effect-size shading, negligible → large (one hue; more contrast against the surface = bigger)
SHADES = {"light": [None, "#cde2fb", "#86b6ef", "#2a78d6"], "dark": [None, "#104281", "#1c5cab", "#3987e5"]}
POPULATION_LABELS = {PARCELS: "Parcels", SALES: "Sales", BOTH: "Sales vs. parcels"}


def is_dark(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


def compact(value: float, kind: str) -> str:
    """Short number for axis ticks and tooltips."""
    if value is None or not math.isfinite(value):
        return "—"
    if kind == PERCENT:
        return f"{value * 100:+.0f}%"
    if kind == RATIO:
        return f"{value:.2f}"
    prefix = "$" if kind == MONEY else ""
    sign = "−" if value < 0 else ""
    v = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= limit:
            text = f"{v / limit:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{prefix}{text}{suffix}"
    return f"{sign}{prefix}{v:,.2f}".rstrip("0").rstrip(".") if v < 100 else f"{sign}{prefix}{v:,.0f}"


def full(value: float | None, kind: str) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if kind == PERCENT:
        return f"{value * 100:+.1f}%"
    if kind == RATIO:
        return f"{value:.3f}"
    sign = "−" if value < 0 else ""
    return f"{sign}{'$' if kind == MONEY else ''}{abs(value):,.2f}"


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = high - low
    if span <= 0:
        return [low]
    step = 10 ** math.floor(math.log10(span / count))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= count:
            step *= m
            break
    first = math.ceil(low / step) * step
    return [first + i * step for i in range(int((high - first) / step) + 1)]


def log_ticks(low: float, high: float) -> list[float]:
    """1-2-5 ticks between low and high (positive)."""
    ticks = []
    for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** exponent
            if low <= v <= high:
                ticks.append(v)
    return ticks


# --- Chart -----------------------------------------------------------------------------------------


class SmallMultiples(QWidget):
    RIGHT, ROW, TOP, AXIS, GAP = 64, 54, 10, 40, 2
    MIN_LEFT, MAX_LEFT_SHARE = 120, 0.4  # label column: fits the longest label, within these bounds

    @property
    def LEFT(self) -> int:  # noqa: N802 - reads like the other layout constants
        if not self.result or not self.result.groups:
            return self.MIN_LEFT
        widest = max(QFontMetrics(self.font()).horizontalAdvance(g.label) for g in self.result.groups) + 20
        return int(min(max(widest, self.MIN_LEFT), self.width() * self.MAX_LEFT_SHARE))

    barClicked = Signal(int, int)  # row, bin (or category)
    rowClicked = Signal(int)  # a row's label: the whole group

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.result: Comparison | None = None
        self.focus: tuple[int, int | None] | None = None  # (row, bin) shown on the map; bin None = whole row

    def set_result(self, result: Comparison | None) -> None:
        self.result = result
        self.focus = None
        rows = len(result.groups) if result else 0
        legend = 22 if result and result.population == BOTH else 0
        self.setMinimumHeight(self.TOP + legend + rows * self.ROW + self.AXIS if rows else 0)
        self.update()

    # Geometry ---------------------------------------------------------------------------------

    def _top(self) -> int:
        return self.TOP + (22 if self.result and self.result.population == BOTH else 0)

    def _n_bins(self) -> int:
        r = self.result
        if r is None:
            return 0
        if r.categories is not None:
            return len(r.categories.names)
        return len(r.histogram.edges) - 1 if r.histogram is not None else 0

    def _plot(self) -> QRectF:
        return QRectF(self.LEFT, self._top(), max(self.width() - self.LEFT - self.RIGHT, 10),
                      len(self.result.groups) * self.ROW)

    def _shares(self):
        r = self.result
        return r.categories.shares if r.categories is not None else r.histogram.shares

    def _x_of(self, value: float, plot: QRectF) -> float | None:
        """x for a value on a continuous axis, or None if off-scale."""
        h = self.result.histogram
        lo, hi = h.edges[0], h.edges[-1]
        if h.log:
            if value <= 0:
                return None
            lo, hi, value = math.log10(lo), math.log10(hi), math.log10(value)
        if not lo <= value <= hi:
            return None
        return plot.left() + (value - lo) / (hi - lo) * plot.width()

    # Painting -------------------------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        r = self.result
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text = pal.color(QPalette.ColorRole.WindowText)
        muted = QColor(text)
        muted.setAlpha(150)
        grid = QColor(text)
        grid.setAlpha(45)
        if r is None or not r.groups or self._n_bins() == 0:
            return
        mode = "dark" if is_dark(self) else "light"
        plot = self._plot()
        bins = self._n_bins()
        width = plot.width() / bins
        shares = self._shares()
        top_share = max((float(s.max()) for s in shares if len(s)), default=0) or 1
        metrics = QFontMetrics(self.font())
        small = QFont(self.font())
        small.setPointSizeF(max(self.font().pointSizeF() - 1, 7))

        if r.population == BOTH:  # legend: color = population
            x = plot.left()
            for population in (PARCELS, SALES):
                painter.fillRect(QRectF(x, self.TOP + 4, 12, 12), QColor(SERIES[mode][population]))
                painter.setPen(text)
                label = "Parcels" if population == PARCELS else "Sales"
                painter.drawText(QPointF(x + 18, self.TOP + 15), label)
                x += 18 + metrics.horizontalAdvance(label) + 20

        for i, group in enumerate(r.groups):
            row_top = plot.top() + i * self.ROW
            base = row_top + self.ROW - 8
            height = self.ROW - 16
            # Label (text ink, never the series color); bold while the whole row is on the map
            row_focused = self.focus == (i, None)
            if row_focused:
                bold = QFont(self.font())
                bold.setBold(True)
                painter.setFont(bold)
            painter.setPen(text)
            label = QFontMetrics(painter.font()).elidedText(group.label, Qt.TextElideMode.ElideMiddle,
                                                            self.LEFT - 12)
            painter.drawText(QRectF(4, row_top, self.LEFT - 12, self.ROW - 4),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)
            painter.setFont(self.font())
            if row_focused:  # a thin bracket around the row's plot area
                painter.setPen(QPen(text, 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(QRectF(plot.left() - 3, row_top + 1, plot.width() + 6, self.ROW - 4), 4, 4)
            painter.setPen(muted)
            painter.setFont(small)
            painter.drawText(QRectF(4, row_top + self.ROW / 2 + 4, self.LEFT - 12, self.ROW / 2 - 4),
                             Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, f"n={r.summaries[i].n:,}")
            painter.setFont(self.font())
            # Baseline
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(plot.left(), base), QPointF(plot.right(), base))
            # Bars: 2px gap between adjacent bars, rounded data end anchored to the baseline
            color = QColor(SERIES[mode][group.population])
            for b, share in enumerate(shares[i]):
                if share <= 0:
                    continue
                h = max(share / top_share * height, 1.5)
                x0 = plot.left() + b * width + self.GAP / 2
                w = max(width - self.GAP, 1)
                path = QPainterPath()
                radius = min(4.0, w / 2, h)
                path.addRoundedRect(QRectF(x0, base - h, w, h + radius), radius, radius)
                painter.save()
                painter.setClipRect(QRectF(x0, base - h, w, h))
                painter.fillPath(path, QBrush(color))
                painter.restore()
                if self.focus == (i, b):  # the bar shown on the map: a ring in text ink, with a surface gap
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(text, 2))
                    painter.drawRoundedRect(QRectF(x0 - 3, base - h - 3, w + 6, h + 3), radius + 2, radius + 2)
            # Median tick (continuous)
            median = r.summaries[i].median
            if r.histogram is not None and median is not None:
                x = self._x_of(median, plot)
                if x is not None:
                    painter.setPen(QPen(text, 1.5))
                    painter.drawLine(QPointF(x, row_top + 4), QPointF(x, base))
            # Off-scale counts
            if r.histogram is not None:
                below, above = r.histogram.below[i], r.histogram.above[i]
                notes = ([f"◂{below:,}"] if below else []) + ([f"{above:,}▸"] if above else [])
                if notes:
                    painter.setPen(muted)
                    painter.setFont(small)
                    painter.drawText(QRectF(plot.right() + 4, row_top, self.RIGHT - 6, self.ROW - 4),
                                     Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, " ".join(notes))
                    painter.setFont(self.font())

        # x-axis
        axis_y = plot.bottom() + 2
        painter.setPen(muted)
        painter.setFont(small)
        if r.categories is not None:
            for b, name in enumerate(r.categories.names):
                rect = QRectF(plot.left() + b * width, axis_y + 2, width, self.AXIS - 8)
                shown = QFontMetrics(small).elidedText(name, Qt.TextElideMode.ElideRight, int(width) - 2)
                painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, shown)
        else:
            h = r.histogram
            ticks = log_ticks(h.edges[0], h.edges[-1]) if h.log else nice_ticks(h.edges[0], h.edges[-1])
            for value in ticks:
                x = self._x_of(value, plot)
                if x is None:
                    continue
                painter.setPen(QPen(grid, 1))
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
                painter.setPen(muted)
                label = compact(value, r.attribute.fmt)
                painter.drawText(QRectF(x - 40, axis_y + 2, 80, 16), Qt.AlignmentFlag.AlignHCenter, label)
            caption = r.attribute.label + (" (log scale)" if h.log else "") + " · tick = median · ◂/▸ off-scale"
            painter.drawText(QRectF(plot.left(), axis_y + 18, plot.width(), 16), Qt.AlignmentFlag.AlignHCenter,
                             caption)

    # Hover --------------------------------------------------------------------------------------

    def _hit(self, pos) -> tuple[int, int | None] | None:
        """(row, bin) under ``pos``; bin None over a row's label; None elsewhere or over an empty bin."""
        r = self.result
        if r is None or not r.groups or self._n_bins() == 0:
            return None
        plot = self._plot()
        row = int((pos.y() - plot.top()) // self.ROW)
        if not 0 <= row < len(r.groups) or pos.y() < plot.top():
            return None
        if 4 <= pos.x() < plot.left() - 4:
            return row, None
        b = int((pos.x() - plot.left()) // (plot.width() / self._n_bins()))
        if not 0 <= b < self._n_bins() or pos.x() < plot.left() or self._shares()[row][b] <= 0:
            return None
        return row, b

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit(event.position())
        if hit is None:
            return
        row, b = hit
        if b is None:
            self.rowClicked.emit(row)
        else:
            self.barClicked.emit(row, b)

    def mouseMoveEvent(self, event) -> None:
        hit = self._hit(event.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.ArrowCursor)
        if hit is None:
            QToolTip.hideText()
            return
        r = self.result
        row, b = hit
        group, summary = r.groups[row], r.summaries[row]
        if b is None:
            noun = "parcels" if group.population == PARCELS else "sold parcels"
            QToolTip.showText(event.globalPosition().toPoint(),
                              f"<b>{group.label}</b><br>Click to show all its {noun} on the map", self)
            return
        noun = "parcels" if group.population == PARCELS else "sales"
        if r.categories is not None:
            what = r.categories.names[b]
            share, count = r.categories.shares[row][b], r.categories.counts[row][b]
        else:
            h = r.histogram
            what = f"{compact(h.edges[b], r.attribute.fmt)} – {compact(h.edges[b + 1], r.attribute.fmt)}"
            share, count = h.shares[row][b], h.counts[row][b]
        tip = f"<b>{group.label}</b><br>{what}: {share * 100:.1f}% of {noun} ({count:,} of {summary.n:,})"
        if summary.median is not None:
            tip += f"<br>Median {full(summary.median, r.attribute.fmt)}"
        shown = self.focus == (row, b)
        tip += "<br><i>" + ("Click again to clear the map" if shown else
                            "Click to show " + ("these parcels" if group.population == PARCELS
                                                else "the parcels of these sales") + " on the map") + "</i>"
        QToolTip.showText(event.globalPosition().toPoint(), tip, self)

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()


# --- Pane --------------------------------------------------------------------------------------------


class ComparePane(QWidget):
    focusRequested = Signal(object)  # parcel indices to show on the map, or None to clear
    zoomRequested = Signal(object)  # parcel indices to fit the map to

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._focus_parcels: np.ndarray | None = None  # what the map is showing, if anything
        self.map_selection = 0  # the neighborhood selected on the map
        self._dirty = True
        self._attributes = attributes(session)

        # Neighborhoods
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter neighborhoods…")
        self.hoods = QListWidget()
        self.hoods.setMaximumHeight(150)
        add = QPushButton("Add map selection")
        add.setToolTip("Check the neighborhood selected on the map")
        add.clicked.connect(self._add_selection)
        clear = QPushButton("Clear")
        clear.setToolTip("Uncheck everything and remove combined groups")
        clear.clicked.connect(self._clear)
        children = QPushButton("Children of selection")
        children.setToolTip("Compare the pieces of a split: checks every child of the neighborhood selected on "
                            "the map, each with its own sub-neighborhoods (replacing the current checks).\n"
                            "Similar children are candidates to merge.")
        children.clicked.connect(self._children_of_selection)
        rest = QPushButton("Selection vs. rest of parent")
        rest.setToolTip(
            "Was this split worth making? Compares the neighborhood selected on the map (with its own "
            "sub-neighborhoods, if any) against everything else under its parent: the parent's own parcels "
            "plus every sibling (replacing the current checks).")
        rest.clicked.connect(self._selection_vs_rest)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(clear)
        buttons.addStretch()
        shortcuts = QHBoxLayout()
        shortcuts.addWidget(children)
        shortcuts.addWidget(rest)
        shortcuts.addStretch()
        self._extras: list[tuple] = []  # keys of combined groups ("subtree" / "rest"), shown above the tree
        hood_box = QGroupBox("Neighborhoods")
        hood_box.setToolTip("A neighborhood in the tree means the parcels directly in it, as elsewhere in the "
                            "app.\n⊕ combined groups (from the shortcuts) include sub-neighborhoods.")
        h_layout = QVBoxLayout(hood_box)
        h_layout.addWidget(self.search)
        h_layout.addWidget(self.hoods)
        h_layout.addLayout(buttons)
        h_layout.addLayout(shortcuts)

        # Attribute and population
        self.attribute = QComboBox()
        self.attribute.setMaxVisibleItems(30)
        model = self.attribute.model()
        heading = None
        for a in self._attributes:
            if a.group != heading:
                heading = a.group
                self.attribute.addItem(f"— {heading} —")
                model.item(self.attribute.count() - 1).setEnabled(False)
            self.attribute.addItem(a.label, a.key)
        self.attribute.setCurrentIndex(1)
        self.population = QComboBox()
        for p in (PARCELS, SALES, BOTH):
            self.population.addItem(POPULATION_LABELS[p], p)
        if session.dataset.sales is None:
            self.population.setEnabled(False)
        self.population.setToolTip("Sales vs. parcels shows, for each neighborhood, its usable sales next to all "
                                   "its parcels: are the sales representative?")
        self.log = QCheckBox("Log scale")
        self.log.setToolTip("For skewed values such as lot size or price. Values ≤ 0 count as off-scale.")
        form = QFormLayout()
        form.addRow("Attribute:", self.attribute)
        form.addRow("Compare:", self.population)
        form.addRow("", self.log)

        top = QHBoxLayout()
        top.addWidget(hood_box, 1)
        top.addLayout(form, 1)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #888")

        self.chart = SmallMultiples()

        self.summary = QTableWidget(0, 0)
        self.summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary.verticalHeader().setVisible(False)
        self.summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.matrix = QTableWidget(0, 0)
        self.matrix.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.matrix.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.matrix.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.matrix_title = QLabel("<b>How different?</b> (each cell: the column compared with the row) ⓘ")
        self.matrix_note = QLabel()  # one-line legend; the full explanation is the tooltip
        self.matrix_note.setStyleSheet("color: #888")
        for table in (self.summary, self.matrix):
            table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustIgnored)

        # Everything below the controls in one scrolling column, each part at its natural height
        body = QWidget()
        b_layout = QVBoxLayout(body)
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.addWidget(self.chart)
        b_layout.addWidget(self.summary)
        b_layout.addWidget(self.matrix_title)
        b_layout.addWidget(self.matrix)
        b_layout.addWidget(self.matrix_note)
        b_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(top)
        layout.addWidget(self.message)
        # What the map is showing (visible only while a bar or row is linked to the map)
        self.focus_label = QLabel()
        self.focus_label.setWordWrap(True)
        zoom = QPushButton("Zoom to")
        zoom.setToolTip("Fit the map to these parcels")
        zoom.clicked.connect(lambda: self.zoomRequested.emit(self._focus_parcels))
        clear_focus = QPushButton("Clear map")
        clear_focus.setToolTip("Stop highlighting; the map returns to normal")
        clear_focus.clicked.connect(self.clear_focus)
        self.focus_bar = QWidget()
        f_layout = QHBoxLayout(self.focus_bar)
        f_layout.setContentsMargins(0, 0, 0, 0)
        f_layout.addWidget(self.focus_label, 1)
        f_layout.addWidget(zoom)
        f_layout.addWidget(clear_focus)
        self.focus_bar.hide()
        layout.addWidget(self.focus_bar)
        self.chart.barClicked.connect(self._bar_clicked)
        self.chart.rowClicked.connect(self._row_clicked)
        layout.addWidget(scroll, 1)

        self.search.textChanged.connect(self._filter)
        self.hoods.itemChanged.connect(self._recompute)
        self.attribute.currentIndexChanged.connect(self._recompute)
        self.population.currentIndexChanged.connect(self._recompute)
        self.log.toggled.connect(self._recompute)
        self.data_changed()

    # Neighborhood list -------------------------------------------------------------------------

    def checked(self) -> list[tuple]:
        """Keys of the checked entries, in list order (combined groups first, then the tree)."""
        return [self.hoods.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.hoods.count())
                if self.hoods.item(i).checkState() == Qt.CheckState.Checked]

    def _rebuild_list(self, check: set[tuple] | None = None) -> None:
        """Rebuild the list from the current tree, keeping checks (or setting them to ``check``)."""
        from .land_table_panel import NeighborhoodPicker

        keep = set(self.checked()) if check is None else check
        model = self.session.neighborhoods
        # Combined groups whose neighborhoods are gone (or that no longer apply) are dropped
        self._extras = [key for key in self._extras if selection_for(model, key) is not None]
        self.hoods.blockSignals(True)
        self.hoods.clear()

        def add(key: tuple, text: str, italic: bool = False) -> None:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in keep else Qt.CheckState.Unchecked)
            if italic:
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setToolTip("A combined group. Clear removes it.")
            self.hoods.addItem(item)

        for key in self._extras:
            add(key, "⊕ " + selection_for(model, key).label, italic=True)
        for index in NeighborhoodPicker._tree_order(model):
            add(("hood", index), "    " * model.depth(index) + model.neighborhoods[index].name)
        self.hoods.blockSignals(False)
        self._filter()

    def _children_of_selection(self) -> None:
        model = self.session.neighborhoods
        kids = model.children(self.map_selection)
        if not kids:
            self.message.setText(f"{model.neighborhoods[self.map_selection].name} has no child neighborhoods.")
            return
        # Each piece of the split is its whole region: a child with sub-neighborhoods includes them
        keys = [("subtree", k) if has_descendants(model, k) else ("hood", k) for k in kids]
        self._extras = [key for key in keys if key[0] == "subtree"]
        self._rebuild_list(check=set(keys))
        self._recompute()

    def _selection_vs_rest(self) -> None:
        model = self.session.neighborhoods
        index = self.map_selection
        rest = rest_of_parent_selection(model, index)
        if rest is None:
            self.message.setText(f"{model.neighborhoods[index].name} is the top neighborhood: it has no parent "
                                 "to compare against. Select a child neighborhood on the map first.")
            return
        # The selection includes its own sub-neighborhoods, so the two groups split the parent between them
        mine = ("subtree", index) if has_descendants(model, index) else ("hood", index)
        self._extras = [key for key in (mine, rest.key) if key[0] != "hood"]
        self._rebuild_list(check={mine, rest.key})
        self._recompute()

    def _filter(self) -> None:
        text = self.search.text().lower()
        for i in range(self.hoods.count()):
            item = self.hoods.item(i)
            item.setHidden(bool(text) and text not in item.text().strip().lower())

    def _add_selection(self) -> None:
        for i in range(self.hoods.count()):
            item = self.hoods.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == ("hood", self.map_selection):
                item.setCheckState(Qt.CheckState.Checked)

    def _clear(self) -> None:
        self._extras = []
        self._rebuild_list(check=set())
        self._recompute()

    # Updates --------------------------------------------------------------------------------------

    def data_changed(self) -> None:
        """Land values, neighborhoods or sale flags changed: refresh now if visible, else when shown."""
        self._rebuild_list()
        self._dirty = True
        self._recompute()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self._recompute()

    def hideEvent(self, event) -> None:
        # Switching to another tab (or closing this one) puts the map back to normal
        self.clear_focus()
        super().hideEvent(event)

    # Map linking ---------------------------------------------------------------------------------

    def _bar_clicked(self, row: int, b: int) -> None:
        if self.chart.focus == (row, b):  # a second click clears
            self.clear_focus()
            return
        r = self.chart.result
        group = r.groups[row]
        if r.categories is not None:
            what = r.categories.names[b]
        else:
            h = r.histogram
            what = f"{full(h.edges[b], r.attribute.fmt)} – {full(h.edges[b + 1], r.attribute.fmt)}"
        self._set_focus((row, b), bin_parcels(r, row, b), f"{group.label}: {r.attribute.label} {what}")

    def _row_clicked(self, row: int) -> None:
        if self.chart.focus == (row, None):
            self.clear_focus()
            return
        r = self.chart.result
        self._set_focus((row, None), group_parcels(r, row), f"{r.groups[row].label}: all with a value")

    def _set_focus(self, focus: tuple, parcels: np.ndarray, description: str) -> None:
        self.chart.focus = focus
        self.chart.update()
        self._focus_parcels = parcels
        group = self.chart.result.groups[focus[0]]
        noun = "parcel" if len(parcels) == 1 else "parcels"
        sold = " (sold)" if group.population == SALES else ""
        self.focus_label.setText(f"<b>On the map:</b> {len(parcels):,} {noun}{sold} · {description}")
        self.focus_bar.show()
        self.focusRequested.emit(parcels)

    def clear_focus(self) -> None:
        if self._focus_parcels is None:
            return
        self._focus_parcels = None
        self.chart.focus = None
        self.chart.update()
        self.focus_bar.hide()
        self.focusRequested.emit(None)

    def current_attribute(self) -> Attribute | None:
        key = self.attribute.currentData()
        return next((a for a in self._attributes if a.key == key), None)

    def _recompute(self, *_args) -> None:
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        attr = self.current_attribute()
        model = self.session.neighborhoods
        hoods = [s for s in (selection_for(model, key) for key in self.checked()) if s is not None]
        population = self.population.currentData() if self.session.dataset.sales is not None else PARCELS
        notes = []
        if attr is not None and attr.sale_level and population == PARCELS:
            population = SALES
            notes.append(f"{attr.label} is only defined for sales, so sales are shown.")
        if attr is None or not hoods:
            self.message.setText("Check one or more neighborhoods to compare." if attr else "Pick an attribute.")
            self._show(None)
            return
        result = compare(self.session, attr, hoods, population, self.log.isChecked())
        self.log.setEnabled(not attr.categorical and result.log_available)
        if attr.categorical:
            notes.append("Categorical: bars are each group's share in each category; the rest fold into Other.")
        notes.append("Bars are shares of each group, on one shared scale.")
        self.message.setText(" ".join(notes))
        self._show(result)

    # Results --------------------------------------------------------------------------------------

    def _show(self, r: Comparison | None) -> None:
        self.clear_focus()  # new bins: whatever the map showed no longer matches a bar
        self.chart.set_result(r)
        self._show_summary(r)
        self._show_matrix(r)
        self._fit_tables()

    MAX_TABLE_ROWS = 8  # beyond this a table scrolls

    def _fit_tables(self) -> None:
        """Size each table to its rows (scrolling past MAX_TABLE_ROWS); hide it when empty."""
        for table in (self.summary, self.matrix):
            table.setVisible(table.rowCount() > 0)
            if not table.rowCount():
                continue
            table.resizeRowsToContents()
            rows = min(table.rowCount(), self.MAX_TABLE_ROWS)
            height = table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth() + 2
            height += sum(table.rowHeight(r) for r in range(rows))
            width = table.verticalHeader().width() + sum(table.columnWidth(c) for c in range(table.columnCount()))
            if width > table.viewport().width():  # room for a horizontal scrollbar
                height += table.horizontalScrollBar().sizeHint().height()
            table.setFixedHeight(height)
        self.matrix_title.setVisible(self.matrix.isVisible())

    def _show_summary(self, r: Comparison | None) -> None:
        table = self.summary
        if r is None or not r.groups:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        fmt = r.attribute.fmt
        if r.attribute.categorical:
            headers = ["Group", "n", "Missing", "Most common", "Share"]
            rows = [[g.label, f"{s.n:,}", f"{s.missing:,}", s.top or "—",
                     "—" if s.top_share is None else f"{s.top_share * 100:.0f}%"]
                    for g, s in zip(r.groups, r.summaries)]
        else:
            headers = ["Group", "n", "Missing", "Median", "Middle 50%", "Min", "Max"]
            rows = [[g.label, f"{s.n:,}", f"{s.missing:,}", full(s.median, fmt),
                     "—" if s.q25 is None else f"{full(s.q25, fmt)} – {full(s.q75, fmt)}",
                     full(s.minimum, fmt), full(s.maximum, fmt)] for g, s in zip(r.groups, r.summaries)]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(i, c, item)

    def _show_matrix(self, r: Comparison | None) -> None:
        table = self.matrix
        if r is None or len(r.groups) < 2:
            table.setRowCount(0)
            table.setColumnCount(0)
            self.matrix_note.setText("Check two or more groups to compare them." if r and r.groups else "")
            return
        k = len(r.groups)
        labels = [g.label for g in r.groups]
        table.setRowCount(k)
        table.setColumnCount(k)
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)
        shades = SHADES["dark" if is_dark(self) else "light"]
        categorical = r.attribute.categorical
        by_pair = {(e.a, e.b): e for e in r.effects}
        for i in range(k):
            for j in range(k):
                if i == j:
                    item = QTableWidgetItem("—")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(i, j, item)
                    continue
                e = by_pair[(min(i, j), max(i, j))]
                flip = i > j  # stored as b vs a with a < b; this cell is column j vs row i
                if e.size is None:
                    item = QTableWidgetItem("n/a")
                    item.setToolTip("Needs values in both groups.")
                else:
                    size = -e.size if flip and not categorical else e.size
                    symbol = "V" if categorical else "δ"
                    value = f"{size:+.2f}".replace("-", "−") if not categorical else f"{size:.2f}"
                    item = QTableWidgetItem(f"{symbol} {value}\n{e.magnitude}")
                    level = MAGNITUDES.index(e.magnitude)
                    if shades[level]:
                        background = QColor(shades[level])
                        item.setBackground(background)
                        item.setForeground(QColor("#ffffff" if background.lightness() < 140 else "#0b0b0b"))
                    item.setToolTip(self._effect_tip(r, i, j, e, flip))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, j, item)
        if categorical:
            levels = CRAMER_LEVELS
            self.matrix_note.setText(f"Cramér's V, 0 = same mix, 1 = no overlap · shading: small ≥ {levels[0]}, "
                                     f"medium ≥ {levels[1]}, large ≥ {levels[2]}")
            tip = ("Cramér's V measures how different the two groups' category mixes are, from 0 (the same "
                   "shares) to 1 (no category in common).")
        else:
            levels = CLIFF_LEVELS
            self.matrix_note.setText(f"Cliff's δ, +1 = column always larger, 0 = no tendency · shading by |δ|: "
                                     f"small ≥ {levels[0]}, medium ≥ {levels[1]}, large ≥ {levels[2]}")
            tip = ("Cliff's δ = P(column > row) − P(column < row). +1 means every value in the column's group is "
                   "larger than every value in the row's; 0 means neither tends to be larger. It's rank-based, so "
                   "outliers and skew don't drive it.\nHover a cell for the median change and the KS distance.")
        tip += ("\n\nThese are effect sizes, not p-values: with many parcels almost any difference is "
                "'significant', so what matters is how big it is.")
        for widget in (self.matrix_title, self.matrix_note, self.matrix):
            widget.setToolTip(tip)

    @staticmethod
    def _effect_tip(r: Comparison, row: int, col: int, e, flip: bool) -> str:
        a, b = r.groups[row].label, r.groups[col].label
        if r.attribute.categorical:
            return f"<b>{b}</b> vs <b>{a}</b><br>Cramér's V {e.size:.2f} ({e.magnitude})"
        delta = -e.size if flip else e.size
        change = e.median_change
        if flip and change is not None:
            change = 1 / (1 + change) - 1 if change != -1 else None
        lines = [f"<b>{b}</b> vs <b>{a}</b>",
                 f"Cliff's δ {delta:+.2f} ({e.magnitude}): "
                 f"{'larger' if delta > 0 else 'smaller' if delta < 0 else 'no tendency'} values in {b}"]
        if change is not None:
            lines.append(f"Median {change * 100:+.1f}%")
        if e.ks is not None:
            lines.append(f"KS distance {e.ks:.2f} (largest gap between the cumulative distributions)")
        return "<br>".join(lines)
