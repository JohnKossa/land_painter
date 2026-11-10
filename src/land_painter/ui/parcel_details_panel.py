"""Parcel details panel: one parcel's imported attributes beside its neighborhood norm.

A non-modal view that follows the current selection — click a parcel on the map or a
sale in the Sales list and it refreshes — so difficult ratios can be run down in place.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..parcel_details import ParcelDetails

FLAG_BRUSH = QBrush(QColor(255, 196, 0, 60))  # subtle amber behind unusual rows
PLACEHOLDER = "Click a parcel on the map, or a sale in the Sales list, to see its details."


class ParcelDetailsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.header = QLabel(PLACEHOLDER)
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.header.setStyleSheet("font-size: 13px")

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Feature", "This parcel", "Neighborhood"])
        self.tree.setRootIsDecorated(False)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.header)
        layout.addWidget(self.tree, 1)

    def clear_view(self) -> None:
        self.header.setText(PLACEHOLDER)
        self.tree.clear()

    def show_parcel(self, details: ParcelDetails, neighborhood_name: str, sale_context: str | None = None) -> None:
        parts = []
        if sale_context:
            parts.append(sale_context)
        parts.append(f"<b>Parcel {details.key}</b> · {neighborhood_name}")
        parts.append(f"{details.n_members:,} parcels in this neighborhood")
        self.header.setText("<br>".join(parts))

        self.tree.clear()
        bold = QFont()
        bold.setBold(True)
        for section in details.sections:
            header = QTreeWidgetItem(self.tree, [section.title])
            header.setFirstColumnSpanned(True)
            header.setFont(0, bold)
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for r in section.rows:
                value = f"{r.value} {r.direction}".strip() if r.flagged and r.direction else r.value
                item = QTreeWidgetItem(header, [r.label, value, r.comparison])
                if r.flagged:
                    for column in range(3):
                        item.setBackground(column, FLAG_BRUSH)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)
