"""Neighborhood pop-up: name, rename/delete, land table editor, and metrics at a glance."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ..neighborhoods import ROOT
from ..session import Session
from .land_table_panel import LandTablePanel
from .metrics_panel import MetricsPanel


class NeighborhoodPopup(QWidget):
    renameRequested = Signal(int)
    deleteRequested = Signal(int)
    splitRequested = Signal(int)

    def __init__(self, session: Session, land_panel: LandTablePanel, metrics_panel: MetricsPanel, parent=None):
        super().__init__(parent)
        self.session = session
        self.index = ROOT
        self.name = QLabel()
        self.name.setStyleSheet("font-size: 15px; font-weight: bold")
        self.name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rename = QPushButton("Rename…")
        rename.clicked.connect(lambda: self.renameRequested.emit(self.index))
        split = QPushButton("Split by filter…")
        split.setToolTip("Move the parcels in this neighborhood that match a filter into a new child neighborhood")
        split.clicked.connect(lambda: self.splitRequested.emit(self.index))
        self.delete = QPushButton("Delete…")
        self.delete.clicked.connect(lambda: self.deleteRequested.emit(self.index))
        header = QHBoxLayout()
        header.addWidget(self.name, 1)
        header.addWidget(rename)
        header.addWidget(split)
        header.addWidget(self.delete)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(land_panel)
        splitter.addWidget(metrics_panel)
        splitter.setSizes([430, 560])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(header)
        layout.addWidget(splitter, 1)

    def show_neighborhood(self, index: int) -> None:
        self.index = index
        self.name.setText(self.session.neighborhoods.neighborhoods[index].name)
        self.delete.setEnabled(index != ROOT)
        self.delete.setToolTip("" if index != ROOT else "\"Everything\" cannot be deleted")


class DeleteDialog(QDialog):
    """Choose which neighborhood receives a deleted neighborhood's parcels."""

    def __init__(self, session: Session, index: int, parent=None):
        super().__init__(parent)
        model = session.neighborhoods
        self.setWindowTitle("Delete neighborhood")
        name = model.neighborhoods[index].name
        n_parcels = int((model.assignment == index).sum())
        n_children = len(model.children(index))

        # Adjacent neighborhoods, closest first; the parent is always offered as a fallback.
        candidates = session.adjacent_neighborhoods(index)
        parent_index = model.neighborhoods[index].parent
        if parent_index not in candidates:
            candidates.append(parent_index)
        counts = np.bincount(model.assignment, minlength=len(model.neighborhoods))
        self.recipient = QComboBox()
        for c in candidates:
            note = " (parent)" if c == parent_index else ""
            self.recipient.addItem(f"{model.neighborhoods[c].name}{note} — {counts[c]:,} parcels", c)

        text = f"Delete <b>{name}</b>? Its {n_parcels:,} parcels will move to the neighborhood you choose " \
               "and be valued with its land table."
        if n_children:
            text += f"<br>Its {n_children:,} child neighborhood{'s' if n_children != 1 else ''} will move up " \
                    f"to {model.neighborhoods[parent_index].name}."
        message = QLabel(text)
        message.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Delete")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(message)
        layout.addWidget(QLabel("Move its parcels to (adjacent neighborhoods, closest first):"))
        layout.addWidget(self.recipient)
        layout.addWidget(buttons)
        self.resize(480, 200)

    def chosen(self) -> int:
        return self.recipient.currentData()
