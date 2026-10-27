"""Upload wizard: choose files and roles, map columns, then review the data.

Nothing about the data's shape is assumed: column choices are pre-filled from
name-based suggestions, and every one is shown to the user to confirm.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget, QWizard,
    QWizardPage,
)

from ..data.config import (
    CHARACTERISTIC_KINDS, DEFAULT_UNITS, DEPRECIATED_COST, FILL_MEDIAN, FILL_NONE, FILL_VALUE, FILL_ZERO,
    MISSING_RULES, NEIGHBORHOOD, REPLACEMENT_COST, ROLE_ATTRIBUTES, ROLE_PARCELS, ROLE_SALES, SALE_DATE,
    SALE_PRICE, SALE_VACANT, Cleaning, ColumnRef, FileSpec, ImportConfig, SalesMapping, suggest,
    suggest_roles, unit_field,
)
from ..data.review import (
    DATE_FORMATS, ERROR, INFO, WARNING, ColumnProfile, Issue, Prepared, clean_numeric, distinct_values,
    numeric_stats, parse_dates, prepare, profile_characteristics,
)
from ..data.sources import BOOLEAN, DATE, FILE_FILTER, NUMERIC, TEXT, SourceInfo, inspect, read_columns
from .tasks import run_with_progress

NONE_LABEL = "(none)"
ROLE_LABELS = {ROLE_PARCELS: "Parcel geometry", ROLE_SALES: "Sales", ROLE_ATTRIBUTES: "Parcel attributes"}
KIND_LABELS = {"continuous": "Continuous", "categorical": "Categorical", "boolean": "Yes/No",
               "date": "Date", "unused": "Not used"}
RULE_LABELS = {"zero": "Zero values", "negative": "Negative values", "placeholder": "Placeholder values",
               "outlier": "Outliers"}
SEVERITY_COLORS = {ERROR: "#e06c75", WARNING: "#e5c07b", INFO: "#98c379"}


def _name(path: str) -> str:
    return Path(path).name


# --- Small widgets ---------------------------------------------------------------------


class ColumnCombo(QComboBox):
    """Choose one column; item data is a ColumnRef (or None)."""

    def __init__(self, options: list[tuple[ColumnRef, str]], allow_none: bool = False, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        if allow_none:
            self.addItem(NONE_LABEL, None)
        for ref, label in options:
            self.addItem(label, ref)

    def ref(self) -> ColumnRef | None:
        return self.currentData()

    def set_ref(self, ref: ColumnRef | None) -> None:
        for i in range(self.count()):
            if self.itemData(i) == ref:
                self.setCurrentIndex(i)
                return
        self.setCurrentIndex(0 if self.count() else -1)


class ColumnMultiPicker(QToolButton):
    """Choose any number of columns (summed)."""

    changed = Signal()

    def __init__(self, options: list[tuple[ColumnRef, str]], parent=None):
        super().__init__(parent)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("QToolButton { text-align: left; padding-left: 6px; }")
        self._menu = QMenu(self)
        self._actions: list[QAction] = []
        for ref, label in options:
            action = self._menu.addAction(label)
            action.setCheckable(True)
            action.setData(ref)
            action.toggled.connect(self._update)
            self._actions.append(action)
        self.setMenu(self._menu)
        self._update()

    def refs(self) -> list[ColumnRef]:
        return [a.data() for a in self._actions if a.isChecked()]

    def set_refs(self, refs: list[ColumnRef]) -> None:
        for a in self._actions:
            a.setChecked(a.data() in refs)
        self._update()

    def _update(self) -> None:
        chosen = [a.text() for a in self._actions if a.isChecked()]
        self.setText(" + ".join(chosen) if chosen else NONE_LABEL)
        self.changed.emit()


# --- Page 1: files and roles ---------------------------------------------------------------


class FilesPage(QWizardPage):
    COL_FILE, COL_ROWS, COL_COLUMNS = 0, 1, 2
    ROLE_COLUMNS = {ROLE_PARCELS: 3, ROLE_SALES: 4, ROLE_ATTRIBUTES: 5}
    EXCLUSIVE = (ROLE_PARCELS, ROLE_SALES)

    def __init__(self, wizard: UploadWizard):
        super().__init__()
        self._wizard = wizard
        self.setTitle("Input files")
        self.setSubTitle("Add your data files and say what each one contains. One file can play several roles. "
                         "The parcel geometry file must contain parcel shapes; its other columns are available "
                         "as parcel attributes.")

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["File", "Rows", "Columns", *ROLE_LABELS.values()])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemChanged.connect(self._role_changed)

        add = QPushButton("Add files…")
        add.clicked.connect(self._add_clicked)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_clicked)
        open_session = QPushButton("Open an existing session instead…")
        open_session.clicked.connect(self._open_session)
        self.problem = QLabel()
        self.problem.setStyleSheet(f"color: {SEVERITY_COLORS[WARNING]}")

        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        buttons.addWidget(open_session)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.problem)

    def add_files(self, paths: list[str]) -> None:
        for path in paths:
            path = str(Path(path).resolve())
            if path in self._wizard.infos:
                continue
            try:
                info = run_with_progress(self, f"Reading {_name(path)}…", lambda p=path: inspect(p))
            except Exception as exc:  # noqa: BLE001 - report any read failure to the user
                QMessageBox.warning(self, "Could not read file", f"{_name(path)}:\n{exc}")
                continue
            self._wizard.infos[path] = info
            self._append_row(info, suggest_roles(info))
        self._enforce_single_parcels()
        self.completeChanged.emit()

    def _append_row(self, info: SourceInfo, roles: list[str]) -> None:
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(info.name)
        name.setData(Qt.ItemDataRole.UserRole, str(info.path))
        name.setToolTip(str(info.path) + ("" if info.geometry_column else "\n(no geometry)"))
        self.table.setItem(row, self.COL_FILE, name)
        self.table.setItem(row, self.COL_ROWS, QTableWidgetItem(f"{info.n_rows:,}"))
        self.table.setItem(row, self.COL_COLUMNS, QTableWidgetItem(f"{len(info.columns):,}"))
        for role, column in self.ROLE_COLUMNS.items():
            item = QTableWidgetItem()
            flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            if role == ROLE_PARCELS and not info.geometry_column:
                flags = Qt.ItemFlag.NoItemFlags
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if role in roles else Qt.CheckState.Unchecked)
            self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    def _enforce_single_parcels(self) -> None:
        """If several files were suggested as parcel geometry, keep only the first."""
        seen = False
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.ROLE_COLUMNS[ROLE_PARCELS])
            if item.checkState() == Qt.CheckState.Checked:
                if seen:
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self.table.item(row, self.ROLE_COLUMNS[ROLE_ATTRIBUTES]).setCheckState(Qt.CheckState.Checked)
                seen = True
        self.table.blockSignals(False)

    def _role_changed(self, item: QTableWidgetItem) -> None:
        role = next((r for r, c in self.ROLE_COLUMNS.items() if c == item.column()), None)
        if role in self.EXCLUSIVE and item.checkState() == Qt.CheckState.Checked:
            self.table.blockSignals(True)
            for row in range(self.table.rowCount()):
                if row != item.row():
                    self.table.item(row, item.column()).setCheckState(Qt.CheckState.Unchecked)
            self.table.blockSignals(False)
        self.completeChanged.emit()

    def _add_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add data files", "", FILE_FILTER)
        self.add_files(paths)

    def _remove_clicked(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            path = self.table.item(row, self.COL_FILE).data(Qt.ItemDataRole.UserRole)
            self._wizard.infos.pop(path, None)
            self.table.removeRow(row)
        self.completeChanged.emit()

    def _open_session(self) -> None:
        from ..app import SESSION_FILTER

        path = QFileDialog.getOpenFileName(self, "Open session", "", SESSION_FILTER)[0]
        if path:
            self._wizard.session_path = path
            self._wizard.reject()

    def file_specs(self) -> list[FileSpec]:
        specs = []
        for row in range(self.table.rowCount()):
            path = self.table.item(row, self.COL_FILE).data(Qt.ItemDataRole.UserRole)
            roles = [r for r, c in self.ROLE_COLUMNS.items()
                     if self.table.item(row, c).checkState() == Qt.CheckState.Checked]
            if roles:
                specs.append(FileSpec(path, roles))
        return specs

    def isComplete(self) -> bool:
        specs = self.file_specs()
        parcels = [s for s in specs if ROLE_PARCELS in s.roles]
        if not parcels:
            self.problem.setText("Choose the file that contains parcel geometry.")
            return False
        self.problem.setText("" if any(ROLE_SALES in s.roles for s in specs)
                             else "No sales file is selected. Metrics need sales; you can continue without them.")
        return True


# --- Page 2: column mapping ---------------------------------------------------------------


class ColumnsPage(QWizardPage):
    def __init__(self, wizard: UploadWizard):
        super().__init__()
        self._wizard = wizard
        self._built_for: list | None = None
        self.setTitle("What the columns mean")
        self.setSubTitle("Choices are pre-filled from column names where possible. Please check each one.")
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._scroll)

    # Options ---------------------------------------------------------------------------

    def _parcel_options(self, kinds=None) -> list[tuple[ColumnRef, str]]:
        files = self._parcel_level_files()
        options = []
        for path in files:
            for column, kind in self._wizard.infos[path].columns.items():
                if kinds is None or kind in kinds:
                    label = column if len(files) == 1 else f"{column}  ({_name(path)})"
                    options.append((ColumnRef(path, column), label))
        return options

    def _file_options(self, path: str, kinds=None) -> list[tuple[ColumnRef, str]]:
        return [(ColumnRef(path, c), c) for c, k in self._wizard.infos[path].columns.items()
                if kinds is None or k in kinds]

    def _parcel_level_files(self) -> list[str]:
        specs = self._wizard.files_page.file_specs()
        parcel = [s.path for s in specs if ROLE_PARCELS in s.roles]
        return parcel + [s.path for s in specs if ROLE_ATTRIBUTES in s.roles and s.path not in parcel]

    def _suggested(self, field_id: str, options: list[tuple[ColumnRef, str]]) -> ColumnRef | None:
        """Suggest among options, preferring the parcel file."""
        by_file: dict[str, dict[str, str]] = {}
        for ref, _ in options:
            by_file.setdefault(ref.file, {})[ref.column] = self._wizard.infos[ref.file].columns[ref.column]
        for path, columns in by_file.items():
            if name := suggest(field_id, columns):
                return ColumnRef(path, name)
        return None

    # Building the form ----------------------------------------------------------------------

    def initializePage(self) -> None:
        specs = self._wizard.files_page.file_specs()
        signature = [(s.path, sorted(s.roles)) for s in specs]
        if signature == self._built_for:
            return  # keep the user's choices when coming back
        self._built_for = signature
        previous = self._wizard.config  # choices made before going back and changing files

        body = QWidget()
        layout = QVBoxLayout(body)
        numeric = (*NUMERIC, TEXT)
        parcel_file = next(s.path for s in specs if ROLE_PARCELS in s.roles)
        sales_file = next((s.path for s in specs if ROLE_SALES in s.roles), None)

        # Keys
        keys = QGroupBox("Parcel keys")
        form = QFormLayout(keys)
        self.parcel_key = ColumnCombo(self._file_options(parcel_file, (TEXT, *NUMERIC)))
        self.parcel_key.set_ref(self._suggested("key", self._file_options(parcel_file, (TEXT, *NUMERIC))))
        form.addRow(f"Key in {_name(parcel_file)}:", self.parcel_key)
        self.attribute_keys: dict[str, ColumnCombo] = {}
        for path in self._parcel_level_files()[1:]:
            options = self._file_options(path, (TEXT, *NUMERIC))
            combo = ColumnCombo(options)
            combo.set_ref(self._suggested("key", options))
            self.attribute_keys[path] = combo
            form.addRow(f"Key in {_name(path)}:", combo)
        layout.addWidget(keys)

        # Sales
        self.sales_file = sales_file
        if sales_file:
            sales = QGroupBox(f"Sales ({_name(sales_file)})")
            form = QFormLayout(sales)
            self.sale_key = self._sales_combo("key", (TEXT, *NUMERIC))
            self.sale_price = self._sales_combo(SALE_PRICE, numeric)
            self.sale_date = self._sales_combo(SALE_DATE, (DATE, TEXT, *NUMERIC))
            self.sale_vacant = self._sales_combo(SALE_VACANT, (BOOLEAN, TEXT, *NUMERIC))
            self.vacant_value = QComboBox()
            self.vacant_value.setEditable(True)
            self.sale_vacant.currentIndexChanged.connect(self._fill_vacant_values)
            for combo in (self.sale_key, self.sale_price, self.sale_date, self.sale_vacant):
                combo.currentIndexChanged.connect(self.completeChanged)
            form.addRow("Parcel key:", self.sale_key)
            form.addRow("Sale price (time-adjusted):", self.sale_price)
            form.addRow("Sale date:", self.sale_date)
            form.addRow("Vacant/improved indicator:", self.sale_vacant)
            form.addRow("Counts as vacant when the value is:", self.vacant_value)
            layout.addWidget(sales)
            self._fill_vacant_values()

        # Costs
        costs = QGroupBox("Improvement costs (building only, excluding land; several columns are summed)")
        form = QFormLayout(costs)
        cost_options = self._parcel_options(numeric)
        self.replacement = ColumnMultiPicker(cost_options)
        self.depreciated = ColumnMultiPicker(cost_options)
        for picker, field_id in ((self.replacement, REPLACEMENT_COST), (self.depreciated, DEPRECIATED_COST)):
            suggestion = self._suggested(field_id, cost_options)
            picker.set_refs([suggestion] if suggestion else [])
        form.addRow("Replacement cost:", self.replacement)
        form.addRow("Depreciated cost:", self.depreciated)
        layout.addWidget(costs)

        # Units of measure
        self.units_box = QGroupBox("Units of measure (lot is automatic: every parcel is 1 lot)")
        self.units_form = QFormLayout(self.units_box)
        self.unit_combos: dict[str, ColumnCombo] = {}
        self._unit_options = cost_options
        for name in DEFAULT_UNITS:
            self._add_unit_row(name)
        add_unit = QPushButton("Add another unit of measure…")
        add_unit.clicked.connect(self._add_unit_clicked)
        self.units_form.addRow(add_unit)
        layout.addWidget(self.units_box)

        # Neighborhoods
        nbhd = QGroupBox("Existing neighborhoods (optional)")
        form = QFormLayout(nbhd)
        self.neighborhood = ColumnCombo(self._parcel_options((TEXT, *NUMERIC)), allow_none=True)
        form.addRow("Import starting neighborhoods from:", self.neighborhood)
        form.addRow(QLabel("Each distinct value becomes a neighborhood under \"Everything\"."))
        layout.addWidget(nbhd)
        layout.addStretch()

        for combo in (self.parcel_key, *self.attribute_keys.values()):
            combo.currentIndexChanged.connect(self.completeChanged)
        if previous is not None:
            self._restore(previous)
        self._scroll.setWidget(body)
        self.completeChanged.emit()

    def _sales_combo(self, field_id: str, kinds) -> ColumnCombo:
        options = self._file_options(self.sales_file, kinds)
        combo = ColumnCombo(options, allow_none=True)
        combo.set_ref(self._suggested(field_id, options))
        return combo

    def _fill_vacant_values(self) -> None:
        ref = self.sale_vacant.ref()
        self.vacant_value.clear()
        if ref is None:
            return
        kind = self._wizard.infos[ref.file].columns[ref.column]
        try:
            values = distinct_values(read_columns(ref.file, [ref.column]).to_series())
        except Exception:  # noqa: BLE001 - leave the box editable if the read fails
            values = []
        self.vacant_value.addItems(values)
        if kind == BOOLEAN and "true" in values:
            self.vacant_value.setCurrentText("true")

    def _add_unit_row(self, name: str) -> None:
        combo = ColumnCombo(self._unit_options, allow_none=True)
        combo.set_ref(self._suggested(unit_field(name), self._unit_options))
        self.unit_combos[name] = combo
        self.units_form.insertRow(len(self.unit_combos) - 1, f"{name}:", combo)

    def _add_unit_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "Add unit of measure", "Name (e.g. \"Canal frontage\"):")
        name = name.strip()
        if ok and name and name not in self.unit_combos and name.lower() != "lot":
            self._add_unit_row(name)

    def _restore(self, config: ImportConfig) -> None:
        """Re-apply earlier choices where the columns still exist."""
        self.replacement.set_refs(config.replacement_cost)
        self.depreciated.set_refs(config.depreciated_cost)
        for name, ref in config.units.items():
            if name not in self.unit_combos:
                self._add_unit_row(name)
            self.unit_combos[name].set_ref(ref)
        self.neighborhood.set_ref(config.neighborhood)

    # Result ---------------------------------------------------------------------------------

    def isComplete(self) -> bool:
        if self.parcel_key.ref() is None or any(c.ref() is None for c in self.attribute_keys.values()):
            return False
        if self.sales_file:
            return all(c.ref() is not None for c in (self.sale_key, self.sale_price, self.sale_date, self.sale_vacant))
        return True

    def build_config(self) -> ImportConfig:
        sales = None
        if self.sales_file:
            sales = SalesMapping(
                key=self.sale_key.ref().column,
                price=self.sale_price.ref().column,
                date=self.sale_date.ref().column,
                vacant=self.sale_vacant.ref().column,
                vacant_values=[self.vacant_value.currentText().strip()],
            )
        return ImportConfig(
            files=self._wizard.files_page.file_specs(),
            parcel_key=self.parcel_key.ref().column,
            attribute_keys={path: combo.ref().column for path, combo in self.attribute_keys.items()},
            sales=sales,
            replacement_cost=self.replacement.refs(),
            depreciated_cost=self.depreciated.refs(),
            units={name: combo.ref() for name, combo in self.unit_combos.items() if combo.ref() is not None},
            neighborhood=self.neighborhood.ref(),
        )


# --- Page 3: review ---------------------------------------------------------------------------


class ReviewPage(QWizardPage):
    FIELD_COLUMNS = ["Field", "Column", "Values", "Missing", "Unreadable", "Zero", "Negative",
                     "Placeholder", "Outliers", "Handling", "Missing after", "Min", "Median", "Max"]

    def __init__(self, wizard: UploadWizard):
        super().__init__()
        self._wizard = wizard
        self.setTitle("Review the data")
        self.setSubTitle("Check the problems found and decide how to handle suspect values. Nothing is changed "
                         "unless you choose it here.")
        self.prepared: Prepared | None = None
        self.profiles: list[ColumnProfile] = []
        self.config: ImportConfig | None = None

        self.issues = QListWidget()
        self.issues.setMaximumHeight(140)
        self.fields = QTableWidget(0, len(self.FIELD_COLUMNS))
        self.fields.setHorizontalHeaderLabels(self.FIELD_COLUMNS)
        self.fields.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fields.verticalHeader().setVisible(False)

        self.characteristics = QTableWidget(0, 7)
        self.characteristics.setHorizontalHeaderLabels(["Column", "File", "Stored as", "Distinct", "Missing",
                                                        "Use as", "Why review"])
        self.characteristics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.characteristics.verticalHeader().setVisible(False)
        self.only_review = QCheckBox("Only show columns that need review")
        self.only_review.setChecked(True)
        self.only_review.toggled.connect(self._filter_characteristics)
        characteristics = QWidget()
        c_layout = QVBoxLayout(characteristics)
        c_layout.addWidget(QLabel("How each parcel characteristic can be used in land table adjustment filters."))
        c_layout.addWidget(self.only_review)
        c_layout.addWidget(self.characteristics)

        tabs = QTabWidget()
        tabs.addTab(self.fields, "Mapped fields")
        tabs.addTab(characteristics, "Characteristics")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Problems found:"))
        layout.addWidget(self.issues)
        layout.addWidget(tabs, 1)

    def initializePage(self) -> None:
        config = self._wizard.columns_page.build_config()
        if self.config is not None:  # keep review choices when returning to this page
            config.cleaning = {k: v for k, v in self.config.cleaning.items()}
            config.characteristic_kinds = dict(self.config.characteristic_kinds)
            if config.sales and self.config.sales and config.sales.date == self.config.sales.date:
                config.sales.date_format = self.config.sales.date_format
        self.config = config
        self._analyze(profile=True)

    def _analyze(self, profile: bool) -> None:
        config, infos = self.config, self._wizard.infos

        def work():
            prepared = prepare(config)
            profiles = profile_characteristics(config, infos) if profile else None
            return prepared, profiles

        try:
            prepared, profiles = run_with_progress(self, "Analyzing data…", work)
        except Exception as exc:  # noqa: BLE001 - show any read/convert failure
            QMessageBox.critical(self, "Could not analyze data", str(exc))
            self.prepared = None
            self.completeChanged.emit()
            return
        self.prepared = prepared
        if profiles is not None:
            self.profiles = profiles
            self._show_characteristics()
        self._show_issues()
        self._show_fields()
        self.completeChanged.emit()

    # Issues ------------------------------------------------------------------------------

    def _show_issues(self) -> None:
        self.issues.clear()
        issues = list(self.prepared.issues)
        if self.config.neighborhood is not None:
            codes = self.prepared.parcels[NEIGHBORHOOD].drop_nulls()
            n = codes.filter(codes.str.strip_chars() != "").n_unique()
            issues.append(Issue(INFO, f"{n:,} neighborhoods will be imported."))
        for issue in sorted(issues, key=lambda i: (ERROR, WARNING, INFO).index(i.severity)):
            item = QListWidgetItem(f"{issue.severity.upper()}: {issue.message}")
            item.setForeground(QBrush(QColor(SEVERITY_COLORS[issue.severity])))
            self.issues.addItem(item)
        if not issues:
            self.issues.addItem("No problems found.")

    # Mapped fields ------------------------------------------------------------------------------

    def _field_rows(self) -> list[tuple[str, str, str, pl.Series | None]]:
        """(field id, label, source column label, values) for every mapped field."""
        rows = []
        config, prepared = self.config, self.prepared
        if config.sales and prepared.sales is not None:
            rows += [(SALE_PRICE, "Sale price", config.sales.price, prepared.sales["price"]),
                     (SALE_DATE, "Sale date", config.sales.date, prepared.sales["date"]),
                     (SALE_VACANT, "Vacant indicator", config.sales.vacant, prepared.sales["vacant"])]
        sources = {REPLACEMENT_COST: config.replacement_cost, DEPRECIATED_COST: config.depreciated_cost}
        sources.update({unit_field(name): [ref] for name, ref in config.units.items()})
        labels = {REPLACEMENT_COST: "Replacement cost", DEPRECIATED_COST: "Depreciated cost"}
        for field_id, refs in sources.items():
            if refs:
                label = labels.get(field_id, field_id.removeprefix("unit:"))
                rows.append((field_id, label, " + ".join(r.column for r in refs), prepared.parcels[field_id]))
        if config.neighborhood:
            rows.append((NEIGHBORHOOD, "Neighborhood", config.neighborhood.column, prepared.parcels[NEIGHBORHOOD]))
        return rows

    def _show_fields(self) -> None:
        rows = self._field_rows()
        self.fields.setRowCount(len(rows))
        for r, (field_id, label, source, values) in enumerate(rows):
            cells = {0: label, 1: source, 2: f"{len(values):,}", 3: f"{values.null_count():,}"}
            if field_id == SALE_DATE:
                cells.update({4: f"{self.prepared.unparseable.get(SALE_DATE, 0):,}",
                              11: str(values.min() or ""), 13: str(values.max() or "")})
                self.fields.setCellWidget(r, 9, self._date_format_combo())
            elif field_id == SALE_VACANT:
                cells[2] = f"{int(values.sum()):,} vacant · {int((~values).sum()):,} improved"
            elif field_id == NEIGHBORHOOD:
                cells[2] = f"{values.drop_nulls().n_unique():,} distinct"
            else:
                stats = numeric_stats(values, self.prepared.unparseable.get(field_id, 0))
                fmt = lambda v: "" if v is None else f"{v:,.2f}".rstrip("0").rstrip(".")  # noqa: E731
                cells.update({4: f"{stats.unparseable:,}", 5: f"{stats.zero:,}", 6: f"{stats.negative:,}",
                              7: f"{stats.placeholder:,}", 8: f"{stats.outlier:,}",
                              11: fmt(stats.minimum), 12: fmt(stats.median), 13: fmt(stats.maximum)})
                self.fields.setCellWidget(r, 9, self._cleaning_widget(field_id, values, stats, r))
                self._update_missing_after(r, field_id, values, stats)
            for c, text in cells.items():
                item = QTableWidgetItem(text)
                if c in (4, 5, 6, 7, 8) and text not in ("", "0"):
                    item.setForeground(QBrush(QColor(SEVERITY_COLORS[WARNING])))
                self.fields.setItem(r, c, item)
            if field_id not in (SALE_DATE, SALE_VACANT, NEIGHBORHOOD) and stats.low_fence is not None:
                self.fields.item(r, 8).setToolTip(f"Outside {fmt(stats.low_fence)} to {fmt(stats.high_fence)} "
                                                  "(3 × IQR of log values)")
        self.fields.resizeColumnsToContents()

    def _date_format_combo(self) -> QWidget:
        combo = QComboBox()
        if self.prepared.date_format is None:
            combo.addItem("Stored as dates", None)
            combo.setEnabled(False)
            return combo
        for fmt, example in DATE_FORMATS.items():
            combo.addItem(f"Format: {example}", fmt)
        combo.setCurrentIndex(combo.findData(self.prepared.date_format))
        combo.currentIndexChanged.connect(lambda: self._date_format_changed(combo.currentData()))
        return combo

    def _date_format_changed(self, fmt: str) -> None:
        self.config.sales.date_format = fmt
        self._analyze(profile=False)

    def _cleaning_widget(self, field_id: str, values: pl.Series, stats, row: int) -> QWidget:
        cleaning = self.config.cleaning.setdefault(field_id, Cleaning())
        counts = {"zero": stats.zero, "negative": stats.negative, "placeholder": stats.placeholder,
                  "outlier": stats.outlier}

        rules = QToolButton()
        rules.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(rules)
        for rule in MISSING_RULES:
            action = menu.addAction(f"{RULE_LABELS[rule]} ({counts[rule]:,})")
            action.setCheckable(True)
            action.setChecked(rule in cleaning.missing)
            action.setEnabled(counts[rule] > 0 or rule in cleaning.missing)
            action.setData(rule)
        rules.setMenu(menu)

        fill = QComboBox()
        fill.addItem("Don't fill", FILL_NONE)
        fill.addItem("Fill: 0", FILL_ZERO)
        fill.addItem("Fill: median", FILL_MEDIAN)
        fill.addItem("Fill: value…", FILL_VALUE)
        fill.setToolTip("How to fill missing values")
        fill.setCurrentIndex(fill.findData(cleaning.fill))
        if cleaning.fill == FILL_VALUE:
            fill.setItemText(3, f"Fill: {cleaning.value:g}")

        def refresh_text():
            chosen = [RULE_LABELS[r].split()[0].lower() for r in cleaning.missing]
            rules.setText("As missing: " + (", ".join(chosen) if chosen else "none"))
            rules.setToolTip("Values to treat as missing before filling")

        def rules_changed():
            cleaning.missing = [a.data() for a in menu.actions() if a.isChecked()]
            refresh_text()
            self._update_missing_after(row, field_id, values, stats)

        def fill_changed():
            mode = fill.currentData()
            if mode == FILL_VALUE:
                value, ok = QInputDialog.getDouble(self, "Fill missing values", "Value:", cleaning.value or 0.0,
                                                   -1e15, 1e15, 4)
                if not ok:
                    fill.blockSignals(True)
                    fill.setCurrentIndex(fill.findData(cleaning.fill))
                    fill.blockSignals(False)
                    return
                cleaning.value = value
                fill.setItemText(3, f"Fill: {value:g}")
            cleaning.fill = mode
            self._update_missing_after(row, field_id, values, stats)

        for action in menu.actions():
            action.toggled.connect(rules_changed)
        fill.currentIndexChanged.connect(fill_changed)
        refresh_text()

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.addWidget(rules)
        layout.addWidget(fill)
        return widget

    def _update_missing_after(self, row: int, field_id: str, values: pl.Series, stats) -> None:
        cleaned = clean_numeric(values, stats, self.config.cleaning.get(field_id, Cleaning()))
        self.fields.setItem(row, 10, QTableWidgetItem(f"{cleaned.null_count():,}"))

    # Characteristics ----------------------------------------------------------------------------

    def _show_characteristics(self) -> None:
        table = self.characteristics
        table.setRowCount(len(self.profiles))
        kinds = self.config.characteristic_kinds
        for r, p in enumerate(self.profiles):
            kinds.setdefault(p.config_key, p.suggested)
            for c, text in enumerate([p.ref.column, _name(p.ref.file), p.storage, f"{p.distinct:,}",
                                      f"{p.missing:,}"]):
                table.setItem(r, c, QTableWidgetItem(text))
            combo = QComboBox()
            for kind in CHARACTERISTIC_KINDS:
                combo.addItem(KIND_LABELS[kind], kind)
            combo.setCurrentIndex(combo.findData(kinds[p.config_key]))
            combo.currentIndexChanged.connect(lambda _, key=p.config_key, cb=combo: kinds.__setitem__(key, cb.currentData()))
            table.setCellWidget(r, 5, combo)
            note = QTableWidgetItem(p.review or "")
            if p.review:
                note.setForeground(QBrush(QColor(SEVERITY_COLORS[WARNING])))
            table.setItem(r, 6, note)
        table.resizeColumnsToContents()
        self._filter_characteristics()

    def _filter_characteristics(self) -> None:
        only = self.only_review.isChecked()
        for r, p in enumerate(self.profiles):
            self.characteristics.setRowHidden(r, only and not p.review)

    def isComplete(self) -> bool:
        return self.prepared is not None and not self.prepared.blocking


# --- Wizard -----------------------------------------------------------------------------------


class UploadWizard(QWizard):
    def __init__(self, files: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Land Painter — New session")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage)
        self.setButtonText(QWizard.WizardButton.FinishButton, "Load")
        self.resize(1300, 820)
        self.infos: dict[str, SourceInfo] = {}
        self.session_path: str | None = None  # set if the user chose to open a session instead

        self.files_page = FilesPage(self)
        self.columns_page = ColumnsPage(self)
        self.review_page = ReviewPage(self)
        self.addPage(self.files_page)
        self.addPage(self.columns_page)
        self.addPage(self.review_page)
        if files:
            self.files_page.add_files(files)

    @property
    def config(self) -> ImportConfig | None:
        """The configuration as reviewed so far (complete once the wizard is accepted)."""
        return self.review_page.config

    def prepared(self) -> Prepared | None:
        return self.review_page.prepared
