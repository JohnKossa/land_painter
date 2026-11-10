"""Session: the loaded inputs, the current state, and the action journal.

The journal is a SQLite database and is also the session file. A new session
journals to memory; after "Save As" it journals straight to the file, so every
later action is persisted as it happens. Undo/redo is linear: a new action
after an undo discards the redo history.

Inputs are referenced by path, not copied: the session stores the import
configuration (files, roles, column mappings, cleaning choices) and rebuilds
the dataset from it on open. A fingerprint of the parcel keys is stored so a
changed or different parcel file is detected.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .actions import (
    Action,
    ApplyTableToChildren,
    ConvertAdjustment,
    CreateNeighborhood,
    DeleteNeighborhood,
    ImportNeighborhoods,
    MergeNeighborhood,
    ReassignParcels,
    RenameNeighborhood,
    SetLandTable,
    SetSaleFlags,
    SplitNeighborhood,
    decode,
)
from .data.config import ImportConfig
from .data.dataset import Dataset, build_dataset
from .land_table import Condition, LandTable, evaluate, split_tables
from .neighborhoods import ROOT, NeighborhoodModel
from .sales import sales_fingerprint

SCHEMA_VERSION = 2
SESSION_SUFFIX = ".lpsession"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS actions (
    seq INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    data BLOB
);
"""


class SessionError(Exception):
    pass


class Journal:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        with conn:
            conn.executescript(_SCHEMA)
        if self.get("schema_version") is None:
            self.set("schema_version", str(SCHEMA_VERSION))
            self.set("cursor", "0")
        elif int(self.get("schema_version")) > SCHEMA_VERSION:
            raise SessionError("Session was saved by a newer version of Land Painter")
        self.path: Path | None = None

    @classmethod
    def memory(cls) -> Journal:
        return cls(sqlite3.connect(":memory:", check_same_thread=False))

    @classmethod
    def open(cls, path: Path) -> Journal:
        if not path.is_file():
            raise SessionError(f"No session file at {path}")
        # Sessions are opened on a worker thread and then used on the UI thread.
        journal = cls(sqlite3.connect(path, check_same_thread=False))
        journal.path = path
        return journal

    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    @property
    def cursor(self) -> int:
        """Number of actions currently applied."""
        return int(self.get("cursor"))

    @cursor.setter
    def cursor(self, value: int) -> None:
        self.set("cursor", str(value))

    @property
    def length(self) -> int:
        return self.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM actions").fetchone()[0]

    def append(self, action: Action) -> None:
        params, data = action.to_record()
        seq = self.cursor + 1
        with self.conn:
            self.conn.execute("DELETE FROM actions WHERE seq >= ?", (seq,))
            self.conn.execute(
                "INSERT INTO actions (seq, kind, params, data) VALUES (?, ?, ?, ?)",
                (seq, action.kind, json.dumps(params), data),
            )
            self.conn.execute("UPDATE meta SET value = ? WHERE key = 'cursor'", (str(seq),))

    def action(self, seq: int) -> Action:
        kind, params, data = self.conn.execute(
            "SELECT kind, params, data FROM actions WHERE seq = ?", (seq,)
        ).fetchone()
        return decode(kind, json.loads(params), data)

    def save_as(self, path: Path) -> None:
        if path.exists():
            path.unlink()
        target = sqlite3.connect(path, check_same_thread=False)
        self.conn.backup(target)
        self.conn.close()
        self.conn = target
        self.path = path


def fingerprint(keys: np.ndarray) -> str:
    digest = hashlib.sha256("\n".join(map(str, keys)).encode())
    return f"{len(keys)}:{digest.hexdigest()}"


def read_config(path: str | Path) -> ImportConfig:
    """The import configuration stored in a session file."""
    journal = Journal.open(Path(path))
    try:
        text = journal.get("import_config")
    finally:
        journal.conn.close()
    if text is None:
        raise SessionError("This session was made by an earlier development version and cannot be opened.")
    return ImportConfig.from_json(text)


def relocate_inputs(config: ImportConfig, session_path: str | Path,
                    locate: Callable[[Path], Path | None] | None = None) -> bool:
    """Find input files that have moved. Returns True if any path changed.

    A missing file is looked for beside the session file, then ``locate`` is
    asked (it returns a replacement path, or None to give up).
    """
    changed = False
    for spec in list(config.files):
        old = Path(spec.path)
        if old.is_file():
            continue
        beside = Path(session_path).parent / old.name
        new = beside if beside.is_file() else (locate(old) if locate else None)
        if new is None:
            raise SessionError(f"Input file not found: {old}")
        config.relocate(spec.path, str(Path(new).resolve()))
        changed = True
    return changed


class Session:
    def __init__(self, dataset: Dataset, journal: Journal):
        self.dataset = dataset
        self.layer = dataset.layer
        self.journal = journal
        self.neighborhoods = NeighborhoodModel(self.layer.n_parcels)
        n_sales = dataset.sales.height if dataset.sales is not None else 0
        self.sale_flags = np.zeros(n_sales, dtype=np.uint8)
        self.land_values = np.full(self.layer.n_parcels, np.nan)  # updated in place

    @classmethod
    def new(cls, dataset: Dataset) -> Session:
        journal = Journal.memory()
        journal.set("import_config", dataset.config.to_json())
        journal.set("parcel_fingerprint", fingerprint(dataset.layer.keys))
        journal.set("sales_fingerprint", sales_fingerprint(dataset.sales))
        session = cls(dataset, journal)
        codes = dataset.neighborhood_codes()
        if codes is not None:
            session.import_neighborhoods(codes)
        return session

    @classmethod
    def open(cls, path: str | Path, config: ImportConfig | None = None) -> Session:
        """Open a session file, rebuild its dataset, and replay its journal.

        Pass ``config`` after ``relocate_inputs`` if input files have moved.
        """
        path = Path(path)
        config = config or read_config(path)
        dataset = build_dataset(config)
        journal = Journal.open(path)
        if fingerprint(dataset.layer.keys) != journal.get("parcel_fingerprint"):
            raise SessionError(
                f"{Path(config.parcel_file).name} does not match the parcels this session was built from")
        stored = journal.get("sales_fingerprint")
        if stored is None:  # sessions saved before sale flags existed
            journal.set("sales_fingerprint", sales_fingerprint(dataset.sales))
        elif sales_fingerprint(dataset.sales) != stored:
            raise SessionError("The sales data does not match the sales this session was built from, "
                               "so sale flags cannot be restored.")
        journal.set("import_config", config.to_json())

        session = cls(dataset, journal)
        for seq in range(1, journal.cursor + 1):
            journal.action(seq).apply(session)
        session.recompute_values()
        return session

    @property
    def path(self) -> Path | None:
        return self.journal.path

    def save_as(self, path: str | Path) -> None:
        self.journal.save_as(Path(path))

    # --- Undo / redo ----------------------------------------------------------

    def do(self, action: Action) -> None:
        action.apply(self)  # apply first: a failed action is never journaled
        self.journal.append(action)
        if action.changes_values:
            self.recompute_values()

    def undo(self) -> Action | None:
        cursor = self.journal.cursor
        if cursor == 0:
            return None
        action = self.journal.action(cursor)
        action.revert(self)
        self.journal.cursor = cursor - 1
        if action.changes_values:
            self.recompute_values()
        return action

    def redo(self) -> Action | None:
        cursor = self.journal.cursor
        if cursor == self.journal.length:
            return None
        action = self.journal.action(cursor + 1)
        action.apply(self)
        self.journal.cursor = cursor + 1
        if action.changes_values:
            self.recompute_values()
        return action

    def undo_label(self) -> str | None:
        cursor = self.journal.cursor
        return self.journal.action(cursor).label if cursor > 0 else None

    def redo_label(self) -> str | None:
        cursor = self.journal.cursor
        return self.journal.action(cursor + 1).label if cursor < self.journal.length else None

    # --- Operations -------------------------------------------------------------

    def create_neighborhood(self, parent: int, parcels: np.ndarray) -> int | None:
        """Create a child of ``parent`` from those ``parcels`` that belong to it.

        Parcels in other neighborhoods are ignored. Returns the new
        neighborhood's index, or None (and records nothing) if none qualified.
        """
        members = self.neighborhoods.members_of(parent, parcels)
        if members.size == 0:
            return None
        index = self.neighborhoods.next_index()
        self.do(CreateNeighborhood(parent, members, f"N{index}", index))
        return index

    def describe_condition(self, condition: Condition) -> str:
        labels = {key: label for key, label, _ in self.dataset.filter_fields()}
        return condition.describe(labels.get(condition.field, condition.field.split("::", 1)[-1]))

    def matching_parcels(self, neighborhood: int, condition: Condition) -> np.ndarray:
        """Parcels directly in ``neighborhood`` (not in its children) that match ``condition``."""
        parcels = np.flatnonzero(self.neighborhoods.assignment == neighborhood)
        return parcels[condition.mask(self.dataset.characteristic(condition.field)[parcels])]

    def split_neighborhood(self, parent: int, condition: Condition, name: str | None = None) -> int | None:
        """Move the parcels directly in ``parent`` that match ``condition`` into a new child.

        The child starts with a copy of the parent's land table. Returns the
        new neighborhood's index, or None (and records nothing) if nothing matched.
        """
        parcels = self.matching_parcels(parent, condition)
        if parcels.size == 0:
            return None
        table = self.neighborhoods.neighborhoods[parent].table
        index = self.neighborhoods.next_index()
        name = (name or "").strip() or self.describe_condition(condition)
        self.do(SplitNeighborhood(parent, parcels, name, index, table, table, table))
        return index

    def convert_adjustment(self, parent: int, position: int, name: str | None = None) -> int | None:
        """Turn adjustment ``position`` of ``parent``'s land table into a child neighborhood.

        Its matching parcels move to the child; land values are unchanged (see
        ``land_table.split_tables``). Returns the new index, or None if nothing matched.
        """
        table = self.neighborhoods.neighborhoods[parent].table
        if table is None or not 0 <= position < len(table.adjustments):
            raise ValueError(f"No adjustment {position} in this neighborhood's land table")
        adjustment = table.adjustments[position]
        parcels = self.matching_parcels(parent, adjustment.condition)
        if parcels.size == 0:
            return None
        parent_table, child_table = split_tables(table, position)
        index = self.neighborhoods.next_index()
        name = (name or "").strip() or adjustment.name or self.describe_condition(adjustment.condition)
        self.do(ConvertAdjustment(parent, parcels, name, index, child_table, parent_table, table))
        return index

    # --- Land values ----------------------------------------------------------------

    def values_for(self, neighborhood: int, table: LandTable | None) -> tuple[np.ndarray, np.ndarray]:
        """(parcels, land values) for a neighborhood's own parcels under ``table`` (e.g. a draft)."""
        parcels = np.flatnonzero(self.neighborhoods.assignment == neighborhood)
        if table is None:
            return parcels, np.full(len(parcels), np.nan)
        units = self.dataset.unit_values(table.unit)[parcels]
        return parcels, evaluate(table, units, self.dataset.characteristic, parcels)

    def parcel_groups(self):
        """Yield (neighborhood index, its parcel indices) for every neighborhood with parcels."""
        assignment = self.neighborhoods.assignment
        order = np.argsort(assignment, kind="stable")
        ordered = assignment[order]
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(ordered)) + 1, [len(order)]])
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end > start:
                yield int(ordered[start]), order[start:end]

    def recompute_values(self) -> None:
        """Recalculate every parcel's land value from its neighborhood's land table."""
        values = np.full(self.layer.n_parcels, np.nan)
        for neighborhood, parcels in self.parcel_groups():
            table = self.neighborhoods.neighborhoods[neighborhood].table
            if table is not None:
                units = self.dataset.unit_values(table.unit)[parcels]
                values[parcels] = evaluate(table, units, self.dataset.characteristic, parcels)
        self.land_values[:] = values

    def set_land_table(self, neighborhood: int, table: LandTable | None) -> bool:
        previous = self.neighborhoods.neighborhoods[neighborhood].table
        if table == previous:
            return False
        self.do(SetLandTable(neighborhood, table, previous))
        return True

    def apply_table_to_children(self, neighborhood: int) -> int:
        """Copy a neighborhood's land table to its direct children; returns how many changed."""
        model = self.neighborhoods
        table = model.neighborhoods[neighborhood].table
        children = [c for c in model.children(neighborhood) if model.neighborhoods[c].table != table]
        if not children:
            return 0
        self.do(ApplyTableToChildren(children, table, [model.neighborhoods[c].table for c in children]))
        return len(children)

    # --- Neighborhood management ---------------------------------------------------------

    ADJACENT_DISTANCE = 0.0005  # degrees (~50 m): parcels this close count as adjacent, across roads

    def rename_neighborhood(self, index: int, name: str) -> bool:
        name = name.strip()
        previous = self.neighborhoods.neighborhoods[index].name
        if not name or name == previous:
            return False
        self.do(RenameNeighborhood(index, name, previous))
        return True

    def adjacent_neighborhoods(self, index: int) -> list[int]:
        """Live neighborhoods with parcels within ADJACENT_DISTANCE of this neighborhood's parcels,
        most strongly connected (most nearby parcels) first."""
        model, layer = self.neighborhoods, self.layer
        parts = np.flatnonzero(model.assignment[layer.part_parcel] == index)
        if parts.size == 0:
            return []
        _, hits = layer.tree.query(layer.parts[parts], predicate="dwithin", distance=self.ADJACENT_DISTANCE)
        neighbors, counts = np.unique(model.assignment[layer.part_parcel[np.unique(hits)]], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        return [int(n) for n in neighbors[order] if n != index and not model.neighborhoods[n].deleted]

    def delete_neighborhood(self, index: int, recipient: int) -> None:
        model = self.neighborhoods
        if index in (ROOT, recipient) or model.neighborhoods[recipient].deleted:
            raise ValueError("Invalid neighborhood deletion")
        parcels = np.flatnonzero(model.assignment == index).astype(np.uint32)
        self.do(DeleteNeighborhood(index, recipient, parcels, model.children(index)))

    def merge_neighborhood(self, source: int, destination: int) -> None:
        """Join ``source`` into ``destination``: its parcels and child neighborhoods move
        there and it is deleted. The source's land table is discarded."""
        model = self.neighborhoods
        if source in (ROOT, destination) or model.neighborhoods[destination].deleted:
            raise ValueError("Invalid neighborhood merge")
        if model.subtree_mask(source)[destination]:
            raise ValueError("Cannot merge a neighborhood into one of its own descendants")
        parcels = np.flatnonzero(model.assignment == source).astype(np.uint32)
        self.do(MergeNeighborhood(source, destination, parcels, model.children(source)))

    def plan_reassign(self, target: int, parcels: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """(parcels that would move, neighborhoods the move would leave empty)."""
        model = self.neighborhoods
        moving = parcels[model.assignment[parcels] != target]
        if moving.size == 0:
            return moving, []
        sources, moved = np.unique(model.assignment[moving], return_counts=True)
        remaining = np.bincount(model.assignment, minlength=len(model.neighborhoods))
        emptied = [int(s) for s, m in zip(sources, moved)
                   if s != ROOT and remaining[s] == m and not model.children(int(s))]
        return moving, emptied

    def reassign(self, target: int, parcels: np.ndarray) -> list[int]:
        """Move parcels into ``target`` and delete the neighborhoods this empties.

        Returns the deleted neighborhoods. Callers confirm with the user first
        (see ``plan_reassign``).
        """
        moving, emptied = self.plan_reassign(target, parcels)
        if moving.size == 0:
            return []
        model = self.neighborhoods
        self.do(ReassignParcels(target, moving.astype(np.uint32), model.assignment[moving].copy(),
                                emptied, [model.children(e) for e in emptied]))
        return emptied

    def set_sale_flags(self, sales: np.ndarray, flag: int) -> bool:
        """Flag the given sales; returns False (recording nothing) if none change."""
        sales = np.asarray(sales, dtype=np.uint32)
        sales = sales[self.sale_flags[sales] != flag]
        if sales.size == 0:
            return False
        self.do(SetSaleFlags(sales, flag, self.sale_flags[sales].copy()))
        return True

    def import_neighborhoods(self, codes, parent: int = ROOT) -> int:
        """Create a child of ``parent`` per distinct code, for parcels currently in ``parent``.

        ``codes`` is aligned to parcels; null or blank codes stay put. Returns
        the number of neighborhoods created.
        """
        codes = np.array([None if c is None or not str(c).strip() else str(c) for c in codes], dtype=object)
        movable = np.flatnonzero((codes != None) & (self.neighborhoods.assignment == parent))  # noqa: E711
        if movable.size == 0:
            return 0
        names = sorted(set(codes[movable]))
        positions = np.searchsorted(np.array(names, dtype=object), codes[movable]).astype(np.uint32)
        self.do(ImportNeighborhoods(parent, names, self.neighborhoods.next_index(), movable.astype(np.uint32),
                                    positions, self.neighborhoods.assignment[movable].astype(np.uint32)))
        return len(names)
