"""Journaled user actions.

Every change to session data is an ``Action``: it can apply and revert itself,
and serialize to a journal record (JSON params plus an optional binary blob
for bulk data such as parcel index arrays). Actions store everything needed
to replay them deterministically, so opening a session is just replaying its
journal against freshly loaded inputs.

To add an action: subclass ``Action``, set a unique ``kind``, implement the
four methods, and decorate it with ``@register``.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from .land_table import LandTable
from .neighborhoods import ROOT, NeighborhoodModel


class State(Protocol):
    neighborhoods: NeighborhoodModel
    sale_flags: np.ndarray  # uint8 per sale, see sales.py


class Action:
    kind: ClassVar[str]
    label: ClassVar[str]  # shown in the Undo/Redo menu items
    changes_values: ClassVar[bool] = True  # whether land values must be recalculated afterwards

    def apply(self, state: State) -> None:
        raise NotImplementedError

    def revert(self, state: State) -> None:
        raise NotImplementedError

    def to_record(self) -> tuple[dict, bytes | None]:
        raise NotImplementedError

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> Action:
        raise NotImplementedError


ACTIONS: dict[str, type[Action]] = {}


def register(cls: type[Action]) -> type[Action]:
    if cls.kind in ACTIONS:
        raise ValueError(f"Duplicate action kind {cls.kind!r}")
    ACTIONS[cls.kind] = cls
    return cls


def decode(kind: str, params: dict, data: bytes | None) -> Action:
    return ACTIONS[kind].from_record(params, data)


def pack_indices(indices: np.ndarray) -> bytes:
    return zlib.compress(np.ascontiguousarray(indices, dtype=np.uint32).tobytes())


def unpack_indices(data: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(data), dtype=np.uint32).copy()


def pack_arrays(*arrays: np.ndarray) -> bytes:
    """Several uint32 arrays in one blob: a length header, then the data."""
    header = np.array([len(a) for a in arrays], dtype=np.uint32)
    return pack_indices(np.concatenate([[len(arrays)], header, *arrays]))


def unpack_arrays(data: bytes) -> list[np.ndarray]:
    flat = unpack_indices(data)
    n = int(flat[0])
    lengths = flat[1:1 + n].astype(np.int64)
    bounds = np.concatenate([[1 + n], 1 + n + np.cumsum(lengths)])
    return [flat[bounds[i]:bounds[i + 1]] for i in range(n)]


@register
@dataclass
class CreateNeighborhood(Action):
    """Move parcels (all currently in ``parent``) into a new child neighborhood."""

    kind: ClassVar[str] = "create_neighborhood"
    label: ClassVar[str] = "Create Neighborhood"

    parent: int
    parcels: np.ndarray
    name: str
    index: int  # expected index of the new neighborhood, checked on replay

    def apply(self, state: State) -> None:
        index = state.neighborhoods.add_child(self.parent, self.parcels, self.name)
        if index != self.index:
            raise RuntimeError(f"Journal out of sync: expected neighborhood {self.index}, got {index}")

    def revert(self, state: State) -> None:
        state.neighborhoods.remove_last(self.parcels)

    def to_record(self) -> tuple[dict, bytes | None]:
        return {"parent": self.parent, "name": self.name, "index": self.index}, pack_indices(self.parcels)

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> CreateNeighborhood:
        return cls(params["parent"], unpack_indices(data), params["name"], params["index"])


@register
@dataclass
class ImportNeighborhoods(Action):
    """Create one child of ``parent`` per name and move parcels into them."""

    kind: ClassVar[str] = "import_neighborhoods"
    label: ClassVar[str] = "Import Neighborhoods"

    parent: int
    names: list[str]
    first: int  # expected index of the first new neighborhood, checked on replay
    parcels: np.ndarray  # parcels moved
    codes: np.ndarray  # for each moved parcel, its position in ``names``
    previous: np.ndarray  # for each moved parcel, its neighborhood before the import

    def apply(self, state: State) -> None:
        model = state.neighborhoods
        first = model.add_children(self.parent, self.names)
        if first != self.first:
            raise RuntimeError(f"Journal out of sync: expected neighborhood {self.first}, got {first}")
        model.assignment[self.parcels] = self.first + self.codes

    def revert(self, state: State) -> None:
        model = state.neighborhoods
        model.assignment[self.parcels] = self.previous
        model.truncate(self.first)

    def to_record(self) -> tuple[dict, bytes | None]:
        return ({"parent": self.parent, "names": self.names, "first": self.first},
                pack_arrays(self.parcels, self.codes, self.previous))

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> ImportNeighborhoods:
        parcels, codes, previous = unpack_arrays(data)
        return cls(params["parent"], params["names"], params["first"], parcels, codes, previous)


@register
@dataclass
class SetSaleFlags(Action):
    """Set the flag of several sales at once."""

    kind: ClassVar[str] = "set_sale_flags"
    label: ClassVar[str] = "Flag Sales"
    changes_values: ClassVar[bool] = False

    sales: np.ndarray  # sale row indices
    flag: int
    previous: np.ndarray  # each sale's flag before the change

    def apply(self, state: State) -> None:
        state.sale_flags[self.sales] = self.flag

    def revert(self, state: State) -> None:
        state.sale_flags[self.sales] = self.previous

    def to_record(self) -> tuple[dict, bytes | None]:
        return {"flag": self.flag}, pack_arrays(self.sales, self.previous)

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> SetSaleFlags:
        sales, previous = unpack_arrays(data)
        return cls(sales, params["flag"], previous.astype(np.uint8))


def _table_dict(table: LandTable | None) -> dict | None:
    return table.to_dict() if table is not None else None


@register
@dataclass
class SetLandTable(Action):
    kind: ClassVar[str] = "set_land_table"
    label: ClassVar[str] = "Edit Land Table"

    neighborhood: int
    table: LandTable | None
    previous: LandTable | None

    def apply(self, state: State) -> None:
        state.neighborhoods.neighborhoods[self.neighborhood].table = self.table

    def revert(self, state: State) -> None:
        state.neighborhoods.neighborhoods[self.neighborhood].table = self.previous

    def to_record(self) -> tuple[dict, bytes | None]:
        return {"neighborhood": self.neighborhood, "table": _table_dict(self.table),
                "previous": _table_dict(self.previous)}, None

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> SetLandTable:
        return cls(params["neighborhood"], LandTable.from_dict(params["table"]), LandTable.from_dict(params["previous"]))


@register
@dataclass
class ApplyTableToChildren(Action):
    """Give each child neighborhood a copy of a land table (the parent's, when recorded)."""

    kind: ClassVar[str] = "apply_table_to_children"
    label: ClassVar[str] = "Apply Land Table to Children"

    children: list[int]
    table: LandTable | None
    previous: list[LandTable | None]

    def apply(self, state: State) -> None:
        for child in self.children:
            state.neighborhoods.neighborhoods[child].table = self.table

    def revert(self, state: State) -> None:
        for child, table in zip(self.children, self.previous):
            state.neighborhoods.neighborhoods[child].table = table

    def to_record(self) -> tuple[dict, bytes | None]:
        # Many children usually share a few distinct previous tables; store each once.
        distinct: list[dict | None] = []
        refs = []
        for table in self.previous:
            d = _table_dict(table)
            if d not in distinct:
                distinct.append(d)
            refs.append(distinct.index(d))
        return {"children": self.children, "table": _table_dict(self.table),
                "previous_tables": distinct, "previous": refs}, None

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> ApplyTableToChildren:
        tables = [LandTable.from_dict(d) for d in params["previous_tables"]]
        return cls(params["children"], LandTable.from_dict(params["table"]), [tables[i] for i in params["previous"]])


@register
@dataclass
class RenameNeighborhood(Action):
    kind: ClassVar[str] = "rename_neighborhood"
    label: ClassVar[str] = "Rename Neighborhood"
    changes_values: ClassVar[bool] = False

    neighborhood: int
    name: str
    previous: str

    def apply(self, state: State) -> None:
        state.neighborhoods.neighborhoods[self.neighborhood].name = self.name

    def revert(self, state: State) -> None:
        state.neighborhoods.neighborhoods[self.neighborhood].name = self.previous

    def to_record(self) -> tuple[dict, bytes | None]:
        return {"neighborhood": self.neighborhood, "name": self.name, "previous": self.previous}, None

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> RenameNeighborhood:
        return cls(params["neighborhood"], params["name"], params["previous"])


def _delete(model: NeighborhoodModel, index: int, children: list[int]) -> None:
    """Mark a neighborhood deleted, moving its children up to its parent."""
    if index == ROOT:
        raise ValueError("The root neighborhood cannot be deleted")
    parent = model.neighborhoods[index].parent
    for child in children:
        model.neighborhoods[child].parent = parent
    model.neighborhoods[index].deleted = True


def _undelete(model: NeighborhoodModel, index: int, children: list[int]) -> None:
    model.neighborhoods[index].deleted = False
    for child in children:
        model.neighborhoods[child].parent = index


@register
@dataclass
class DeleteNeighborhood(Action):
    """Delete a neighborhood, moving its parcels to ``recipient`` and its children to its parent."""

    kind: ClassVar[str] = "delete_neighborhood"
    label: ClassVar[str] = "Delete Neighborhood"

    neighborhood: int
    recipient: int
    parcels: np.ndarray  # the deleted neighborhood's parcels
    children: list[int]  # its child neighborhoods

    def apply(self, state: State) -> None:
        model = state.neighborhoods
        model.assignment[self.parcels] = self.recipient
        _delete(model, self.neighborhood, self.children)

    def revert(self, state: State) -> None:
        model = state.neighborhoods
        _undelete(model, self.neighborhood, self.children)
        model.assignment[self.parcels] = self.neighborhood

    def to_record(self) -> tuple[dict, bytes | None]:
        return ({"neighborhood": self.neighborhood, "recipient": self.recipient, "children": self.children},
                pack_indices(self.parcels))

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> DeleteNeighborhood:
        return cls(params["neighborhood"], params["recipient"], unpack_indices(data), params["children"])


@register
@dataclass
class ReassignParcels(Action):
    """Move parcels from any neighborhoods into ``target``; delete neighborhoods this empties."""

    kind: ClassVar[str] = "reassign_parcels"
    label: ClassVar[str] = "Reassign Parcels"

    target: int
    parcels: np.ndarray
    previous: np.ndarray  # each parcel's neighborhood before the move
    deleted: list[int]  # neighborhoods emptied by the move, deleted in this order
    children: list[list[int]]  # for each deleted neighborhood, its child neighborhoods

    def apply(self, state: State) -> None:
        model = state.neighborhoods
        model.assignment[self.parcels] = self.target
        for index, children in zip(self.deleted, self.children):
            _delete(model, index, children)

    def revert(self, state: State) -> None:
        model = state.neighborhoods
        for index, children in reversed(list(zip(self.deleted, self.children))):
            _undelete(model, index, children)
        model.assignment[self.parcels] = self.previous

    def to_record(self) -> tuple[dict, bytes | None]:
        return ({"target": self.target, "deleted": self.deleted, "children": self.children},
                pack_arrays(self.parcels, self.previous))

    @classmethod
    def from_record(cls, params: dict, data: bytes | None) -> ReassignParcels:
        parcels, previous = unpack_arrays(data)
        return cls(params["target"], parcels, previous.astype(np.uint16), params["deleted"], params["children"])
