"""Neighborhood tree and parcel assignment.

Every parcel belongs to exactly one neighborhood. Neighborhoods form a tree
rooted at "Everything" with no depth limit. Neighborhoods are referenced
internally by index; their user-facing identifier is a renamable string.

Deleted neighborhoods stay in the list (flagged ``deleted``, with no parcels
and no children) so that indices recorded in the journal remain valid.

The model has no undo logic of its own: changes go through actions (see
``actions.py``), which know how to apply and revert themselves.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from .land_table import LandTable

ROOT = 0


@dataclass
class Neighborhood:
    name: str
    parent: int | None
    color: tuple[int, int, int]
    table: LandTable | None = None
    deleted: bool = False


def _color(index: int) -> tuple[int, int, int]:
    if index == ROOT:
        return (150, 150, 150)
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (round(r * 255), round(g * 255), round(b * 255))


class NeighborhoodModel:
    def __init__(self, n_parcels: int):
        self.assignment = np.zeros(n_parcels, dtype=np.uint16)
        self.neighborhoods: list[Neighborhood] = [Neighborhood("Everything", None, _color(ROOT))]

    def members_of(self, neighborhood: int, parcels: np.ndarray) -> np.ndarray:
        """The subset of ``parcels`` currently assigned to ``neighborhood``."""
        return parcels[self.assignment[parcels] == neighborhood]

    def next_index(self) -> int:
        return len(self.neighborhoods)

    def add_children(self, parent: int, names: list[str]) -> int:
        """Append children of ``parent`` (no parcels yet); returns the first new index.

        Children start with a copy of the parent's land table (tables are
        immutable, so the copy is the same object until either is edited).
        """
        first = self.next_index()
        if first + len(names) - 1 > np.iinfo(self.assignment.dtype).max:
            raise OverflowError("Too many neighborhoods")
        table = self.neighborhoods[parent].table
        self.neighborhoods.extend(Neighborhood(name, parent, _color(first + i), table) for i, name in enumerate(names))
        return first

    def children(self, index: int) -> list[int]:
        return [i for i, n in enumerate(self.neighborhoods) if n.parent == index and not n.deleted]

    def live(self) -> list[int]:
        """Indices of neighborhoods that have not been deleted."""
        return [i for i, n in enumerate(self.neighborhoods) if not n.deleted]

    def is_empty(self, index: int) -> bool:
        """No parcels directly in it and no child neighborhoods."""
        return not (self.assignment == index).any() and not self.children(index)

    def depth(self, index: int) -> int:
        depth = 0
        while (index := self.neighborhoods[index].parent) is not None:
            depth += 1
        return depth

    def truncate(self, count: int) -> None:
        """Drop neighborhoods from index ``count`` on. Their parcels must already be reassigned."""
        del self.neighborhoods[count:]

    def add_child(self, parent: int, parcels: np.ndarray, name: str) -> int:
        index = self.add_children(parent, [name])
        self.assignment[parcels] = index
        return index

    def remove_last(self, parcels: np.ndarray) -> None:
        """Undo ``add_child``: return its parcels to the parent and drop it."""
        child = self.neighborhoods.pop()
        self.assignment[parcels] = child.parent

    def subtree_mask(self, index: int) -> np.ndarray:
        """Boolean mask over neighborhoods: ``index`` and all its descendants."""
        mask = np.zeros(len(self.neighborhoods), dtype=bool)
        mask[index] = True
        # Children are always created after their parents, so one forward pass suffices.
        for i in range(index + 1, len(self.neighborhoods)):
            parent = self.neighborhoods[i].parent
            if parent is not None and mask[parent]:
                mask[i] = True
        return mask

    def palette(self) -> list[dict]:
        return [{"name": n.name, "parent": n.parent, "color": n.color} for n in self.neighborhoods]
