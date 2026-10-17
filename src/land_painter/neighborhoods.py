"""Neighborhood tree and parcel assignment.

Every parcel belongs to exactly one neighborhood. Neighborhoods form a tree
rooted at "Everything" with no depth limit. Neighborhoods are referenced
internally by index; their user-facing identifier is a renamable string.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

ROOT = 0


@dataclass
class Neighborhood:
    name: str
    parent: int | None
    color: tuple[int, int, int]


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

    def create_child(self, parent: int, parcels: np.ndarray) -> int | None:
        """Move the given parcels that belong to ``parent`` into a new child.

        Parcels in any other neighborhood are ignored. Returns the new
        neighborhood's index, or None if no parcels qualified.
        """
        parcels = parcels[self.assignment[parcels] == parent]
        if parcels.size == 0:
            return None
        index = len(self.neighborhoods)
        if index > np.iinfo(self.assignment.dtype).max:
            raise OverflowError("Too many neighborhoods")
        self.neighborhoods.append(Neighborhood(f"N{index}", parent, _color(index)))
        self.assignment[parcels] = index
        return index

    def palette(self) -> list[dict]:
        return [{"name": n.name, "parent": n.parent, "color": n.color} for n in self.neighborhoods]
