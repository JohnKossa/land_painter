"""Land tables and land value calculation.

A land table has a single unit of measure, a rate model, and adjustments:

    land value = rate_model(units) × Π(multipliers) + Σ(additions)

Multipliers apply only to the value implied by the unit of measure; additions
are applied after all multiplications. Each adjustment targets the parcels
matching one condition on a characteristic: a comparison for continuous
fields, or a set of values for categorical (and yes/no) fields.

Land tables are immutable, so a child neighborhood can share its parent's
table object; editing produces a new table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

LOT = "Lot"  # every parcel is exactly 1 lot
OPS = (">", ">=", "<", "<=", "=", "≠")
IN = "in"
MULTIPLY, ADD = "multiply", "add"


@dataclass(frozen=True)
class Condition:
    field: str  # characteristic key, "path::column"
    op: str  # one of OPS (continuous fields) or IN (categorical / yes-no fields)
    value: float | None = None  # for OPS
    values: tuple[str, ...] = ()  # for IN

    def mask(self, data: np.ndarray) -> np.ndarray:
        """Which entries of ``data`` match. Missing values never match."""
        if self.op == IN:
            present = (np.array([v is not None for v in data], dtype=bool)
                       if data.dtype == object else np.ones(len(data), bool))
            return present & np.isin(data.astype(str), list(self.values))
        x, v = data.astype(float), float(self.value)
        with np.errstate(invalid="ignore"):
            result = {">": x > v, ">=": x >= v, "<": x < v, "<=": x <= v, "=": x == v, "≠": x != v}[self.op]
        return result & ~np.isnan(x)

    def describe(self, field_label: str) -> str:
        if self.op == IN:
            shown = ", ".join(self.values[:4]) + (f", … ({len(self.values)} values)" if len(self.values) > 4 else "")
            return f"{field_label} is {shown}"
        return f"{field_label} {self.op} {self.value:g}"

    def to_dict(self) -> dict:
        return {"field": self.field, "op": self.op, "value": self.value, "values": list(self.values)}

    @classmethod
    def from_dict(cls, d: dict) -> Condition:
        return cls(d["field"], d["op"], d["value"], tuple(d["values"]))


@dataclass(frozen=True)
class Adjustment:
    condition: Condition
    kind: str  # MULTIPLY or ADD
    amount: float
    name: str = ""

    def describe(self) -> str:
        if self.kind == MULTIPLY:
            return f"× {self.amount:g}"
        return f"{'+' if self.amount >= 0 else '−'} ${abs(self.amount):,.2f}"

    def to_dict(self) -> dict:
        return {"condition": self.condition.to_dict(), "kind": self.kind, "amount": self.amount, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> Adjustment:
        return cls(Condition.from_dict(d["condition"]), d["kind"], d["amount"], d.get("name", ""))


# --- Rate models -------------------------------------------------------------------
# Only a flat rate for now. Log curves and breakpoint schedules will be further
# RateModel subclasses registered in RATE_MODELS.


class RateModel:
    type: ClassVar[str]

    def value(self, units: np.ndarray) -> np.ndarray:
        """Land value implied by the unit of measure, before adjustments."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class FlatRate(RateModel):
    type: ClassVar[str] = "flat"
    rate: float = 0.0

    def value(self, units: np.ndarray) -> np.ndarray:
        return units * self.rate

    def to_dict(self) -> dict:
        return {"type": self.type, "rate": self.rate}


RATE_MODELS: dict[str, Callable[[dict], RateModel]] = {
    FlatRate.type: lambda d: FlatRate(d["rate"]),
}


@dataclass(frozen=True)
class LandTable:
    unit: str  # LOT or the name of a mapped unit of measure
    rate: RateModel = field(default_factory=FlatRate)
    adjustments: tuple[Adjustment, ...] = ()

    def to_dict(self) -> dict:
        return {"unit": self.unit, "rate": self.rate.to_dict(), "adjustments": [a.to_dict() for a in self.adjustments]}

    @classmethod
    def from_dict(cls, d: dict | None) -> LandTable | None:
        if d is None:
            return None
        return cls(d["unit"], RATE_MODELS[d["rate"]["type"]](d["rate"]),
                   tuple(Adjustment.from_dict(a) for a in d["adjustments"]))


def components(table: LandTable, units: np.ndarray, field_data: Callable[[str], np.ndarray],
               parcels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(base value, combined multiplier, combined addition) for ``parcels`` under ``table``.

    ``units`` is the quantity of the table's unit for each of ``parcels``;
    ``field_data(key)`` returns a characteristic for *all* parcels.
    """
    multiplier = np.ones(len(parcels))
    addition = np.zeros(len(parcels))
    for adjustment in table.adjustments:
        matched = adjustment.condition.mask(field_data(adjustment.condition.field)[parcels])
        if adjustment.kind == MULTIPLY:
            multiplier[matched] *= adjustment.amount
        else:
            addition[matched] += adjustment.amount
    return table.rate.value(units), multiplier, addition


def split_tables(table: LandTable, position: int) -> tuple[LandTable, LandTable]:
    """(parent table, child table) when adjustment ``position`` becomes its own child neighborhood.

    The child takes the parcels the adjustment matched, so both tables keep
    every land value unchanged:

    - the parent drops the adjustment (none of its remaining parcels match it);
    - a multiplier on a flat rate is folded into the child's rate, so the
      child's rate is the sub-market's actual rate;
    - otherwise (an addition, or a rate model a multiplier cannot fold into)
      the child keeps the adjustment, which now matches every parcel in it.

    The parent's other adjustments carry over to the child.
    """
    adjustment = table.adjustments[position]
    others = table.adjustments[:position] + table.adjustments[position + 1:]
    parent = LandTable(table.unit, table.rate, others)
    if adjustment.kind == MULTIPLY and isinstance(table.rate, FlatRate):
        child = LandTable(table.unit, FlatRate(table.rate.rate * adjustment.amount), others)
    else:
        child = table
    return parent, child


def evaluate(table: LandTable, units: np.ndarray, field_data: Callable[[str], np.ndarray],
             parcels: np.ndarray) -> np.ndarray:
    """Land values for ``parcels`` under ``table``; NaN where units are missing."""
    base, multiplier, addition = components(table, units, field_data, parcels)
    return base * multiplier + addition
