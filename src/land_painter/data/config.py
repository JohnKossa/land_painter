"""Import configuration: which files play which roles, and what each column means.

This is everything the user decides on the upload and review screens. It is
stored in the session file and re-applied to the input files on open, so it
must be JSON-serializable and must not depend on any particular data layout.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .sources import BOOLEAN, DATE, NUMERIC, TEXT, SourceInfo

ROLE_PARCELS = "parcels"
ROLE_SALES = "sales"
ROLE_ATTRIBUTES = "attributes"
ROLES = (ROLE_PARCELS, ROLE_SALES, ROLE_ATTRIBUTES)

DEFAULT_UNITS = ("Area", "Road frontage", "Lake frontage", "River frontage")

# Cleaning choices for numeric fields.
MISSING_RULES = ("zero", "negative", "placeholder", "outlier")
FILL_NONE, FILL_ZERO, FILL_MEDIAN, FILL_VALUE = "none", "zero", "median", "value"

# Characteristic kinds, used later by land table adjustment filters.
CONTINUOUS, CATEGORICAL, FLAG, WHEN, UNUSED = "continuous", "categorical", "boolean", "date", "unused"
CHARACTERISTIC_KINDS = (CONTINUOUS, CATEGORICAL, FLAG, WHEN, UNUSED)


@dataclass(frozen=True)
class ColumnRef:
    file: str  # path, as listed in ImportConfig.files
    column: str


@dataclass
class FileSpec:
    path: str
    roles: list[str] = field(default_factory=list)


@dataclass
class SalesMapping:
    key: str
    price: str
    date: str
    vacant: str
    vacant_values: list[str]  # values of the vacant column (as text) that mean "vacant"
    date_format: str | None = None  # strptime format for text dates; None = detect


@dataclass
class Cleaning:
    missing: list[str] = field(default_factory=list)  # subset of MISSING_RULES treated as missing
    fill: str = FILL_NONE
    value: float | None = None


@dataclass
class ImportConfig:
    files: list[FileSpec]
    parcel_key: str = ""
    attribute_keys: dict[str, str] = field(default_factory=dict)  # attribute file path -> key column
    sales: SalesMapping | None = None
    replacement_cost: list[ColumnRef] = field(default_factory=list)  # summed
    depreciated_cost: list[ColumnRef] = field(default_factory=list)  # summed
    # Current roll values (optional, summed): what the existing assessment books as building and as
    # land. Used to audit the roll's land/building split, not to compute land values.
    roll_building_value: list[ColumnRef] = field(default_factory=list)
    roll_land_value: list[ColumnRef] = field(default_factory=list)
    units: dict[str, ColumnRef] = field(default_factory=dict)  # unit name -> column
    neighborhood: ColumnRef | None = None
    cleaning: dict[str, Cleaning] = field(default_factory=dict)  # field id -> cleaning
    characteristic_kinds: dict[str, str] = field(default_factory=dict)  # "path::column" -> kind

    # --- Files by role ------------------------------------------------------------

    def files_with(self, role: str) -> list[str]:
        return [f.path for f in self.files if role in f.roles]

    @property
    def parcel_file(self) -> str:
        return self.files_with(ROLE_PARCELS)[0]

    @property
    def sales_file(self) -> str | None:
        files = self.files_with(ROLE_SALES)
        return files[0] if files else None

    def key_for(self, path: str) -> str:
        """Key column of a parcel-level file (the parcel file or an attribute file)."""
        return self.parcel_key if path == self.parcel_file else self.attribute_keys[path]

    def parcel_level_files(self) -> list[str]:
        files = [self.parcel_file]
        return files + [p for p in self.files_with(ROLE_ATTRIBUTES) if p not in files]

    def relocate(self, old: str, new: str) -> None:
        """Point every reference at a moved file."""
        def move(ref: ColumnRef | None) -> ColumnRef | None:
            return ColumnRef(new, ref.column) if ref is not None and ref.file == old else ref

        for f in self.files:
            if f.path == old:
                f.path = new
        if old in self.attribute_keys:
            self.attribute_keys[new] = self.attribute_keys.pop(old)
        self.replacement_cost = [move(r) for r in self.replacement_cost]
        self.depreciated_cost = [move(r) for r in self.depreciated_cost]
        self.roll_building_value = [move(r) for r in self.roll_building_value]
        self.roll_land_value = [move(r) for r in self.roll_land_value]
        self.units = {name: move(r) for name, r in self.units.items()}
        self.neighborhood = move(self.neighborhood)
        self.characteristic_kinds = {
            (f"{new}::{k.split('::', 1)[1]}" if k.split("::", 1)[0] == old else k): v
            for k, v in self.characteristic_kinds.items()
        }

    # --- Serialization ----------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> ImportConfig:
        d = json.loads(text)
        ref = lambda r: ColumnRef(**r) if r else None  # noqa: E731
        return cls(
            files=[FileSpec(**f) for f in d["files"]],
            parcel_key=d["parcel_key"],
            attribute_keys=d["attribute_keys"],
            sales=SalesMapping(**d["sales"]) if d["sales"] else None,
            replacement_cost=[ref(r) for r in d["replacement_cost"]],
            depreciated_cost=[ref(r) for r in d["depreciated_cost"]],
            # Added later: sessions saved before then have neither
            roll_building_value=[ref(r) for r in d.get("roll_building_value", [])],
            roll_land_value=[ref(r) for r in d.get("roll_land_value", [])],
            units={k: ref(r) for k, r in d["units"].items()},
            neighborhood=ref(d["neighborhood"]),
            cleaning={k: Cleaning(**c) for k, c in d["cleaning"].items()},
            characteristic_kinds=d["characteristic_kinds"],
        )


# --- Field ids ------------------------------------------------------------------------
# Mapped fields that get reviewed and cleaned. Units use "unit:<name>".

SALE_PRICE, SALE_DATE, SALE_VACANT = "sale_price", "sale_date", "sale_vacant"
REPLACEMENT_COST, DEPRECIATED_COST, NEIGHBORHOOD = "replacement_cost", "depreciated_cost", "neighborhood"
ROLL_BUILDING, ROLL_LAND = "roll_building_value", "roll_land_value"


def unit_field(name: str) -> str:
    return f"unit:{name}"


# --- Suggestions ------------------------------------------------------------------------
# Name-based guesses to pre-fill the column choices. The user confirms every one.

_PATTERNS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # field: (regexes in priority order, acceptable kinds)
    "key": ((r"^key$", r"^parcel_?(id|key|no|num|number)$", r"^(pin|parid|apn|strap|folio|account)$"),
            (TEXT, *NUMERIC)),
    SALE_PRICE: ((r"time_?adj", r"adj\w*_price|price\w*_adj", r"^sale_?price$", r"price"), NUMERIC),
    SALE_DATE: ((r"^sale_?date$", r"sale\w*date", r"date"), (DATE, TEXT)),
    SALE_VACANT: ((r"vacant_?sale|sale_?vacant", r"vacant"), (BOOLEAN, TEXT, *NUMERIC)),
    REPLACEMENT_COST: ((r"replacement", r"\brcn\b|^rcn_"), NUMERIC),
    DEPRECIATED_COST: ((r"depreciated", r"rcnld"), NUMERIC),
    # Assessed / just / market values. Building-ish names that aren't costs; land values, not land areas.
    ROLL_BUILDING: ((r"^(bldg|building|impr|improvement)s?_?(val|value|assessed|av|jv|just|mkt|market)$",
                     r"(just|assessed|mkt|market|appr\w*)_?(bldg|building|impr\w*)",
                     r"(bldg|building|impr\w*)_?(val|value|assessed|just|mkt|market)"), NUMERIC),
    ROLL_LAND: ((r"^(lnd|land)_?(val|value|assessed|av|jv|just|mkt|market)$",
                 r"(just|assessed|mkt|market|appr\w*)_?(lnd|land)(_?val\w*)?$",
                 r"(lnd|land)_?(val|value|assessed|just|mkt|market)"), NUMERIC),
    unit_field("Area"): ((r"^land_?area_?sq_?ft$", r"land_?area", r"acre", r"area"), NUMERIC),
    unit_field("Road frontage"): ((r"total_?frontage", r"road_?frontage", r"^frontage"), NUMERIC),
    unit_field("Lake frontage"): ((r"lake\w*frontage|frontage\w*lake",), NUMERIC),
    unit_field("River frontage"): ((r"river\w*frontage|frontage\w*river",), NUMERIC),
}


def suggest(field_id: str, columns: dict[str, str]) -> str | None:
    """Best guess for ``field_id`` among ``columns`` (name -> kind), or None.

    Numeric fields fall back to text columns, since delimited files often
    store numbers as text (e.g. "$1,000").
    """
    patterns, kinds = _PATTERNS.get(field_id, ((), ()))
    passes = [kinds, (TEXT,)] if set(kinds) <= set(NUMERIC) else [kinds]
    for allowed in passes:
        for pattern in patterns:
            for name, kind in columns.items():
                if kind in allowed and re.search(pattern, name, re.IGNORECASE):
                    return name
    return None


def suggest_roles(info: SourceInfo) -> list[str]:
    looks_like_sales = suggest(SALE_PRICE, info.columns) is not None and suggest(SALE_DATE, info.columns) is not None
    if looks_like_sales:
        return [ROLE_SALES]
    if info.geometry_column:
        return [ROLE_PARCELS]  # the parcel file always supplies attributes too
    return [ROLE_ATTRIBUTES]


def parcel_only_config(info: SourceInfo, key: str | None = None) -> ImportConfig:
    """A minimal configuration using just a parcel file (for tests and the self-test)."""
    key = key or suggest("key", info.columns)
    if key is None:
        raise ValueError(f"Could not find a parcel key column in {info.name}; pass one explicitly")
    return ImportConfig(files=[FileSpec(str(info.path.resolve()), [ROLE_PARCELS])], parcel_key=key)
