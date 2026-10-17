# Purpose

An interactive desktop application to support the iteration on neighborhoods and land tables

# Platform & Scale

Desktop application, targeting **Windows only** for now.

Must support a very high number of parcels (hundreds of thousands) with responsive map rendering and metric recalculation.

The application is built to run **fully offline**: no network access is required, and all JavaScript, map assets, and any basemap are bundled or stored locally. This can be revisited if offline operation causes problems.

# Technology Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.14+ | Map front-end requires a small amount of JavaScript |
| GUI shell | PySide6 (Qt) | Native widgets for upload/reconciliation, sales pane, land table editor, ratio banner. Large tables via `QTableView` + custom model |
| Map rendering | deck.gl in `QWebEngineView` | GPU rendering of hundreds of thousands of polygons; neighborhood coloring via per-parcel attribute buffers |
| Bulk Python ↔ map transfer | Raw typed binary buffers (the layout deck.gl consumes directly) via a custom `QWebEngineUrlSchemeHandler` | Geometry is sent to the map once; subsequent updates send only small typed arrays (e.g. neighborhood ID per parcel) |
| Python ↔ map events | `QWebChannel` | Small messages only: clicks, lasso polygons, selection events |
| Geometry I/O | pyogrio + GeoPandas | Shapefile, GeoPackage, GeoParquet, etc. |
| Spatial operations | Shapely 2 (`STRtree`) | Lasso polygon → intersecting parcel keys |
| Tabular data, joins, metrics | Polars (DuckDB as an alternative) | Parcel-key joins, type casting, null fill, per-neighborhood ratio statistics |
| Columnar format / export | PyArrow | Parquet and CSV export; internal snapshots |
| Sessions & undo journal | SQLite | Append-only action log for undo/replay; data snapshots stored as Parquet alongside |

## Architecture principles
* All computation (land values, metrics, evidence, lasso selection) happens in Python; the map is a rendering and input layer only.
* Parcel geometry is loaded to the map once. Neighborhood changes, flags, and edits update attributes, never re-send geometry.
* Metrics are recomputed incrementally, only for neighborhoods affected by an edit.
* The rate model lives in Python so that future log-curve and breakpoint rates require no map changes.
* JavaScript and map assets are bundled locally so the app runs fully offline. If a basemap is added, it is a local PMTiles file, not online tiles.

## First milestone: performance spike
Before building the rest of the app, load a real parcel dataset into PySide6 + deck.gl and measure initial load time, lasso selection over a large area, and recolor latency. All three should feel instant.

# Major Features

Map-driven interface

Ability to draw neighborhoods

Sanity tests / metrics for each neighborhood
* comparison to vacant sales
* land ratio studies
* spread of land values in the region (COD)
* implied allocation ratio given the land table
* implied allocation ratio given by the vacant sales
* check for land values in excess of total sale price

Ability to set a land table for each neighborhood

Ability to subdivide neighborhoods into sub-neighborhoods and have them inherit the land table from the parent neighborhood

## Data Inputs

All input files must include the **parcel key**, which is the join key across every file.

Parcel geometry is supplied by the user. It should preferably contain:
* parcel key
* geometry (parcel polygon)
* lat/long

Other inputs (CSVs):
* Vacant sales
* Improved sales — in scope, as they provide evidence for land allocation ratios and cost residuals
* Parcel characteristics
* Cost values — building-only costs (excluding land), in dollars:
  * replacement cost
  * depreciated cost
* Any other relevant data

Sale prices are assumed to already be time-adjusted by the user. The application does not perform time adjustment.

Bulk (multi-parcel) sales are not supported.

A parcel may have multiple sales. All of its sales that are not disqualified are used.

Additional data can be added to a session after the initial upload, though this is not expected to be common.

## Data Reconciliation

After upload, the application parses the files and lets the user reconcile the data:
* Select the data type for any column whose type is ambiguous
* Option to fill null or empty values
* Map a source column to each potential unit of measure (area, road frontage, lake frontage, river frontage, etc.)
* Parcel keys with no matching geometry are flagged to the user; processing continues without them

## Land table parts
Unit of measure (area, road frontage, lake frontage, river frontage, etc., or lot). A land table uses a **single** unit of measure (multi-unit tables are out of scope for now). Unit-of-measure quantities come from the columns mapped during reconciliation. Lot is not supplied as data: every parcel is assumed to be exactly 1 lot.

Rate (per unit of measure). The initial implementation uses a **flat rate**. The rate model must be designed to be extensible, since a log curve or user-defined breakpoints will likely be needed in the future.

Adjustments, each either additive or multiplicative, for parcels within this region compared to their neighbors:
* Each adjustment targets parcels selected by a **filter**:
  * a boolean condition on a continuous field (e.g. `acreage > 5`), or
  * a set of categories for a categorical field (e.g. `view in {golf, water}`)
* **Multiplicative** adjustments apply only to the value implied by the unit of measure (units × rate).
* **Additive** adjustments are applied after all multiplications.

So for a parcel: `land value = (units × rate) × Π(multipliers) + Σ(additions)`

## Neighborhood Inheritance

When a sub-neighborhood is created, it receives a **copy** of its parent's land table at that moment. After creation, parent and child are independent: changes to either do not propagate to the other. This divergence is intended.

## Neighborhood Management

Neighborhood identifiers are **strings**. New neighborhoods may be given generated IDs, but users are expected to rename them.

There is **no depth limit** on neighborhood nesting.

Lasso selection applies only to parcels within the **currently selected** neighborhood. Lassoing inside a sub-neighborhood therefore creates a child of that sub-neighborhood. Parcels touched by the lasso that belong to any other neighborhood (siblings, ancestors, or the selected neighborhood's existing children) are silently ignored and stay where they are.

Neighborhoods can be renamed or deleted. On delete:
1. The user is prompted to choose a recipient neighborhood from a dropdown of all neighborhoods **adjacent** to the one being deleted.
2. All parcels of the deleted neighborhood move to the recipient, so no parcels are orphaned.
3. The moved parcels are valued with the recipient neighborhood's land table.

## Evidence
Implied range of land values per UOM from vacant sales

Implied range of land values per UOM from Cost Residuals (improved sale price minus depreciated cost of improvements)

Implied range of land values per UOM from Sale-based allocation ratio (land allocation computed from improved sales rather than the current land table; see Metrics)

Implied best UOM for the land table based on the IAAO standard of using the measure with the lowest spread, where spread is measured by **COV** (coefficient of variation)

Each evidence range is displayed as `min ≤ avg ≤ max`.

## Sale Flags

Each sale is **known good**, **unflagged**, or **disqualified**.
* Disqualified sales are excluded from all ratio studies, metrics, and evidence.
* Known good and unflagged sales are both used, with equal weight.
* Future: known good sales will receive increased weight in metrics.

## Metrics

Land ratio study statistics, computed from **vacant sales** (land value ÷ sale price):
* Median ratio
* COD (coefficient of dispersion)
* PRD (price-related differential)
* PRB (price-related bias)

Land allocation ratio: the percentage of the full sale price attributed to land.
* For vacant land this should be 100%.
* For improved parcels it will be less, often derived from the cost values.

All allocation ratios use the same formula for built parcels:

`land allocation = 1 − improved allocation = 1 − (cost / total)`

where `cost` is the **depreciated cost** of the improvements and `total = land + cost`. They differ only in where `land` comes from:

| Allocation ratio | `land` | `total` |
|---|---|---|
| Table-based (implied by the land table) | land value derived from the neighborhood's land schedule | land + cost |
| Implied by vacant sales | average land value per UOM from the neighborhood's vacant sales × parcel units | land + cost |
| Sale-based (improved sales) | residual: sale price − cost | the improved sale price |

The sale-based allocation is a residual, so its implied land values are the same quantity as the Cost Residual evidence.

"Average land value per UOM" is the vacant sale price divided by the parcel's quantity in the land table's unit of measure (e.g. price ÷ acres, price ÷ front feet). For the lot UOM, it is the full sale price, since every parcel is 1 lot.

Future: users will be able to edit a depreciation schedule, applied to replacement cost, in place of the supplied depreciated cost.

Spread of land values in the region is measured by **COD**.

The UI displays **"Undefined"** in place of a metric score only when the metric is mathematically impossible to compute because there are too few sales (e.g. no sales at all, or a statistic that requires at least two observations). There is no minimum sample size beyond that: a metric computed from very few sales is still shown.

## Interface
User should be presented with a screen before the map in which they upload their parcel geometry and CSVs of their vacant sales, improved sales, characteristics, cost values, and any other relevant data. The application should parse the files and allow the user to perform reconciliation of their data (see Data Reconciliation).

User should have a pane for vacant sales that allows them to flag "known good" sales and "disqualified" sales. These will affect the metrics supplied to the user.

Map should treat everything initially as a single neighborhood and allow the user to create sub-neighborhoods. The user should be able to click on a neighborhood to see the land table and metrics for that neighborhood.

Map panel should have an unobtrusive but always-present current land ratio study results somewhere near the top of the window.

User should be able to click a neighborhood and have a pop-up that allows them to edit the land table for that neighborhood and also presents the metrics, tests, and evidence at a glance.

User should be able to shift click and drag to lasso parcels within the currently selected neighborhood to form a new sub-neighborhood. Neighborhood splits are done by parcel lasso: any parcel the lasso touches or crosses becomes part of the new neighborhood (parcels are never split).

## Sessions & Undo

Sessions can be saved and reopened.

User actions (boundary changes, land table edits, sale flags, reconciliation choices, etc.) are journaled, allowing them to be undone.

## Export

Export to **Parquet or CSV**: parcel-level data with the neighborhood assignment tagged onto each parcel, along with the land tables.

## Intended Workflow
1. User uploads parcel geometry and CSVs of vacant sales, improved sales, characteristics, and cost values
2. User reconciles the data and flags known good and disqualified sales
3. User creates a land table for the "everything" neighborhood and sees the metrics for that neighborhood
4. User selects the "everything" neighborhood and lassos parcels to form sub-neighborhoods
5. User selects each sub-neighborhood and edits the land table for that neighborhood and sees the metrics for that neighborhood
6. User iterates on the land tables and boundaries until satisfied with the results
7. User exports the parcels tagged with neighborhoods, plus the land tables, for use in their mass appraisal system

## Open Questions
None at this time.
