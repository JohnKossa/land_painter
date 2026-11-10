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
* Current roll values (optional), in dollars, each possibly summed from several columns:
  * assessed building value (e.g. main building + extra features)
  * assessed land value
  
  Used only to audit the existing assessment's land/building split (whether location value is booked as building) and to compare the roll with the app's land tables. They never set land values. The building audit divides roll building value by depreciated cost, so it's only meaningful when the mapped depreciated cost is location-free (cost book only, no neighborhood or market factor). Sessions saved before these fields existed load with them unmapped.
* Any other relevant data

Sale prices are assumed to already be time-adjusted by the user. The application does not perform time adjustment.

Bulk (multi-parcel) sales are not supported.

A parcel may have multiple sales. All of its sales that are not disqualified are used.

Additional data can be added to a session after the initial upload, though this is not expected to be common.

## Data Reconciliation

The application must not hard-code assumptions about the shape of the input data (column names, which flag marks a vacant sale, which cost columns to use, etc.). Uploaded data varies between jurisdictions; the user reviews it after upload and tells the application what each column means. Reference sample data (Guilford County) is a fair example of what to expect, not a fixed schema.

* A single file may fill several roles (e.g. one "universe" file supplying parcel geometry, characteristics, and cost values).
* The user maps columns to roles: parcel key, geometry, sale price (time-adjusted), sale date, vacant/improved indicator, replacement cost, depreciated cost, and each unit of measure. Where a role could reasonably combine several columns (e.g. building cost plus outbuilding cost), the user decides.
* The application may *suggest* mappings (e.g. by column name), but the user confirms them.
* The review shows summary statistics for mapped columns and flags suspect values for the user to decide on, rather than silently fixing them: keys that don't match across files, zero or negative values in unit-of-measure columns, implausible outliers or sentinel values, and nulls.

After upload, the application parses the files and lets the user reconcile the data:
* Select the data type for any column whose type is ambiguous
* Option to fill null or empty values
* Map a source column to each potential unit of measure (area, road frontage, lake frontage, river frontage, etc.)
* Parcel keys with no matching geometry are flagged to the user; processing continues without them
* Optionally choose an existing neighborhood column to import as starting neighborhoods (see Neighborhood Import)

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

An **"Apply land table to all children"** action copies a neighborhood's current land table to all of its direct children, replacing theirs. It is a single undoable action. This is how a parent's table is pushed to many neighborhoods at once (e.g. after an import).

## Neighborhood Management

Neighborhood identifiers are **strings**. New neighborhoods may be given generated IDs, but users are expected to rename them.

There is **no depth limit** on neighborhood nesting.

Lasso selection applies only to parcels within the **currently selected** neighborhood. Lassoing inside a sub-neighborhood therefore creates a child of that sub-neighborhood. Parcels touched by the lasso that belong to any other neighborhood (siblings, ancestors, or the selected neighborhood's existing children) are silently ignored and stay where they are.

**Reassign lasso** (Ctrl+Shift+drag): moves every parcel the lasso touches, from **any** neighborhood, into the currently selected neighborhood. No new neighborhood is created. Moved parcels are valued with the selected neighborhood's land table. This is how boundaries between sibling neighborhoods are adjusted. If a reassign would leave any neighborhood with no parcels, a confirmation pop-up names the neighborhood(s) that would be emptied, before anything changes:
* **Confirm:** the transfer happens and the emptied neighborhood(s) are deleted, all as one undoable action.
* **Cancel:** the transfer does not happen; nothing changes.

Neighborhoods can be renamed or deleted. On delete:
1. The user is prompted to choose a recipient neighborhood from a dropdown of all neighborhoods **adjacent** to the one being deleted.
2. All parcels of the deleted neighborhood move to the recipient, so no parcels are orphaned.
3. The moved parcels are valued with the recipient neighborhood's land table.
4. Child neighborhoods of the deleted neighborhood move up to its parent.

Implementation notes:
* **Adjacent** means having parcels within ~50 m (0.0005°) of the neighborhood's parcels, so neighborhoods separated only by a road count as adjacent. Choices are ordered by how many nearby parcels they share (closest first); the parent is always offered as a fallback, even if not adjacent. "Everything" cannot be deleted.
* For the reassign lasso, a neighborhood counts as **emptied** when it has no parcels left and no child neighborhoods. "Everything" is never deleted.
* Deleted neighborhoods are kept internally (hidden) so undo and the session journal stay consistent.

## Neighborhood Import

Existing neighborhood assignments can be imported from a column in the parcel or characteristics data. The import can be chosen during upload or run later from a menu.
* Each distinct value in the column becomes a child neighborhood of "Everything", named by that value.
* Parcels with a null or empty value stay in "Everything".
* Imported neighborhoods copy the parent's land table at the moment of import, following the normal inheritance rule. If "Everything" has no land table yet, they start without one; use "Apply land table to all children" to seed them.
* Imported neighborhoods behave exactly like drawn ones: they can be renamed, deleted, subdivided by lasso, and adjusted with the reassign lasso. They may be non-contiguous.
* The import is a single undoable action. The journal stores the resulting assignment, so replay does not depend on the source column.

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

### Metric details (as implemented — confirm or adjust)
* **Scope:** a neighborhood's metrics use the parcels directly in it (the parcels its land table values) and the sales on those parcels. The always-visible banner uses all vacant sales county-wide.
* **Sales used:** not disqualified, with a positive price and a known vacant/improved value. The ratio study additionally needs the parcel to have a land value.
* **PRB:** IAAO regression of (ratio − median) / median on log₂ of the value proxy ½·price + ½·(land value / median ratio).
* **CHD (coefficient of horizontal dispersion):** COD of land value per unit (the land table's unit) across the neighborhood's parcels with a land value and positive units. It needs no sales, so it measures horizontal uniformity (similar land valued alike) where the ratio study can't. Highlighted above 15, and flagged as unreliable with fewer than 15 parcels. Both are rules of thumb from the Center for Land Economics' "Mass Appraisal for the Masses", not an IAAO standard.
* **Spatial clustering of ratios (residual spatial autocorrelation):** for the selected neighborhood only. Uses the ratio study's sales (usable vacant sales with a positive land value). Residual = ln(land value / price) − the neighborhood's median ln ratio. r = correlation of each residual with the mean residual of its 8 nearest sales in the neighborhood (fewer if it has fewer sales), never counting a sale on the same parcel. One-sided p-value from 999 permutations of the residuals among the neighborhood's sales (fixed seed). A time-controlled r excludes neighbors sold within 90 days, refilling with the next-nearest; pairs with a missing date are kept. Highlighted at p < 0.05. Not computed for the county-wide banner or the land table draft preview.
* **Check against market (amenity allocation):** on demand, from an adjustment in the land table panel. The adjustment's condition is the amenity. On the same usable improved sales (with a land value, a positive depreciated cost and every chosen control), fits ln(sale price), ln(land value) and ln(depreciated cost) on the amenity indicator plus the chosen controls. Controls are units and continuous characteristics, entered as ln(x) when all positive, else linearly. The default control is the table's unit unless it's Lot. "All neighborhoods" adds neighborhood fixed effects (within-neighborhood demeaning); "this neighborhood only" uses none. Land values include the neighborhood's draft table. Each coefficient is reported as e^β − 1 with a 95% range, and in dollars at the sample median. Building share = building $ ÷ (land $ + building $), undefined when the two have opposite signs. The assessed premium (land $ + building $) is compared with the market premium. Highlighted: a building share over 50%, an assessed premium outside 75–125% of the market or of opposite sign, and a market range that spans 0.
* **Land/building split check (current roll; Tools menu, on demand):** needs roll building value and depreciated cost. Parcels are grouped by the imported neighborhood codes (default, when mapped) or the session's neighborhoods. Groups with fewer than 5 qualifying parcels are left out.
  * **5a:** improved parcels with a positive roll building value and depreciated cost. ratio_b = roll building ÷ depreciated cost. One-way ANOVA of ln(ratio_b) by group gives η², ω² (adjusted) and a one-way random-effects ICC. ω² is highlighted at 0.06 or above (Cohen's "medium").
  * **5b:** the same decomposition for ln(roll land value per unit) on the same parcels. The session's land tables are shown alongside for reference. Leakage = building ω² ÷ roll land ω², highlighted at 0.75 or above (location split across both components).
  * A per-neighborhood table (median building ratio, median roll and session land per unit) is sorted by building ratio, so the neighborhoods where land is most likely under-booked come first.
* **Allocation ratios** are reported as the median across built parcels (depreciated cost > 0), or across improved sales for the sale-based ratio. The vacant-sales version uses the **mean** vacant sale price per unit (the spec's "average").
* **Land value in excess of sale price:** count of usable sales (vacant and improved) whose parcel's land value exceeds the sale price.
* **Previous roll** (shown when roll land value is mapped), for the neighborhood's parcels:
  * Change: new ÷ roll − 1 per parcel with a positive roll value and a new land value (a new value of $0 counts as −100%). Reported as the median and middle half.
  * Big movers: parcels changing more than ±25% either way, split up and down. The threshold is one constant (`MOVE_THRESHOLD`), passed as a parameter so it can become a user setting.
  * Coverage: parcels with no roll value, and parcels with a roll value but no new land value.
  * The roll's ratio study on the same vacant sales. It appears as a second row ("Previous roll") under "Land tables" in the neighborhood's ratio study table, so each statistic is shown once and the two compare head to head. Without roll values, the table has only the land tables row.
  * The land table panel's draft preview also shows the change from the roll.
  * The evidence table gains a "Previous roll: land per unit" range per unit. It's a reference, not market evidence, so it doesn't count toward the score or the best unit. The column is hidden when roll land value isn't mapped.
* **Evidence** ranges are shown for every unit of measure, so the best unit can be compared. Cost residual and sale-based allocation evidence are one row, since they are the same quantity.
* **Highlighting:** ratio statistics outside IAAO ranges are highlighted: median 0.90–1.10, COD ≤ 25 (vacant land), PRD 0.98–1.03, PRB −0.05 to 0.05.
* While a land table is being edited, the draft's ratio study is previewed before it is applied.

The UI displays **"Undefined"** in place of a metric score only when the metric is mathematically impossible to compute because there are too few sales (e.g. no sales at all, or a statistic that requires at least two observations). There is no minimum sample size beyond that: a metric computed from very few sales is still shown.

### Compare tab
A dock tab beside Sales that compares one attribute's distribution across the neighborhoods the user checks (parcels directly in each, as elsewhere).
* **Attributes:** land value and land value per unit, change from previous roll, sale ratio (vacant sales), sale price and price per unit, units, depreciated cost, roll building and land values, roll building ÷ cost, and every continuous or categorical characteristic. Options appear only when their data is mapped.
* **Compare:** parcels, sales (parcel attributes read off each usable sale's parcel), or sales vs. parcels (both per neighborhood, to check that sales represent the parcels). Sale-only attributes always show sales.
* **Chart:** small multiples, one row per group, on shared bins, a shared x-axis and a shared y-scale. Bars are each group's share, so groups of different sizes compare. A tick marks each median. The range is the pooled 1st–99th percentile, but a tail is clipped only when it lies more than half that range beyond it. Clipped counts show as ◂n / n▸. There's an optional log scale; values ≤ 0 count as off-scale. Categorical attributes show shares of the 12 most common categories, with the rest folded into "Other". Hover shows a bar's share and count. Colour only separates parcels from sales, using the validated reference palette's slots 1–2 with a legend. Row labels carry neighborhood identity.
* **Summary table:** n, missing, median, middle 50%, min, max (or most common category and its share).
* **"How different?" matrix:** effect sizes, not p-values. For continuous attributes, Cliff's δ = P(column > row) − P(column < row), with the KS distance and median change in the tooltip. For categorical attributes, Cramér's V. Shading is one sequential hue by magnitude (Romano thresholds 0.147 / 0.33 / 0.474 for |δ|; 0.1 / 0.3 / 0.5 for V), with the magnitude word written in each cell.
* **Shortcuts** for the neighborhood selected on the map. Each replaces the current checks.
  * **Children of selection:** every child, each with its own sub-neighborhoods. Used to compare the pieces of a split; similar children are candidates to merge.
  * **Selection vs. rest of parent:** the selection with its sub-neighborhoods, against everything else under its parent (the parent's own parcels plus all siblings and their sub-neighborhoods). The two groups split the parent exactly. Used to test whether a split was worth making. Not available for the top neighborhood.
  * Combined groups ("+ sub-neighborhoods", "Rest of …") appear as ⊕ entries above the tree. They're rebuilt from the current tree on every change, so they follow renames and reassignments, and they're dropped if their neighborhood is deleted. Clear removes them.
* **Map linking:** clicking a bar shows its parcels on the map. For a sales row, these are the sold parcels. Clicking a row's label shows all of that group's parcels. Shown parcels are drawn near-opaque and every other parcel fades; there's no outline-only mode, so a scattered set stays visible at county zoom. The clicked bar gets a ring, and a row's label goes bold. A strip above the chart says what's on the map, with **Zoom to** (fit the map to those parcels) and **Clear map**. Clicking the same bar again also clears. The highlight clears when the comparison changes (new bins), or when the tab is hidden. A bar's parcels use the same bin assignment as its count (one shared function), so the two always agree.
* It recomputes when land values, neighborhoods or sale flags change, but only while the tab is visible.

## Interface
User should be presented with a screen before the map in which they upload their parcel geometry and CSVs of their vacant sales, improved sales, characteristics, cost values, and any other relevant data. The application should parse the files and allow the user to perform reconciliation of their data (see Data Reconciliation).

User should have a pane for vacant sales that allows them to flag "known good" sales and "disqualified" sales. These will affect the metrics supplied to the user.
* The pane shows vacant sales by default and can switch to improved or all sales; it can also filter by flag and to the selected neighborhood (including its sub-neighborhoods).
* Columns include each mapped unit of measure and, for vacant sales, price per unit, to help spot outliers.
* Flags apply to all selected rows (buttons, context menu, or keys G / U / D). Each flag change is one undoable action.
* Shown sales appear as dots on the map, colored by flag. Selecting a sale outlines its parcel; double-clicking flies the map to it; clicking a dot selects its row.

Map should treat everything initially as a single neighborhood and allow the user to create sub-neighborhoods. The user should be able to click on a neighborhood to see the land table and metrics for that neighborhood.

Map panel should have an unobtrusive but always-present current land ratio study results somewhere near the top of the window.

User should be able to click a neighborhood and have a pop-up that allows them to edit the land table for that neighborhood and also presents the metrics, tests, and evidence at a glance.
* The pop-up is a floating window that opens beside the click and follows later clicks; it can be docked into the main window. Opening on click can be turned off (View menu).
* It shows the neighborhood's name with Rename and Delete, the land table editor (with a searchable neighborhood picker), and the metrics and evidence side by side.

User should be able to shift click and drag to lasso parcels within the currently selected neighborhood to form a new sub-neighborhood, and ctrl+shift click and drag to reassign lassoed parcels from any neighborhood into the selected one. Neighborhood splits are done by parcel lasso: any parcel the lasso touches or crosses becomes part of the new neighborhood (parcels are never split).

## Sessions & Undo

Sessions can be saved and reopened.

User actions (boundary changes, land table edits, sale flags, reconciliation choices, etc.) are journaled, allowing them to be undone.

* The session file (`.lpsession`) is a SQLite database holding the action journal and references to the input files; inputs are not copied into it. Opening a session reloads the inputs and replays the journal.
* A fingerprint of the parcel keys is stored; opening a session against a different or modified parcel file is refused. Likewise for the sales (key, price, date of every sale), since sale flags are tied to specific sales. If an input file has moved, the app looks beside the session file, then asks the user to locate it.
* A new session is unsaved until "Save As". After that, every action is written to the file as it happens.
* Undo/redo is linear: a new action after an undo discards the redo history. Undo history survives closing and reopening the session.

## Export

Export to **Parquet or CSV**: parcel-level data with the neighborhood assignment tagged onto each parcel, along with the land tables.

File → Export writes three files side by side:
* `<name>`: one row per parcel, in the chosen format: key, neighborhood, neighborhood path (e.g. `Everything / North / Lake`), unit of measure, units, rate, base value, combined multiplier, combined addition, land value (`base × multiplier + addition`). Parcels without a land value have empty values. As Parquet this is written as **GeoParquet** — it includes the parcel geometry (WGS84), so it opens and renders directly in QGIS and other GIS tools; as CSV it is attributes only (CSV has no native geometry).
* `<name>_land_tables.csv`: one row per neighborhood (deleted ones excluded): neighborhood, path, parent, parcel count, unit of measure, rate model, rate, number of adjustments. Neighborhoods without a land table have an empty rate.
* `<name>_adjustments.csv`: one row per adjustment: neighborhood, order, name, field and its source file, operator, value or values (`|`-separated), readable condition, effect (multiply/add), amount.

The two reference tables are **always CSV** (they have no geometry, and CSV opens anywhere, e.g. LibreOffice), regardless of the parcel file's format. Only applied land tables are exported; the user is warned about unapplied edits. CSV text values are quoted so keys keep leading zeros.

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
