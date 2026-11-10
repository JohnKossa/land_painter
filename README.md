# Land Painter

An interactive desktop tool for mass appraisers to **draw neighborhoods and tune land tables on a
live map**, watching assessment-quality metrics update as they work — then export the result for use
in a CAMA system.

It's built to handle hundreds of thousands of parcels with responsive map rendering, and to make no
assumptions about the shape of your data: you tell it what each column means after uploading.

> **New here?** See **[QUICKSTART.md](QUICKSTART.md)** to install and run it (no programming
> background needed). The full design is in **[SPEC.md](SPEC.md)**.

---

## How you use it to dial in land values

In Land Painter, every parcel's land value comes from the **land table** of the neighborhood it
belongs to:

```
land value = (units × rate) × (all multipliers) + (all additions)
```

where *units* is the parcel's measure for that table (acreage, front feet, or simply "1 lot"), and
multipliers/additions are **adjustments** that target parcels matching a filter (e.g. `view in
{water, golf}`, or `acreage > 5`). Your job is to carve the map into neighborhoods and get each
neighborhood's table right. The app grades your work against the sales as you go, so the loop is:

1. **Upload** your parcel geometry and CSVs (vacant sales, improved sales, characteristics, cost
   values, optionally current roll values), then **reconcile** the data — confirm the parcel key,
   the sale price/date/vacant flag, units of measure, and so on. The app suggests mappings; you
   confirm them.
2. **Flag sales** — mark known-good and disqualified sales so the metrics lean on the right ones.
3. **Start with "Everything"** — give the whole county a first land table and read its metrics.
4. **Lasso sub-neighborhoods** on the map where values clearly differ, and give each its own table.
5. **Read the evidence and tune.** For each neighborhood the app shows a ratio study
   (median, COD, PRD, PRB), land-allocation ratios, the spread of land value per unit, the best
   unit of measure to use, implied value ranges from vacant sales and cost residuals, a check for
   land values above sale price, and — where ratios cluster in one part of a neighborhood — a hint
   that it should be split or is missing an adjustment. Adjust rates, add adjustments, move
   boundaries, and the numbers update live.
6. **Compare and audit.** Use the Compare tab to put neighborhoods side by side, and the
   land/building split check to spot where the current roll has booked location value as building
   (so land is under-assessed).
7. **Iterate** until the metrics look right, then **export** the parcels tagged with their
   neighborhoods plus the land tables.

The point is to make the *evidence* for a land value visible while you set it, instead of valuing
blind and checking later.

---

## What's in it

- **Map-driven neighborhoods** — draw sub-neighborhoods by lasso, reassign parcels between them,
  split by a filter, merge, or import existing neighborhood codes. No depth limit; sub-neighborhoods
  inherit a copy of their parent's table.
- **Land tables** — a single unit of measure, a rate, and any number of additive/multiplicative
  adjustments driven by filters on your characteristics. Designed so future rate curves and
  breakpoints can be added.
- **Metrics, tests, and evidence** — IAAO ratio statistics, allocation ratios (from the table, from
  vacant sales, and from improved-sale residuals), horizontal-uniformity and spatial-clustering
  checks, per-unit evidence with a recommended best unit, and a comparison against the previous roll
  when roll values are mapped.
- **Sales pane** — browse, filter, and flag sales; see them on the map.
- **Compare tab** — distributions and effect sizes across the neighborhoods you pick.
- **Land/building split check** — audits whether the current roll hides location value in building
  values.
- **Sessions & undo** — every action is journaled; undo/redo is unlimited and survives reopening.
  Sessions reference your input files rather than copying them.
- **Export** — parcels tagged with neighborhood and land value, plus the land tables and
  adjustments, to Parquet or CSV.

---

## Getting started

Full instructions (including installing the one tool you need) are in
**[QUICKSTART.md](QUICKSTART.md)**. The short version, once set up:

```powershell
uv sync            # one time: fetches Python and the libraries
uv run land-painter
```

---

## Notes & status

- **Windows only** for now.
- Runs **offline**. An optional street-map backdrop (via a Carto key) can be enabled — see
  QUICKSTART — and a bundled offline basemap is planned to replace it.
- **Sale prices are assumed already time-adjusted** by you; the app does not time-adjust.
- Actively developed and in usability testing — expect rough edges.

---

## Development

```powershell
uv sync --extra dev
uv run pytest          # tests
uv run ruff check .    # lint
```

Project layout: application code in `src/land_painter/` (domain logic at the top level, UI widgets
under `ui/`, the map front-end under `map/`), tests in `tests/`.

---

## Acknowledgments

Special thanks to the **Lee County Property Appraiser's office** for their insight, domain
expertise, and support of this project. Their guidance on assessment practice and real-world
appraisal workflows shaped the design throughout.

---

## License

Land Painter is licensed under the **GNU Affero General Public License, version 3.0** — see
[LICENSE](LICENSE) for the full text. In short: you may use, study, modify, and share it freely,
but any modified version you distribute **or run as a network service** must be offered to its users
under the same license, with source.

Copyright © 2026 John Kossa
