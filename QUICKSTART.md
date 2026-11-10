# Land Painter — Quick Start

This guide gets Land Painter running on your Windows PC. **You don't need to know how to
program.** You'll copy a few commands into a terminal; each step says what it does and roughly
how long it takes. The first-time setup is about 10–15 minutes, and starting the app after that
is two quick commands.

A single tool called **uv** does all the heavy lifting — it installs the right version of Python
and every library Land Painter needs, so you don't have to install those yourself.

---

## What you'll need

- A **Windows** PC (Land Painter is Windows-only for now).
- The **Land Painter project folder** on your PC. If you don't have it yet, clone the Git
  repository or copy the folder your team shared with you. Remember where you put it.
- About 10–15 minutes for the one-time setup.

---

## Step 1 — Install uv (one time)

Open **Windows PowerShell** (click Start, type "PowerShell", press Enter) and paste this line,
then press Enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

This installs uv. When it finishes, **close that PowerShell window and open a new one** — this
lets your PC find the newly installed tool.

> You only ever do this once per PC.

---

## Step 2 — Open a terminal in the project folder

1. Open **File Explorer** and go to the Land Painter project folder (the one containing this
   `QUICKSTART.md` file).
2. Click the **address bar** at the top (where the folder path is shown), type `powershell`,
   and press Enter.

A terminal window opens, already pointed at the project folder. You'll run the next commands here.

---

## Step 3 — Install Land Painter (one time)

In that terminal, type:

```powershell
uv sync
```

The first time, this downloads Python and all of Land Painter's libraries. It can take a few
minutes and will print a lot of lines — that's normal. When it's done you're ready to go, and you
won't need to run this again unless the app is updated.

---

## Step 4 — Start Land Painter

```powershell
uv run land-painter
```

The **upload wizard** opens. Point it at your parcel geometry file and your CSVs (vacant sales,
improved sales, characteristics, cost values), then follow the on-screen steps to confirm what
each column means. From there you're in the map.

### Just want to see it work first? (optional)

You can generate a small set of pretend parcels and open them, no real data required:

```powershell
uv run python scripts/make_synthetic_parcels.py data/sample.parquet --count 2000
uv run land-painter data/sample.parquet
```

---

## Optional — the background map

Land Painter can show a street-map backdrop behind your parcels, but **this is optional**. Without
it, everything still works — your parcels just draw on a plain background instead of over a map.

The backdrop uses a service called **Carto**, which needs an access key. If you have one (ask
whoever set up your team's account), tell Land Painter about it once:

```powershell
setx LAND_PAINTER_CARTO_KEY "your-key-goes-here"
```

Then **close and reopen the terminal** so the change takes effect, and start the app again. If you
don't set a key, Land Painter simply skips the backdrop — no error, nothing else changes.

> The background map is a temporary convenience; a built-in offline map is planned, after which no
> key will be needed.

---

## Opening a saved session

Land Painter saves your work in a `.lpsession` file. To reopen one, either use **File → Open
Session** inside the app, or point to it when you start:

```powershell
uv run land-painter "C:\path\to\my-work.lpsession"
```

(The quotes matter if the path has spaces in it.)

---

## Everyday use (after the one-time setup)

Once Steps 1–3 are done, starting Land Painter is just:

1. Open the project folder in File Explorer, type `powershell` in the address bar, Enter.
2. Run `uv run land-painter`.

---

## If something goes wrong

- **"uv is not recognized…"** — Close the terminal and open a new one (uv needs a fresh window to
  be found). If it still doesn't work, restart your PC and try again.
- **Step 3 or 4 is slow the first time** — It's downloading Python and libraries. Later runs are
  fast.
- **No street map behind the parcels** — That's the optional background map (see above). The app
  is working fine; you just haven't set a Carto key.
- **You opened the wrong folder** — Step 2's terminal must be in the folder that contains this
  `QUICKSTART.md`. Re-do Step 2 if `uv run land-painter` says it can't find the app.
- **Still stuck?** — Copy the red error text from the terminal and send it to whoever maintains
  Land Painter; it usually says exactly what's missing.
