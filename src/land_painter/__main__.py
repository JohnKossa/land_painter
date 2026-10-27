"""Entry point.

    python -m land_painter                      # upload wizard
    python -m land_painter data1.parquet ...    # upload wizard with these files added
    python -m land_painter my.lpsession         # open a session
    python -m land_painter parcels.parquet --selftest   # parcel-only session, timed lasso/undo/redo
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .map import register_scheme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="land-painter")
    parser.add_argument("paths", nargs="*", help="data files to start the upload wizard with, or one .lpsession file")
    parser.add_argument("--key", help="with --selftest: parcel key column (default: suggested from column names)")
    parser.add_argument("--selftest", action="store_true",
                        help="build a parcel-only session from one file without the wizard, lasso the central "
                             "area, undo, redo, print timings as JSON, and exit")
    parser.add_argument("--screenshot", metavar="PNG", help="with --selftest: save a screenshot after the lasso")
    args = parser.parse_args(argv)

    register_scheme()
    app = QApplication(sys.argv[:1])

    from .app import new_session, open_session, show_window  # after QApplication exists
    from .session import SESSION_SUFFIX, Session

    t0 = time.perf_counter()
    if args.selftest:
        from .data.config import parcel_only_config
        from .data.dataset import build_dataset
        from .data.sources import inspect

        session = Session.new(build_dataset(parcel_only_config(inspect(args.paths[0]), args.key)))
    elif len(args.paths) == 1 and args.paths[0].lower().endswith(SESSION_SUFFIX):
        session = open_session(args.paths[0])
    else:
        session = new_session(files=args.paths)
    if session is None:
        return 1
    load_ms = (time.perf_counter() - t0) * 1000
    layer = session.layer
    print(f"Loaded {layer.n_parcels:,} parcels ({layer.n_parts:,} parts, "
          f"{len(layer.positions) // 2:,} vertices) in {load_ms:,.0f} ms", flush=True)

    window = show_window(session)
    if args.selftest:
        _run_selftest(app, window, {"python_load": load_ms}, args.screenshot)
    return app.exec()


def _run_selftest(app: QApplication, window, timings: dict, screenshot: str | None) -> None:
    window._discard_on_close = True
    minx, miny, maxx, maxy = window.layer.bounds
    w, h = maxx - minx, maxy - miny
    # A box over the central ~40% of the extent: a large lasso.
    box = [[minx + 0.3 * w, miny + 0.3 * h], [maxx - 0.3 * w, miny + 0.3 * h],
           [maxx - 0.3 * w, maxy - 0.3 * h], [minx + 0.3 * w, maxy - 0.3 * h]]
    assignment = window.session.neighborhoods.assignment
    steps = [  # (name, action, check after the map has re-rendered)
        ("lasso", lambda: window.lasso(box), lambda: (assignment == 1).sum() > 0),
        ("undo", window.undo, lambda: (assignment == 0).all()),
        ("redo", window.redo, lambda: (assignment == 1).sum() > 0),
    ]
    results = {}

    def run_next():
        if not steps:
            return finish()
        name, act, _ = steps[0]
        t0 = time.perf_counter()
        act()
        results[name] = t0

    def finish():
        if screenshot:
            # Grab from the screen: QWidget.grab() misses GPU-composited web content.
            window.screen().grabWindow(window.winId()).save(screenshot)
        timings["lassoed_parcels"] = int((assignment == 1).sum())
        print(json.dumps({k: round(float(v), 1) for k, v in timings.items()}), flush=True)
        app.quit()

    def on_timing(name: str, ms: float):
        if name == "first_render":
            timings[name] = ms
            QTimer.singleShot(500, run_next)
        elif name == "recolor" and steps:
            step, _, check = steps.pop(0)
            timings[f"{step}_total"] = (time.perf_counter() - results[step]) * 1000
            if not check():
                print(f"selftest FAILED: state wrong after {step}", flush=True)
                app.exit(1)
                return
            QTimer.singleShot(500 if screenshot and not steps else 0, run_next)
        elif name not in ("recolor",):
            timings[name] = ms

    window.timing.connect(on_timing)
    QTimer.singleShot(120_000, lambda: (print("selftest timed out", timings, flush=True), app.exit(1)))


if __name__ == "__main__":
    sys.exit(main())
