"""Entry point: ``python -m land_painter [parcel_file] [--selftest]``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from .map import register_scheme
from .parcels import load_parcels

PARCEL_FILTER = "Parcel geometry (*.parquet *.geoparquet *.gpkg *.shp *.geojson *.fgb);;All files (*)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="land-painter")
    parser.add_argument("parcels", nargs="?", help="parcel geometry file")
    parser.add_argument("--key", default="key", help="parcel key column (default: key)")
    parser.add_argument("--selftest", action="store_true",
                        help="load, run one lasso over the central area, print timings as JSON, and exit")
    parser.add_argument("--screenshot", metavar="PNG", help="with --selftest: save a screenshot after the lasso")
    args = parser.parse_args(argv)

    register_scheme()
    app = QApplication(sys.argv[:1])

    path = args.parcels or QFileDialog.getOpenFileName(None, "Open parcel geometry", "", PARCEL_FILTER)[0]
    if not path:
        return 0

    from .app import MainWindow  # after QApplication exists

    t0 = time.perf_counter()
    try:
        layer = load_parcels(path, args.key)
    except Exception as exc:
        QMessageBox.critical(None, "Could not load parcels", str(exc))
        return 1
    load_ms = (time.perf_counter() - t0) * 1000

    window = MainWindow(layer)
    window.show()
    timings = {"python_load": load_ms}
    print(f"Loaded {layer.n_parcels:,} parcels ({layer.n_parts:,} parts, "
          f"{len(layer.positions) // 2:,} vertices) in {load_ms:,.0f} ms", flush=True)

    if args.selftest:
        _run_selftest(app, window, timings, args.screenshot)
    return app.exec()


def _run_selftest(app: QApplication, window, timings: dict, screenshot: str | None) -> None:
    minx, miny, maxx, maxy = window.layer.bounds
    w, h = maxx - minx, maxy - miny
    # A box over the central ~40% of the extent: a large lasso.
    box = [[minx + 0.3 * w, miny + 0.3 * h], [maxx - 0.3 * w, miny + 0.3 * h],
           [maxx - 0.3 * w, maxy - 0.3 * h], [minx + 0.3 * w, maxy - 0.3 * h]]

    def finish():
        if screenshot:
            # Grab from the screen: QWidget.grab() misses GPU-composited web content.
            window.screen().grabWindow(window.winId()).save(screenshot)
        timings["lassoed_parcels"] = int((window.model.assignment == 1).sum())
        print(json.dumps({k: round(v, 1) for k, v in timings.items()}), flush=True)
        app.quit()

    def on_timing(name: str, ms: float):
        timings[name] = ms
        if name == "first_render":
            QTimer.singleShot(500, lambda: window.lasso(box))
        elif name == "recolor":
            QTimer.singleShot(500 if screenshot else 0, finish)

    window.timing.connect(on_timing)
    QTimer.singleShot(120_000, lambda: (print("selftest timed out", timings, flush=True), app.exit(1)))


if __name__ == "__main__":
    sys.exit(main())
