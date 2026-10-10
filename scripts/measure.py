"""The results: outie run on every mass: face by face, its default, by components and by parity, each scored by backfacingness and
timed, results/backfacingness.csv, which every figure reads to choose what it draws. Time is taken
twice, as the wall clock waits and as the processor works, summed over every core. The other methods run on one mass only, in
compare_methods.py.

    uv run --group figures python scripts/measure.py
"""

import csv
import time

import numpy as np

import outie
from load import DATA, RESULTS, load, score


def turn(triangles: np.ndarray, flip: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(flip, bool)[:, None], triangles[:, ::-1], triangles)


METHODS = {
    "outie": lambda v, f: turn(f, outie.reorient_facets_raycast(v, f)[0]),
    "outie, components": lambda v, f: turn(f, outie.reorient_facets_raycast(v, f, facet_wise=False)[0]),
    "outie, parity": lambda v, f: turn(f, outie.reorient_facets_raycast(v, f, use_parity=True)[0]),
}


def run(methods: dict, name: str) -> None:
    """Every method on every mass, scored and timed, written to results/<name>.csv, each method's mean printed."""
    rows = []
    for path in sorted(DATA.glob("*.off")):
        v, f = load(path.stem)
        rows.append({"mass": path.stem, "triangles": len(f), "method": "as published", "backfacingness": score(v, f), "seconds": "", "cpu_seconds": ""})
        for label, method in methods.items():
            wall, cpu = time.perf_counter(), time.process_time()
            result = method(v, f)
            wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
            rows.append({"mass": path.stem, "triangles": len(f), "method": label, "backfacingness": score(v, result), "seconds": f"{wall:.3f}", "cpu_seconds": f"{cpu:.3f}"})
        print(path.stem, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, ["mass", "triangles", "method", "backfacingness", "seconds", "cpu_seconds"])
        writer.writeheader()
        writer.writerows({**r, "backfacingness": f"{r['backfacingness']:.4f}"} for r in rows)
    for label in ["as published", *methods]:
        mine = [r for r in rows if r["method"] == label]
        scores = np.array([r["backfacingness"] for r in mine])
        seconds = sum(float(r["seconds"] or 0) for r in mine), sum(float(r["cpu_seconds"] or 0) for r in mine)
        print(f"{label:32s} {scores.mean():.4f} ± {scores.std():.4f}   {seconds[0]:6.1f} s   {seconds[1]:7.1f} s of processor")


if __name__ == "__main__":
    run(METHODS, "backfacingness")
