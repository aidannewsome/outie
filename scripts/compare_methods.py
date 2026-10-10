"""One mass as published and as each method leaves it, side by side, gold fronts and grey backs, each named beneath with
the time it took, as the pictures show how well each did: figures/methods.png. Each method's backfacingness and time on
that mass go in results/methods.csv for the README. The mass is the largest
well mixed one in the results, the most triangles among those scoring between 0.4 and 0.6 as published. Time is taken twice, as the wall clock waits
and as the processor works, summed over every core: libigl runs on one core and outie on all.

The methods with code to run, others first and ours last: trimesh's fix_normals, winding made to agree and then
turned by signed volume; libigl's orient_outward, each component turned away from its centre; libigl's
reorient_facets_raycast, the paper's own code, which outie ports; and outie with its defaults. Outie's own settings
are compared in the results and the components figure.

    uv run --group figures python scripts/compare_methods.py
"""

import csv
import time

import igl
import igl.embree
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from draw import draw
from load import GAP, HEIGHT, OUT, RESULTS, find_largest_mixed, load, scale_to_height, score
from measure import METHODS as OUTIE
from measure import turn


def fix_with_trimesh(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    return np.asarray(mesh.faces)


METHODS = {  # others first, ours last, as comparisons are laid out
    "trimesh fix_normals": fix_with_trimesh,
    "libigl orient_outward": lambda v, f: igl.orient_outward(v, f, igl.orientable_patches(f)[:, None])[0],
    "libigl reorient_facets_raycast": lambda v, f: turn(f, igl.embree.reorient_facets_raycast(v, f, rays_total=100 * len(f))[0]),
    "outie": OUTIE["outie"],
}

if __name__ == "__main__":
    name = find_largest_mixed()
    v, f = load(name)
    rows, images = [{"mass": name, "method": "as published", "backfacingness": score(v, f), "seconds": "", "cpu_seconds": ""}], [draw(v, f, size=700)]
    for label, method in METHODS.items():
        wall, cpu = time.perf_counter(), time.process_time()
        result = method(v, f)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
        rows.append({"mass": name, "method": label, "backfacingness": score(v, result), "seconds": f"{wall:.3f}", "cpu_seconds": f"{cpu:.3f}"})
        images.append(draw(v, result, size=700))
    with open(RESULTS / "methods.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, ["mass", "method", "backfacingness", "seconds", "cpu_seconds"])
        writer.writeheader()
        writer.writerows({**r, "backfacingness": f"{r['backfacingness']:.4f}"} for r in rows)
    drawn = np.argwhere(np.any([(i < 0.995).any(axis=2) for i in images], axis=0))
    (top, left), (bottom, right) = drawn.min(axis=0), drawn.max(axis=0) + 1
    tiles = [scale_to_height(i[top:bottom, left:right], HEIGHT) for i in images]
    space = 2 * GAP  # between columns, room for their names
    row = np.hstack([t for tile in tiles for t in (tile, np.ones((HEIGHT, space, 3)))][:-1])
    label = 4 * GAP  # beneath, for a name over two lines and its backfacingness
    out = np.ones((HEIGHT + 2 * GAP + label, row.shape[1] + 2 * space, 3))
    out[GAP : GAP + HEIGHT, space : space + row.shape[1]] = row
    figure = plt.figure(figsize=(out.shape[1] / 100, out.shape[0] / 100), dpi=100)
    figure.figimage(out)
    left = space
    for tile, r in zip(tiles, rows):
        lines = [r["method"]] if r["method"] == "as published" else [*r["method"].split(" ", 1), f"{float(r['seconds']):.2f} s"]
        for n, line in enumerate(lines):
            figure.text((left + tile.shape[1] / 2) / out.shape[1], (label - (0.5 + 0.8 * n) * GAP) / out.shape[0], line, ha="center", va="top", fontsize=13, color="#66686c" if line.endswith(" s") else "#1b1c1e")
        left += tile.shape[1] + space
    figure.savefig(OUT / "methods.png", dpi=100, facecolor="white")
    print("drew figures/methods.png")
    for r in rows:
        print(f"{r['method']:32s} {r['backfacingness']:.4f}   {r['seconds']:>7} s   {r['cpu_seconds']:>7} s of processor")
