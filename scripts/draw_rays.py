"""How many rays are enough: outie on every mass at a range of rays a triangle, backfacingness against the time it took,
results/rays.csv and figures/rays.png. Each face's own minimum, rays_minimum, is lowered to one, so the number of rays
is what each setting says; at its default of ten, every face would get at least ten whatever the setting.

    uv run --group figures python scripts/draw_rays.py
"""

import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load import DATA, OUT, RESULTS, load, orient, score

RAYS = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # a triangle; 100 is the paper's and outie's default
INK, QUIET, GOLD = "#1b1c1e", "#8a8a85", "#d49a1f"

if __name__ == "__main__":
    masses = [load(path.stem) for path in sorted(DATA.glob("*.off"))]
    for v, f in masses:  # once untimed, so the first setting does not pay for starting Embree and loading the masses
        orient(v, f, rays_total=len(f))
    rows = []
    for per in RAYS:
        timings = []
        for _ in range(3):  # the fastest of three, as other work on the machine only slows a run
            clock, oriented = time.perf_counter(), []
            for v, f in masses:
                oriented.append((v, orient(v, f, rays_total=per * len(f), rays_minimum=1)[0]))
            timings.append(time.perf_counter() - clock)
        seconds = min(timings)
        scores = [score(v, f) for v, f in oriented]
        rows.append({"rays_a_triangle": per, "backfacingness": f"{np.mean(scores):.4f}", "seconds": f"{seconds:.2f}"})
        print(rows[-1])
    with open(RESULTS / "rays.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, ["rays_a_triangle", "backfacingness", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    seconds, scores = [float(r["seconds"]) for r in rows], [float(r["backfacingness"]) for r in rows]
    figure, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(seconds, scores, color=GOLD, linewidth=2, marker="o", markersize=7)
    for r, x, y in zip(rows, seconds, scores):
        ax.annotate(f"{r['rays_a_triangle']} ray{'s' if r['rays_a_triangle'] > 1 else ''}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("seconds for all 60 masses, log scale", color=INK)
    ax.set_ylabel("backfacingness, mean", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(QUIET)
    ax.tick_params(colors=QUIET)
    ax.grid(color="#ececea", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(OUT / "rays.png", facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print("drew figures/rays.png")
