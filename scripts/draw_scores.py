"""Every mass's backfacingness as published and after outie, from the results: one row a mass, a grey dot before and a
gold one after, sorted by where they started: figures/scores.png.

    uv run --group figures python scripts/draw_scores.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load import OUT, read_results

INK, QUIET, GOLD, GREY = "#1b1c1e", "#8a8a85", "#d49a1f", "#4a4a4a"

if __name__ == "__main__":
    scores = read_results()[0]
    names = sorted(scores["outie"], key=lambda n: scores["as published"][n])
    before = np.array([scores["as published"][n] for n in names])
    after = np.array([scores["outie"][n] for n in names])
    y = np.arange(len(names))
    figure, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.hlines(y, after, before, color="#dcdcd8", linewidth=1.2, zorder=1)
    ax.scatter(before, y, color=GREY, s=16, zorder=2, label="as published", clip_on=False)
    ax.scatter(after, y, color=GOLD, s=16, zorder=2, label="after outie", clip_on=False)
    ax.set_xlim(-0.02, 1)
    ax.set_yticks([])
    ax.set_xlabel("backfacingness, one row a mass", color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(QUIET)
    ax.tick_params(colors=QUIET)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2)
    figure.tight_layout()
    figure.savefig(OUT / "scores.png", facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print("drew figures/scores.png")
