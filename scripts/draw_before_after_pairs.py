"""The three masses outie improves most, by how far their backfacingness falls in the results, each as a pair, as
published on the left and after on the right, in a row, every pair cropped to its mass and the rows of one
height, as the paper's figure of results: figures/before-after-pairs.png.

    uv run --group figures python scripts/draw_before_after_pairs.py
"""

import numpy as np

from draw import draw
from load import GAP, HEIGHT, load, orient, read_results, save, scale_to_height


def find_most_improved(count: int) -> list[str]:
    """The masses whose backfacingness outie lowers most."""
    scores = read_results()[0]
    return sorted(scores["outie"], key=lambda name: scores["outie"][name] - scores["as published"][name])[:count]


def draw_pair(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A mass as published and after, cut to one box so both stand the same, scaled to a row's height."""
    vertices, triangles = load(name)
    before, after = draw(vertices, triangles, size=700), draw(vertices, orient(vertices, triangles)[0], size=700)
    drawn = np.argwhere(((before < 0.995) | (after < 0.995)).any(axis=2))
    (top, left), (bottom, right) = drawn.min(axis=0), drawn.max(axis=0) + 1
    return scale_to_height(before[top:bottom, left:right], HEIGHT), scale_to_height(after[top:bottom, left:right], HEIGHT)


def arrange_rows(pairs: list[np.ndarray], per_row: int = 3) -> np.ndarray:
    """Pairs in rows, each row centred, every gap the same."""
    lines = [np.hstack([p for q in pairs[i : i + per_row] for p in (q, np.ones((HEIGHT, GAP, 3)))][:-1]) for i in range(0, len(pairs), per_row)]
    width = max(line.shape[1] for line in lines) + 2 * GAP
    out = np.ones((GAP + len(lines) * (HEIGHT + GAP), width, 3))
    for n, line in enumerate(lines):
        left = (width - line.shape[1]) // 2
        out[GAP + n * (HEIGHT + GAP) : GAP + n * (HEIGHT + GAP) + HEIGHT, left : left + line.shape[1]] = line
    return out


if __name__ == "__main__":
    pairs = [np.hstack([before, np.ones((HEIGHT, GAP // 3, 3)), after]) for before, after in map(draw_pair, find_most_improved(3))]
    save(arrange_rows(pairs), "before-after-pairs")
