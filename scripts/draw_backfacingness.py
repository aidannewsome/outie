"""The largest well mixed mass in the results, as published, beside the six views its backfacingness counts, fronts red
and backs blue, as the paper draws its motorcycle: figures/backfacingness.png.

    uv run --group figures python scripts/draw_backfacingness.py
"""

import numpy as np

import outie
from draw import draw
from load import find_largest_mixed, load, save

RED, BLUE = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])  # the paper's two colours
ORDER = [0, 1, 2, 3, 4, 5]  # two sides then the view from above along the top, their opposites below, the view from beneath last


def compose(vertices: np.ndarray, triangles: np.ndarray, tile: int = 440, gap: int = 40) -> np.ndarray:
    """The mesh in a square two rows high on the left, its six views in two rows of three on the right, the views from above and beneath at the end, every gap and
    the margin the same, on white."""
    model = 2 * tile + gap
    image = np.ones((gap + model + gap, gap + model + 3 * (gap + tile) + gap, 3))
    image[gap : gap + model, gap : gap + model] = draw(vertices, triangles, size=model)
    views = outie.draw_backfacing_views(vertices, triangles, tile)
    for place, v in enumerate(ORDER):
        square = np.zeros((tile, tile, 3))
        square[views[v] == 1], square[views[v] == 2] = RED, BLUE
        row, col = divmod(place, 3)
        top, left = gap + row * (tile + gap), gap + model + gap + col * (tile + gap)
        image[top : top + tile, left : left + tile] = np.rot90(square[::-1], count_quarter_turns(v))
    return image


def count_quarter_turns(v: int) -> int:
    """Quarter turns anticlockwise that bring z to the top of view v: the paper's cameras take y as up, these masses z."""
    axis = np.vstack([np.eye(3), -np.eye(3)])[v]
    up = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])[v]
    if abs(axis[2]) > 0.5 or up[2] > 0.5:
        return 0  # looking along z, or z already up
    across = round(np.cross(-axis, up)[2])
    return {-1: 3, 1: 1}.get(across, 2)


if __name__ == "__main__":
    save(compose(*load(find_largest_mixed())), "backfacingness")
