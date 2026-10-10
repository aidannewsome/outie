"""Six of Toronto's best-known buildings from the city's 3D Massing, each as published, its faces' backs grey, and as
outie turns them, two to a row with their names and backfacingness before and after: figures/landmarks.png, the figure at the top of the README.

    uv run --group figures python scripts/draw_landmarks.py
"""

import numpy as np

from draw import aim_camera, draw, read_off
from load import DATA, GAP, orient, remove_duplicates, save, scale_to_height, score

LANDMARKS = {  # each file in data/landmarks/, its name, and the way it is seen from, degrees round from east
    "ocad-university": ("OCAD University", -55),
    "royal-ontario-museum": ("Royal Ontario Museum", -55),
    "art-gallery-of-ontario": ("Art Gallery of Ontario", 50),
    "roy-thomson-hall": ("Roy Thomson Hall", -55),
    "city-hall": ("City Hall", -55),
    "rogers-centre": ("Rogers Centre", -55),
}
TILE = 420  # pixels a drawing stands


def draw_pair(file: str, azimuth: float) -> tuple[np.ndarray, float, float]:
    """A building as published and after, cut to one box so both stand the same, side by side, with each's score."""
    vertices, triangles = read_off(DATA / "landmarks" / f"{file}.off")
    triangles = remove_duplicates(triangles)
    view = aim_camera(azimuth=azimuth)
    turned = orient(vertices, triangles)[0]
    before, after = draw(vertices, triangles, view, size=900), draw(vertices, turned, view, size=900)
    drawn = np.argwhere(((before < 0.995) | (after < 0.995)).any(axis=2))
    (top, left), (bottom, right) = drawn.min(axis=0), drawn.max(axis=0) + 1
    side = max(bottom - top, right - left)  # a square round the building, so every pair is one size
    top, left = max(0, (top + bottom - side) // 2), max(0, (left + right - side) // 2)
    pad = lambda image: np.pad(image[top : top + side, left : left + side], ((0, side - min(side, 900 - top)), (0, side - min(side, 900 - left)), (0, 0)), constant_values=1)  # noqa: E731
    pair = np.hstack([scale_to_height(pad(before), TILE), np.ones((TILE, GAP // 3, 3)), scale_to_height(pad(after), TILE)])
    return pair, score(vertices, triangles), score(vertices, turned)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [(name, *draw_pair(file, azimuth)) for file, (name, azimuth) in LANDMARKS.items()]
    width, head, row_height = pairs[0][1].shape[1], 50, TILE + 70
    size = (2 * width + 3 * GAP, head + 3 * row_height + GAP)
    figure = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    for column in range(2):
        for side, words in enumerate(("As published", "After outie")):
            x = GAP + column * (width + GAP) + side * (width + GAP // 3) / 2
            figure.text(x / size[0], 1 - (GAP + 10) / size[1], words, fontsize=15, family="Helvetica", color="#8a8a85", va="baseline")
    for n, (name, image, before, after) in enumerate(pairs):
        row, column = divmod(n, 2)
        left, top = GAP + column * (width + GAP), head + GAP // 2 + row * row_height
        axes = figure.add_axes((left / size[0], 1 - (top + TILE) / size[1], width / size[0], TILE / size[1]))
        axes.imshow(image)
        axes.set_axis_off()
        axes.text(0, TILE + 34, name, fontsize=17, family="Helvetica", va="baseline", color="#1b1c1e")
        axes.text(width, TILE + 34, f"backfacingness {before:.2f} to {after:.2f}", fontsize=15, family="Helvetica", va="baseline", ha="right", color="#8a8a85")
    figure.canvas.draw()
    save(np.asarray(figure.canvas.buffer_rgba())[..., :3] / 255, "landmarks")
