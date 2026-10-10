"""The three masses whose backfacingness stays highest after outie in the results, as the paper shows what it could not
put right: figures/remaining.png.

    uv run --group figures python scripts/draw_remaining.py
"""

import numpy as np

from draw import draw
from load import GAP, HEIGHT, crop, load, orient, read_results, save, scale_to_height

if __name__ == "__main__":
    after = read_results()[0]["outie"]
    tiles = []
    for name in sorted(after, key=lambda n: -after[n])[:3]:
        vertices, triangles = load(name)
        tiles.append(scale_to_height(crop(draw(vertices, orient(vertices, triangles)[0], size=500)), HEIGHT))
    row = np.hstack([t for tile in tiles for t in (tile, np.ones((HEIGHT, GAP, 3)))][:-1])
    out = np.ones((HEIGHT + 2 * GAP, row.shape[1] + 2 * GAP, 3))
    out[GAP : GAP + HEIGHT, GAP : GAP + row.shape[1]] = row
    save(out, "remaining")
