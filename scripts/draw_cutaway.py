"""The mass with the most triangles in the results, the likeliest to keep what a modeller left inside, as the
paper opens its bus and train: as published and after outie, whole above and below cut open by a clipping
plane through the middle, framed on what is left: figures/cutaway.png.

    uv run --group figures python scripts/draw_cutaway.py
"""

from draw import draw, make_clipping_plane
from load import arrange_grid, load, orient, read_results, save

if __name__ == "__main__":
    triangles = read_results()[1]
    whole, opened = [], []
    for name in sorted(triangles, key=lambda n: -triangles[n])[:1]:
        v, f = load(name)
        for t in (f, orient(v, f)[0]):
            whole.append(draw(v, t, size=500))
            opened.append(draw(v, t, size=500, plane=make_clipping_plane(v)))
    save(arrange_grid([whole, opened]), "cutaway")
