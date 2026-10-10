"""The paper's backfacingness measure: a closed box seen from its six sides shows only fronts when wound out, only
backs when wound in, and one face's share when one face is wound in, as each view sees exactly one face."""

import numpy as np

import outie

CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
OUT = np.array([  # two triangles a side, each wound anticlockwise seen from outside
    [0, 2, 1], [0, 3, 2],  # floor
    [4, 5, 6], [4, 6, 7],  # roof
    [0, 1, 5], [0, 5, 4],  # south
    [1, 2, 6], [1, 6, 5],  # east
    [2, 3, 7], [2, 7, 6],  # north
    [3, 0, 4], [3, 4, 7],  # west
])


def test_a_box_wound_out_shows_no_backs_and_wound_in_only_backs():
    assert outie.measure_backfacingness(CORNERS, OUT, 64) == 0.0
    assert outie.measure_backfacingness(CORNERS, OUT[:, ::-1], 64) == 1.0


def test_one_face_wound_in_is_one_view_in_six():
    one_in = OUT.copy()
    one_in[:2] = one_in[:2, ::-1]
    assert abs(outie.measure_backfacingness(CORNERS, one_in, 64) - 1 / 6) < 1e-9


def test_the_views_are_six_squares_of_what_each_pixel_shows():
    views = outie.draw_backfacing_views(CORNERS, OUT, 32)
    assert views.shape == (6, 32, 32) and set(np.unique(views)) == {0, 1}
    assert (views[0] == 1).sum() == (views[3] == 1).sum() > 0  # a side and its opposite cover the same pixels
