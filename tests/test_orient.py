"""The shapes the method must get right. Each is polygons with some faces wound wrong; every face must come out facing out."""

import numpy as np
import pytest

import outie


def normal(face):
    return np.cross(face[1] - face[0], face[2] - face[0])


def quad(*corners):
    return np.array(corners, dtype=float)


def box():
    """A unit box with two faces wound inward."""
    return [
        quad([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])[::-1],  # floor, wrong
        quad([0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]),  # roof
        quad([0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]),  # south, wrong
        quad([1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]),  # east
        quad([1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]),  # north
        quad([0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]),  # west
    ]


def open_box():
    """Two opposite walls and a roof, no floor, the north wall wound wrong."""
    return [
        quad([0, 0, 0], [10, 0, 0], [10, 0, 5], [0, 0, 5]),
        quad([0, 10, 0], [0, 10, 5], [10, 10, 5], [10, 10, 0]),
        quad([0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]),
    ]


def courtyard():
    """A ring building around a courtyard: the courtyard's south wall must face north, into the court."""
    return [
        quad([0, 0, 0], [30, 0, 0], [30, 0, 9], [0, 0, 9]),  # outer south wall, faces south
        quad([10, 10, 0], [20, 10, 0], [20, 10, 9], [10, 10, 9]),  # court south wall, wound the same way, wrong
        quad([0, 0, 9], [30, 0, 9], [30, 10, 9], [0, 10, 9]),
        quad([0, 20, 9], [30, 20, 9], [30, 30, 9], [0, 30, 9]),
        quad([0, 10, 9], [10, 10, 9], [10, 20, 9], [0, 20, 9]),
        quad([20, 10, 9], [30, 10, 9], [30, 20, 9], [20, 20, 9]),
        quad([0, 30, 0], [30, 30, 0], [30, 30, 9], [0, 30, 9])[::-1],
    ]


def fins():
    """A wall behind fins under an overhanging roof, the wall wound wrong: it must still face out through the gaps."""
    wall = quad([0, 0, 0], [10, 0, 0], [10, 0, 20], [0, 0, 20])[::-1]
    fins = [quad([x, -0.5, 0], [x, -0.5, 20], [x, 0, 20], [x, 0, 0]) for x in (2, 4, 6, 8)]
    overhang = quad([0, -1, 20], [10, -1, 20], [10, 10, 20], [0, 10, 20])
    back = quad([0, 10, 0], [0, 10, 20], [10, 10, 20], [10, 10, 0])
    return [wall, *fins, overhang, back]


def test_box():
    out = outie.orient(box())
    centre = np.array([0.5, 0.5, 0.5])
    assert all(np.dot(normal(f), f.mean(axis=0) - centre) > 0 for f in out)


def test_open_box():
    out = outie.orient(open_box())
    assert normal(out[0])[1] < 0  # south wall faces south
    assert normal(out[1])[1] > 0  # north wall faces north
    assert normal(out[2])[2] > 0  # roof faces up


def test_courtyard():
    out = outie.orient(courtyard())
    assert normal(out[0])[1] < 0  # outer south wall faces south
    assert normal(out[1])[1] > 0  # court south wall faces north, into the court


def test_fins():
    out = outie.orient(fins())
    assert normal(out[0])[1] < 0  # the wall faces south, out through the fins


def test_triangles_match_polygons():
    faces = box()
    corners = np.vstack(faces)
    vertices, inverse = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    rings = np.asarray(inverse).reshape(-1).reshape(len(faces), 4)
    triangles = np.vstack([[[r[0], r[1], r[2]], [r[0], r[2], r[3]]] for r in rings])
    flip, patch = outie.reorient_facets_raycast(vertices, triangles)
    assert flip.dtype == bool and flip.shape == (12,)
    assert patch.max() == 0  # one closed box, one patch
    assert flip.sum() == 4  # the two wrong quads, two triangles each
    fixed = outie.orient_mesh(vertices, triangles)
    centre = np.array([0.5, 0.5, 0.5])
    for tri in vertices[fixed]:
        assert np.dot(normal(tri), tri.mean(axis=0) - centre) > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deterministic(seed):
    a = outie.orient(courtyard(), seed=seed)
    b = outie.orient(courtyard(), seed=seed)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
