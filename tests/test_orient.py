"""The shapes the method must get right. Each is polygons with some faces wound wrong; every face must come out facing out."""

import numpy as np
import pytest

import outie


def normal(face):
    return np.cross(face[1] - face[0], face[2] - face[0])


def quad(*corners):
    return np.array(corners, dtype=float)


def box():
    """A unit box with two faces wound inward, the floor and the south wall."""
    return [
        quad([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]),  # floor, wound to face up: wrong
        quad([0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]),  # roof
        quad([0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1])[::-1],  # south, wound to face north: wrong
        quad([1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]),  # east
        quad([1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]),  # north
        quad([0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]),  # west
    ]


def outward(faces, centre):
    return [np.dot(normal(f), np.asarray(f).mean(axis=0) - centre) > 0 for f in faces]


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
    centre = np.array([0.5, 0.5, 0.5])
    assert outward(box(), centre).count(False) == 2  # the fixture really has two faces wrong
    assert all(outward(outie.orient(box()), centre))


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


def test_facet_wise():
    """Every face judged alone still comes out right on a plain box."""
    centre = np.array([0.5, 0.5, 0.5])
    assert all(outward(outie.orient(box(), facet_wise=True), centre))


def test_use_parity():
    """The parity vote agrees with the escape vote on the courtyard."""
    out = outie.orient(courtyard(), use_parity=True)
    assert normal(out[0])[1] < 0
    assert normal(out[1])[1] > 0


def test_closed_by_volume():
    """A closed box is settled by its volume with no rays, and comes out the same as the vote."""
    centre = np.array([0.5, 0.5, 0.5])
    assert all(outward(outie.orient(box(), closed_by_volume=True), centre))
    assert all(outward(outie.orient(box(), closed_by_volume=True, rays_total=0, rays_minimum=0), centre))  # no rays at all
    out = outie.orient(courtyard(), closed_by_volume=True)  # open pieces still go to the vote
    assert normal(out[0])[1] < 0 and normal(out[1])[1] > 0


def keyhole_roof():
    """Four walls, no floor, and a roof with a courtyard cut into it as one ring that runs in through a slit: the
    roof wound wrong, and the courtyard walls wound wrong too."""
    outer = [(0, 0), (30, 0), (30, 30), (0, 30)]
    inner = [(10, 10), (10, 20), (20, 20), (20, 10)]  # the other way round, as a hole runs
    ring = [*outer, (0, 0), (10, 10), *inner[1:], (10, 10)]
    roof = np.array([(x, y, 9.0) for x, y in ring])[::-1]
    walls = [quad([0, 0, 0], [30, 0, 0], [30, 0, 9], [0, 0, 9]), quad([30, 0, 0], [30, 30, 0], [30, 30, 9], [30, 0, 9]), quad([30, 30, 0], [0, 30, 0], [0, 30, 9], [30, 30, 9]), quad([0, 30, 0], [0, 0, 0], [0, 0, 9], [0, 30, 9])]
    court = [quad([10, 10, 0], [20, 10, 0], [20, 10, 9], [10, 10, 9]), quad([20, 10, 0], [20, 20, 0], [20, 20, 9], [20, 10, 9]), quad([20, 20, 0], [10, 20, 0], [10, 20, 9], [20, 20, 9]), quad([10, 20, 0], [10, 10, 0], [10, 10, 9], [10, 20, 9])]
    return [roof, *walls, *[c[::-1] for c in court]]


def test_keyhole_roof():
    """A ring with a hole cut into it is one piece: its triangles agree, and it comes out facing up with its walls."""
    tris, owner = outie.triangulate([keyhole_roof()[0]])
    assert np.all(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2] < 0)  # all wound as the ring is
    assert abs(np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2 - 800) < 1e-6  # and cover its area
    out = outie.orient(keyhole_roof())
    assert normal(out[0])[2] > 0  # the roof faces up
    assert normal(out[5])[1] > 0  # the courtyard's south wall faces north, into the court


def test_dart_quad():
    """A quad with a reflex corner splits along the diagonal inside it, so both halves agree with the quad."""
    dart = quad([0, 0, 0], [10, 4, 0], [20, 0, 0], [10, 10, 0])
    tris, _ = outie.triangulate([dart])
    signs = np.sign(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2])
    assert len(tris) == 2 and signs[0] == signs[1] == 1
    assert abs(np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2 - 60) < 1e-9


def test_no_area():
    """A collapsed polygon gives no triangles and passes through orient untouched."""
    flat = quad([0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1])
    tris, owner = outie.triangulate([flat, *box()])
    assert 0 not in owner
    assert np.array_equal(outie.orient([flat, *box()])[0], flat)


def test_ground_occluder():
    """A step standing on the ground: alone, its underside is as open as its top; with the ground given, it faces up."""
    tread = quad([0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1])[::-1]  # wound to face down: wrong
    riser = quad([0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1])
    ground = quad([-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0])
    out = outie.orient([tread, riser], occluders=[ground])
    assert normal(out[0])[2] > 0  # the tread faces up
    assert normal(out[1])[1] < 0  # the riser faces out
