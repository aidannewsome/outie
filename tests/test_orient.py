"""The shapes the method must get right. Each is quads with some wound wrong, cut into two triangles each as a caller
would cut them; every quad must come out facing out."""

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


def mesh(quads):
    """Quads as a triangle mesh: their corners once each, each quad cut into two triangles wound as it is."""
    corners = np.vstack(quads)
    vertices, inverse = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    rings = np.asarray(inverse).reshape(-1, 4)
    return vertices, np.vstack([[[r[0], r[1], r[2]], [r[0], r[2], r[3]]] for r in rings])


def reoriented(quads, **settings):
    """Each quad turned round where outie flips its triangles, which must agree."""
    vertices, triangles = mesh(quads)
    flip, _ = outie.reorient_facets_raycast(vertices, triangles, **settings)
    assert len(flip) == 2 * len(quads) and (flip[0::2] == flip[1::2]).all()
    return [q[::-1] if f else q for q, f in zip(quads, flip[0::2])]


def test_box():
    centre = np.array([0.5, 0.5, 0.5])
    assert outward(box(), centre).count(False) == 2  # the fixture really has two faces wrong
    assert all(outward(reoriented(box()), centre))


def test_outputs():
    """libigl's outputs: per triangle, whether to flip it, and its component; with facet_wise=False the closed box is one
    component, and exactly the two wrong quads' triangles are flipped."""
    vertices, triangles = mesh(box())
    flip, component = outie.reorient_facets_raycast(vertices, triangles, facet_wise=False)
    assert flip.dtype == bool and flip.shape == (12,)
    assert component.max() == 0  # one closed box, one component
    assert flip.sum() == 4  # the two wrong quads' triangles


def test_open_box():
    out = reoriented(open_box())
    assert normal(out[0])[1] < 0  # south wall faces south
    assert normal(out[1])[1] > 0  # north wall faces north
    assert normal(out[2])[2] > 0  # roof faces up


def test_courtyard():
    out = reoriented(courtyard())
    assert normal(out[0])[1] < 0  # outer south wall faces south
    assert normal(out[1])[1] > 0  # court south wall faces north, into the court


def test_fins():
    """A wall behind thin fins, decided component by component: single-faced thin parts are where components help, as the paper's
    Figure 5 shows, since face by face their halves can come out opposite."""
    out = reoriented(fins(), facet_wise=False)
    assert normal(out[0])[1] < 0  # the wall faces south, out through the fins


def test_a_seed_repeats():
    vertices, faces = mesh(courtyard())
    a, b = (outie.reorient_facets_raycast(vertices, faces, seed=7) for _ in range(2))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_inside_a_closed_room():
    """A wall inside a closed box, near one end, its front facing the near end: no ray escapes either side, so it turns
    to face the side its rays travel further, the far end."""
    room = box()
    room[0], room[2] = room[0][::-1], room[2][::-1]  # the box's own faces all out
    wall = quad([0.8, 0.2, 0.2], [0.8, 0.8, 0.2], [0.8, 0.8, 0.8], [0.8, 0.2, 0.8])  # faces +x, the near end
    out = reoriented([*room, wall])
    assert normal(out[-1])[0] < 0


def test_a_bad_corner_number_is_refused():
    vertices, faces = mesh(box())
    with pytest.raises(IndexError):
        outie.reorient_facets_raycast(vertices, faces + len(vertices))


def test_an_empty_mesh():
    flip, component = outie.reorient_facets_raycast(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert len(flip) == len(component) == 0


def test_use_parity():
    """The parity vote agrees with the escape vote on the courtyard."""
    out = reoriented(courtyard(), use_parity=True)
    assert normal(out[0])[1] < 0
    assert normal(out[1])[1] > 0



def test_reorient_returns_the_faces_turned():
    """reorient applies the flips: every quad of the box comes back facing out, the input left as it was."""
    vertices, faces = mesh(box())
    before = faces.copy()
    turned = outie.reorient(vertices, faces)
    assert np.array_equal(faces, before)
    flip, _ = outie.reorient_facets_raycast(vertices, faces)
    assert np.array_equal(turned, np.where(flip[:, None], faces[:, ::-1], faces))
