"""The shapes the method must get right. Each is faces with some wound wrong, given as polygons; every face must come out facing out."""

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


def mesh(polygons):
    """Polygons as a mesh: their corners once each, and each polygon as a ring of corner numbers."""
    corners = np.vstack(polygons)
    vertices, inverse = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    ends = np.cumsum([len(p) for p in polygons])
    return vertices, [inverse[e - len(p):e] for p, e in zip(polygons, ends)]


def triangles(quads):
    """Quads as triangles given by corners, k by 3 by 3, as occluders are."""
    return np.vstack([[[q[0], q[1], q[2]], [q[0], q[2], q[3]]] for q in quads])


def reoriented(polygons, **settings):
    """Each polygon turned round where outie says so."""
    vertices, faces = mesh(polygons)
    flip, _ = outie.reorient_facets_raycast(vertices, faces, **settings)
    assert len(flip) == len(polygons)
    return [p[::-1] if f else p for p, f in zip(polygons, flip)]


def test_box():
    centre = np.array([0.5, 0.5, 0.5])
    assert outward(box(), centre).count(False) == 2  # the fixture really has two faces wrong
    assert all(outward(reoriented(box()), centre))


def test_outputs():
    vertices, faces = mesh(box())
    flip, patch = outie.reorient_facets_raycast(vertices, faces)
    assert flip.dtype == bool and flip.shape == (6,)  # one answer a face
    assert patch.max() == 0  # one closed box, one patch
    assert flip.sum() == 2  # the two wrong faces


def test_triangles_as_libigl():
    """Given triangles, faces are triangles: an m by 3 array works as libigl's does."""
    vertices, faces = mesh(box())
    triangles = np.vstack([[[r[0], r[1], r[2]], [r[0], r[2], r[3]]] for r in faces])
    flip, patch = outie.reorient_facets_raycast(vertices, triangles)
    assert flip.shape == (12,) and flip.sum() == 4 and patch.max() == 0


def test_any_number_of_corners():
    """A roof with a courtyard cut into it as one ring running in through a slit, its corners touching themselves, is
    one face: it comes out facing up with its walls, and the court's walls face into the court."""
    outer = [(0, 0), (30, 0), (30, 30), (0, 30)]
    inner = [(10, 10), (10, 20), (20, 20), (20, 10)]
    ring = [*outer, (0, 0), (10, 10), *inner[1:], (10, 10)]
    roof = np.array([(x, y, 9.0) for x, y in ring])[::-1]  # wound to face down: wrong
    walls = [quad([0, 0, 0], [30, 0, 0], [30, 0, 9], [0, 0, 9]), quad([30, 0, 0], [30, 30, 0], [30, 30, 9], [30, 0, 9]), quad([30, 30, 0], [0, 30, 0], [0, 30, 9], [30, 30, 9]), quad([0, 30, 0], [0, 0, 0], [0, 0, 9], [0, 30, 9])]
    court = [quad([10, 10, 0], [20, 10, 0], [20, 10, 9], [10, 10, 9]), quad([20, 10, 0], [20, 20, 0], [20, 20, 9], [20, 10, 9]), quad([20, 20, 0], [10, 20, 0], [10, 20, 9], [20, 20, 9]), quad([10, 20, 0], [10, 10, 0], [10, 10, 9], [10, 20, 9])]
    out = reoriented([roof, *walls, *[c[::-1] for c in court]])
    assert normal(out[0])[2] > 0  # the roof faces up
    assert normal(out[5])[1] > 0  # the courtyard's south wall faces north, into the court


def test_a_reflex_corner():
    """An L-shaped roof, wound wrong, over its walls: a ring with a reflex corner is cut inside itself and decided whole."""
    plan = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
    roof = np.array([(x, y, 6.0) for x, y in plan])[::-1]
    walls = [quad([*a, 0], [*b, 0], [*b, 6], [*a, 6]) for a, b in zip(plan, plan[1:] + plan[:1])]
    out = reoriented([roof, *walls])
    assert normal(out[0])[2] > 0


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
    out = reoriented(fins())
    assert normal(out[0])[1] < 0  # the wall faces south, out through the fins


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deterministic(seed):
    vertices, faces = mesh(courtyard())
    a = outie.reorient_facets_raycast(vertices, faces, seed=seed)
    b = outie.reorient_facets_raycast(vertices, faces, seed=seed)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_facet_wise():
    """Every face judged alone still comes out right on a plain box."""
    centre = np.array([0.5, 0.5, 0.5])
    assert all(outward(reoriented(box(), facet_wise=True), centre))


def test_use_parity():
    """The parity vote agrees with the escape vote on the courtyard."""
    out = reoriented(courtyard(), use_parity=True)
    assert normal(out[0])[1] < 0
    assert normal(out[1])[1] > 0


def test_ground_occluder():
    """A step standing on the ground: alone, its underside is as open as its top; with the ground given, it faces up."""
    tread = quad([0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1])[::-1]  # wound to face down: wrong
    riser = quad([0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1])
    ground = quad([-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0])
    out = reoriented([tread, riser], occluders=triangles([ground]))
    assert normal(out[0])[2] > 0  # the tread faces up
    assert normal(out[1])[1] < 0  # the riser faces out


def podium():
    """A podium with no floor and a tower standing on part of its roof, every face its own patch, the podium roof wound to face down."""
    def shell(x0, y0, z0, x1, y1, z1, nudge):
        c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
        faces = [c[[4, 5, 6, 7]], c[[0, 1, 5, 4]], c[[1, 2, 6, 5]], c[[2, 3, 7, 6]], c[[3, 0, 4, 7]]]
        return [f + nudge * (i + 1) for i, f in enumerate(faces)]  # moved apart a hair, so no face shares an edge
    base = shell(0, 0, 0, 40, 40, 10, 1e-4)
    base[0] = base[0][::-1]
    return base + shell(5, 5, 10, 25, 25, 80, -1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_podium_roof_needs_the_ground(seed):
    """The podium roof's upward rays hit the tower; without the ground its downward ones leave through the open bottom."""
    ground = quad([-120, -120, 0], [160, -120, 0], [160, 160, 0], [-120, 160, 0])
    out = reoriented(podium(), occluders=triangles([ground]), rays_total=400, seed=seed)
    assert normal(out[0])[2] > 0


@pytest.mark.parametrize("seed", range(5))
def test_ground_at_the_lowest_point(seed):
    """A floor that is its own patch faces down with the ground at its level: its downward rays start a hair below the
    ground and escape. Lowered even a centimetre, the ground catches them and the floor turns up."""
    walls = [quad([0, 0, 40], [10, 0, 40], [10, 10, 40], [0, 10, 40]), quad([0, 0, 0], [10, 0, 0], [10, 0, 40], [0, 0, 40]), quad([10, 0, 0], [10, 10, 0], [10, 10, 40], [10, 0, 40]), quad([10, 10, 0], [0, 10, 0], [0, 10, 40], [10, 10, 40]), quad([0, 10, 0], [0, 0, 0], [0, 0, 40], [0, 10, 40])]
    floor = quad([0.5, 0.5, 0], [0.5, 9.5, 0], [9.5, 9.5, 0], [9.5, 0.5, 0])  # faces down, apart from the walls
    level = quad([-30, -30, 0], [40, -30, 0], [40, 40, 0], [-30, 40, 0])
    assert normal(reoriented([*walls, floor], occluders=triangles([level]), rays_total=200, seed=seed)[-1])[2] < 0
    lowered = level - [0, 0, 0.01]
    assert normal(reoriented([*walls, floor], occluders=triangles([lowered]), rays_total=200, seed=seed)[-1])[2] > 0
