"""Stage 2: the transliteration made to run, behind the API the package will keep."""

import random
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).parents[2] / "port"))
import reorient_facets_raycast as port  # noqa: E402


def reorient_facets_raycast(vertices, faces, rays_total=None, rays_minimum=10, facet_wise=False, use_parity=False, seed=0):
    """Per triangle, whether to flip it, and the patch it belongs to."""
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    random.seed(seed)
    np.random.seed(seed)
    I = np.zeros(0, dtype=int)
    C = np.zeros(0, dtype=int)
    port.reorient_facets_raycast(V, F, rays_total if rays_total is not None else F.shape[0] * 100, rays_minimum, facet_wise, use_parity, False, I, C, seeded=True)
    return I.astype(bool), C


def reoriented(vertices, faces, **kw):
    """The same triangles, turned as needed."""
    F = np.asarray(faces, dtype=int)
    flip, _ = reorient_facets_raycast(vertices, F, **kw)
    out = F.copy()
    out[flip] = out[flip][:, ::-1]
    return out


def orient(polygons, **kw):
    """Polygons, each an n by 3 ring, turned as needed. Each is triangulated, its triangles wound as it is, and flipped with it."""
    triangles, owner = triangulate(polygons)
    if not len(triangles):
        return list(polygons)
    corners = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    F = np.asarray(inverse).reshape(-1, 3)
    flip, _ = reorient_facets_raycast(vertices, F, **kw)
    turned = np.zeros(len(polygons), dtype=bool)
    for f, o in zip(flip, owner):
        turned[o] = turned[o] or f
    return [p[::-1] if t else p for p, t in zip(polygons, turned)]


def triangulate(polygons):
    """Every polygon as triangles wound as it is, and the polygon each came from."""
    out, owner = [], []
    for n, f in enumerate(polygons):
        f = np.asarray(f, dtype=float)
        if len(f) < 3:
            continue
        if len(f) <= 4:
            out += [np.stack([f[0], f[i], f[i + 1]]) for i in range(1, len(f) - 1)]
            owner += [n] * (len(f) - 2)
            continue
        normal = np.cross(f, np.roll(f, -1, axis=0)).sum(axis=0)
        drop = int(np.argmax(np.abs(normal)))
        keep = [i for i in range(3) if i != drop]
        flat = shapely.Polygon(f[:, keep])
        if not flat.is_valid or flat.area == 0:
            out += [np.stack([f[0], f[i], f[i + 1]]) for i in range(1, len(f) - 1)]
            owner += [n] * (len(f) - 2)
            continue
        lookup = {tuple(np.round(v[keep], 6)): v for v in f}
        for t in shapely.get_parts(shapely.constrained_delaunay_triangles(flat)):
            corners = [lookup.get(tuple(np.round(c, 6))) for c in np.asarray(t.exterior.coords)[:3]]
            if any(c is None for c in corners):
                continue
            tri = np.stack(corners)
            if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), normal) < 0:
                tri = tri[::-1]
            out.append(tri)
            owner.append(n)
    return (np.array(out), np.array(owner)) if out else (np.zeros((0, 3, 3)), np.zeros(0, int))
