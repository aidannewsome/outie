"""Turns every face of a polygon mesh to point out, even when the mesh is not closed.

A port of libigl's reorient_facets_raycast at commit 7100764, the reference code for Takayama, Jacobson, Kavan and
Sorkine-Hornung, "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray Casting", 2014,
written in Rust. The paper's faces are polygons, and so are outie's: a face is a ring of corners, any number of them,
and given triangles it is libigl's function exactly. Faces that share edges are gathered into patches that agree. Each
face is cut into triangles only to cast rays from, off the front and the back of each patch; the side from which more
rays escape is the outside, and on a tie the side whose rays travel further before hitting anything. A face is decided
with its patch, whole.

One function, libigl's, with its name, arguments and outputs, and one addition: occluders, faces rays can hit that are
never turned, such as the ground a model stands on. Mending a face that crosses itself is the caller's.
"""

import numpy as np

from outie import _core


def reorient_facets_raycast(vertices, faces, rays_total=None, rays_minimum=10, facet_wise=False, use_parity=False, seed=0, occluders=()):
    """libigl's function and outputs, for faces of any number of corners: per face, whether to turn it, and the patch it
    belongs to.

    vertices are n by 3; faces are rings of corner numbers, an m by k array or a list of any lengths; occluders are
    triangles as corners, k by 3 by 3, that rays can hit but that are never turned. rays_total counts the rays for the
    whole mesh, 100 a triangle of it when not given.
    """
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        counts = np.full(len(faces), faces.shape[1], dtype=np.int64)
        corners = faces.astype(np.int64).ravel()
    else:
        rings = [np.asarray(f, dtype=np.int64).ravel() for f in faces]
        counts = np.array([len(r) for r in rings], dtype=np.int64)
        corners = np.concatenate(rings) if rings else np.zeros(0, dtype=np.int64)
    blocking = np.ascontiguousarray(np.asarray(occluders, dtype=np.float64).reshape(-1, 3, 3))
    return _core.reorient_facets_raycast(
        np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3)),
        np.ascontiguousarray(corners), counts,
        rays_total, rays_minimum, facet_wise, use_parity, seed, blocking if len(blocking) else None,
    )
