"""Turns every face of a triangle mesh to point out, even when the mesh is not closed.

A port of libigl's reorient_facets_raycast at commit 7100764, the reference code for Takayama, Jacobson, Kavan and
Sorkine-Hornung, "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray Casting", 2014,
written in Rust. Faces that share edges are gathered into patches that agree. Then rays are shot from random points
on each patch, spread by area, off its front and its back; the side from which more rays escape is the outside, and
on a tie the side whose rays travel further before hitting anything.

One function, libigl's, with its name, arguments and outputs, and one addition: occluders, faces rays can hit that are
never turned, such as the ground a model stands on. Cutting polygons into triangles, and mending them, is the caller's.
"""

import numpy as np

from outie import _core


def reorient_facets_raycast(vertices, faces, rays_total=None, rays_minimum=10, facet_wise=False, use_parity=False, seed=0, occluders=()):
    """libigl's function and outputs: per triangle, whether to turn it, and the patch it belongs to.

    vertices are n by 3, faces m by 3 triangles by corner; occluders are triangles as corners, k by 3 by 3, that rays
    can hit but that are never turned.
    """
    blocking = np.ascontiguousarray(np.asarray(occluders, dtype=np.float64).reshape(-1, 3, 3))
    return _core.reorient_facets_raycast(
        np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3)),
        np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3)),
        rays_total, rays_minimum, facet_wise, use_parity, seed, blocking if len(blocking) else None,
    )
