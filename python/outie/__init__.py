"""Turns every face of a triangle mesh to point out, even when the mesh is not closed.

A port of libigl's reorient_facets_raycast at commit 7100764, the reference code for Takayama, Jacobson, Kavan and
Sorkine-Hornung, "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray Casting", 2014,
written in Rust. Triangles that share edges are gathered into components that agree. Rays are shot off the front and the
back of each component; the side from which more rays escape is the outside, and on a tie the side whose rays travel further
before hitting anything. A triangle is decided with its component.

Two functions of the paper's: reorient_facets_raycast, libigl's, with its name, arguments and outputs, and
backfacingness, the paper's measure of the result, with the six views it counts. Cutting polygons into triangles and
removing duplicate faces are the caller's, as they are in the paper's own command.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from outie import _core

__all__ = ["reorient_facets_raycast", "draw_backfacing_views", "backfacingness"]


def _make_mesh(vertices: ArrayLike, faces: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    return (
        np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3)),
        np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3)),
    )


def reorient_facets_raycast(
    vertices: ArrayLike,
    faces: ArrayLike,
    rays_total: int | None = None,
    rays_minimum: int = 10,
    facet_wise: bool = True,
    use_parity: bool = False,
    seed: int = 0,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """libigl's function and outputs: per triangle, whether to flip it, and the component it belongs to.

    vertices are n by 3 corner positions, faces m by 3 corner numbers, each a triangle. rays_total counts the rays for
    the whole mesh, 100 a triangle when not given.
    """
    return _core.reorient_facets_raycast(*_make_mesh(vertices, faces), rays_total, rays_minimum, facet_wise, use_parity, seed)


def reorient(vertices: ArrayLike, faces: ArrayLike, **options) -> NDArray[np.int64]:
    """The faces, each turned to point out: reorient_facets_raycast's flips applied, with its options."""
    faces = np.array(faces, dtype=np.int64).reshape(-1, 3)
    flip, _ = reorient_facets_raycast(vertices, faces, **options)
    faces[flip] = faces[flip][:, ::-1]
    return faces


def draw_backfacing_views(vertices: ArrayLike, faces: ArrayLike, resolution: int = 1024) -> NDArray[np.uint8]:
    """The six views the paper's backfacingness measure counts, 6 by resolution by resolution: 0 where nothing is
    drawn, 1 where a triangle shows its front, 2 where it shows its back. Orthographic, from the six sides of a box 5%
    larger than the mesh, as the paper's measure_backfacingness draws them."""
    return _core.draw_backfacing_views(*_make_mesh(vertices, faces), resolution)


def measure_backfacingness(vertices: ArrayLike, faces: ArrayLike, resolution: int = 1024) -> float:
    """The paper's measure of how plausible a mesh's facing is from outside: of the pixels drawn in its six views, the
    share showing a triangle's back. 0 when every face seen from outside shows its front."""
    views = draw_backfacing_views(vertices, faces, resolution)
    drawn = np.count_nonzero(views)
    return np.count_nonzero(views == 2) / drawn if drawn else 0.0
