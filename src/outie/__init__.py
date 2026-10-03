"""Turns every face of a triangle mesh to point out, even when the mesh is not closed.

A port of libigl's reorient_facets_raycast at commit 7100764, the reference code for Takayama, Jacobson, Kavan and
Sorkine-Hornung, "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray
Casting", 2014. Faces that share edges are gathered into patches that agree. Then rays are shot from
random points on each patch, spread by area, off its front and its back; the side from which more rays
escape is the outside, and on a tie the side whose rays travel further before hitting anything.

One function, libigl's, with its name, arguments and outputs, and one addition: occluders, faces rays can hit that are
never turned, such as the ground a model stands on. Cutting polygons into triangles, and mending them, is the caller's.
"""

from collections import deque

import numpy as np
import trimesh

GRAZING = 0.1  # a ray closer than this to lying in its face is drawn again
EPSILON = 1e-4  # how far along itself a ray starts, so it does not hit its own face


def reorient_facets_raycast(vertices, faces, rays_total=None, rays_minimum=10, facet_wise=False, use_parity=False, seed=0, occluders=()):
    """libigl's function and outputs: per triangle, whether to turn it, and the patch it belongs to.

    One addition libigl does not have: occluders, triangles as corners, k by 3 by 3, that rays can hit but that are
    never turned, for the ground a model stands on or the things around it; without them a model with no floor is
    as open below as above.
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    if not len(F):
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
    rays_total = len(F) * 100 if rays_total is None else rays_total
    rng = np.random.default_rng(seed)
    if facet_wise:
        FF, patch = F.copy(), np.arange(len(F))
    else:
        FF, patch = bfs_orient(F)
    normals = np.cross(V[FF[:, 1]] - V[FF[:, 0]], V[FF[:, 2]] - V[FF[:, 0]])
    double_area = np.linalg.norm(normals, axis=1)
    patches = int(patch.max()) + 1
    area = np.bincount(patch, weights=double_area, minlength=patches)
    rays = np.maximum((rays_total * area / max(area.sum(), 1e-300)).astype(int), rays_minimum)
    rays[area == 0] = 0
    if not rays.any():
        return np.any(FF != F, axis=1), patch
    t = sample(patch, double_area, rays, rng)  # rays start on triangles chosen by area within their patch, at random points on them, Turk 1990
    s, u = rng.random(len(t)), rng.random(len(t))
    root = np.sqrt(u)
    weights = np.column_stack([1 - root, (1 - s) * root, s * root])
    origins = np.einsum("ri,rik->rk", weights, V[FF[t]])
    d = random_directions(normals[t] / double_area[t, None], rng)
    blocking = np.asarray(occluders, dtype=float).reshape(-1, 3, 3)
    scene_vertices = np.vstack([V, blocking.reshape(-1, 3)]) if len(blocking) else V
    scene_faces = np.vstack([FF, len(V) + np.arange(3 * len(blocking)).reshape(-1, 3)]) if len(blocking) else FF
    mesh = trimesh.Trimesh(vertices=scene_vertices, faces=scene_faces, process=False)
    by = patch[t]
    if use_parity:
        front = np.bincount(by, weights=parity(mesh, origins, d), minlength=patches)
        back = np.bincount(by, weights=parity(mesh, origins, -d), minlength=patches)
        voted = front > back
    else:
        front_escaped, front_distance = first_hits(mesh, origins, d)
        back_escaped, back_distance = first_hits(mesh, origins, -d)
        escaped = np.bincount(by, weights=front_escaped, minlength=patches), np.bincount(by, weights=back_escaped, minlength=patches)
        distance = np.bincount(by, weights=front_distance, minlength=patches), np.bincount(by, weights=back_distance, minlength=patches)
        voted = (escaped[0] < escaped[1]) | ((escaped[0] == escaped[1]) & (distance[0] < distance[1]))
    return voted[patch] ^ np.any(FF != F, axis=1), patch  # a face bfs_orient already turned is reported the other way round


def sample(patch, double_area, rays, rng):
    """The triangle each ray starts on: rays[p] of them in patch p, each triangle as likely as its area."""
    order = np.argsort(patch, kind="stable")
    grouped = patch[order]
    reach = np.cumsum(double_area[order])
    first = np.searchsorted(grouped, np.arange(len(rays)))
    last = np.searchsorted(grouped, np.arange(len(rays)), side="right") - 1
    below = np.where(first > 0, reach[first - 1], 0.0)
    p = np.repeat(np.arange(len(rays)), rays)
    target = below[p] + rng.random(len(p)) * (reach[last[p]] - below[p])
    picked = np.clip(np.searchsorted(reach, target, side="right"), first[p], last[p])
    return order[picked]


def first_hits(mesh, origins, directions):
    """Per ray: 1 when it escapes to infinity else 0, and how far it travels before its first hit else 0."""
    where, ray, _ = mesh.ray.intersects_location(origins + directions * EPSILON, directions, multiple_hits=False)  # the nearest hit per ray
    escaped = np.ones(len(origins))
    distance = np.zeros(len(origins))
    escaped[ray] = 0
    distance[ray] = np.linalg.norm(where - origins[ray], axis=1)
    return escaped, distance


def parity(mesh, origins, directions):
    """Per ray, the parity of how many faces it passes through."""
    _, ray, _ = mesh.ray.intersects_location(origins + directions * EPSILON, directions, multiple_hits=True)
    return np.bincount(ray, minlength=len(origins)) % 2


def random_directions(normals, rng):
    """A random direction off the front of each face, none grazing it."""
    d = rng.normal(size=normals.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    along = np.einsum("ij,ij->i", d, normals)
    grazing = np.abs(along) < GRAZING
    while grazing.any():
        again = rng.normal(size=(int(grazing.sum()), 3))
        d[grazing] = again / np.linalg.norm(again, axis=1, keepdims=True)
        along = np.einsum("ij,ij->i", d, normals)
        grazing = np.abs(along) < GRAZING
    d[along < 0] *= -1
    return d


def bfs_orient(faces):
    """libigl's bfs_orient: faces wound to agree with their neighbours across the edges exactly two share, and the patch each belongs to."""
    F = np.asarray(faces, dtype=int)
    edges = np.sort(np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]]), axis=1)
    _, inverse, count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    face_of = np.tile(np.arange(len(F)), 3)
    order = np.argsort(inverse, kind="stable")
    shared = count[inverse[order]] == 2  # the sorted edge list, kept where exactly two faces share the edge
    pairs = face_of[order][shared].reshape(-1, 2)  # those edges' two faces, side by side
    neighbours = [[] for _ in range(len(F))]
    for a, b in pairs.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    FF = F.copy()
    patch = np.full(len(F), -1)
    for start in range(len(F)):
        if patch[start] >= 0:
            continue
        patch[start] = start
        queue = deque([start])
        while queue:
            f = queue.popleft()
            for n in neighbours[f]:
                if patch[n] >= 0:
                    continue
                patch[n] = start
                if shares_directed_edge(FF[f], FF[n]):
                    FF[n] = FF[n][::-1]
                queue.append(n)
    _, patch = np.unique(patch, return_inverse=True)
    return FF, np.asarray(patch).reshape(-1)


def shares_directed_edge(f, n):
    """Whether two triangles run a shared edge the same way, which means one of them is wound against the other."""
    f0, f1, f2 = int(f[0]), int(f[1]), int(f[2])
    n0, n1, n2 = int(n[0]), int(n[1]), int(n[2])
    mine = {(f1, f2), (f2, f0), (f0, f1)}
    return (n1, n2) in mine or (n2, n0) in mine or (n0, n1) in mine
