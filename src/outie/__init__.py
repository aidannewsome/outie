"""Turns every face of a mesh to point out, even when the mesh is not closed.

A port of libigl's reorient_facets_raycast, the reference code for Takayama, Jacobson, Kavan and
Sorkine-Hornung, "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray
Casting", 2014. Faces that share edges are gathered into patches that agree. Then rays are shot from
random points on each patch, spread by area, off its front and its back; the side from which more rays
escape is the outside, and on a tie the side whose rays travel further before hitting anything.
"""

from collections import deque

import numpy as np
import shapely
import trimesh

GRAZING = 0.1  # a ray closer than this to lying in its face is drawn again
EPSILON = 1e-4  # how far along itself a ray starts, so it does not hit its own face


def orient(polygons, **settings):
    """Polygons, each an n by 3 ring, with those that pointed in turned round."""
    triangles, owner = triangulate(polygons)
    if not len(triangles):
        return [np.asarray(p) for p in polygons]
    corners = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(-1, 3)
    flip = flips(vertices, faces, **settings)
    turned = np.zeros(len(polygons), dtype=bool)
    np.logical_or.at(turned, owner, flip)
    return [np.asarray(p)[::-1] if t else np.asarray(p) for p, t in zip(polygons, turned)]


def orient_mesh(vertices, faces, **settings):
    """Triangles, m by 3 into vertices n by 3, with those that pointed in turned round."""
    faces = np.asarray(faces, dtype=int)
    flip = flips(vertices, faces, **settings)
    out = faces.copy()
    out[flip] = out[flip][:, ::-1]
    return out


def flips(vertices, faces, rays_total=None, rays_minimum=10, facet_wise=False, use_parity=False, seed=0):
    """Per triangle, whether it should be turned, by the paper's vote with libigl's defaults."""
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    if not len(F):
        return np.zeros(0, dtype=bool)
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
    chosen = []  # rays start on triangles chosen by area within their patch, at random points on them, Turk 1990
    for p in np.flatnonzero(rays):
        mine = np.flatnonzero(patch == p)
        chosen.append(rng.choice(mine, size=rays[p], p=double_area[mine] / double_area[mine].sum()))
    if not chosen:
        return np.zeros(len(F), dtype=bool)
    t = np.concatenate(chosen)
    s, u = rng.random(len(t)), rng.random(len(t))
    root = np.sqrt(u)
    weights = np.column_stack([1 - root, (1 - s) * root, s * root])
    origins = np.einsum("ri,rik->rk", weights, V[FF[t]])
    d = random_directions(normals[t] / double_area[t, None], rng)
    mesh = trimesh.Trimesh(vertices=V, faces=FF, process=False)
    by = patch[t]
    if use_parity:
        front = np.bincount(by, weights=parity(mesh, origins, d), minlength=patches)
        back = np.bincount(by, weights=parity(mesh, origins, -d), minlength=patches)
        vote = front > back
    else:
        front_escaped, front_distance = first_hits(mesh, origins, d)
        back_escaped, back_distance = first_hits(mesh, origins, -d)
        escaped = np.bincount(by, weights=front_escaped, minlength=patches), np.bincount(by, weights=back_escaped, minlength=patches)
        distance = np.bincount(by, weights=front_distance, minlength=patches), np.bincount(by, weights=back_distance, minlength=patches)
        vote = (escaped[0] < escaped[1]) | ((escaped[0] == escaped[1]) & (distance[0] < distance[1]))
    return vote[patch] ^ np.any(FF != F, axis=1)  # a face bfs_orient already turned is reported the other way round


def first_hits(mesh, origins, directions):
    """Per ray: 1 when it escapes to infinity else 0, and how far it travels before its first hit else 0."""
    where, ray, _ = mesh.ray.intersects_location(origins + directions * EPSILON, directions, multiple_hits=False)
    escaped = np.ones(len(origins))
    distance = np.zeros(len(origins))
    if len(ray):
        far = np.linalg.norm(where - origins[ray], axis=1)
        order = np.lexsort((far, ray))  # nearest hit per ray first
        ray, far = ray[order], far[order]
        first = np.concatenate([[True], ray[1:] != ray[:-1]])
        escaped[ray[first]] = 0
        distance[ray[first]] = far[first]
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


def bfs_orient(F):
    """Faces wound to agree with their neighbours across the edges exactly two share, and the patch each belongs to."""
    edges = np.sort(np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]]), axis=1)
    _, inverse, count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    face_of = np.tile(np.arange(len(F)), 3)
    neighbours = [[] for _ in range(len(F))]
    order = np.argsort(inverse, kind="stable")
    for e in np.flatnonzero(count == 2):  # an edge with more than two faces is a seam and joins nothing
        a, b = face_of[order[np.searchsorted(inverse[order], e) : np.searchsorted(inverse[order], e, side="right")]]
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
    mine = {(f[1], f[2]), (f[2], f[0]), (f[0], f[1])}
    return any(pair in mine for pair in ((n[1], n[2]), (n[2], n[0]), (n[0], n[1])))


def triangulate(polygons):
    """Every polygon as triangles wound as it is, and the polygon each came from."""
    out, owner = [], []
    for k, f in enumerate(polygons):
        f = np.asarray(f, dtype=float)
        if len(f) < 3:
            continue
        if len(f) <= 4:
            out += [np.stack([f[0], f[i], f[i + 1]]) for i in range(1, len(f) - 1)]
            owner += [k] * (len(f) - 2)
            continue
        normal = np.cross(f, np.roll(f, -1, axis=0)).sum(axis=0)
        drop = int(np.argmax(np.abs(normal)))
        keep = [i for i in range(3) if i != drop]
        flat = shapely.Polygon(f[:, keep])
        if not flat.is_valid or flat.area == 0:
            out += [np.stack([f[0], f[i], f[i + 1]]) for i in range(1, len(f) - 1)]
            owner += [k] * (len(f) - 2)
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
            owner.append(k)
    return (np.array(out), np.array(owner)) if out else (np.zeros((0, 3, 3)), np.zeros(0, int))
