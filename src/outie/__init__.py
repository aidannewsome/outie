"""Turns every face of a mesh to point out, even when the mesh is not closed.

A port of libigl's reorient_facets_raycast at commit 7100764, the reference code for Takayama, Jacobson, Kavan and
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


def orient(polygons, occluders=(), **settings):
    """Polygons, each an n by 3 ring, with those that pointed in turned round.

    occluders are polygons that block rays but are not turned: the ground a model stands on, or its neighbours.
    """
    triangles, owner = triangulate(polygons)
    if not len(triangles):
        return [np.asarray(p) for p in polygons]
    corners = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(-1, 3)
    blocking, _ = triangulate(occluders)
    flip, _ = reorient_facets_raycast(vertices, faces, occluders=blocking, **settings)
    turned = np.zeros(len(polygons), dtype=bool)
    np.logical_or.at(turned, owner, flip)
    return [np.asarray(p)[::-1] if t else np.asarray(p) for p, t in zip(polygons, turned)]


def orient_mesh(vertices, faces, **settings):
    """Triangles, m by 3 into vertices n by 3, with those that pointed in turned round. occluders, if given, are triangles as corners, k by 3 by 3."""
    faces = np.asarray(faces, dtype=int)
    flip, _ = reorient_facets_raycast(vertices, faces, **settings)
    out = faces.copy()
    out[flip] = out[flip][:, ::-1]
    return out


def reorient_facets_raycast(vertices, faces, rays_total=None, rays_minimum=10, facet_wise=False, use_parity=False, seed=0, closed_by_volume=False, occluders=()):
    """libigl's function and outputs: per triangle, whether to turn it, and the patch it belongs to.

    Two additions libigl does not have, both off unless asked for. closed_by_volume: a patch whose every edge
    exactly two faces share is a closed surface, and its signed volume says which way it faces exactly, with no
    rays. occluders: triangles as corners, k by 3 by 3, that rays can hit but that are never turned, for the
    ground a model stands on or the things around it; without them a model with no floor is as open below as above.
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
    settled = np.zeros(patches, dtype=bool)
    vote = np.zeros(patches, dtype=bool)
    if closed_by_volume and not facet_wise:
        volume = signed_volumes(V, FF, patch, patches)
        settled = closed(FF, patch, patches) & (np.abs(volume) > 1e-9 * np.maximum(area, 1e-300))
        vote[settled] = volume[settled] < 0
        rays[settled] = 0
    if not rays.any():
        return vote[patch] ^ np.any(FF != F, axis=1), patch
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
    vote[~settled] = voted[~settled]
    return vote[patch] ^ np.any(FF != F, axis=1), patch  # a face bfs_orient already turned is reported the other way round


def closed(FF, patch, patches):
    """Per patch, whether every edge is shared by exactly two of its faces: a closed surface."""
    edges = np.sort(np.concatenate([FF[:, [1, 2]], FF[:, [2, 0]], FF[:, [0, 1]]]), axis=1)
    _, inverse, count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    open_edge = count[np.asarray(inverse).reshape(-1)] != 2
    broken = np.zeros(patches, dtype=bool)
    broken[np.tile(patch, 3)[open_edge]] = True
    return ~broken


def signed_volumes(V, FF, patch, patches):
    """Per patch, six times its signed volume about its own centre, positive when its faces point out."""
    centre = np.zeros((patches, 3))
    np.add.at(centre, patch, V[FF].mean(axis=1))
    centre /= np.maximum(np.bincount(patch, minlength=patches), 1)[:, None]
    a, b, c = (V[FF[:, i]] - centre[patch] for i in range(3))
    return np.bincount(patch, weights=np.einsum("ij,ij->i", a, np.cross(b, c)), minlength=patches)


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


def triangulate(polygons):
    """Every polygon as triangles wound as it is, and the polygon each came from.

    Every triangle agrees with its polygon's own normal, so a polygon is one consistent piece whatever its shape: a
    quad is split along the diagonal that lies inside it, and a longer ring is triangulated in its plane after
    mending, since a ring that touches or crosses itself, as a ring with a hole cut into it does, would otherwise
    fan into triangles wound both ways. A polygon with no area gives no triangles.
    """
    rings = [np.asarray(f, dtype=float) for f in polygons]
    out, owner = [], []
    quads = [k for k, f in enumerate(rings) if len(f) == 4]
    if quads:
        q = np.stack([rings[k] for k in quads])
        normal = np.cross(q, np.roll(q, -1, axis=1)).sum(axis=1)
        first = np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1)  # split along 0-2
        second = np.stack([q[:, [1, 2, 3]], q[:, [1, 3, 0]]], axis=1)  # split along 1-3
        agree = np.einsum("qtk,qk->qt", np.cross(first[:, :, 1] - first[:, :, 0], first[:, :, 2] - first[:, :, 0]), normal) > 0
        split = np.where(agree.all(axis=1)[:, None, None, None], first, second)  # the diagonal inside a dart is the one both halves agree on
        has_area = np.linalg.norm(normal, axis=1) > 0
        out += list(split[has_area].reshape(-1, 3, 3))
        owner += np.repeat(np.asarray(quads)[has_area], 2).tolist()
    rest = [k for k, f in enumerate(rings) if len(f) >= 3 and len(f) != 4]
    if rest:
        triangles = [k for k in rest if len(rings[k]) == 3]
        out += [rings[k] for k in triangles]
        owner += triangles
        longer = [k for k in rest if len(rings[k]) > 4]
        for k, normal in zip(longer, newell([rings[k] for k in longer])):
            if not np.linalg.norm(normal):
                continue
            pieces = in_plane(rings[k], normal)
            against = np.cross(pieces[:, 1] - pieces[:, 0], pieces[:, 2] - pieces[:, 0]) @ normal < 0
            pieces[against] = pieces[against][:, ::-1]
            out += list(pieces)
            owner += [k] * len(pieces)
    return (np.array(out), np.array(owner)) if out else (np.zeros((0, 3, 3)), np.zeros(0, int))


def newell(rings):
    """Each ring's normal, twice its area long, by Newell's method, for every ring at once."""
    if not rings:
        return np.zeros((0, 3))
    corners = np.concatenate(rings)
    starts = np.cumsum([0, *map(len, rings)])
    following = np.arange(1, len(corners) + 1)
    following[starts[1:] - 1] = starts[:-1]  # each ring's last corner is followed by its first
    return np.add.reduceat(np.cross(corners, corners[following]), starts[:-1], axis=0)


def in_plane(f, normal):
    """A ring of five or more corners as triangles: mended where it touches or crosses itself, then a constrained
    Delaunay triangulation in the plane it lies flattest in. Corners the mending adds are lifted back onto that plane."""
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    flat = shapely.Polygon(f[:, keep])
    if not flat.is_valid:
        flat = shapely.make_valid(flat, method="structure", keep_collapsed=False)
    parts = [p for p in shapely.get_parts(flat) if p.geom_type == "Polygon" and p.area > 0]
    if not parts:
        return np.stack([np.stack([f[0], f[i], f[i + 1]]) for i in range(1, len(f) - 1)])
    corners = np.concatenate([shapely.get_coordinates(shapely.constrained_delaunay_triangles(p)).reshape(-1, 4, 2)[:, :3] for p in parts])
    lifted = np.empty((*corners.shape[:2], 3))
    lifted[:, :, keep] = corners
    lifted[:, :, drop] = (np.dot(normal, f[0]) - corners @ normal[keep]) / normal[drop]
    keys = np.round(f[:, keep], 6) @ [1, 1j]  # each corner of the ring as one number, to find it among the triangles' corners
    order = np.argsort(keys)
    at = np.searchsorted(keys[order], (np.round(corners, 6) @ [1, 1j]).reshape(-1))
    at = np.clip(at, 0, len(keys) - 1)
    known = keys[order][at] == (np.round(corners, 6) @ [1, 1j]).reshape(-1)
    lifted.reshape(-1, 3)[known] = f[order[at[known]]]  # a corner of the ring keeps its own height, off the plane or not
    return lifted
