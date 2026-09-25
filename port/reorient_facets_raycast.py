# Stage 1 of the port: a literal transliteration of libigl's C++ into Python, line for line.
# Not good Python and not expected to run cleanly. It exists to be checked against the original.
#
# Sources, libigl main at 7100764c2a28 (2026-09-04):
#   include/igl/embree/reorient_facets_raycast.cpp
#   include/igl/bfs_orient.cpp
#   include/igl/orientable_patches.cpp
#   include/igl/random_dir.cpp
#   include/igl/per_face_normals.cpp
#   include/igl/doublearea.cpp
#   include/igl/embree/EmbreeIntersector.h  (only what intersectRay does: every hit along a ray, nearest first)

import math
import random
import time
from collections import deque

import numpy as np
import scipy.sparse as sp
import trimesh


# ---------------------------------------------------------------- Hit.h
class Hit:
    def __init__(self, id, gid, u, v, t):
        self.id = id    # primitive id
        self.gid = gid  # geometry id
        self.u = u      # barycentric coordinates so that
        self.v = v      # x = u*v1 + v*v2 + (1-u-v)*v0
        self.t = t      # distance along the ray


# ---------------------------------------------------------------- EmbreeIntersector.h
class EmbreeIntersector:
    def init(self, V, F):
        self.mesh = trimesh.Trimesh(vertices=np.asarray(V, dtype=np.float32), faces=np.asarray(F), process=False)

    def intersectRay(self, origin, direction, hits, num_rays, tnear=0, tfar=float("inf"), mask=0xFFFFFFFF):
        # the vector<Hit> overload: every hit along the ray, sorted by t, nearest first
        hits.clear()
        where, ray, tri = self.mesh.ray.intersects_location(np.asarray([origin]), np.asarray([direction]), multiple_hits=True)
        ts = [float(np.linalg.norm(w - np.asarray(origin))) for w in where]
        for t, id in sorted(zip(ts, tri)):
            if t > tnear and t < tfar:
                hits.append(Hit(int(id), 0, 0.0, 0.0, t))
        num_rays = len(hits) + 1
        return len(hits) > 0


# ---------------------------------------------------------------- random_dir.cpp
PI = math.pi
RAND_MAX = 2147483647


def rand():
    return random.randint(0, RAND_MAX)


def random_dir():
    z = float(rand()) / float(RAND_MAX) * 2.0 - 1.0
    t = float(rand()) / float(RAND_MAX) * 2.0 * PI
    r = math.sqrt(1.0 - z * z)
    x = r * math.cos(t)
    y = r * math.sin(t)
    return np.array([x, y, z])


# ---------------------------------------------------------------- per_face_normals.cpp
def per_face_normals(V, F, N):
    # N = cross(v1 - v0, v2 - v0), normalized; a degenerate face gets Z (the default) which is the zero vector here
    Z = np.zeros(3)
    N.resize((F.shape[0], 3), refcheck=False)
    for i in range(F.shape[0]):
        v1 = V[F[i, 1]] - V[F[i, 0]]
        v2 = V[F[i, 2]] - V[F[i, 0]]
        N[i] = np.cross(v1, v2)
        r = np.linalg.norm(N[i])
        if r == 0:
            N[i] = Z
        else:
            N[i] /= r


# ---------------------------------------------------------------- doublearea.cpp
def doublearea(V, F, dblA):
    # twice the area of each triangle, from the cross product (the 3D branch of doublearea)
    dblA.resize(F.shape[0], refcheck=False)
    for i in range(F.shape[0]):
        v1 = V[F[i, 1]] - V[F[i, 0]]
        v2 = V[F[i, 2]] - V[F[i, 0]]
        dblA[i] = np.linalg.norm(np.cross(v1, v2))


# ---------------------------------------------------------------- orientable_patches.cpp
def orientable_patches(F, C, A):
    assert F.shape[1] == 3
    m = F.shape[0]
    # allE: every directed edge of every face, three per face
    allE = np.zeros((m * 3, 2), dtype=int)
    allE[0 * m : 1 * m, 0] = F[:, 1]
    allE[0 * m : 1 * m, 1] = F[:, 2]
    allE[1 * m : 2 * m, 0] = F[:, 2]
    allE[1 * m : 2 * m, 1] = F[:, 0]
    allE[2 * m : 3 * m, 0] = F[:, 0]
    allE[2 * m : 3 * m, 1] = F[:, 1]
    # sort(allE,2,true,sortallE,IX): each row sorted ascending
    sortallE = np.sort(allE, axis=1)
    # unique_rows(sortallE,uE,IA,IC): uE the unique rows, IC maps each row to its unique row
    uE, IC = np.unique(sortallE, axis=0, return_inverse=True)
    IC = np.asarray(IC).reshape(-1)
    # uE2FT: faces by unique edges, 1 where the face has the edge
    rows = [e % m for e in range(IC.shape[0])]
    cols = [IC[e] for e in range(IC.shape[0])]
    uE2FT = sp.lil_matrix((m, uE.shape[0]))
    for r, c in zip(rows, cols):
        uE2FT[r, c] += 1
    # drop edges with more than two faces (non-manifold): zero their column
    uE2FT = uE2FT.tocsc()
    for j in range(uE2FT.shape[1]):
        degree = uE2FT[:, j].count_nonzero()
        if degree > 2:
            uE2FT[:, j] = 0
    uE2FT.eliminate_zeros()
    uE2F = uE2FT.transpose()
    A_ = (uE2FT @ uE2F).tolil()
    # cap at 1
    A_[A_ > 1] = 1
    A[0] = A_.tocsr()
    # vertex_components(A,C): connected components of the adjacency
    n_components, labels = sp.csgraph.connected_components(A[0], directed=False)
    C.resize(m, refcheck=False)
    C[:] = labels


# ---------------------------------------------------------------- bfs_orient.cpp
def bfs_orient(F, FF, C):
    A = [None]
    orientable_patches(F, C, A)
    A = A[0]
    m = F.shape[0]
    num_cc = int(C.max()) + 1
    seen = np.zeros(m, dtype=int)
    ES = [[1, 2], [2, 0], [0, 1]]
    if FF is not F:
        FF[:] = F
    for c in range(num_cc):  # parallel_for in the original
        Q = deque()
        for f in range(FF.shape[0]):
            if C[f] == c:
                Q.append(f)
                break
        assert len(Q) > 0
        while len(Q) > 0:
            f = Q.popleft()
            if seen[f] > 0:
                continue
            seen[f] += 1
            row = A.getrow(f)
            for n, value in zip(row.indices, row.data):  # InnerIterator over column f of A, A is symmetric
                if value != 0 and n != f:
                    assert n != f
                    for efi in range(3):
                        ef = (FF[f, ES[efi][0]], FF[f, ES[efi][1]])
                        for eni in range(3):
                            en = (FF[n, ES[eni][0]], FF[n, ES[eni][1]])
                            if ef[0] == en[0] and ef[1] == en[1]:
                                FF[n] = FF[n][::-1]
                    Q.append(n)


# ---------------------------------------------------------------- reorient_facets_raycast.cpp
def reorient_facets_raycast(V, F, rays_total, rays_minimum, facet_wise, use_parity, is_verbose, I, C, seeded=False):
    assert F.shape[1] == 3
    assert V.shape[1] == 3
    m = F.shape[0]
    Fi = F.astype(int)
    FF = np.zeros_like(Fi)
    if facet_wise:
        FF[:] = Fi
        C.resize(m, refcheck=False)
        for i in range(m):
            C[i] = i
    else:
        if is_verbose:
            print("extracting patches... ", end="")
        bfs_orient(Fi, FF, C)
    if is_verbose:
        print(str(int(C.max()) + 1) + " components. ", end="")
    num_cc = int(C.max()) + 1
    ei = EmbreeIntersector()
    ei.init(V.astype(np.float32), FF)
    N = np.zeros((0, 3))
    per_face_normals(V, FF, N)
    A = np.zeros(0)
    doublearea(V, FF, A)
    area_total = A.sum()
    area_per_component = np.zeros(num_cc)
    for f in range(m):
        area_per_component[C[f]] += A[f]
    num_rays_per_component = np.zeros(num_cc, dtype=int)
    for c in range(num_cc):
        num_rays_per_component[c] = max(int(rays_total * area_per_component[c] / area_total), rays_minimum)
    rays_total = int(num_rays_per_component.sum())
    if is_verbose:
        print("generating rays... ", end="")
    if not seeded:
        random.seed(time.time())  # prng.seed(time(nullptr))
    ray_face = []
    ray_ori = []
    ray_dir = []
    for c in range(num_cc):
        if area_per_component[c] == 0:
            continue
        CF = []  # set of faces per component
        CF_area = []
        for f in range(m):
            if C[f] == c:
                CF.append(f)
                CF_area.append(A[f])
        # std::discrete_distribution over CF weighted by CF_area
        weights = np.array(CF_area) / np.sum(CF_area)
        for i in range(num_rays_per_component[c]):
            f = CF[np.random.choice(len(CF), p=weights)]  # select face with probability proportional to face area
            s = random.random()  # random barycentric coordinate (Turk, Graphics Gems I 1990)
            t = random.random()
            sqrt_t = math.sqrt(t)
            a = 1 - sqrt_t
            b = (1 - s) * sqrt_t
            cc = s * sqrt_t
            p = a * V[FF[f, 0]] + b * V[FF[f, 1]] + cc * V[FF[f, 2]]  # be careful with the index!!!
            n = N[f]
            if not np.any(n):
                continue
            while True:
                d = random_dir()
                ndotd = float(np.dot(n, d))
                if abs(ndotd) < 0.1:
                    continue
                if ndotd < 0:
                    d = d * -1.0
                break
            ray_face.append(f)
            ray_ori.append(p)
            ray_dir.append(d)
            if is_verbose and len(ray_face) % (rays_total // 10) == 0:
                print(".", end="")
    if is_verbose:
        print(str(len(ray_face)) + " rays. ", end="")
    C_vote_distance = [[0.0, 0.0] for _ in range(num_cc)]  # sum of distance between ray origin and intersection
    C_vote_infinity = [[0, 0] for _ in range(num_cc)]  # number of rays reaching infinity
    C_vote_parity = [[0, 0] for _ in range(num_cc)]  # sum of parity count for each ray
    if is_verbose:
        print("shooting rays... ", end="")
    for i in range(len(ray_face)):
        f = ray_face[i]
        o = ray_ori[i]
        d = ray_dir[i]
        c = C[f]
        hits_front = []
        hits_back = []
        num_rays_front = 0
        num_rays_back = 0
        ei.intersectRay(o, d, hits_front, num_rays_front)
        ei.intersectRay(o, -d, hits_back, num_rays_back)
        if len(hits_front) > 0 and hits_front[0].id == f:
            hits_front.pop(0)
        if len(hits_back) > 0 and hits_back[0].id == f:
            hits_back.pop(0)
        if use_parity:
            C_vote_parity[c][0] += len(hits_front) % 2
            C_vote_parity[c][1] += len(hits_back) % 2
        else:
            if len(hits_front) == 0:
                C_vote_infinity[c][0] += 1
            else:
                C_vote_distance[c][0] += hits_front[0].t
            if len(hits_back) == 0:
                C_vote_infinity[c][1] += 1
            else:
                C_vote_distance[c][1] += hits_back[0].t
    I.resize(m, refcheck=False)
    for f in range(m):
        c = C[f]
        if use_parity:
            # Ideally, parity for the front/back side should be 1/0 (i.e., parity sum for all rays should be smaller on the front side)
            I[f] = 1 if C_vote_parity[c][0] > C_vote_parity[c][1] else 0
        else:
            I[f] = (
                1
                if (C_vote_infinity[c][0] == C_vote_infinity[c][1] and C_vote_distance[c][0] < C_vote_distance[c][1])
                or C_vote_infinity[c][0] < C_vote_infinity[c][1]
                else 0
            )
        if not np.array_equal(Fi[f], FF[f]):
            I[f] = 1 - I[f]
    if is_verbose:
        print("done!")


def reorient_facets_raycast_default(V, F, FF, I):
    # the overload with the defaults filled in, writing the reoriented faces to FF
    rays_total = F.shape[0] * 100
    rays_minimum = 10
    facet_wise = False
    use_parity = False
    is_verbose = False
    C = np.zeros(0, dtype=int)
    reorient_facets_raycast(V, F, rays_total, rays_minimum, facet_wise, use_parity, is_verbose, I, C)
    FF.resize(F.shape, refcheck=False)
    for i in range(I.shape[0]):
        if I[i]:
            FF[i] = F[i][::-1]
        else:
            FF[i] = F[i]
