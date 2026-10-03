//! Outie: libigl's `reorient_facets_raycast`, the reference code for Takayama, Jacobson, Kavan and Sorkine-Hornung,
//! "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray Casting", 2014, in Rust, with
//! one addition: occluders, triangles rays can hit that are never turned, such as the ground a model stands on.
//!
//! The paper's faces are polygons, and so are outie's: a face is a ring of corners, any number of them, with any holes
//! after it. Faces that share
//! edges are gathered into patches that agree. Each face is cut into triangles only to cast rays from: rays are shot
//! from random points on each patch, spread by area, off its front and its back; the side from which more rays escape
//! is the outside, and on a tie the side whose rays travel further before hitting anything. A face is decided with its
//! patch, whole. Given triangles, it is libigl's function exactly.

use std::collections::{HashMap, VecDeque};

use parry3d_f64::math::Vector;
use parry3d_f64::query::{Ray, RayCast};
use parry3d_f64::shape::TriMesh;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

/// A face: its outer ring, then any holes, each a ring of corner numbers.
pub type Face = Vec<Vec<usize>>;

const GRAZING: f64 = 0.1; // a ray closer than this to lying in its face is drawn again
const EPSILON: f64 = 1e-4; // how far along itself a ray starts, so it does not hit its own face

/// The paper's settings, with libigl's defaults.
#[derive(Clone, Debug)]
pub struct Settings {
    pub rays_total: Option<usize>, // rays for the whole mesh; 100 a triangle when None
    pub rays_minimum: usize,       // at least this many a patch
    pub facet_wise: bool,          // every triangle its own patch
    pub use_parity: bool,          // the parity vote, for closed meshes
    pub seed: u64,                 // so a run repeats
}

impl Default for Settings {
    fn default() -> Self {
        Settings { rays_total: None, rays_minimum: 10, facet_wise: false, use_parity: false, seed: 0 }
    }
}

/// Per face, whether to turn it round, and the patch it belongs to.
pub fn reorient_facets_raycast(vertices: &[[f64; 3]], faces: &[Face], occluders: &[[[f64; 3]; 3]], settings: &Settings) -> (Vec<bool>, Vec<usize>) {
    if faces.is_empty() {
        return (vec![], vec![]);
    }
    let (wound, patch) = if settings.facet_wise { (faces.to_vec(), (0..faces.len()).collect()) } else { bfs_orient(faces) };
    let turned: Vec<bool> = wound.iter().zip(faces).map(|(a, b)| a != b).collect(); // a face bfs_orient turned is reported the other way round
    let (triangles, owner) = triangulate(vertices, &wound); // for casting rays only, each wound as its face
    let normals: Vec<[f64; 3]> = triangles.iter().map(|f| cross(sub(vertices[f[1]], vertices[f[0]]), sub(vertices[f[2]], vertices[f[0]]))).collect();
    let double_area: Vec<f64> = normals.iter().map(|n| norm(*n)).collect();
    let in_patch: Vec<usize> = owner.iter().map(|f| patch[*f]).collect();
    let patches = patch.iter().max().map_or(0, |p| p + 1);
    let mut area = vec![0.0; patches];
    for (p, a) in in_patch.iter().zip(&double_area) {
        area[*p] += a;
    }
    let total: f64 = area.iter().sum::<f64>().max(1e-300);
    let rays_total = settings.rays_total.unwrap_or(triangles.len() * 100) as f64;
    let rays: Vec<usize> = area.iter().map(|a| if *a == 0.0 { 0 } else { ((rays_total * a / total) as usize).max(settings.rays_minimum) }).collect();
    if rays.iter().all(|r| *r == 0) {
        return (turned, patch);
    }

    let mut rng = Pcg64::seed_from_u64(settings.seed);
    let starts = sample(&in_patch, &double_area, &rays, &mut rng); // rays start on triangles chosen by area within their patch, at random points on them, Turk 1990
    let origins: Vec<[f64; 3]> = starts
        .iter()
        .map(|t| {
            let (s, u): (f64, f64) = (rng.random(), rng.random());
            let root = u.sqrt();
            let [a, b, c] = triangles[*t].map(|i| vertices[i]);
            let w = [1.0 - root, (1.0 - s) * root, s * root];
            [0, 1, 2].map(|k| w[0] * a[k] + w[1] * b[k] + w[2] * c[k])
        })
        .collect();
    let directions: Vec<[f64; 3]> = starts.iter().map(|t| random_direction(scale(normals[*t], 1.0 / double_area[*t]), &mut rng)).collect();

    let mut points: Vec<Vector> = vertices.iter().map(|v| Vector::new(v[0], v[1], v[2])).collect();
    let mut indices: Vec<[u32; 3]> = triangles.iter().map(|f| f.map(|i| i as u32)).collect();
    for triangle in occluders {
        let at = points.len() as u32;
        points.extend(triangle.iter().map(|v| Vector::new(v[0], v[1], v[2])));
        indices.push([at, at + 1, at + 2]);
    }
    let scene = TriMesh::new(points, indices).expect("a mesh of triangles");

    let cast: Vec<((f64, f64), (f64, f64))> = origins
        .par_iter()
        .zip(&directions)
        .map(|(o, d)| {
            if settings.use_parity {
                (parity(&scene, *o, *d), parity(&scene, *o, scale(*d, -1.0)))
            } else {
                (first_hit(&scene, *o, *d), first_hit(&scene, *o, scale(*d, -1.0)))
            }
        })
        .collect();
    let mut front = vec![(0.0, 0.0); patches]; // per patch: rays escaped, or crossings odd, and distance travelled
    let mut back = vec![(0.0, 0.0); patches];
    for (t, (f, b)) in starts.iter().zip(cast) {
        let p = in_patch[*t];
        front[p].0 += f.0;
        front[p].1 += f.1;
        back[p].0 += b.0;
        back[p].1 += b.1;
    }
    let voted: Vec<bool> = (0..patches)
        .map(|p| if settings.use_parity { front[p].0 > back[p].0 } else { front[p].0 < back[p].0 || (front[p].0 == back[p].0 && front[p].1 < back[p].1) })
        .collect();
    (patch.iter().zip(&turned).map(|(p, t)| voted[*p] ^ t).collect(), patch)
}

/// Every face as triangles wound as it is, and the face each came from: a lone triangle as it is, anything else cut by
/// earcut, holes and all, in the plane it lies flattest in. A face with no area gives none.
fn triangulate(vertices: &[[f64; 3]], faces: &[Face]) -> (Vec<[usize; 3]>, Vec<usize>) {
    let mut cutter = earcut::Earcut::<f64>::new();
    let mut cut: Vec<u32> = Vec::new();
    let (mut triangles, mut owner) = (Vec::new(), Vec::new());
    for (f, rings) in faces.iter().enumerate() {
        let Some(outer) = rings.first() else { continue };
        if rings.len() == 1 && outer.len() == 3 {
            triangles.push([outer[0], outer[1], outer[2]]);
            owner.push(f);
            continue;
        }
        if outer.len() < 3 {
            continue;
        }
        let normal = newell(vertices, outer);
        if norm(normal) == 0.0 {
            continue;
        }
        let drop = (0..3).max_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs())).unwrap_or(2);
        let keep: Vec<usize> = (0..3).filter(|k| *k != drop).collect();
        let corners: Vec<usize> = rings.iter().flatten().copied().collect();
        let holes: Vec<u32> = rings.iter().scan(0, |at, ring| { let start = *at; *at += ring.len(); Some(start as u32) }).skip(1).collect();
        cutter.earcut(corners.iter().map(|i| [vertices[*i][keep[0]], vertices[*i][keep[1]]]), &holes, &mut cut);
        for piece in cut.chunks_exact(3) {
            let mut triangle = [corners[piece[0] as usize], corners[piece[1] as usize], corners[piece[2] as usize]];
            if dot(cross(sub(vertices[triangle[1]], vertices[triangle[0]]), sub(vertices[triangle[2]], vertices[triangle[0]])), normal) < 0.0 {
                triangle.swap(1, 2); // wound as its face
            }
            triangles.push(triangle);
            owner.push(f);
        }
    }
    (triangles, owner)
}

/// A ring's normal, twice its area long, by Newell's method, which holds for a bent or a concave ring.
fn newell(vertices: &[[f64; 3]], ring: &[usize]) -> [f64; 3] {
    let mut n = [0.0; 3];
    for (k, i) in ring.iter().enumerate() {
        let (a, b) = (vertices[*i], vertices[ring[(k + 1) % ring.len()]]);
        n = add(n, cross(a, b));
    }
    n
}

/// The triangle each ray starts on: rays[p] of them in patch p, each triangle as likely as its area.
fn sample(patch: &[usize], double_area: &[f64], rays: &[usize], rng: &mut Pcg64) -> Vec<usize> {
    let mut members: Vec<Vec<usize>> = vec![vec![]; rays.len()];
    for (t, p) in patch.iter().enumerate() {
        members[*p].push(t);
    }
    let mut out = Vec::with_capacity(rays.iter().sum());
    for (p, count) in rays.iter().enumerate() {
        let reach: Vec<f64> = members[p]
            .iter()
            .scan(0.0, |sum, t| {
                *sum += double_area[*t];
                Some(*sum)
            })
            .collect();
        let Some(&whole) = reach.last() else { continue };
        for _ in 0..*count {
            let target = rng.random::<f64>() * whole;
            let at = reach.partition_point(|r| *r <= target).min(members[p].len() - 1);
            out.push(members[p][at]);
        }
    }
    out
}

/// A random direction off the front of a face, never grazing it.
fn random_direction(normal: [f64; 3], rng: &mut Pcg64) -> [f64; 3] {
    loop {
        let d: [f64; 3] = [rng.sample(StandardNormal), rng.sample(StandardNormal), rng.sample(StandardNormal)];
        let length = norm(d);
        if length == 0.0 {
            continue;
        }
        let d = scale(d, 1.0 / length);
        let along = dot(d, normal);
        if along.abs() >= GRAZING {
            return if along < 0.0 { scale(d, -1.0) } else { d };
        }
    }
}

fn ray(start: [f64; 3], direction: [f64; 3]) -> Ray {
    Ray::new(Vector::new(start[0], start[1], start[2]), Vector::new(direction[0], direction[1], direction[2]))
}

/// A ray's first hit: 1 and 0 when it escapes to infinity, else 0 and how far it travels before it.
fn first_hit(scene: &TriMesh, origin: [f64; 3], direction: [f64; 3]) -> (f64, f64) {
    match scene.cast_local_ray(&ray(add(origin, scale(direction, EPSILON)), direction), f64::MAX, false) {
        Some(time) => (0.0, time + EPSILON),
        None => (1.0, 0.0),
    }
}

/// The parity of how many faces a ray passes through.
fn parity(scene: &TriMesh, origin: [f64; 3], direction: [f64; 3]) -> (f64, f64) {
    let mut start = add(origin, scale(direction, EPSILON));
    let mut crossings = 0;
    while let Some(time) = scene.cast_local_ray(&ray(start, direction), f64::MAX, false) {
        crossings += 1;
        start = add(start, scale(direction, time + EPSILON));
    }
    ((crossings % 2) as f64, 0.0)
}

/// libigl's bfs_orient, for faces of any number of corners and holes: faces wound to agree with their neighbours across
/// the edges exactly two share, and the patch each belongs to, numbered in the order the patches are found.
pub fn bfs_orient(faces: &[Face]) -> (Vec<Face>, Vec<usize>) {
    let mut sharing: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (f, rings) in faces.iter().enumerate() {
        for (x, y) in rings.iter().flat_map(|ring| edges(ring)) {
            if x != y {
                sharing.entry((x.min(y), x.max(y))).or_default().push(f);
            }
        }
    }
    let mut neighbours: Vec<Vec<usize>> = vec![vec![]; faces.len()];
    for owners in sharing.values() {
        if let [a, b] = owners[..] {
            if a != b {
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }
    let mut wound = faces.to_vec();
    let mut patch = vec![usize::MAX; faces.len()];
    let mut next = 0;
    for start in 0..faces.len() {
        if patch[start] != usize::MAX {
            continue;
        }
        patch[start] = next;
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            for &n in &neighbours[f] {
                if patch[n] != usize::MAX {
                    continue;
                }
                patch[n] = next;
                if shares_directed_edge(&wound[f], &wound[n]) {
                    for ring in &mut wound[n] {
                        ring.reverse(); // a face turned round, its holes with it
                    }
                }
                queue.push_back(n);
            }
        }
        next += 1;
    }
    (wound, patch)
}

/// A ring's edges, each as it runs, the last back to the first.
fn edges(ring: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    ring.iter().enumerate().map(|(k, a)| (*a, ring[(k + 1) % ring.len()]))
}

/// Whether two faces run a shared edge the same way, which means one of them is wound against the other.
fn shares_directed_edge(f: &Face, n: &Face) -> bool {
    let mine: Vec<(usize, usize)> = f.iter().flat_map(|ring| edges(ring)).collect();
    n.iter().flat_map(|ring| edges(ring)).any(|e| mine.contains(&e))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

mod python {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
    use pyo3::prelude::*;

    /// libigl's function and outputs, for faces of any number of corners and holes: per face, whether to turn it, and its patch.
    #[pyfunction]
    #[pyo3(signature = (vertices, corners, ring_sizes, face_sizes, rays_total=None, rays_minimum=10, facet_wise=false, use_parity=false, seed=0, occluders=None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reorient_facets_raycast<'py>(
        py: Python<'py>,
        vertices: PyReadonlyArray2<'py, f64>,
        corners: PyReadonlyArray1<'py, i64>,
        ring_sizes: PyReadonlyArray1<'py, i64>,
        face_sizes: PyReadonlyArray1<'py, i64>,
        rays_total: Option<usize>,
        rays_minimum: usize,
        facet_wise: bool,
        use_parity: bool,
        seed: u64,
        occluders: Option<PyReadonlyArray3<'py, f64>>,
    ) -> PyResult<(Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<i64>>)> {
        let vertices: Vec<[f64; 3]> = vertices.as_array().rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
        let corners = corners.as_array();
        let mut rings: Vec<Vec<usize>> = Vec::with_capacity(ring_sizes.len()?);
        let mut at = 0;
        for size in ring_sizes.as_array().iter() {
            let next = at + *size as usize;
            if *size < 0 || next > corners.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("The rings' corners run out before their sizes do."));
            }
            rings.push(corners.slice(numpy::ndarray::s![at..next]).iter().map(|i| *i as usize).collect());
            at = next;
        }
        let mut faces: Vec<super::Face> = Vec::with_capacity(face_sizes.len()?);
        let mut rings = rings.into_iter();
        for size in face_sizes.as_array().iter() {
            let face: super::Face = rings.by_ref().take((*size).max(0) as usize).collect();
            if face.len() != *size as usize {
                return Err(pyo3::exceptions::PyValueError::new_err("The faces' rings run out before their sizes do."));
            }
            faces.push(face);
        }
        let occluders: Vec<[[f64; 3]; 3]> = occluders
            .map(|o| o.as_array().outer_iter().map(|t| [0, 1, 2].map(|i| [t[[i, 0]], t[[i, 1]], t[[i, 2]]])).collect())
            .unwrap_or_default();
        if let Some(bad) = faces.iter().flatten().flatten().find(|i| **i >= vertices.len()) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!("A face names corner {bad}, but there are {} corners.", vertices.len())));
        }
        let settings = super::Settings { rays_total, rays_minimum, facet_wise, use_parity, seed };
        let (flip, patch) = py.detach(|| super::reorient_facets_raycast(&vertices, &faces, &occluders, &settings));
        Ok((flip.into_pyarray(py), patch.into_iter().map(|p| p as i64).collect::<Vec<_>>().into_pyarray(py)))
    }

    #[pymodule(gil_used = false)]  // nothing shared between calls, so a free-threaded Python runs it on many threads at once
    fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(reorient_facets_raycast, module)?)
    }
}
