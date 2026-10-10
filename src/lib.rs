//! Outie: libigl's `reorient_facets_raycast`, the reference code for Takayama, Jacobson, Kavan and Sorkine-Hornung,
//! "A Simple Method for Correcting Facet Orientations in Polygon Meshes Based on Ray Casting", 2014, in Rust.
//!
//! A mesh of triangles in; per triangle, whether to flip it and the component it belongs to, out. Triangles that share an
//! edge no other triangle shares are gathered into components that agree. Rays are shot from random points on each component,
//! spread by area, off its front and its back; the side from which more rays escape is the outside, and on a tie the
//! side whose rays travel less far before hitting something is the inside. A triangle is decided with its component. With
//! the paper's measure of how plausible the result is, backfacingness, drawn as its six views.

use std::collections::{HashMap, VecDeque};

use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

mod bind;
mod cast;
mod measure;

pub use measure::draw_backfacing_views;

const GRAZING: f64 = 0.1; // a ray closer than this to lying in its triangle is drawn again, as libigl's

/// The paper's settings, with the paper's defaults: each triangle decided on its own, as its results were made.
#[derive(Clone, Debug)]
pub struct Settings {
    pub rays_total: Option<usize>, // rays for the whole mesh; 100 a triangle when None
    pub rays_minimum: usize,       // at least this many a component
    pub facet_wise: bool,          // every triangle its own component
    pub use_parity: bool,          // the parity vote, for closed meshes
    pub seed: u64,                 // so a run repeats
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            rays_total: None,
            rays_minimum: 10,
            facet_wise: true,
            use_parity: false,
            seed: 0,
        }
    }
}

/// Per triangle, whether to flip it, and the component it belongs to.
pub fn reorient_facets_raycast(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    settings: &Settings,
) -> (Vec<bool>, Vec<usize>) {
    if triangles.is_empty() {
        return (vec![], vec![]);
    }
    let (wound, component) = if settings.facet_wise {
        (triangles.to_vec(), (0..triangles.len()).collect())
    } else {
        bfs_orient(triangles)
    };
    let turned: Vec<bool> = wound.iter().zip(triangles).map(|(a, b)| a != b).collect(); // a triangle bfs_orient turned is reported the other way round
    let normals: Vec<[f64; 3]> = wound
        .iter()
        .map(|f| {
            cross(
                sub(vertices[f[1]], vertices[f[0]]),
                sub(vertices[f[2]], vertices[f[0]]),
            )
        })
        .collect();
    let double_area: Vec<f64> = normals.iter().map(|n| norm(*n)).collect();
    let components = component.iter().max().map_or(0, |p| p + 1);
    let mut area = vec![0.0; components];
    for (p, a) in component.iter().zip(&double_area) {
        area[*p] += a;
    }
    let total: f64 = area.iter().sum::<f64>().max(1e-300);
    let rays_total = settings.rays_total.unwrap_or(triangles.len() * 100) as f64;
    let rays: Vec<usize> = area
        .iter()
        .map(|a| {
            if *a == 0.0 {
                0
            } else {
                ((rays_total * a / total) as usize).max(settings.rays_minimum)
            }
        })
        .collect();
    if rays.iter().all(|r| *r == 0) {
        return (turned, component);
    }

    let mut rng = Pcg64::seed_from_u64(settings.seed);
    let starts = sample(&component, &double_area, &rays, &mut rng); // rays start on triangles chosen by area within their component, at random points on them, Turk 1990
    let origins: Vec<[f64; 3]> = starts
        .iter()
        .map(|t| {
            let (s, u): (f64, f64) = (rng.random(), rng.random());
            let root = u.sqrt();
            let [a, b, c] = wound[*t].map(|i| vertices[i]);
            let w = [1.0 - root, (1.0 - s) * root, s * root];
            [0, 1, 2].map(|k| w[0] * a[k] + w[1] * b[k] + w[2] * c[k])
        })
        .collect();
    let directions: Vec<[f64; 3]> = starts
        .iter()
        .map(|t| draw_direction(scale(normals[*t], 1.0 / double_area[*t]), &mut rng))
        .collect();

    let scene = cast::Scene::new(vertices, &wound);

    let cast: Vec<((f64, f64), (f64, f64))> = starts
        .par_iter()
        .zip(&origins)
        .zip(&directions)
        .map(|((t, o), d)| {
            let own = *t;
            if settings.use_parity {
                (
                    count_crossings(&scene, own, *o, *d),
                    count_crossings(&scene, own, *o, scale(*d, -1.0)),
                )
            } else {
                (
                    cast_ray(&scene, own, *o, *d),
                    cast_ray(&scene, own, *o, scale(*d, -1.0)),
                )
            }
        })
        .collect();
    let mut front = vec![(0.0, 0.0); components]; // per component: rays escaped, or crossings odd, and distance travelled
    let mut back = vec![(0.0, 0.0); components];
    for (t, (f, b)) in starts.iter().zip(cast) {
        let p = component[*t];
        front[p].0 += f.0;
        front[p].1 += f.1;
        back[p].0 += b.0;
        back[p].1 += b.1;
    }
    let voted: Vec<bool> = (0..components)
        .map(|p| {
            if settings.use_parity {
                front[p].0 > back[p].0
            } else {
                front[p].0 < back[p].0 || (front[p].0 == back[p].0 && front[p].1 < back[p].1)
            }
        })
        .collect();
    (
        component
            .iter()
            .zip(&turned)
            .map(|(p, t)| voted[*p] ^ t)
            .collect(),
        component,
    )
}

/// The triangle each ray starts on: rays[p] of them in component p, each triangle as likely as its area.
fn sample(component: &[usize], double_area: &[f64], rays: &[usize], rng: &mut Pcg64) -> Vec<usize> {
    let mut members: Vec<Vec<usize>> = vec![vec![]; rays.len()];
    for (t, p) in component.iter().enumerate() {
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
            let at = reach
                .partition_point(|r| *r <= target)
                .min(members[p].len() - 1);
            out.push(members[p][at]);
        }
    }
    out
}

/// A random direction off the front of a face, never grazing it.
fn draw_direction(normal: [f64; 3], rng: &mut Pcg64) -> [f64; 3] {
    loop {
        let d: [f64; 3] = [
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
        ];
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

/// A ray's first hit on any triangle but the one it leaves, as the paper's code drops a hit on its own face: 1 and 0
/// when it escapes to infinity, else 0 and how far it travels before it.
fn cast_ray(scene: &cast::Scene, own: usize, origin: [f64; 3], direction: [f64; 3]) -> (f64, f64) {
    match scene.cast_ray(own, origin, direction, 0.0) {
        Some(distance) => (0.0, distance as f64),
        None => (1.0, 0.0),
    }
}

/// The parity of how many triangles a ray passes through, its own left out: hit after hit, each next one sought just
/// past the last, as the paper's code gathers them.
fn count_crossings(
    scene: &cast::Scene,
    own: usize,
    origin: [f64; 3],
    direction: [f64; 3],
) -> (f64, f64) {
    let mut crossings = 0;
    let mut start = 0.0f32;
    while let Some(distance) = scene.cast_ray(own, origin, direction, start) {
        crossings += 1;
        start = distance + 1e-5 * (1.0 + distance);
    }
    ((crossings % 2) as f64, 0.0)
}

/// libigl's bfs_orient: triangles wound to agree with their neighbours across the edges exactly two share, and the
/// component each belongs to, numbered in the order the components are found.
pub fn bfs_orient(triangles: &[[usize; 3]]) -> (Vec<[usize; 3]>, Vec<usize>) {
    let mut sharing: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (f, t) in triangles.iter().enumerate() {
        for (x, y) in list_edges(t) {
            if x != y {
                sharing.entry((x.min(y), x.max(y))).or_default().push(f);
            }
        }
    }
    let mut neighbours: Vec<Vec<usize>> = vec![vec![]; triangles.len()];
    for owners in sharing.values() {
        if let [a, b] = owners[..]
            && a != b
        {
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }
    let mut wound = triangles.to_vec();
    let mut component = vec![usize::MAX; triangles.len()];
    let mut next = 0;
    for start in 0..triangles.len() {
        if component[start] != usize::MAX {
            continue;
        }
        component[start] = next;
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            for &n in &neighbours[f] {
                if component[n] != usize::MAX {
                    continue;
                }
                component[n] = next;
                if shares_directed_edge(&wound[f], &wound[n]) {
                    wound[n].reverse();
                }
                queue.push_back(n);
            }
        }
        next += 1;
    }
    (wound, component)
}

/// A triangle's edges, each as it runs, the last back to the first.
fn list_edges(t: &[usize; 3]) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..3).map(move |k| (t[k], t[(k + 1) % 3]))
}

fn shares_directed_edge(f: &[usize; 3], n: &[usize; 3]) -> bool {
    list_edges(f).any(|e| list_edges(n).any(|o| o == e))
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
