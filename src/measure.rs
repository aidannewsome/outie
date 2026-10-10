//! The paper's measure of how plausible a mesh's facing is from outside, backfacingness, drawn as the six views its
//! measure_backfacingness counts.

use rayon::prelude::*;

use crate::{add, cross, dot, scale, sub};

/// The six views of the paper's backfacingness measure, per pixel 0 where nothing is drawn, 1 where a triangle shows
/// its front and 2 where it shows its back: as the paper's measure_backfacingness draws them, orthographic, from the
/// six sides of a box 5% larger than the mesh, each triangle's front and back in two colours, depth tested. Each view
/// is resolution by resolution, its rows from the bottom up as OpenGL reads them.
pub fn draw_backfacing_views(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    resolution: usize,
) -> Vec<u8> {
    let mut views = vec![0u8; 6 * resolution * resolution];
    if triangles.is_empty() || resolution == 0 {
        return views;
    }
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for t in triangles {
        for v in t {
            for k in 0..3 {
                low[k] = low[k].min(vertices[*v][k]);
                high[k] = high[k].max(vertices[*v][k]);
            }
        }
    }
    let centre = [0, 1, 2].map(|k| (low[k] + high[k]) / 2.0);
    let r = (0..3).map(|k| high[k] - low[k]).fold(0.0, f64::max) * 1.05 / 2.0;
    if r == 0.0 {
        return views;
    }
    let axes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ];
    let ups = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    views
        .par_chunks_mut(resolution * resolution)
        .enumerate()
        .for_each(|(i, view)| {
            let eye = add(centre, scale(axes[i], 2.0 * r));
            let forward = scale(axes[i], -1.0); // gluLookAt: toward the centre
            let side = cross(forward, ups[i]);
            let up = cross(side, forward);
            let window = |p: [f64; 3]| {
                let d = sub(p, eye);
                [
                    (dot(side, d) / r + 1.0) / 2.0 * resolution as f64,
                    (dot(up, d) / r + 1.0) / 2.0 * resolution as f64,
                    dot(forward, d),
                ]
            };
            let mut depth = vec![f64::INFINITY; resolution * resolution];
            for t in triangles {
                let [a, b, c] = t.map(|v| window(vertices[v]));
                let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                if area == 0.0 {
                    continue;
                }
                let colour = if area > 0.0 { 1 } else { 2 }; // anticlockwise on screen, OpenGL's front
                let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
                let x1 = (a[0].max(b[0]).max(c[0]).ceil() as usize).min(resolution);
                let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
                let y1 = (a[1].max(b[1]).max(c[1]).ceil() as usize).min(resolution);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                        let wa = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area;
                        let wb = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area;
                        let wc = 1.0 - wa - wb;
                        if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                            continue;
                        }
                        let z = wa * a[2] + wb * b[2] + wc * c[2];
                        let at = y * resolution + x;
                        if z < depth[at] && (r..=3.0 * r).contains(&z) {
                            depth[at] = z;
                            view[at] = colour;
                        }
                    }
                }
            }
        });
    views
}
