//! The Python binding: numpy arrays in, numpy arrays out, Python's lock let go while the work runs.

use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

type Mesh = (Vec<[f64; 3]>, Vec<[usize; 3]>);

fn read_mesh(
    vertices: PyReadonlyArray2<'_, f64>,
    triangles: PyReadonlyArray2<'_, i64>,
) -> PyResult<Mesh> {
    let vertices: Vec<[f64; 3]> = vertices
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let triangles: Vec<[usize; 3]> = triangles
        .as_array()
        .rows()
        .into_iter()
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    if let Some(bad) = triangles.iter().flatten().find(|i| **i >= vertices.len()) {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "A triangle names corner {bad}, but there are {} corners.",
            vertices.len()
        )));
    }
    Ok((vertices, triangles))
}

/// libigl's function and outputs: per triangle, whether to flip it, and its component.
#[pyfunction]
#[pyo3(signature = (vertices, triangles, rays_total=None, rays_minimum=10, facet_wise=true, use_parity=false, seed=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn reorient_facets_raycast<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, i64>,
    rays_total: Option<usize>,
    rays_minimum: usize,
    facet_wise: bool,
    use_parity: bool,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<i64>>)> {
    let (vertices, triangles) = read_mesh(vertices, triangles)?;
    let settings = crate::Settings {
        rays_total,
        rays_minimum,
        facet_wise,
        use_parity,
        seed,
    };
    let (flip, component) =
        py.detach(|| crate::reorient_facets_raycast(&vertices, &triangles, &settings));
    Ok((
        flip.into_pyarray(py),
        component
            .into_iter()
            .map(|p| p as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}

/// The six views of the paper's backfacingness measure: 0 nothing, 1 a front, 2 a back, 6 by resolution by resolution.
#[pyfunction]
#[pyo3(signature = (vertices, triangles, resolution=1024))]
fn draw_backfacing_views<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, i64>,
    resolution: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let (vertices, triangles) = read_mesh(vertices, triangles)?;
    let views = py.detach(|| crate::draw_backfacing_views(&vertices, &triangles, resolution));
    Ok(
        numpy::ndarray::Array3::from_shape_vec((6, resolution, resolution), views)
            .expect("six square views")
            .into_pyarray(py),
    )
}

#[pymodule(gil_used = false)] // nothing shared between calls, so a free-threaded Python runs it on many threads at once
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(reorient_facets_raycast, module)?)?;
    module.add_function(wrap_pyfunction!(draw_backfacing_views, module)?)
}
