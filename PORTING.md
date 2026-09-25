# How this was ported

Outie is a port of `igl::embree::reorient_facets_raycast` from libigl, taken at commit
7100764c2a28 of libigl's main branch, 2026-09-04. The function and the helpers it calls were
fetched from that commit:

- `include/igl/embree/reorient_facets_raycast.cpp` and `.h`
- `include/igl/bfs_orient.cpp`
- `include/igl/orientable_patches.cpp`
- `include/igl/random_dir.cpp`
- `include/igl/per_face_normals.cpp`
- `include/igl/doublearea.cpp`
- `include/igl/embree/EmbreeIntersector.h`, for what `intersectRay` promises: every hit along a ray,
  nearest first, with the primitive id and the distance.

The port went in three stages, one commit each, so that each can be checked on its own.

## Stage 1: transliteration

The C++ was rewritten into Python line for line, by hand, keeping the original's names, order,
loops and comments, in `port/reorient_facets_raycast.py`. Eigen matrices became numpy arrays,
`std::vector` became lists, the sparse adjacency became scipy, the C random number generator
became Python's `random`, and Embree became trimesh's ray query with the embreex engine. Nothing
was reorganised. It is not good Python and was not expected to run cleanly.

## Stage 2: made to run

The transliteration was run against the test meshes in `tests/` and fixed until every test
passed, changing as little as possible. Each fix is a line in the stage 2 commit message.

## Stage 3: made Python

The working transliteration was rewritten into `src/outie/__init__.py` in the flavour of
Python: rays generated and traced in one vectorised pass, votes tallied with `numpy.bincount`,
plain functions with the paper's parameters as keyword arguments, polygons accepted as well as
triangles. The behaviour is that of stage 2; the tests are the same. The transliteration was then
deleted from the tree and lives in the history.

## What was not ported

`facet_wise` and `use_parity` are carried through as the original has them. Verbose printing was
dropped. `random_dir_stratified` is not used by the function and was left out.
