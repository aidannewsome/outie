# Outie

A mesh whose faces all point out, like an outie, even when the mesh is not closed.

A Python port of `reorient_facets_raycast` from [libigl](https://libigl.github.io), which implements
Takayama, Jacobson, Kavan and Sorkine-Hornung, *A Simple Method for Correcting Facet Orientations in
Polygon Meshes Based on Ray Casting*, Journal of Computer Graphics Techniques 3(4), 2014
([paper](https://jcgt.org/published/0003/04/02/paper.pdf), [project page](https://igl.ethz.ch/projects/facet-orientation/)).

Hand-made meshes, SketchUp models, CAD exports and city models rarely close. Faces are drawn one at
a time, from both sides, and nobody makes the normals agree. Tests that work on closed surfaces, a
signed volume or a winding number, have nothing to say about a wall with fins in front of it or a
courtyard. This method does not need a closed surface. It asks a simpler question of every patch of
faces: from which side do rays escape?

## How it works

1. Faces are gathered into patches that agree across shared edges. Winding is spread from face to
   neighbour by breadth-first search, and never across an edge that more than two faces share.
2. Rays are shot from random points on each patch, spread by area, in random directions off the
   front and off the back of the face they start on.
3. The side from which more rays escape to infinity is the outside. When the two sides tie, the
   side whose rays travel further before hitting anything. The patch is flipped if that is its back.

Rays are traced by [Embree](https://www.embree.org) through [trimesh](https://trimesh.org) and
[embreex](https://github.com/trimesh/embreex).

## Use

```python
import outie

flip, patch = outie.reorient_facets_raycast(vertices, faces)  # triangles, n by 3 and m by 3
faces = outie.reoriented(vertices, faces)                     # the same faces, turned as needed
polygons = outie.orient(polygons)                              # a list of rings, n by 3 each
```

The parameters are libigl's, with its defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `rays_total` | 100 per face | rays shot over the whole mesh, shared out to patches by area |
| `rays_minimum` | 10 | rays every patch gets, however small |
| `facet_wise` | `False` | judge every face alone rather than by patch |
| `use_parity` | `False` | vote by the parity of hits along each ray instead of escapes and distance |

## What it is not

It does not close holes, weld vertices, or remove duplicate faces. A mesh that is a soup goes in as a
soup and comes out as a soup whose faces point out. It is not a substitute for a signed volume
test on a mesh that is watertight, where that test is exact and free; it is what to reach for when
that test cannot be applied.

## Licence

MPL-2.0, as libigl is. This is a port of libigl's reference implementation, line for line where the
languages allow, and so a derivative of it. The method is the paper's authors'; the port is
Aidan Newsome's.
