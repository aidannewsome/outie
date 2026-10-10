"""Component by component against face by face, as the paper weighs them: each component its own colour and backs darker. First the
mass face by face does better on, then the one components do better on, by the results, each decided both ways, side by
side: figures/components.png.

    uv run --group figures python scripts/draw_components.py
"""

import numpy as np

from draw import draw
from load import arrange_grid, load, orient, read_results, save


def draw_both_ways(name: str, rng: np.random.Generator) -> list[np.ndarray]:
    vertices, triangles = load(name)
    by_component, component = orient(vertices, triangles, facet_wise=False)
    by_face, _ = orient(vertices, triangles)
    component_colours = (0.55 + 0.45 * rng.random((component.max() + 1, 3)))[component]
    face_colours = 0.55 + 0.45 * rng.random((len(triangles), 3))
    return [draw(vertices, by_component, size=600, colours=component_colours), draw(vertices, by_face, size=600, colours=face_colours)]


if __name__ == "__main__":
    scores = read_results()[0]
    components, faces = scores["outie, components"], scores["outie"]
    faces_win, components_win = max(components, key=lambda n: components[n] - faces[n]), max(components, key=lambda n: faces[n] - components[n])
    rng = np.random.default_rng(1)
    save(arrange_grid([draw_both_ways(faces_win, rng) + draw_both_ways(components_win, rng)]), "components")
