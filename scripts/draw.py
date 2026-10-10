"""A small renderer for outie's figures, as the paper draws its own: each face gold where it shows its front and dark
grey where it shows its back, lit from above, its triangles' edges thin, seen orthographically from above at an angle.
Which triangle each pixel shows is found with a depth buffer, so nothing is drawn in front of what hides it."""

from pathlib import Path

import numpy as np

GOLD, GREY = np.array([0.96, 0.72, 0.20]), np.array([0.30, 0.30, 0.30])
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """An OFF file's corners and triangles."""
    words = Path(path).read_text().split()
    vertices, triangles = int(words[1]), int(words[2])
    numbers = np.array(words[4 : 4 + 3 * vertices], dtype=float).reshape(-1, 3)
    faces = np.array(words[4 + 3 * vertices :], dtype=np.int64).reshape(triangles, 4)[:, 1:]
    return numbers, faces


def aim_camera(elevation: float = 30, azimuth: float = -55) -> np.ndarray:
    """The rows of a view: right, up and toward the viewer, for a camera looking down at an angle."""
    e, a = np.radians(elevation), np.radians(azimuth)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross([0, 0, 1], toward)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(toward, right), toward])


def find_visible(vertices: np.ndarray, triangles: np.ndarray, view: np.ndarray, size: int, plane: tuple | None = None) -> np.ndarray:
    """Per pixel, the triangle it shows, -1 for none, rows from the top: an orthographic view filling the image. Given
    a plane, a point and a normal, everything on the side the normal points to is clipped away, pixel by pixel."""
    seen = vertices @ view.T
    side = (vertices - plane[0]) @ plane[1] if plane else np.zeros(len(vertices))
    framed = seen[side <= 0] if (side <= 0).any() else seen  # a cut-away frames what is left of the mesh
    low, high = framed[:, :2].min(axis=0), framed[:, :2].max(axis=0)
    scale = 0.92 * size / (high - low).max()
    middle = (low + high) / 2
    x = (seen[:, 0] - middle[0]) * scale + size / 2
    y = size / 2 - (seen[:, 1] - middle[1]) * scale
    depth, shown = np.full((size, size), -np.inf), np.full((size, size), -1)
    tie = 1e-6 * (seen[:, 2].max() - seen[:, 2].min() + 1)  # nearer than this is a tie: faces drawn twice on one plane show the first whole, not speckled
    for t, (i, j, k) in enumerate(triangles):
        xs, ys = x[[i, j, k]], y[[i, j, k]]
        x0, x1 = max(int(xs.min()), 0), min(int(np.ceil(xs.max())) + 1, size)
        y0, y1 = max(int(ys.min()), 0), min(int(np.ceil(ys.max())) + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0])
        if area == 0:
            continue
        px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        a = ((xs[1] - px) * (ys[2] - py) - (ys[1] - py) * (xs[2] - px)) / area
        b = ((xs[2] - px) * (ys[0] - py) - (ys[2] - py) * (xs[0] - px)) / area
        c = 1 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        z = a * seen[i, 2] + b * seen[j, 2] + c * seen[k, 2]
        kept = a * side[i] + b * side[j] + c * side[k] <= 0
        window = depth[y0:y1, x0:x1]
        nearer = inside & kept & (z > window + tie)
        window[nearer] = z[nearer]
        shown[y0:y1, x0:x1][nearer] = t
    return shown


def draw(vertices: np.ndarray, triangles: np.ndarray, view: np.ndarray | None = None, size: int = 900, colours: np.ndarray | None = None, plane: tuple | None = None) -> np.ndarray:
    """A mesh drawn as the paper draws it, size by size by 3: gold fronts and grey backs, or each triangle's own colour
    darkened where it shows its back, on white, its edges thin, clipped by a plane where one is given. Drawn at twice
    the size and halved, for smooth edges."""
    view = aim_camera() if view is None else view
    big = 2 * size
    shown = find_visible(vertices, triangles, view, big, plane)
    a, b, c = (vertices[triangles[:, n]] for n in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    front = normal @ view[2] > 0
    light = 0.55 + 0.45 * np.abs(normal @ LIGHT)
    base = np.where(front[:, None], GOLD, GREY) if colours is None else np.where(front[:, None], colours, colours * 0.35)
    shade = base * light[:, None]
    image = np.ones((big, big, 3))
    drawn = shown >= 0
    image[drawn] = shade[shown[drawn]]
    edge = np.zeros_like(drawn)
    edge[:-1] |= shown[:-1] != shown[1:]
    edge[:, :-1] |= shown[:, :-1] != shown[:, 1:]
    image[edge & drawn] *= 0.72
    return image.reshape(size, 2, size, 2, 3).mean(axis=(1, 3))


def make_clipping_plane(vertices: np.ndarray, view: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A vertical plane through the mesh's middle, square to the viewer, clipping away the half nearer them: a cut-away."""
    view = aim_camera() if view is None else view
    toward = view[2] * [1, 1, 0]
    return (vertices.min(axis=0) + vertices.max(axis=0)) / 2, toward / np.linalg.norm(toward)
