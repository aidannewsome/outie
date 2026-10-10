"""What every script shares: a mass read from data/ with its duplicate triangles removed, as the paper's own command
removes them; outie's answer for it; the results table measure.py writes; and laying images out."""

import csv
from pathlib import Path

import numpy as np

import outie
from draw import read_off

DATA = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent.parent / "figures"
RESULTS = Path(__file__).parent.parent / "results"
HEIGHT, GAP = 520, 40  # pixels a row stands, and between pairs, rows and the margin; a pair's two sides a third of it apart


def remove_duplicates(triangles: np.ndarray) -> np.ndarray:
    """Each triangle once, the first of any on the same three corners, either way round: libigl's unique_simplices,
    the paper's -u. A face drawn twice otherwise fights itself in every view and in the rays."""
    _, first = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    return triangles[np.sort(first)]


def score(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Backfacingness, as the paper measures it."""
    return outie.measure_backfacingness(vertices, triangles)


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A mass's corners and triangles, each triangle once."""
    vertices, triangles = read_off(DATA / f"{name}.off")
    return vertices, remove_duplicates(triangles)


def orient(vertices: np.ndarray, triangles: np.ndarray, **settings) -> tuple[np.ndarray, np.ndarray]:
    """The triangles as outie turns them, and the component each belongs to."""
    flip, component = outie.reorient_facets_raycast(vertices, triangles, **settings)
    return np.where(flip[:, None], triangles[:, ::-1], triangles), component


def read_results() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """From results/backfacingness.csv: each method's backfacingness by mass, and each mass's triangles."""
    scores: dict[str, dict[str, float]] = {}
    triangles: dict[str, int] = {}
    with open(RESULTS / "backfacingness.csv") as f:
        for row in csv.DictReader(f):
            scores.setdefault(row["method"], {})[row["mass"]] = float(row["backfacingness"])
            triangles[row["mass"]] = int(row["triangles"])
    return scores, triangles


def find_largest_mixed() -> str:
    """The mass with the most triangles among those wound well both ways, scoring between 0.4 and 0.6 as published."""
    scores, triangles = read_results()
    return max((n for n, score in scores["as published"].items() if 0.4 <= score <= 0.6), key=triangles.get)


def save(image: np.ndarray, name: str) -> None:
    """An image of the figure's own, figures/<name>.png."""
    import matplotlib.pyplot as plt

    path = OUT / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0, 1))  # three bands, or four with transparency
    print(f"drew {path.relative_to(OUT.parent)}")


def arrange_grid(images: list[list[np.ndarray]], gap: int = 40) -> np.ndarray:
    """Images of one size in rows, every gap and the margin the same, on white."""
    h, w = images[0][0].shape[:2]
    out = np.ones((gap + len(images) * (h + gap), gap + len(images[0]) * (w + gap), 3))
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            out[gap + r * (h + gap) : gap + r * (h + gap) + h, gap + c * (w + gap) : gap + c * (w + gap) + w] = image
    return out


def crop(image: np.ndarray) -> np.ndarray:
    """An image cut to what is drawn in it."""
    drawn = np.argwhere((image < 0.995).any(axis=2))
    (top, left), (bottom, right) = drawn.min(axis=0), drawn.max(axis=0) + 1
    return image[top:bottom, left:right]


def scale_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """An image scaled to a height, its width with it, by nearest pixel."""
    rows = (np.arange(height) * image.shape[0] / height).astype(int)
    cols = (np.arange(round(image.shape[1] * height / image.shape[0])) * image.shape[0] / height).astype(int)
    return image[rows][:, cols]
