"""Toronto's worst-wound building masses, extracted from the City of Toronto's 3D Massing as OFF files, one mass a file.

The city models each mass from SketchUp and BIM drawings, whose faces point whichever way they were left. Every mass in
the boxes below is read from the city's geodatabase, its faces cut into triangles wound as the faces are, and scored by
the paper's backfacingness as published. The worst are the most mixed, whose score is furthest from both nought and
one: a mass wound all inward, scoring one, is put right by turning it whole, while one wound both ways is the hard case.
The fifty most mixed are written here, and with them the ten masses with the most triangles, the likeliest to keep
the floors and furniture a modeller left in, for cut-aways. A mass is named
by the city's tile layer and its place in it, as the city gives masses no ids of their own. The city publishes in Web
Mercator, which stretches Toronto's plan by about 1.38; coordinates here are true metres in MTM zone 10 (EPSG:2952),
Toronto's own system, heights as the city gives them, shifted so each mass's lowest corner is at nought.

    uv run --with pyogrio --with pyproj --with mapbox_earcut python data/extract_toronto.py 3DMassingMultipatch_2025_WGS84.zip

The geodatabase is the city's download, https://open.toronto.ca/dataset/3d-massing/, licensed as LICENSE says.
"""

import struct
import sys
import zipfile
from pathlib import Path

import mapbox_earcut
import numpy as np
import pyogrio
from pyproj import Transformer

import outie

BOXES = {  # longitude and latitude: the Financial District and Kensington Market
    "financial-district": (-79.392, 43.645, -79.376, 43.656),
    "kensington-market": (-79.411, 43.651, -79.397, 43.659),
}
KEEP = 50  # masses kept, the worst first
LARGEST_KEPT = 10  # masses kept for having the most triangles
SMALLEST = 300  # triangles a mass must have to be kept: a house of a few boxes says little
LARGEST = 1_500_000  # bytes one mass may take
HERE = Path(__file__).parent
METRES = "EPSG:2952"  # NAD83(CSRS) MTM zone 10, the city's survey system

pyogrio.set_gdal_config_options({"OPENFILEGDB_USE_SPATIAL_INDEX": "NO"})  # GDAL 3.10's index finds nothing in this geodatabase


def main(download: str) -> None:
    model = find_geodatabase(download)
    system = pyogrio.read_info(model, layer="Context_Tiles")["crs"]
    to_city = Transformer.from_crs("EPSG:4326", system, always_xy=True)
    to_metres = Transformer.from_crs(system, METRES, always_xy=True)
    scored, seen = [], set()
    for place, (west, south, east, north) in BOXES.items():
        box = (*to_city.transform(west, south), *to_city.transform(east, north))
        meta, _, _, values = pyogrio.raw.read(model, layer="Context_Tiles", bbox=box)
        tiles = [str(t) for t in values[list(meta["fields"]).index("Tile_Name")]]
        for tile in tiles:
            try:
                _, fids, drawn, _ = pyogrio.raw.read(model, layer=f"Multipatch_{tile}", bbox=box, columns=[], return_fids=True)
            except pyogrio.errors.DataLayerError:
                continue  # the index names one tile that has no layer
            for fid, wkb in zip(fids, drawn):
                name = f"{tile}-{int(fid):04d}"
                if wkb is None or name in seen:
                    continue
                seen.add(name)
                vertices, triangles = make_mesh([f for f in read_faces(wkb) if len(f[0]) >= 3], to_metres)
                if len(triangles):
                    scored.append((outie.measure_backfacingness(vertices, triangles), name, vertices, triangles))
        print(f"{place}: {len(scored)} masses scored")
    scored.sort(key=lambda s: -min(s[0], 1 - s[0]))
    for old in HERE.glob("*.off"):
        old.unlink()
    kept = 0
    chosen = set()
    for score, name, vertices, triangles in sorted(scored, key=lambda s: -len(s[3])):
        text = write_off(vertices, triangles)
        if len(text) > LARGEST:
            continue
        (HERE / f"{name}.off").write_text(text)
        print(f"{name}: {len(triangles)} triangles, backfacingness {score:.4f}")
        chosen.add(name)
        if len(chosen) == LARGEST_KEPT:
            break
    for score, name, vertices, triangles in scored:
        if name in chosen:
            continue
        text = write_off(vertices, triangles)
        if len(triangles) < SMALLEST or len(text) > LARGEST:
            continue
        (HERE / f"{name}.off").write_text(text)
        print(f"{name}: backfacingness {score:.4f}, {len(triangles)} triangles")
        kept += 1
        if kept == KEEP:
            break


def find_geodatabase(download: str) -> str:
    """The geodatabase inside the city's zip, as GDAL opens it in place."""
    for entry in zipfile.ZipFile(download).namelist():
        if ".gdb/" in entry:
            return f"/vsizip/{Path(download).resolve()}/{entry.split('.gdb/')[0]}.gdb"
    raise ValueError(f"{download} holds no geodatabase.")


def read_faces(wkb: bytes) -> list[list[np.ndarray]]:
    """Every polygon of a geometry as a face, its outer ring then its holes, read straight from GDAL's WKB, so a ring
    too short for shapely is kept as the city drew it."""
    found: list[list[np.ndarray]] = []

    def read(at: int) -> int:
        order = "<" if wkb[at] == 1 else ">"
        (code,) = struct.unpack_from(order + "I", wkb, at + 1)
        at += 5
        kind = code % 1000 if code < 0x80000000 else code & 0xFF
        z, m = code // 1000 in (1, 3) or code >= 0x80000000, code // 1000 in (2, 3)
        width = 2 + z + m

        def run(at: int) -> tuple[np.ndarray, int]:
            (count,) = struct.unpack_from(order + "I", wkb, at)
            corners = np.frombuffer(wkb, dtype=order + "f8", count=count * width, offset=at + 4).reshape(count, width)
            return corners, at + 4 + count * width * 8

        if kind == 1:
            return at + width * 8
        if kind == 2:
            return run(at)[1]
        (count,) = struct.unpack_from(order + "I", wkb, at)
        at += 4
        if kind not in (3, 17):  # a multi, a collection, a polyhedral surface or a TIN: each part in turn; 17 is a triangle, a polygon
            for _ in range(count):
                at = read(at)
            return at
        rings = []
        for _ in range(count):
            corners, at = run(at)
            points = np.column_stack([corners[:, :2], corners[:, 2] if z else np.zeros(len(corners))])
            rings.append(points[:-1] if len(points) > 1 and (points[0] == points[-1]).all() else points)
        found.append(rings)
        return at

    read(0)
    return found


def make_mesh(polygons: list[list[np.ndarray]], to_metres: Transformer) -> tuple[np.ndarray, np.ndarray]:
    """Faces in metres, cut into triangles wound as each face is, by earcut in the plane each lies flattest in, as one
    mesh whose corners are shared and whose lowest corner is at nought."""
    corners, triangles = [], []
    count = 0
    for rings in polygons:
        rings = [np.column_stack([*to_metres.transform(r[:, 0], r[:, 1]), r[:, 2]]) for r in rings]
        points = np.vstack(rings)
        normal = sum(np.cross(r, np.roll(r, -1, axis=0)).sum(axis=0) for r in rings)
        if not normal.any():
            continue
        flat = np.delete(points, int(np.argmax(np.abs(normal))), axis=1)
        cut = mapbox_earcut.triangulate_float64(flat, np.cumsum([len(r) for r in rings]).astype(np.uint32)).reshape(-1, 3)
        if not len(cut):
            continue
        a, b, c = points[cut[:, 0]], points[cut[:, 1]], points[cut[:, 2]]
        cut = np.where((np.cross(b - a, c - a) @ normal < 0)[:, None], cut[:, ::-1], cut)
        corners.append(points)
        triangles.append(cut + count)
        count += len(points)
    if not triangles:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    points = np.vstack(corners)
    points = points - points.min(axis=0)
    shared, place = np.unique(np.round(points, 3), axis=0, return_inverse=True)
    return shared, place.reshape(-1)[np.vstack(triangles)]


def write_off(vertices: np.ndarray, triangles: np.ndarray) -> str:
    """A mesh as an OFF file, to the millimetre."""
    lines = ["OFF", f"{len(vertices)} {len(triangles)} 0"]
    lines += [f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in vertices]
    lines += [f"3 {a} {b} {c}" for a, b, c in triangles]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(sys.argv[1])
