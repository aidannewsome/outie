"""Six of Toronto's best-known buildings, from the City of Toronto's 3D Massing as OFF files in landmarks/, for the
figure at the top of the README. Each is the mass with the most faces whose plan holds the building's point, as the
city draws one mass a building or a block. Read, cut and placed as extract_toronto.py reads its masses.

    uv run --with pyproj --with mapbox_earcut --with "pyogrio==0.13.0+12.g86d8bb4" \
        --find-links https://github.com/aidannewsome/pyogrio/releases/expanded_assets/maquette-0.13.0-12 \
        python data/extract_landmarks.py 3DMassingMultipatch_2025_WGS84.zip

Stock pyogrio cannot read the tiles of City Hall and the Rogers Centre (geopandas/pyogrio#703), so a build with
that fix reads them.
"""

import sys
from pathlib import Path

import numpy as np
import pyogrio
from extract_toronto import find_geodatabase, make_mesh, read_faces, write_off
from pyproj import Transformer

LANDMARKS = {  # latitude and longitude of a point inside each
    "OCAD University": (43.65310, -79.39140),
    "Royal Ontario Museum": (43.66770, -79.39480),
    "Art Gallery of Ontario": (43.65390, -79.39270),
    "Roy Thomson Hall": (43.64660, -79.38630),
    "City Hall": (43.65340, -79.38410),
    "Rogers Centre": (43.64140, -79.38940),
}
HERE = Path(__file__).parent / "landmarks"

pyogrio.set_gdal_config_options({"OPENFILEGDB_USE_SPATIAL_INDEX": "NO"})


def main(download: str) -> None:
    model = find_geodatabase(download)
    system = pyogrio.read_info(model, layer="Context_Tiles")["crs"]
    to_city = Transformer.from_crs("EPSG:4326", system, always_xy=True)
    to_metres = Transformer.from_crs(system, "EPSG:2952", always_xy=True)
    HERE.mkdir(exist_ok=True)
    for name, (latitude, longitude) in LANDMARKS.items():
        x, y = to_city.transform(longitude, latitude)
        box = (x - 1, y - 1, x + 1, y + 1)
        meta, _, _, values = pyogrio.raw.read(model, layer="Context_Tiles", bbox=box)
        found = []
        for tile in {str(t) for t in values[list(meta["fields"]).index("Tile_Name")]}:
            for wkb in pyogrio.raw.read(model, layer=f"Multipatch_{tile}", bbox=box, columns=[])[2]:
                faces = [f for f in read_faces(wkb) if len(f[0]) >= 3] if wkb is not None else []
                if faces and holds(faces, x, y):
                    found.append(faces)
        vertices, triangles = make_mesh(max(found, key=len), to_metres)
        (HERE / f"{name.lower().replace(' ', '-')}.off").write_text(write_off(vertices, triangles))
        print(f"{name}: {len(triangles)} triangles")


def holds(faces: list[list[np.ndarray]], x: float, y: float) -> bool:
    """Whether a mass's plan, the box round its corners, holds a point."""
    corners = np.vstack([f[0] for f in faces])
    return bool((corners[:, 0].min() <= x <= corners[:, 0].max()) and (corners[:, 1].min() <= y <= corners[:, 1].max()))


if __name__ == "__main__":
    main(sys.argv[1])
