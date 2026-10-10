"""How the method decides, in three isometric drawings of see-through buildings in a row, each titled with its
setting and rule, and one legend beneath for them all: figures/method.svg.

    uv run python scripts/draw_method.py
"""

from math import cos, radians, sin
from pathlib import Path

BEFORE, AFTER = "#d1382b", "#2e9e4f"  # a face's front as published, and as outie turns it
GOLD, GOLD_SIDE, GREY, INK, RAY, QUIET = "#e7ae36", "#c98f1f", "#55575a", "#1b1c1e", "#d1382b", "#8a8a85"
SCALE = 26  # pixels a unit


def iso(x: float, y: float, z: float, origin: tuple[float, float]) -> tuple[float, float]:
    """A point seen isometrically, x running down to the right, y down to the left, z up."""
    return origin[0] + (x - y) * cos(radians(30)) * SCALE, origin[1] + ((x + y) * sin(radians(30)) - z) * SCALE


def polygon(points, origin, fill, opacity=1.0, width=1.4) -> str:
    corners = " ".join(f"{a:.1f},{b:.1f}" for a, b in (iso(*p, origin) for p in points))
    return f'<polygon points="{corners}" fill="{fill}" fill-opacity="{opacity}" stroke="{INK}" stroke-width="{width}" stroke-linejoin="round"/>'


def segment(a, b, origin, dash="") -> str:
    (x1, y1), (x2, y2) = iso(*a, origin), iso(*b, origin)
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{RAY}" stroke-width="2.2" stroke-linecap="round"{dashes}/>'


def start(p, origin) -> str:
    """Where a ray starts: a ring."""
    x, y = iso(*p, origin)
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="white" stroke="{INK}" stroke-width="2"/>'


def hit(p, origin) -> str:
    """Where a ray hits a face: a cross."""
    x, y = iso(*p, origin)
    return f'<path d="M{x - 5:.1f},{y - 5:.1f} L{x + 5:.1f},{y + 5:.1f} M{x - 5:.1f},{y + 5:.1f} L{x + 5:.1f},{y - 5:.1f}" stroke="{RAY}" stroke-width="2.6" stroke-linecap="round"/>'


def text(x, y, words, size=15, colour=INK, anchor="start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" fill="{colour}" text-anchor="{anchor}">{words}</text>'


def arrow(a, b, origin, colour) -> str:
    """A face's front, the way it points."""
    (x1, y1), (x2, y2) = iso(*a, origin), iso(*b, origin)
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    head = [(x2, y2), (x2 - 11 * ux + 6 * uy, y2 - 11 * uy - 6 * ux), (x2 - 11 * ux - 6 * uy, y2 - 11 * uy + 6 * ux)]
    shaft = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2 - 8 * ux:.1f}" y2="{y2 - 8 * uy:.1f}" stroke="{colour}" stroke-width="3.5"/>'
    return shaft + f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in head)}" fill="{colour}"/>'


def box(x0, y0, z0, x1, y1, z1, origin, opacity=0.18, grey=None, colour=None) -> str:
    """A see-through box: its three far faces first, then its three near ones, the named face grey."""
    faces = {
        "-x": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        "-y": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        "-z": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        "+x": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        "+y": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        "+z": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
    }
    shade = {"+x": GOLD_SIDE, "+y": GOLD, "+z": GOLD, "-x": GOLD, "-y": GOLD_SIDE, "-z": GOLD} if colour is None else dict.fromkeys(("+x", "+y", "+z", "-x", "-y", "-z"), colour)
    out = []
    for name in ("-x", "-y", "-z", "+x", "+y", "+z"):
        is_grey = name == grey
        out.append(polygon(faces[name], origin, GREY if is_grey else shade[name], 0.55 if is_grey else opacity, 1.2 if name[0] == "-" else 1.6))
    return "".join(out)


def escape(origin) -> str:
    """A wall whose front faces in: its front rays hit the far wall, its back rays escape, so it is turned round."""
    out = [box(0, 0, 0, 7, 4, 4, origin, grey="+x")]
    for begin, far in [((7, 1.0, 1.0), (0, 2.6, 1.6)), ((7, 2.0, 2.6), (0, 0.8, 3.4)), ((7, 3.0, 1.6), (0, 3.6, 0.6))]:
        beyond = tuple(s - 0.7 * (f - s) for s, f in zip(begin, far))
        out += [segment(begin, far, origin), hit(far, origin), segment(begin, beyond, origin, "7 5"), start(begin, origin)]
    out.append(arrow((7, 3.6, 0.4), (5.4, 3.6, 0.4), origin, BEFORE))
    out.append(arrow((7, 3.6, 0.4), (9.0, 3.6, 0.4), origin, AFTER))
    return "".join(out)


def interior(origin) -> str:
    """A wall inside a closed room: no ray escapes, so the side whose rays go further is the outside."""
    out = [box(0, 0, 0, 7, 4, 4, origin, opacity=0.12)]
    out.append(polygon([(5.4, 0.4, 0.4), (5.4, 3.6, 0.4), (5.4, 3.6, 3.6), (5.4, 0.4, 3.6)], origin, GREY, 0.55))
    for begin, near, far in [((5.4, 1.2, 1.2), (7, 1.4, 1.4), (0, 0.5, 0.4)), ((5.4, 2.4, 2.6), (7, 2.2, 2.8), (0, 3.2, 1.8)), ((5.4, 2.0, 1.8), (7, 2.2, 1.6), (0, 1.3, 2.9))]:
        out += [segment(begin, near, origin), hit(near, origin), segment(begin, far, origin, "7 5"), hit(far, origin), start(begin, origin)]
    out.append(arrow((5.4, 2.0, 0.8), (6.6, 2.0, 0.8), origin, BEFORE))
    out.append(arrow((5.4, 2.0, 0.8), (3.8, 2.0, 0.8), origin, AFTER))
    return "".join(out)


def parity(origin) -> str:
    """A closed block, one wall's front facing in. From a point on it, a ray each way, every face it passes counted:
    inward it passes the far wall, one, odd, so that side is inside the block; outward none, even, so that is outside."""
    out = [box(0, 0, 0, 7, 4, 4, origin, grey="+x")]
    begin = (7, 2.0, 2.0)
    out += [segment(begin, (-3.0, 2.0, 2.0), origin), hit((0, 2.0, 2.0), origin), segment(begin, (11.5, 2.0, 2.0), origin, "7 5"), start(begin, origin)]
    out.append(arrow((7, 3.6, 0.4), (5.4, 3.6, 0.4), origin, BEFORE))
    out.append(arrow((7, 3.6, 0.4), (9.0, 3.6, 0.4), origin, AFTER))
    return "".join(out)


PANELS = [
    (escape, "Default: more rays escape outside"),
    (interior, "Default, closed room: rays go further outside"),
    (parity, "use_parity=True: odd crossings mean inside"),
]
LEGEND = [
    ("grey", "a face showing its back"), ("gold", "a face showing its front"),
    ("ray", "a ray off the front"), ("dashed", "the same ray off the back"), ("start", "where a ray starts"), ("hit", "where a ray hits a face"),
    ("before", "a face's front, as published"), ("after", "and as outie turns it"),
]


def key(kind: str, x: float, y: float) -> str:
    """A legend's mark."""
    if kind in ("grey", "gold"):
        return f'<rect x="{x}" y="{y - 12}" width="22" height="16" fill="{ {"grey": GREY, "gold": GOLD}[kind] }" stroke="{INK}"/>'
    if kind in ("ray", "dashed"):
        dashes = ' stroke-dasharray="6 4"' if kind == "dashed" else ""
        return f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" stroke="{RAY}" stroke-width="2.2"{dashes}/>'
    if kind in ("before", "after"):
        colour = BEFORE if kind == "before" else AFTER
        return f'<line x1="{x}" y1="{y - 4}" x2="{x + 16}" y2="{y - 4}" stroke="{colour}" stroke-width="3.5"/><polygon points="{x + 22},{y - 4} {x + 13},{y - 9} {x + 13},{y + 1}" fill="{colour}"/>'
    if kind == "start":
        return f'<circle cx="{x + 11}" cy="{y - 4}" r="5" fill="white" stroke="{INK}" stroke-width="2"/>'
    if kind == "hit":
        return f'<path d="M{x + 6},{y - 9} L{x + 16},{y + 1} M{x + 6},{y + 1} L{x + 16},{y - 9}" stroke="{RAY}" stroke-width="2.6" stroke-linecap="round"/>'
    return f'<circle cx="{x + 11}" cy="{y - 4}" r="10" fill="{INK}"/>' + text(x + 11, y + 1, kind, 13, "white", "middle")


WIDTH, TOP, DRAWING, ROW = 640, 56, 360, 28


def svg(width: float, height: float, body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}"><rect width="{width}" height="{height}" fill="white"/>{body}</svg>\n'


def diagram() -> str:
    """The three drawings in a row, each titled, and one legend for them all beneath, in three columns."""
    body = []
    for n, (draw, title) in enumerate(PANELS):
        x, y = n * WIDTH, 0
        body.append(f'<g transform="translate({x},{y})">{text(24, 36, title, 16)}<g transform="translate(0,{TOP})">{draw((300, 110))}</g></g>')
    below = TOP + DRAWING + 10
    third = (len(LEGEND) + 2) // 3
    for n, (kind, words) in enumerate(LEGEND):
        x, y = 24 + (n // third) * WIDTH, below + ROW * (n % third) + 18
        body += [key(kind, x, y), text(x + 32, y, words, 14)]
    return svg(3 * WIDTH, below + ROW * third + 16, "".join(body))


if __name__ == "__main__":
    (Path(__file__).parent.parent / "figures" / "method.svg").write_text(diagram())
    print("drew figures/method.svg")
