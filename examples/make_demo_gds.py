"""Generate demo_resonator.gds: two perforated proof masses on straight beams, with comb electrodes.

Layers: 1/0 device, 2/0 handle, 3/0 release, 4/0 vias. Units: micrometres.
"""

from __future__ import annotations

import pathlib
import sys

import gdstk

DEVICE, HANDLE, RELEASE, VIAS = (1, 0), (2, 0), (3, 0), (4, 0)


def rect(x0: float, y0: float, x1: float, y1: float, layer: tuple[int, int]) -> gdstk.Polygon:
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer[0], datatype=layer[1])


def perforated_plate(x0: float, y0: float, x1: float, y1: float, hole: float = 8.0, pitch: float = 15.0):
    plate = gdstk.rectangle((x0, y0), (x1, y1))
    nx, ny = int((x1 - x0) // pitch), int((y1 - y0) // pitch)
    mx, my = ((x1 - x0) - nx * pitch) / 2, ((y1 - y0) - ny * pitch) / 2
    holes = []
    for i in range(nx):
        for j in range(ny):
            cx, cy = x0 + mx + pitch * (i + 0.5), y0 + my + pitch * (j + 0.5)
            holes.append(gdstk.rectangle((cx - hole / 2, cy - hole / 2), (cx + hole / 2, cy + hole / 2)))
    return gdstk.boolean(plate, holes, "not", layer=DEVICE[0], datatype=DEVICE[1])


def build() -> gdstk.Library:
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("RESONATOR")
    cell.add(rect(0, 0, 1200, 1200, HANDLE))

    for x0 in (250, 650):  # two proof masses
        cell.add(*perforated_plate(x0, 450, x0 + 300, 750))
        for px in (x0 + 25,):  # pads above and below
            cell.add(rect(px, 900, px + 250, 1050, DEVICE))
            cell.add(rect(px, 150, px + 250, 300, DEVICE))
        for bx in (x0 + 48, x0 + 248):  # four suspension beams, 4 µm wide, 150 µm long
            cell.add(rect(bx, 750, bx + 4, 900, DEVICE))
            cell.add(rect(bx, 300, bx + 4, 450, DEVICE))
    cell.add(rect(550, 598, 650, 602, DEVICE))  # coupling beam

    # Comb drives: electrode bodies with stub fingers interdigitated with mass fingers.
    cell.add(rect(120, 480, 180, 720, DEVICE))
    cell.add(rect(1020, 480, 1080, 720, DEVICE))
    for k in range(18):
        y = 500 + 12 * k
        cell.add(rect(180, y, 230, y + 3, DEVICE))
        cell.add(rect(200, y + 6, 250, y + 9, DEVICE))
        cell.add(rect(970, y, 1020, y + 3, DEVICE))
        cell.add(rect(950, y + 6, 1000, y + 9, DEVICE))

    # A thin T-shaped tether touching three bodies: rejected as a beam, shown as such.
    cell.add(rect(560, 150, 640, 300, DEVICE))
    cell.add(rect(550, 460, 650, 462, DEVICE))
    cell.add(rect(599, 300, 601, 460, DEVICE))

    cell.add(rect(135, 580, 165, 620, VIAS))  # left electrode shorted to the handle
    cell.add(rect(560, 560, 640, 640, RELEASE))
    return lib


def main(argv: list[str]) -> int:
    out = pathlib.Path(argv[1] if len(argv) > 1 else pathlib.Path(__file__).with_name("demo_resonator.gds"))
    build().write_gds(str(out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
