"""Colour tables over paint groups, shared by the 2D and the 3D view.

A table has one RGBA row per paint group (see :class:`connflex_app.geometry.Groups`). The
views mix each row's colour over the base colour of the silicon by the row's alpha, so a row
with alpha 0 leaves its group as the layer colour. Tables are tiny, and both views read them
from a texture, so recolouring by net, by selection, or by voltage costs no geometry.
"""

from __future__ import annotations

import colorsys

import numpy as np
from numpy.typing import NDArray

from connflex_app.geometry import Groups

Table = NDArray[np.float32]

PICK_A = (0.984, 0.749, 0.141)  # amber: the first of two selected members
PICK_B = (0.133, 0.827, 0.933)  # cyan: the second
NEUTRAL = (0.42, 0.45, 0.52)  # silicon at zero volts
POSITIVE = (0.96, 0.33, 0.24)
NEGATIVE = (0.25, 0.55, 0.98)
FLOATING = (0.66, 0.50, 0.94)
ANCHOR = (0.45, 0.50, 0.60)
HANDLE_SHADE = 0.72  # the handle wears a darker shade of its net's colour


def categorical(index: int, saturation: float = 0.62, lightness: float = 0.6) -> tuple[float, float, float]:
    """Well-separated colour for category ``index`` from a golden-angle hue walk."""
    hue = (index * 0.618033988749895 + 0.11) % 1.0
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def _palette(count: int, anchor_first: bool = False) -> NDArray[np.float32]:
    colors = np.array([categorical(k) for k in range(max(count, 1))], dtype=np.float32)
    if anchor_first and count:
        colors[0] = ANCHOR
    return colors


def blank(groups: Groups) -> Table:
    """A table that leaves every group in its layer colour."""
    return np.zeros((groups.count, 4), dtype=np.float32)


def by_category(groups: Groups, kind: str, weight: float = 0.92) -> Table:
    """Each group in the colour of its net, body, or electrode.

    Beams and bridges belong to no body and no electrode, so they keep the layer colour
    unless the table is by net.
    """
    table = blank(groups)
    if groups.count == 0:
        return table
    electrodes = groups.electrodes
    if kind == "net":
        table[:, :3] = _palette(groups.nets)[groups.net]
        table[:, 3] = weight
    elif kind == "body":
        bodies = groups.body[:electrodes]
        if bodies.size:
            table[:electrodes, :3] = _palette(int(bodies.max()) + 1, anchor_first=True)[bodies]
            table[:electrodes, 3] = weight
    elif kind == "electrode":
        table[:electrodes, :3] = _palette(electrodes)
        table[:electrodes, 3] = weight
    else:
        raise ValueError(f"unknown category {kind!r}")
    table[groups.handle, :3] *= HANDLE_SHADE
    return table


def with_picks(table: Table, first=(), second=(), hover=(), dim: float = 0.4) -> Table:
    """``table`` with the groups of two selected members in amber and cyan and a hovered
    member lightened. With a selection, everything else fades toward the layer colour."""
    result = table.copy()
    first = [g for g in first if 0 <= g < len(result)]
    second = [g for g in second if 0 <= g < len(result)]
    hover = [g for g in hover if 0 <= g < len(result)]
    if first or second:
        result[:, 3] *= dim
    if hover:
        result[hover, :3] = result[hover, :3] + (1.0 - result[hover, :3]) * 0.35
        result[hover, 3] = np.maximum(result[hover, 3], 0.75)
    if second:
        result[second] = (*PICK_B, 1.0)
    if first:
        result[first] = (*PICK_A, 1.0)
    return result


def voltage_scale(values) -> float:
    """Largest voltage of ``values`` in magnitude, or 1 when all are zero."""
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return peak if np.isfinite(peak) and peak > 0 else 1.0


def voltage_color(value: float, scale: float) -> tuple[float, float, float]:
    """Colour of one potential on the diverging map: cool below zero, warm above."""
    level = float(np.clip(value / scale, -1.0, 1.0)) if scale > 0 else 0.0
    target = POSITIVE if level >= 0 else NEGATIVE
    strength = abs(level) ** 0.6  # keep small voltages visible
    return tuple(n + (t - n) * strength for n, t in zip(NEUTRAL, target))  # type: ignore[return-value]


def group_potentials(groups: Groups, electrodes, nets) -> NDArray[np.float64]:
    """Potential of every group from potentials per electrode and per net: an electrode shows
    its own, the beams of a net show the net's."""
    return np.concatenate([np.asarray(electrodes, dtype=np.float64), np.asarray(nets, dtype=np.float64)])


def by_voltage(groups: Groups, potentials, scale: float, floating=(), weight: float = 0.95) -> Table:
    """Each group coloured by its potential on the diverging map.

    ``potentials`` holds one value per group (see :func:`group_potentials`), or one per net,
    which every group of the net then shares. Nets in ``floating`` get a tint of their own
    blended with their potential, since a floating net follows the bodies.
    """
    table = blank(groups)
    if groups.count == 0:
        return table
    potentials = np.asarray(potentials, dtype=np.float64)
    if len(potentials) != groups.count:
        potentials = potentials[groups.net]
    level = np.clip(potentials / scale if scale > 0 else np.zeros_like(potentials), -1.0, 1.0)
    strength = (np.abs(level) ** 0.6)[:, None]
    target = np.where(level[:, None] >= 0, np.array(POSITIVE)[None, :], np.array(NEGATIVE)[None, :])
    colors = np.array(NEUTRAL)[None, :] + (target - np.array(NEUTRAL)[None, :]) * strength
    floats = np.isin(groups.net, np.asarray(list(floating), dtype=np.intp))
    colors[floats] = 0.5 * colors[floats] + 0.5 * np.array(FLOATING)
    table[:, :3] = colors
    table[:, 3] = weight
    table[groups.handle, :3] *= HANDLE_SHADE
    return table


def with_outline(table: Table, selected=(), hover=()) -> Table:
    """``table`` with a selected member lightened a little and a hovered one a little more.

    The lightening is slight on purpose: where colour means voltage, a strong highlight would
    read as another voltage.
    """
    result = table.copy()
    for groups_, amount in ((selected, 0.15), (hover, 0.3)):
        picked = [int(g) for g in groups_ if 0 <= g < len(result)]
        if picked:
            result[picked, :3] = result[picked, :3] + (1.0 - result[picked, :3]) * amount
            result[picked, 3] = 1.0
    return result
