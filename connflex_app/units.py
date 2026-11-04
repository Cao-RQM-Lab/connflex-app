"""Human-friendly number formatting with SI prefixes."""

from __future__ import annotations

import math

_PREFIXES = (
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)


def fmt_si(value: float | None, unit: str = "", digits: int = 3) -> str:
    """Format ``value`` with an SI prefix, e.g. ``fmt_si(1.5e-6, "m")`` gives ``1.5 µm``."""
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0:
        return f"0 {unit}".strip()
    magnitude = abs(value)
    for factor, prefix in _PREFIXES:
        if magnitude >= factor * 0.9995:
            return f"{value / factor:.{digits}g} {prefix}{unit}".strip()
    factor, prefix = _PREFIXES[-1]
    return f"{value / factor:.{digits}g} {prefix}{unit}".strip()


def fmt_length(metres: float | None) -> str:
    return fmt_si(metres, "m")


def fmt_area(square_metres: float | None) -> str:
    if square_metres is None or not math.isfinite(square_metres):
        return "—"
    return f"{square_metres * 1e12:.4g} µm²"


def fmt_freq(hertz: float | None) -> str:
    return fmt_si(hertz, "Hz", 4)


def fmt_mass(kilograms: float | None) -> str:
    return fmt_si(None if kilograms is None else kilograms * 1e3, "g")


def fmt_stiffness(value: float | None, rotational: bool = False) -> str:
    return fmt_si(value, "N·m/rad" if rotational else "N/m")


def fmt_point(xyz, unit: float = 1e6) -> str:
    return "(" + ", ".join(f"{float(v) * unit:.1f}" for v in xyz) + ") µm"
