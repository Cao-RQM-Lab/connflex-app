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


def fmt_capacitance(farads: float | None, digits: int = 4) -> str:
    return fmt_si(farads, "F", digits)


_PARSE_PREFIXES = {"": 1.0, "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}


def parse_si(text: str, unit: str = "") -> float:
    """Number a user typed, with an optional SI prefix and unit: ``"180 µs"``, ``"40"``, ``"2.5k"``.

    ``unit`` is the unit the field expects; it may be typed or left out. Raises ``ValueError``
    with a message for the user when the text is not a number in that form.
    """
    raw = text.strip().replace(",", "").replace("−", "-")
    if not raw:
        raise ValueError("enter a number")
    split = len(raw)
    while split > 0 and not (raw[split - 1].isdigit() or raw[split - 1] == "."):
        split -= 1
    number, suffix = raw[:split].strip(), raw[split:].strip()
    try:
        value = float(number)
    except ValueError:
        raise ValueError(f"{text.strip()!r} is not a number") from None
    for candidate in (unit, unit.lower()) if unit else ():
        if candidate and suffix.endswith(candidate):
            suffix = suffix[: -len(candidate)]
            break
    if suffix not in _PARSE_PREFIXES:
        expected = f"a number, optionally with a prefix such as µ, m, or k{' and the unit ' + unit if unit else ''}"
        raise ValueError(f"{text.strip()!r} is not understood; enter {expected}")
    value *= _PARSE_PREFIXES[suffix]
    if not math.isfinite(value):
        raise ValueError(f"{text.strip()!r} is not a finite number")
    return value
