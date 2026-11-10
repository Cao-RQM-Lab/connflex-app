"""Process description files (TOML).

connflex-core takes every parameter as a Python object and reads no files, so the
application parses its own. A file holds the tables ``[stack.device]``, ``[stack.box]``,
``[stack.handle]``, and ``[extraction]``, and optionally ``[solve]``, ``[electrostatics]``,
``[transient]``, and ``[materials.<name>]``. Each stack table gives a ``thickness`` and names
its ``material``: a preset (``silicon`` or ``oxide``) or a ``[materials.<name>]`` table, which
overrides fields of the same-named preset or defines a new material. Unknown keys are errors.
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any

from connflex_core import (
    ElectrostaticParams,
    ExtractionParams,
    Layer,
    Material,
    SolveOptions,
    Stack,
    TransientOptions,
)

PRESETS: dict[str, Callable[..., Material]] = {"silicon": Material.silicon, "oxide": Material.oxide}
ROLES = ("device", "box", "handle")
TABLES = ("stack", "materials", "extraction", "solve", "electrostatics", "transient")


@dataclass(frozen=True)
class ProcessConfig:
    """What one process file sets. Tables the file leaves out are None."""

    stack: Stack
    params: ExtractionParams
    options: SolveOptions
    electrostatics: dict[str, Any] | None = None
    transient: dict[str, Any] | None = None


def load_config(path: str | PathLike[str]) -> ProcessConfig:
    """Read a process file; raises ``ValueError`` naming the table and key on any mistake."""
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    _reject_unknown(data, set(TABLES), "the top level")
    materials = {name: _material(name, table) for name, table in _table(data, "materials", required=False).items()}
    stack_table = _table(data, "stack")
    _reject_unknown(stack_table, set(ROLES), "[stack]")
    layers = {role: _layer(role, _table(stack_table, role, "[stack]"), materials) for role in ROLES}
    params = _build(ExtractionParams, _table(data, "extraction"), "[extraction]")
    options = _build(SolveOptions, _table(data, "solve", required=False), "[solve]")
    electrostatics = transient = None
    if "electrostatics" in data:
        electrostatics = _checked(ElectrostaticParams, _table(data, "electrostatics"), "[electrostatics]")
    if "transient" in data:
        transient = _checked(TransientOptions, _table(data, "transient"), "[transient]", build=False)
    return ProcessConfig(Stack(**layers), params, options, electrostatics, transient)


def _table(data: Mapping[str, Any], key: str, context: str = "the top level", *, required: bool = True) -> dict[str, Any]:
    if key not in data:
        if required:
            raise ValueError(f"{context} is missing the [{key}] table")
        return {}
    table = data[key]
    if not isinstance(table, dict):
        raise ValueError(f"{key} in {context} must be a table")
    return table


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {context}: {', '.join(unknown)}")


def _values(table: Mapping[str, Any]) -> dict[str, Any]:
    return {key: tuple(value) if isinstance(value, list) else value for key, value in table.items()}


def _build(cls, table: Mapping[str, Any], context: str):
    fields = dataclasses.fields(cls)
    _reject_unknown(table, {f.name for f in fields}, context)
    missing = [
        f.name
        for f in fields
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in table
    ]
    if missing:
        raise ValueError(f"{context} is missing {', '.join(missing)}")
    try:
        return cls(**_values(table))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {exc}") from exc


def _checked(cls, table: Mapping[str, Any], context: str, build: bool = True) -> dict[str, Any]:
    """The table's values after the parameter class has accepted them (or its keys alone)."""
    _reject_unknown(table, {f.name for f in dataclasses.fields(cls)}, context)
    values = _values(table)
    if build:
        try:
            cls(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: {exc}") from exc
    return values


def _material(name: str, table: Mapping[str, Any]) -> Material:
    if not isinstance(table, dict):
        raise ValueError(f"[materials.{name}] must be a table")
    context = f"[materials.{name}]"
    _reject_unknown(table, {f.name for f in dataclasses.fields(Material)}, context)
    try:
        if name in PRESETS:
            return PRESETS[name](**table)
        return _build(Material, table, context)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {exc}") from exc


def _layer(role: str, table: Mapping[str, Any], materials: Mapping[str, Material]) -> Layer:
    context = f"[stack.{role}]"
    _reject_unknown(table, {"thickness", "material"}, context)
    for key in ("thickness", "material"):
        if key not in table:
            raise ValueError(f"{context} is missing {key}")
    name = table["material"]
    if name in materials:
        material = materials[name]
    elif name in PRESETS:
        material = PRESETS[name]()
    else:
        known = sorted({*PRESETS, *materials})
        raise ValueError(f"{context}: unknown material {name!r}; known materials are {', '.join(known)}")
    try:
        return Layer(thickness=table["thickness"], material=material)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {exc}") from exc
