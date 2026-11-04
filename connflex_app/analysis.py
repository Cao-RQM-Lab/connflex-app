"""Stiffness inspection and static responses on the assembled system matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from connflex_core import Structure
from connflex_core.mechanics.assembly import DOF_NAMES, SystemMatrices

CONDITIONS: tuple[tuple[str, str], ...] = (
    ("others_fixed", "All other bodies fixed"),
    ("anchor_only", "All other bodies free (only the anchor is fixed)"),
    ("custom", "Custom: choose which bodies are fixed"),
)
ROTATIONAL = ("tx", "ty", "tz")


def dof_offsets(system: SystemMatrices) -> dict[int, int]:
    """Row of the first degree of freedom of each movable body."""
    return {body: row for row, (body, name) in enumerate(system.dofs) if name == "ux"}


def _rows(offsets: dict[int, int], bodies) -> NDArray[np.int64]:
    bodies = list(bodies)
    if not bodies:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(offsets[b], offsets[b] + 6) for b in bodies])


@dataclass(frozen=True)
class StiffnessReport:
    """Stiffness felt by one body under a boundary condition on the others.

    Attributes:
        body: Inspected body.
        fixed: Other bodies held fixed; the anchor is always fixed.
        stiffness: 6 by 6 condensed stiffness at the body's centre of mass.
        compliance: Its inverse, or None when singular.
        direct: Diagonal of ``stiffness``: force per unit motion with the body's other
            degrees of freedom locked.
        effective: One over the diagonal of ``compliance``: force per unit motion with the
            body's other degrees of freedom free to move.
        singular: Whether a solve needed a least-squares fallback.
        frequency: Single-degree-of-freedom estimate from ``effective`` and the body's mass or inertia.
    """

    body: int
    fixed: tuple[int, ...]
    stiffness: NDArray[np.float64]
    compliance: NDArray[np.float64] | None
    direct: NDArray[np.float64]
    effective: NDArray[np.float64]
    singular: bool
    frequency: NDArray[np.float64]


def condensed_stiffness(
    structure: Structure, system: SystemMatrices, body: int, fixed: set[int]
) -> StiffnessReport:
    """Condense the system stiffness onto ``body`` with ``fixed`` bodies held and the rest free."""
    offsets = dof_offsets(system)
    if body not in offsets:
        raise ValueError(f"body {body} is the anchor or does not exist")
    rows_i = _rows(offsets, [body])
    free = [b for b in offsets if b != body and b not in fixed]
    rows_o = _rows(offsets, free)
    k = system.stiffness
    kii = k[np.ix_(rows_i, rows_i)]
    singular = False
    if rows_o.size:
        koo = k[np.ix_(rows_o, rows_o)]
        koi = k[np.ix_(rows_o, rows_i)]
        try:
            if np.linalg.cond(koo) > 1e14:
                raise np.linalg.LinAlgError("ill-conditioned")
            x = np.linalg.solve(koo, koi)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(koo, koi, rcond=None)[0]
            singular = True
        keff = kii - koi.T @ x
    else:
        keff = kii.copy()
    keff = 0.5 * (keff + keff.T)
    compliance: NDArray[np.float64] | None
    try:
        if np.linalg.cond(keff) > 1e14:
            raise np.linalg.LinAlgError("ill-conditioned")
        compliance = np.linalg.inv(keff)
    except np.linalg.LinAlgError:
        compliance = None
        singular = True
    direct = np.diag(keff).copy()
    if compliance is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            effective = 1.0 / np.diag(compliance)
        effective = np.where(np.isfinite(effective) & (np.diag(compliance) > 0), effective, np.nan)
    else:
        effective = np.full(6, np.nan)
    inertia_diag = np.diag(structure.bodies[body].inertial.matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        frequency = np.sqrt(np.clip(effective, 0.0, None) / inertia_diag) / (2.0 * np.pi)
    return StiffnessReport(
        body=body,
        fixed=tuple(sorted(fixed)),
        stiffness=keff,
        compliance=compliance,
        direct=direct,
        effective=effective,
        singular=singular,
        frequency=frequency,
    )


def static_response(
    system: SystemMatrices, body: int, dof: int, fixed: set[int]
) -> NDArray[np.float64]:
    """System displacement for a unit generalized force on ``body`` along ``dof`` (0 to 5)."""
    offsets = dof_offsets(system)
    if body not in offsets:
        raise ValueError(f"body {body} is the anchor or does not exist")
    free = [b for b in offsets if b not in fixed or b == body]
    rows = _rows(offsets, free)
    k = system.stiffness[np.ix_(rows, rows)]
    force = np.zeros(rows.size)
    force[int(np.flatnonzero(rows == offsets[body] + dof)[0])] = 1.0
    try:
        u = np.linalg.solve(k, force)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(k, force, rcond=None)[0]
    full = np.zeros(len(system.dofs))
    full[rows] = u
    return full


def dof_name(index: int) -> str:
    return DOF_NAMES[index]
