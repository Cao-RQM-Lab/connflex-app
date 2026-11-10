"""User-editable configuration mirrored onto connflex-core objects."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from connflex_core import (
    ElectrostaticParams,
    ExtractionParams,
    Layer,
    LayerMap,
    Material,
    SolveOptions,
    Stack,
    TransientOptions,
)

OXIDE_PERMITTIVITY = 3.9


@dataclass
class MaterialSettings:
    """One layer's material. ``resistivity`` (ohm metres) matters for the device layer only,
    ``relative_permittivity`` for the buried oxide only; None leaves each unset."""

    density: float = 2330.0
    youngs_modulus: float = 169e9
    poisson_ratio: float = 0.28
    resistivity: float | None = None
    relative_permittivity: float | None = None

    def to_material(self) -> Material:
        return Material(
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
            resistivity=self.resistivity,
            relative_permittivity=self.relative_permittivity,
        )

    @classmethod
    def from_material(cls, material: Material) -> MaterialSettings:
        return cls(
            material.density,
            material.youngs_modulus,
            material.poisson_ratio,
            material.resistivity,
            material.relative_permittivity,
        )


SILICON = MaterialSettings(2330.0, 169e9, 0.28)
OXIDE = MaterialSettings(2200.0, 70e9, 0.17, relative_permittivity=OXIDE_PERMITTIVITY)
MATERIAL_PRESETS: dict[str, MaterialSettings] = {"Silicon": SILICON, "Oxide": OXIDE}


def preset_name(material: MaterialSettings) -> str:
    for name, preset in MATERIAL_PRESETS.items():
        if (
            abs(preset.density - material.density) < 1e-6
            and abs(preset.youngs_modulus - material.youngs_modulus) < 1.0
            and abs(preset.poisson_ratio - material.poisson_ratio) < 1e-6
        ):
            return name
    return "Custom"


@dataclass
class LayerSettings:
    thickness: float
    material: MaterialSettings

    def to_layer(self) -> Layer:
        return Layer(thickness=self.thickness, material=self.material.to_material())


@dataclass
class ElectroSettings:
    """Electrostatic model. ``element_size`` and ``interaction_distance`` of None are chosen
    from the narrowest gap between device electrodes (see ``electrical.suggest_lengths``)."""

    include_plates: bool = True
    include_sidewalls: bool = True
    include_fringe: bool = True
    element_size: float | None = None
    interaction_distance: float | None = None
    ambient_permittivity: float = 1.0
    dense_cutoff: float = 10.0
    plates_cutoff: float = 100.0
    surrogate_tolerance: float | None = 1e-4
    max_elements: int = 8000
    workers: int | None = None

    def params(self, element_size: float | None = None, interaction_distance: float | None = None) -> ElectrostaticParams:
        """connflex-core parameters, with the given lengths where this leaves them automatic.

        Without a gap to size them from (one device electrode, or none near another), there
        are no sidewalls to solve and the model keeps the plates alone.
        """
        size = self.element_size if self.element_size is not None else element_size
        reach = self.interaction_distance if self.interaction_distance is not None else interaction_distance
        sidewalls = self.include_sidewalls and size is not None and reach is not None
        return ElectrostaticParams(
            ambient_permittivity=self.ambient_permittivity,
            element_size=size if sidewalls else None,
            interaction_distance=reach if sidewalls else None,
            dense_cutoff=self.dense_cutoff,
            plates_cutoff=max(self.plates_cutoff, self.dense_cutoff),
            include_plates=self.include_plates,
            include_sidewalls=sidewalls,
            include_fringe=self.include_fringe and sidewalls,
            surrogate_tolerance=self.surrogate_tolerance,
            max_elements=self.max_elements,
        )


@dataclass
class TransientSettings:
    """Options of a transient run. ``rise_time`` of None is a fiftieth of the shortest step."""

    quality_factor: float = 50.0
    rise_time: float | None = None
    samples: int = 200
    method: str = "radau"
    tolerance: float = 1e-6
    max_step: float | None = None
    contact_clearance: float = 0.1
    source_resistance: float = 0.0
    settle_first: bool = False

    def options(self) -> TransientOptions:
        return TransientOptions(
            quality_factor=self.quality_factor,
            tolerance=self.tolerance,
            max_step=self.max_step,
            contact_clearance=self.contact_clearance,
            method=self.method,
        )


@dataclass
class DriveNet:
    """One net of the drive: where its source connects, and its voltage in every step.

    ``point`` is a point in metres on the electrode the source is bonded to, so a saved drive
    finds its net again after the layout is extracted anew.
    """

    point: tuple[float, float]
    layer: str = "device"
    mode: str = "drive"  # "drive": a voltage source; "float": no source, the net keeps zero charge
    volts: list[float] = field(default_factory=list)


@dataclass
class DriveSettings:
    """Voltage steps of the electromechanical tab. A duration of None is chosen from the lowest mode."""

    durations: list[float | None] = field(default_factory=lambda: [None])
    nets: list[DriveNet] = field(default_factory=list)


def _from_table(cls, table: dict[str, Any]):
    """A settings dataclass from a JSON table, ignoring keys it does not have."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in table.items() if key in names})


@dataclass
class Settings:
    """Everything the user chooses: layer roles, stack, extraction, solve, electrostatics, and drive."""

    device: tuple[int, int] | None = None
    handle: tuple[int, int] | None = None
    release: tuple[int, int] | None = None
    vias: tuple[int, int] | None = None
    box: tuple[int, int] | None = None
    cell: str | None = None
    invert_device: bool = False
    invert_handle: bool = False
    invert_release: bool = False
    invert_vias: bool = False
    invert_box: bool = False

    stack_device: LayerSettings = field(default_factory=lambda: LayerSettings(25e-6, replace(SILICON)))
    stack_box: LayerSettings = field(default_factory=lambda: LayerSettings(2e-6, replace(OXIDE)))
    stack_handle: LayerSettings = field(default_factory=lambda: LayerSettings(400e-6, replace(SILICON)))

    release_hole_max_area: float = 100e-12
    release_distance: float = 5e-6
    body_min_width: float = 10e-6
    heal_distance: float = 10e-9
    simplify_tolerance: float | None = None
    beam_overlap_margin: float | None = None
    beam_width_min: float = 0.0
    beam_width_max: float | None = None
    anchor: int | tuple[float, float] | None = None
    buffer_resolution: int = 8
    precision_grid: float | None = None
    via_resistance: float = 0.0

    n_modes: int | None = None
    method: str = "dense"

    electro: ElectroSettings = field(default_factory=ElectroSettings)
    transient: TransientSettings = field(default_factory=TransientSettings)
    drive: DriveSettings = field(default_factory=DriveSettings)

    def inverted_roles(self) -> set[str]:
        """Roles whose layer is replaced by its complement within the cell bounds."""
        return {role for role in ("device", "handle", "release", "vias", "box") if getattr(self, f"invert_{role}")}

    def layer_map(self) -> LayerMap:
        if self.device is None or self.handle is None:
            raise ValueError("choose the device and handle layers first")
        return LayerMap(
            device=tuple(self.device),
            handle=tuple(self.handle),
            release=None if self.release is None else tuple(self.release),
            vias=None if self.vias is None else tuple(self.vias),
        )

    def stack(self) -> Stack:
        return Stack(
            device=self.stack_device.to_layer(),
            box=self.stack_box.to_layer(),
            handle=self.stack_handle.to_layer(),
        )

    def params(self) -> ExtractionParams:
        anchor = self.anchor
        if isinstance(anchor, list):
            anchor = tuple(anchor)
        return ExtractionParams(
            release_hole_max_area=self.release_hole_max_area,
            release_distance=self.release_distance,
            body_min_width=self.body_min_width,
            heal_distance=self.heal_distance,
            simplify_tolerance=self.simplify_tolerance,
            beam_overlap_margin=self.beam_overlap_margin,
            beam_width_min=self.beam_width_min,
            beam_width_max=self.beam_width_max,
            anchor=anchor,
            buffer_resolution=self.buffer_resolution,
            precision_grid=self.precision_grid,
            via_resistance=self.via_resistance,
        )

    def solve_options(self) -> SolveOptions:
        return SolveOptions(n_modes=self.n_modes, method=self.method)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Settings:
        data: dict[str, Any] = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        settings = cls()
        for key, value in data.items():
            if not hasattr(settings, key):
                continue
            if key.startswith("stack_") and isinstance(value, dict):
                material = _from_table(MaterialSettings, value.get("material", {}))
                value = LayerSettings(thickness=float(value["thickness"]), material=material)
            elif key == "electro" and isinstance(value, dict):
                value = _from_table(ElectroSettings, value)
            elif key == "transient" and isinstance(value, dict):
                value = _from_table(TransientSettings, value)
            elif key == "drive" and isinstance(value, dict):
                nets = []
                for table in value.get("nets", []):
                    net = _from_table(DriveNet, table)
                    net.point = (float(net.point[0]), float(net.point[1]))
                    net.volts = [float(v) for v in net.volts]
                    nets.append(net)
                durations = [None if d is None else float(d) for d in value.get("durations", [None])] or [None]
                value = DriveSettings(durations=durations, nets=nets)
            elif key in ("device", "handle", "release", "vias", "box") and value is not None:
                value = (int(value[0]), int(value[1]))
            elif key == "anchor" and isinstance(value, list):
                value = (float(value[0]), float(value[1]))
            setattr(settings, key, value)
        return settings

    def with_config(self, path: str) -> Settings:
        """Copy with the stack, extraction parameters, solve options, and, where the file has
        them, electrostatic and transient options read from a TOML process file."""
        from connflex_app.config import load_config

        config = load_config(path)
        stack, params, options = config.stack, config.params, config.options
        new = replace(self)
        if config.electrostatics is not None:
            new.electro = replace(self.electro, **{k: v for k, v in config.electrostatics.items() if hasattr(self.electro, k)})
        if config.transient is not None:
            new.transient = replace(self.transient, **{k: v for k, v in config.transient.items() if hasattr(self.transient, k)})
        new.stack_device = LayerSettings(stack.device.thickness, MaterialSettings.from_material(stack.device.material))
        new.stack_box = LayerSettings(stack.box.thickness, MaterialSettings.from_material(stack.box.material))
        new.stack_handle = LayerSettings(stack.handle.thickness, MaterialSettings.from_material(stack.handle.material))
        for name in (
            "release_hole_max_area",
            "release_distance",
            "body_min_width",
            "heal_distance",
            "simplify_tolerance",
            "beam_overlap_margin",
            "beam_width_min",
            "beam_width_max",
            "anchor",
            "buffer_resolution",
            "precision_grid",
            "via_resistance",
        ):
            setattr(new, name, getattr(params, name))
        new.n_modes = options.n_modes
        new.method = options.method
        return new
