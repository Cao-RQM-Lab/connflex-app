"""User-editable configuration mirrored onto connflex-core objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from connflex_core import ExtractionParams, Layer, LayerMap, Material, SolveOptions, Stack, load_config


@dataclass
class MaterialSettings:
    density: float = 2330.0
    youngs_modulus: float = 169e9
    poisson_ratio: float = 0.28

    def to_material(self) -> Material:
        return Material(
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
        )

    @classmethod
    def from_material(cls, material: Material) -> MaterialSettings:
        return cls(material.density, material.youngs_modulus, material.poisson_ratio)


SILICON = MaterialSettings(2330.0, 169e9, 0.28)
OXIDE = MaterialSettings(2200.0, 70e9, 0.17)
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
class Settings:
    """Everything the user chooses: layer roles, stack, extraction parameters, solve options."""

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

    n_modes: int | None = None
    method: str = "dense"

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
                material = MaterialSettings(**value.get("material", {}))
                value = LayerSettings(thickness=float(value["thickness"]), material=material)
            elif key in ("device", "handle", "release", "vias", "box") and value is not None:
                value = (int(value[0]), int(value[1]))
            elif key == "anchor" and isinstance(value, list):
                value = (float(value[0]), float(value[1]))
            setattr(settings, key, value)
        return settings

    def with_config(self, path: str) -> Settings:
        """Copy with stack, extraction parameters, and solve options read from a TOML file."""
        stack, params, options = load_config(path)
        new = replace(self)
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
        ):
            setattr(new, name, getattr(params, name))
        new.n_modes = options.n_modes
        new.method = options.method
        return new
