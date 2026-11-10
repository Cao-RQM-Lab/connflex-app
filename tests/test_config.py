"""Process files and settings: what connflex-core no longer reads, the application does."""

from __future__ import annotations

import pytest

from connflex_app.config import load_config
from connflex_app.data import demo_config
from connflex_app.settings import DriveNet, DriveSettings, ElectroSettings, Settings, TransientSettings
from connflex_app.units import fmt_si, parse_si
from connflex_app.worker import is_worker

PROCESS = """
[stack.device]
thickness = 10e-6
material = "doped"

[stack.box]
thickness = 1e-6
material = "oxide"

[stack.handle]
thickness = 300e-6
material = "silicon"

[materials.doped]
density = 2330.0
youngs_modulus = 160e9
poisson_ratio = 0.22
resistivity = 1e-4

[materials.oxide]
relative_permittivity = 4.2

[extraction]
release_hole_max_area = 50e-12
release_distance = 4e-6
body_min_width = 8e-6
via_resistance = 2.0
anchor = [1e-4, 2e-4]

[solve]
n_modes = 6

[electrostatics]
element_size = 1e-6
interaction_distance = 20e-6
include_fringe = false

[transient]
quality_factor = 200.0
method = "bdf"
"""


def test_the_demo_process_file_loads():
    config = load_config(demo_config())
    assert config.stack.device.thickness == 25e-6 and config.stack.box.material.relative_permittivity == 3.9
    assert config.params.release_distance == 5e-6 and config.options.method == "dense"
    assert config.electrostatics is None and config.transient is None


def test_every_table_reaches_the_settings(tmp_path):
    path = tmp_path / "process.toml"
    path.write_text(PROCESS)
    settings = Settings().with_config(str(path))
    assert settings.stack_device.thickness == 10e-6 and settings.stack_device.material.resistivity == 1e-4
    assert settings.stack_box.material.relative_permittivity == 4.2
    assert settings.stack().device.material.resistivity == 1e-4
    assert settings.params().via_resistance == 2.0 and settings.params().anchor == (1e-4, 2e-4)
    assert settings.n_modes == 6
    assert settings.electro.element_size == 1e-6 and settings.electro.include_fringe is False
    assert settings.electro.include_plates is True  # what the file leaves out keeps its value
    assert settings.transient.quality_factor == 200.0 and settings.transient.method == "bdf"
    assert settings.transient.options().method == "bdf"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (PROCESS.replace("[solve]", "[solver]"), "unknown key"),
        (PROCESS.replace('material = "doped"', 'material = "gold"'), "unknown material 'gold'"),
        (PROCESS.replace("body_min_width = 8e-6", ""), "missing body_min_width"),
        (PROCESS.replace("release_distance = 4e-6", "release_distance = -1.0"), r"\[extraction\]"),
        (PROCESS.replace("include_fringe = false", "mesh = 3"), r"unknown key\(s\) in \[electrostatics\]"),
        (PROCESS.replace("element_size = 1e-6", "element_size = -1.0"), r"\[electrostatics\]"),
    ],
)
def test_mistakes_name_their_table(tmp_path, text, message):
    path = tmp_path / "process.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_settings_round_trip_with_electrical_and_drive():
    settings = Settings(device=(1, 0), handle=(2, 0), via_resistance=1.5)
    settings.stack_device.material.resistivity = 2e-4
    settings.electro = ElectroSettings(element_size=1e-6, surrogate_tolerance=None, workers=3)
    settings.transient = TransientSettings(quality_factor=120.0, rise_time=2e-6, settle_first=True)
    settings.drive = DriveSettings(durations=[None, 5e-5], nets=[DriveNet((1e-4, 2e-4), "device", "drive", [40.0, 0.0]), DriveNet((0.0, 0.0), "handle", "float", [])])
    again = Settings.from_json(settings.to_json())
    assert again == settings
    assert again.drive.nets[0].point == (1e-4, 2e-4) and again.drive.durations == [None, 5e-5]
    assert again.params().via_resistance == 1.5
    old = Settings.from_dict({"device": [1, 0], "stack_box": {"thickness": 2e-6, "material": {"density": 2200.0, "youngs_modulus": 70e9, "poisson_ratio": 0.17}}})
    assert old.stack_box.material.relative_permittivity is None  # settings saved by 0.1 had no permittivity


def test_numbers_with_prefixes():
    assert parse_si("40", "V") == 40.0 and parse_si("40 V", "V") == 40.0
    assert parse_si("-5.5mV", "V") == pytest.approx(-5.5e-3)
    assert parse_si("180 µs", "s") == pytest.approx(180e-6) and parse_si("180u", "s") == pytest.approx(180e-6)
    assert parse_si("0.18 ms", "s") == pytest.approx(180e-6) and parse_si("1e-4", "s") == 1e-4
    assert parse_si("3 um", "m") == pytest.approx(3e-6) and parse_si("1.5 m", "m") == 1.5
    assert parse_si("2k", "Ω") == 2000.0 and parse_si("1,000", "") == 1000.0
    assert parse_si(fmt_si(1.5e-6, "m"), "m") == pytest.approx(1.5e-6)
    for bad in ("", "volts", "4 x", "1e"):
        with pytest.raises(ValueError):
            parse_si(bad, "V")


def test_worker_command_line():
    assert is_worker(["ConnFlex", "-m", "connflex_core._serve"])
    assert not is_worker(["ConnFlex"]) and not is_worker(["ConnFlex", "layout.gds"]) and not is_worker(["ConnFlex", "-m", "other"])
