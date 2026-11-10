"""Electrodes as paint groups, and capacitance between nets, bodies, and electrodes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from connflex_app import paint
from connflex_app.electrical import (
    couplings,
    describe_net,
    grouping,
    member_between,
    member_total,
    mutual_matrix,
    net_potentials,
    pair_report,
    suggest_lengths,
)
from connflex_app.geometry import KIND_BEAM, KIND_BOX, Groups, silicon_pieces
from connflex_app.settings import ElectroSettings

STATOR, SHUTTLE, LEFT = (1050e-6, 600e-6), (800e-6, 600e-6), (150e-6, 600e-6)


def test_every_electrode_is_one_piece_and_one_group(demo, demo_project):
    structure = demo["structure"]
    groups = Groups.of(structure)
    assert groups.count == len(structure.electrodes) + structure.nets.count
    pieces = silicon_pieces(structure)
    assert sorted(piece.group for piece in pieces) == [e.index for e in structure.electrodes]
    for piece in pieces:
        electrode = structure.electrodes[piece.group]
        assert piece.layer == electrode.layer and piece.body == electrode.body
        assert piece.geometry.covers(electrode.region.geoms[0].representative_point())
    mesh = demo_project.mesh
    assert set(np.unique(mesh.group[mesh.kind == KIND_BOX])) == {-1}
    assert set(np.unique(mesh.group[mesh.kind == KIND_BEAM])) == {groups.of_net(structure.beams[0].net)}
    assert set(np.unique(mesh.group)) >= {e.index for e in structure.electrodes}


def test_picking_by_group_finds_the_topmost_silicon(demo, demo_project):
    structure, render = demo["structure"], demo_project.render
    stator = structure.electrode_at(STATOR, "device")
    assert render.group_at(*STATOR) == stator.index
    assert render.group_at(*SHUTTLE) == structure.electrode_at(SHUTTLE, "device").index
    handle = render.group_at(20e-6, 20e-6)  # bare handle in a corner of the die
    assert structure.electrodes[handle].layer == "handle"
    beam = structure.beams[0]
    middle = ((beam.start[0] + beam.end[0]) / 2, (beam.start[1] + beam.end[1]) / 2)
    assert render.group_at(*middle) == render.groups.of_net(beam.net)
    assert render.group_at(-1.0, -1.0) == -1


def test_lengths_are_sized_from_the_narrowest_gap(demo):
    suggestion = suggest_lengths(demo["structure"])
    assert suggestion.gap == pytest.approx(3e-6, rel=1e-6)  # the comb's finger gap
    assert suggestion.element_size == pytest.approx(1.5e-6, rel=1e-6)
    assert suggestion.interaction_distance == pytest.approx(12e-6, rel=1e-6)
    params = ElectroSettings().params(suggestion.element_size, suggestion.interaction_distance)
    assert params.include_sidewalls and params.element_size == suggestion.element_size
    alone = ElectroSettings().params(None, None)  # nothing to size sidewalls from: plates alone
    assert not alone.include_sidewalls and not alone.include_fringe and alone.include_plates
    explicit = ElectroSettings(element_size=1e-6, interaction_distance=20e-6).params(None, None)
    assert explicit.include_sidewalls and explicit.interaction_distance == 20e-6


def test_groupings_reduce_the_matrix(demo, plates_model):
    structure, model = demo["structure"], plates_model.model
    nets = grouping(structure, "net")
    assert nets.keys == (0, 1, 2) and nets.labels == ("N0", "N1", "N2")
    assert np.allclose(nets.reduce(model.matrix), model.net_matrix)
    assert "handle" in describe_net(structure, structure.electrode_at(LEFT, "device").net)
    bodies = grouping(structure, "body")
    assert bodies.keys == (0, 1, 2) and bodies.labels[0] == "Anchor"
    electrodes = grouping(structure, "electrode")
    assert electrodes.count == len(structure.electrodes)
    assert np.allclose(electrodes.reduce(model.matrix), model.matrix)
    for group in (nets, bodies, electrodes):
        mutual = mutual_matrix(model, group)
        assert np.allclose(mutual, mutual.T)
        off = mutual - np.diag(np.diag(mutual))
        assert np.allclose(off.sum(axis=1), np.diag(mutual))  # a member's total is the sum of its couplings
        assert member_total(model, group, 1) == pytest.approx(mutual[1, 1])
        assert member_between(model, group, 0, 1) == pytest.approx(mutual[0, 1], abs=1e-24)


def test_groups_map_onto_members(demo):
    structure = demo["structure"]
    groups = Groups.of(structure)
    stator = structure.electrode_at(STATOR, "device")
    nets, bodies, electrodes = (grouping(structure, kind) for kind in ("net", "body", "electrode"))
    beams = groups.of_net(stator.net)
    assert nets.member_of_group(stator.index, groups) == nets.index(stator.net)
    assert nets.member_of_group(beams, groups) == nets.index(stator.net)
    assert bodies.member_of_group(stator.index, groups) == bodies.index(stator.body)
    assert bodies.member_of_group(beams, groups) == -1  # a beam is silicon of a net and of no body
    assert electrodes.member_of_group(stator.index, groups) == stator.index
    assert nets.member_of_group(-1, groups) == -1
    assert set(nets.paint_groups(nets.index(stator.net), groups)) == {stator.index, beams}
    assert bodies.paint_groups(-1, groups) == []


def test_plate_capacitance_of_a_proof_mass(demo, plates_model):
    structure, model = demo["structure"], plates_model.model
    shuttle = structure.electrode_at(SHUTTLE, "device")
    bodies = grouping(structure, "body")
    report = pair_report(model, bodies, bodies.index(shuttle.body), bodies.index(0))
    assert report.value == pytest.approx(sum(report.parts.values()), rel=1e-12)
    assert report.parts["sidewall"] == 0 and report.parts["fringe"] == 0 and report.parts["oxide"] == 0
    # A released plate of this area over a 2 um gap, less its release holes.
    area = shuttle.region.area - structure.openings.intersection(shuttle.region).area
    assert report.value == pytest.approx(8.8541878128e-12 * area / 2e-6, rel=0.02)
    rows = report.by_body()
    assert rows and rows[0][0] == shuttle.body
    assert rows[0][1][2] == pytest.approx(-report.value / 2e-6, rel=0.02)  # closing the gap raises it
    alone = pair_report(model, bodies, bodies.index(shuttle.body), bodies.index(shuttle.body))
    assert alone.value == pytest.approx(alone.total_a) and alone.value == pytest.approx(report.value)
    assert [member for member, _value in couplings(model, bodies, bodies.index(shuttle.body))] == [bodies.index(0)]


def test_comb_capacitance_with_sidewalls(demo, field_model):
    structure, model = demo["structure"], field_model.model
    assert field_model.params.include_sidewalls and "cluster" in field_model.summary()
    assert np.allclose(model.matrix.sum(axis=1), 0.0, atol=1e-18)
    nets = grouping(structure, "net")
    stator = structure.electrode_at(STATOR, "device")
    shuttle = structure.electrode_at(SHUTTLE, "device")
    report = pair_report(model, nets, nets.index(stator.net), nets.index(shuttle.net))
    assert report.value == pytest.approx(98.1e-15, rel=0.03)
    assert report.parts["sidewall"] > report.parts["fringe"] > 0
    assert report.value == pytest.approx(sum(report.parts.values()), rel=1e-9)
    body, six = report.by_body()[0]
    assert body == shuttle.body and six[0] > 0  # the shuttle moving toward the stator engages the comb
    ranked = couplings(model, nets, nets.index(stator.net))
    assert [value for _member, value in ranked] == sorted((value for _member, value in ranked), reverse=True)
    assert field_model.matches(dataclasses.replace(field_model.settings, workers=4))
    assert not field_model.matches(dataclasses.replace(field_model.settings, include_fringe=False))


def test_net_potentials_follow_electrodes(demo):
    structure = demo["structure"]
    potentials = np.array([[float(e.net) for e in structure.electrodes], [10.0 * e.net for e in structure.electrodes]])
    assert np.allclose(net_potentials(structure, potentials), [[0, 1, 2], [0, 10, 20]])


def test_paint_tables(demo):
    groups = Groups.of(demo["structure"])
    by_net = paint.by_category(groups, "net")
    assert by_net.shape == (groups.count, 4) and by_net.dtype == np.float32 and np.all(by_net[:, 3] > 0)
    by_body = paint.by_category(groups, "body")
    assert np.all(by_body[groups.electrodes :, 3] == 0)  # beams belong to no body: layer colour
    picked = paint.with_picks(by_net, first=[0], second=[1], hover=[2])
    assert tuple(picked[0, :3]) == pytest.approx(paint.PICK_A) and tuple(picked[1, :3]) == pytest.approx(paint.PICK_B)
    assert picked[3, 3] < by_net[3, 3] and picked[2, 3] >= 0.75
    volts = paint.by_voltage(groups, np.array([40.0, 0.0, -40.0]), 40.0, floating=[1])
    per_group = paint.group_potentials(groups, np.array([40.0, 0.0, -40.0])[groups.net[: groups.electrodes]], [40.0, 0.0, -40.0])
    assert np.array_equal(paint.by_voltage(groups, per_group, 40.0, floating=[1]), volts)  # per net or per group
    electrodes = demo["structure"].electrodes
    hot = next(e.index for e in electrodes if e.net == 0 and e.layer == "device")
    cold = next(e.index for e in electrodes if e.net == 2 and e.layer == "device")
    assert tuple(volts[hot, :3]) == pytest.approx(paint.POSITIVE, abs=1e-6)
    assert tuple(volts[cold, :3]) == pytest.approx(paint.NEGATIVE, abs=1e-6)
    assert paint.voltage_scale([]) == 1.0 and paint.voltage_scale([-3.0, 2.0]) == 3.0
    assert paint.voltage_color(0.0, 40.0) == pytest.approx(paint.NEUTRAL)
    assert not np.array_equal(paint.with_outline(volts, selected=[hot]), volts)
