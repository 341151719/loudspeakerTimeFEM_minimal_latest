from pathlib import Path

import numpy as np

from loudspeaker_time_fem.comsol_mesh import load_comsol_mphtxt_mesh
from loudspeaker_time_fem.native_acoustic import build_native_acoustic


ROOT = Path(__file__).resolve().parents[1]


def test_transient_native_mesh_contract():
    mesh = load_comsol_mphtxt_mesh(ROOT / "inputs/comsol_transient_mesh.mphtxt")
    assert mesh.points_rz_m.shape == (5015, 2)
    assert mesh.cells["tri"].shape == (8010, 3)
    assert mesh.cells["quad"].shape == (950, 4)
    domains = set(np.unique(mesh.entities["tri"])) | set(
        np.unique(mesh.entities["quad"])
    )
    assert domains >= {1, 2, 3, 5, 6}
    interfaces = mesh.acoustic_interface_edges(
        {1, 2, 3, 5, 6}, {4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19}
    )
    assert len(interfaces) > 0
    assert all(boundary >= 1 for *_, boundary in interfaces)
    assert mesh.line_tags.min() == 1
    assert mesh.line_tags.max() == 92


def test_uniform_acoustic_refinement_is_conforming_and_conservative():
    path = ROOT / "inputs/comsol_transient_mesh.mphtxt"
    coarse = build_native_acoustic(path, {2, 3, 5}, set(), 0)
    refined = build_native_acoustic(path, {2, 3, 5}, set(), 1)
    assert len(refined.triangles_global) == 4 * len(coarse.triangles_global)
    assert len(refined.acoustic_nodes_global) > len(coarse.acoustic_nodes_global)
    assert refined.edge_midpoint_nodes
    np.testing.assert_allclose(refined.Mp.sum(), coarse.Mp.sum(), rtol=2e-13)
    np.testing.assert_allclose(
        refined.Kp @ np.ones(refined.Kp.shape[0]), 0.0, atol=2e-11
    )


def test_p2_acoustic_partition_of_unity_and_integral_conservation():
    path = ROOT / "inputs/comsol_transient_mesh.mphtxt"
    p1 = build_native_acoustic(path, {2, 3, 5}, set(), 0, 1)
    p2 = build_native_acoustic(path, {2, 3, 5}, set(), 0, 2)
    assert p2.pressure_order == 2
    assert p2.triangles_global.shape[1] == 6
    np.testing.assert_allclose(p2.Mp.sum(), p1.Mp.sum(), rtol=2e-12)
    np.testing.assert_allclose(p2.Kp @ np.ones(p2.Kp.shape[0]), 0.0, atol=8e-11)
