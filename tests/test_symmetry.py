import torch
import numpy as np
from rfdiffusion.inference.symmetry import SymGen

def test_screw_symmetry_right_handed():
    # Test H_R_6.3_8.2_3.5 symmetry
    sym_gen = SymGen('H_R_6.3_8.2_3.5', recenter=False, radius=0)
    assert sym_gen.order == int(round(6.3 * 3.5))

    translation_per_unit = 8.2 / 6.3
    rotation_per_unit = 360.0 / 6.3

    assert torch.allclose(sym_gen.sym_trans[1], torch.tensor([0., 0., translation_per_unit]))

    # Create some dummy coordinates
    subunit_len = 10
    coords_in = torch.randn(subunit_len * sym_gen.order, 14, 3)
    seq_in = torch.zeros(subunit_len * sym_gen.order, 22)
    seq_in[:subunit_len, 0] = 1

    coords_out, seq_out = sym_gen.apply_symmetry(coords_in, seq_in)

    # Check the second subunit
    rot = torch.tensor(
        np.array(
            [[np.cos(np.deg2rad(rotation_per_unit)), -np.sin(np.deg2rad(rotation_per_unit)), 0],
             [np.sin(np.deg2rad(rotation_per_unit)), np.cos(np.deg2rad(rotation_per_unit)), 0],
             [0, 0, 1]]
        ), dtype=torch.float32)

    expected_coords = torch.einsum('bnj,kj->bnk', coords_in[:subunit_len], rot) + torch.tensor([0,0,translation_per_unit])
    assert torch.allclose(coords_out[subunit_len:2*subunit_len], expected_coords, atol=1e-5)

def test_screw_symmetry_left_handed():
    # Test H_L_6.3_8.2_3.5 symmetry
    sym_gen = SymGen('H_L_6.3_8.2_3.5', recenter=False, radius=0)
    assert sym_gen.order == int(round(6.3 * 3.5))

    translation_per_unit = 8.2 / 6.3
    rotation_per_unit = -360.0 / 6.3

    assert torch.allclose(sym_gen.sym_trans[1], torch.tensor([0., 0., translation_per_unit]))

    # Create some dummy coordinates
    subunit_len = 10
    coords_in = torch.randn(subunit_len * sym_gen.order, 14, 3)
    seq_in = torch.zeros(subunit_len * sym_gen.order, 22)
    seq_in[:subunit_len, 0] = 1

    coords_out, seq_out = sym_gen.apply_symmetry(coords_in, seq_in)

    # Check the second subunit
    rot = torch.tensor(
        np.array(
            [[np.cos(np.deg2rad(rotation_per_unit)), -np.sin(np.deg2rad(rotation_per_unit)), 0],
             [np.sin(np.deg2rad(rotation_per_unit)), np.cos(np.deg2rad(rotation_per_unit)), 0],
             [0, 0, 1]]
        ), dtype=torch.float32)

    expected_coords = torch.einsum('bnj,kj->bnk', coords_in[:subunit_len], rot) + torch.tensor([0,0,translation_per_unit])
    assert torch.allclose(coords_out[subunit_len:2*subunit_len], expected_coords, atol=1e-5)