import torch
import numpy as np
from rfdiffusion.inference.symmetry import SymGen

def test_screw_symmetry():
    # Test H6_10 symmetry
    sym_gen = SymGen('H6_10', recenter=False, radius=0)
    assert sym_gen.order == 6
    assert torch.allclose(sym_gen.sym_trans[1], torch.tensor([0., 0., 10.]))
    assert torch.allclose(sym_gen.sym_trans[5], torch.tensor([0., 0., 50.]))

    # Create some dummy coordinates
    coords_in = torch.randn(60, 14, 3)
    seq_in = torch.zeros(60, 22)
    seq_in[:10, 0] = 1

    coords_out, seq_out = sym_gen.apply_symmetry(coords_in, seq_in)

    # Check shape of output
    assert coords_out.shape == coords_in.shape
    assert seq_out.shape == seq_in.shape

    # Check that the first subunit is unchanged
    assert torch.allclose(coords_out[:10], coords_in[:10])

    # Check the second subunit
    # Rotate the first subunit by 60 degrees around Z and translate by 10A
    rot = torch.tensor([
        [np.cos(np.pi/3), -np.sin(np.pi/3), 0],
        [np.sin(np.pi/3), np.cos(np.pi/3), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    expected_coords = torch.einsum('bnj,kj->bnk', coords_in[:10], rot) + torch.tensor([0,0,10.0])
    assert torch.allclose(coords_out[10:20], expected_coords, atol=1e-5)