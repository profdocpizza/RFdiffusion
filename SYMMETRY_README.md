# Symmetry in RFdiffusion

This document provides a detailed explanation of how symmetry is implemented within the RFdiffusion inference code.

## Overview

Symmetry is a powerful constraint in protein design, enabling the generation of homo-oligomeric complexes with beautiful and complex architectures. In RFdiffusion, symmetry is enforced during the denoising process. At each step of the reverse diffusion trajectory, the model's prediction is made symmetric by applying the specified symmetry operations.

The core idea is to generate a single protomer (the asymmetric unit) and then duplicate it according to the symmetry operations to form the full oligomer. The diffusion process, however, operates on the full symmetric assembly. Gradients from each protomer are averaged together, which enforces that each subunit is identical and has an identical structural environment.

## Implementation Details

The key files for the symmetry implementation are:
- `rfdiffusion/inference/symmetry.py`: This file contains the `SymGen` class, which is responsible for generating the symmetry matrices for the specified point group.
- `rfdiffusion/inference/model_runners.py`: The `Sampler` (and its child) classes in this file instantiate the `SymGen` class and use it to apply symmetry at each step of the diffusion process.

### `SymGen` Class

The `SymGen` class is initialized with a symmetry group (e.g., 'C3', 'D2', 'tetrahedral'). Based on this input, it generates a set of rotation matrices (`self.sym_rots`) that define the symmetry operations. For standard point groups like Cyclic (C), Dihedral (D), Tetrahedral (T), Octahedral (O), and Icosahedral (I), these matrices are either calculated on the fly or loaded from a pre-computed file (`rfdiffusion/inference/sym_rots.npz`).

### `Sampler` and `SelfConditioning` Classes

Within `model_runners.py`, the `Sampler` class initializes `SymGen` if the `inference.symmetry` config option is provided. The key method is `apply_symmetry`, which is called at two main points:
1.  **Initialization (`sample_init`):** After the initial coordinates and sequence are generated, `apply_symmetry` is called to create the full symmetric assembly to start the diffusion process.
2.  **Denoising Step (`sample_step`):** After each denoising step where the model predicts the structure at `x_0`, the resulting coordinates are made symmetric before being noised again to `x_{t-1}`. This is the crucial step that enforces the symmetry constraint throughout the generation process. In the `SelfConditioning` model runner, the predicted structure is also symmetrized before being fed back into the model as a template.

## Supported Symmetries

The current implementation supports the following point group symmetries:
-   **Cyclic (C#):** e.g., `C2`, `C3`, `C4`, etc.
-   **Dihedral (D#):** e.g., `D2`, `D3`, `D4`, etc.
-   **Tetrahedral (T)**
-   **Octahedral (O)**
-   **Icosahedral (I)**

## Usage

To run a symmetric design, you need to specify the symmetry group in the configuration. The primary parameter is:
-   `inference.symmetry`: The desired symmetry group (e.g., `C5`, `D2`, `octahedral`).

For certain symmetries, especially those with off-axis symmetry axes, you may need to use additional parameters to ensure the object is centered correctly:
-   `inference.recenter`: (boolean) Whether to recenter the asymmetric unit at each step.
-   `inference.radius`: (float) The radius at which to place the asymmetric unit from the origin.

## Advanced Topics & FAQ

### How do I use helical symmetry?

Helical symmetry (a type of screw symmetry) is supported with a detailed set of controls, allowing for the generation of complex helical assemblies.

To use helical symmetry, specify the symmetry in the following format:
`inference.symmetry=H_<R/L>_<units_per_turn>_<rise_per_turn>_<num_turns>`

-   `<R/L>`: The handedness of the helix (`R` for right-handed, `L` for left-handed).
-   `<units_per_turn>`: The number of subunits in one full 360° turn of the helix. This can be a floating point number.
-   `<rise_per_turn>`: The translation in Angstroms along the Z-axis for one full turn. This can be a floating point number.
-   `<num_turns>`: The total number of full turns in the helix. This can be a floating point number.

The total number of subunits in the assembly is calculated as `round(units_per_turn * num_turns)`. The length of the asymmetric unit (a single subunit) is defined by the input contig string (e.g., `contigs: [50-50]` for a 50-residue subunit).

**Example:**

To generate a right-handed helix with 6.3 subunits per turn, a rise of 8.2Å per turn, and a total of 3.5 turns, you would use:
`inference.symmetry=H_R_6.3_8.2_3.5`

This would generate a helix with `round(6.3 * 3.5) = 22` subunits.

### Is it possible to run a 3x stacked C3 symmetries such that each unit would be diffused identically (9x identical units)?

This describes a hierarchical symmetry, where a C3 oligomer is itself subjected to another C3 symmetry operation. This is also **not directly supported** by the current implementation. The `SymGen` class is designed to handle a single set of symmetry operations that form a single point group.

To achieve a design with 3x stacked C3 symmetries (a C3 trimer of C3 trimers), you would effectively be describing a P3 point-group symmetry, which is not implemented. While you could run a single C9 generation, this would produce a 9-mer ring, not a stack of trimers. Generating such a hierarchical assembly would likely require a more complex, multi-stage design process or significant modifications to the symmetry generation code to handle nested or layered symmetry operations. The current framework is built around diffusing a single asymmetric unit and applying one set of symmetry operations.