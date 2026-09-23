# Agent entrypoint: loudspeakerTimeFEM

Frequency snapshot baseline and import closure: [`docs/FREQUENCY_SNAPSHOT_AUDIT_CN.md`](docs/FREQUENCY_SNAPSHOT_AUDIT_CN.md).

This repository is the independent 2-D axisymmetric, time-domain loudspeaker FEM project. Treat the checked-in files and current branch as authoritative; do not assume another checkout is synchronized.

## Read first

1. [`docs/AGENT_MAP_CN.md`](docs/AGENT_MAP_CN.md) for task-to-file routing.
2. [`README_CN.md`](README_CN.md) for the production config, physical contracts, diagnostic gates, and validation order.
3. [`docs/PROJECT_RELATIONSHIP_CN.md`](docs/PROJECT_RELATIONSHIP_CN.md) before changing or comparing the frequency-domain dependency.

## Main route

- Current production config recorded in `README_CN.md`: `configs/transient_70Hz_nonlinear_comsol_physical_abc.json`.
- Production magnetic-law input: `inputs/nonlinear_magnetic_law_20260728.json`.
- CLI: `cli.py`; model assembly: `src/loudspeaker_time_fem/model.py`; time solvers: `solver.py` and `nonlinear_solver.py`.
- `model.py` loads frequency-domain structures from the bundled `inputs/frequency_mainline/` tree by default. Preserve that tree as a runtime dependency.

## Project boundary

Use this repository for transient waveforms, nonlinear large-signal coupling, and time-domain acoustic response. Use [`loudspeakerFEM`](https://github.com/341151719/loudspeakerFEM_minimal_latest) for harmonic frequency sweeps, frequency-domain modes, native blocked-coil impedance, and its enclosure/FR10 branches. The two repositories are independent Git projects; a change in one is not a change in the other.

## Operating constraints

- The tensor co-energy branch is diagnostic-only until its documented mesh-convergence gate passes. Keep it out of the production config and do not manufacture an input table to bypass its guard.
- Keep COMSOL as an offline benchmark; production Python runs must not use COMSOL results as runtime corrections.
- Put outputs in a fresh `runs/` directory or outside the repository, never in `inputs/`. `cli.py run` rotates an existing output directory to `.previous` and removes an existing `.previous`, so do not reuse a valuable output path.
- Do not edit the production config for experiments. Treat the historical metrics in `README_CN.md` as evidence summaries, not as results reproduced by the current checkout.
- Use the ordered acceptance steps in `README_CN.md`; a solver residual or passing unit tests alone does not establish mesh convergence or physical validity.
