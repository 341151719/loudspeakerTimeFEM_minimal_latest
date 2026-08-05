# Curated benchmark record

This directory is a compact, public-safe snapshot of selected results from
`loudspeakerTimeFEM_codex_full_20260802`. It keeps small machine-readable metrics and
decision records while excluding COMSOL solved models, virtual environments, logs, long
time series, field snapshots, and generated animations.

The files are archival evidence from completed runs, not CI tests. The Python model,
configs, and comparison tools remain in the main project. Re-running the external COMSOL
comparisons requires COMSOL 6.3 and the corresponding private/raw COMSOL inputs.

The selection intentionally includes both accepted engineering checks and open or rejected
experiments. See [`README_CN.md`](README_CN.md) for the detailed Chinese description,
field conventions, limitations, and re-check commands. The machine-readable catalog is
[`manifest.json`](manifest.json).

| Benchmark | Role | Result |
|---|---|---|
| `comsol_refined_70hz` | Python vs refined COMSOL at 70 Hz / 10 V peak | Registered engineering criteria pass; the PML rear probe is diagnostic only |
| `comsol_time_convergence_70hz` | COMSOL tighter time step | Primary signals pass the 1% / 1° gate |
| `comsol_mesh_convergence_70hz` | COMSOL refined mesh | Strict 1% / 1° spatial closure remains open |
| `interface_motion_substitution_70hz` | Interface-layer isolation | Complex interface L2 error is about 0.213% |
| `native_fft_crosscheck_70hz` | Native FFT vs external harmonic fit | THD difference is about 0.0055 percentage points |
| `spherical_dtn_diagnostic_70hz` | Frequency-only DtN ablation | Diagnostic only; not the production boundary |
| `tensor_coenergy_pilot_20260801` | Native tensor co-energy pilot | Mesh gate failed; no tensor law was promoted |
