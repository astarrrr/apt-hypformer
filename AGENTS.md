# Repository Guidelines

## Project Structure & Module Organization
This repository contains the hyperbolic encoder components that extend PIDSMaker. Core modules live at the repo root by domain: `encoders/` for the dual-branch Hyperbolic Transformer, `decoders/` for edge decoding, `objectives/` for training objectives, `losses/` for Lorentz-space losses, and `manifolds/` for geometry utilities and layers. Integration points are in `factory_ext.py` and `optimizer.py`. Runtime configs live in `configs/`, operational notes in `docs/`, and ad hoc sweep outputs in `sweep_results.csv`.

## Build, Test, and Development Commands
Use the existing scripts rather than inventing new entry points.

- `./run.sh CADETS_E3`: symlinks `configs/hyp_pids.yml` into the local PIDSMaker checkout and launches training.
- `./run.sh THEIA_E5 --wandb`: passes extra PIDSMaker CLI flags through unchanged.
- `python sweep.py`: runs the staged hyperparameter sweep and appends results to `sweep_results.csv`.
- `python -m py_compile $(find . -name '*.py')`: quick syntax validation for all Python modules.

This project assumes PIDSMaker is available at `/home/astar/projects/PIDSMaker/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and short docstrings on public modules/classes. Keep tensor shape assumptions explicit, especially the Lorentz convention where `node_out_dim` means spatial dimensions and runtime tensors are `dim + 1`. Match existing config names such as `trans_num_layers`, `hyp_lr`, and `intra_graph_batch_size`.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Before opening a PR, run a syntax pass and at least one smoke test through `run.sh` against a small or known dataset. If you touch encoder or manifold math, add a minimal reproducible check in the PR description showing expected tensor shapes or loss behavior.

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so follow standard imperative commit subjects such as `Add Lorentz midpoint guard`. Keep commits scoped to one concern. PRs should include: a short problem statement, the files changed, the validation command(s) run, and any config or PIDSMaker dependency changes. Include logs or screenshots only when they clarify training or sweep behavior.

## Configuration & Environment Tips
Avoid hard-coding new machine-specific paths; reuse the existing PIDSMaker path pattern or make paths configurable. Do not commit generated `__pycache__/` contents or experiment artifacts unless they are intentionally versioned results.
