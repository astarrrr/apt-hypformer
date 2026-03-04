#!/usr/bin/env python3
"""Staged greedy hyperparameter sweep for ADP optimization on CADETS_E3.

Runs PIDSMaker with different hyperparameter combinations, logging results
to sweep_results.csv. Uses a staged greedy approach (~20 runs) instead of
full grid search (~400+ combos).

Usage:
    cd /home/astar/projects/myproject
    python sweep.py
"""

import csv
import itertools
import os
import re
import subprocess
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PIDSMAKER_DIR = "/home/astar/projects/PIDSMaker"
MYPROJECT_DIR = "/home/astar/projects/myproject"
RESULTS_CSV = "/home/astar/projects/myproject/sweep_results.csv"
DATASET = "CADETS_E3"
MODEL = "hyp_pids"

# CLI arg paths for each parameter
PARAM_PATHS = {
    "node_out_dim": "training.node_out_dim",
    "trans_num_layers": "training.encoder.hyperbolic_transformer.trans_num_layers",
    "trans_num_heads": "training.encoder.hyperbolic_transformer.trans_num_heads",
    "k": "training.encoder.hyperbolic_transformer.k",
    "hyp_lr": "training.hyp_lr",
    "patience": "training.patience",
    "intra_graph_batch_size": "batching.intra_graph_batching.edges.intra_graph_batch_size",
    "power_k": "training.encoder.hyperbolic_transformer.power_k",
}

# Default values (matching hyp_pids.yml)
DEFAULTS = {
    "node_out_dim": 128,
    "trans_num_layers": 2,
    "trans_num_heads": 4,
    "k": 1.0,
    "hyp_lr": 0.0001,
    "patience": 5,
    "intra_graph_batch_size": 4096,
    "power_k": 2,
}

# Staged greedy sweep definition
# Each stage: list of (param_name, candidates) tuples to sweep jointly
STAGES = [
    # Stage 1: embedding dimension
    [("node_out_dim", [128,256,320])],
    # Stage 2: curvature
    [("k", [0.5, 1.0, 2.0])],
    # Stage 3: transformer architecture
    [("trans_num_layers", [2, 3]), ("trans_num_heads", [2, 4, 8])],
    # Stage 4: learning rate & attention kernel
    [("hyp_lr", [0.0001, 0.001]), ("power_k", [2, 3])],
    # Stage 5: training patience
    [("patience", [5, 10])],
]

# Parameters that require restarting from batching (not just training)
BATCHING_PARAMS = {"intra_graph_batch_size"}


def _get_env():
    """Build environment with PYTHONPATH so PIDSMaker finds both itself and myproject."""
    env = os.environ.copy()
    extra = os.pathsep.join([PIDSMAKER_DIR, MYPROJECT_DIR])
    env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
    return env


def parse_adp_score(output):
    """Extract the last adp_score from PIDSMaker output."""
    matches = re.findall(r"adp_score:\s*([\d.]+)", output)
    if matches:
        return float(matches[-1])
    return None


def build_command(params):
    """Build the PIDSMaker CLI command with parameter overrides."""
    # Determine force_restart level
    changed_batching = any(
        p in BATCHING_PARAMS and params[p] != DEFAULTS[p] for p in params
    )
    restart_from = "batching" if changed_batching else "training"

    cmd = [
        sys.executable, "-m", "pidsmaker.main",
        MODEL, DATASET,
        "--force_restart", restart_from,
    ]

    for name, value in params.items():
        path = PARAM_PATHS[name]
        cmd.append(f"--{path}={value}")

    return cmd


def run_experiment(params, run_id):
    """Run a single experiment and return results."""
    cmd = build_command(params)
    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if v != DEFAULTS[k])
    if not param_str:
        param_str = "(defaults)"

    print(f"\n{'='*70}")
    print(f"Run {run_id}: {param_str}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=PIDSMAKER_DIR,
            env=_get_env(),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per run
        )
        output = result.stdout + "\n" + result.stderr
        adp = parse_adp_score(output)

        if result.returncode != 0:
            print(f"  [WARN] Non-zero exit code: {result.returncode}")
            # Still try to parse ADP in case evaluation completed before error
            if adp is None:
                # Print last 30 lines of output for debugging
                lines = output.strip().split("\n")
                print("  Last 30 lines of output:")
                for line in lines[-30:]:
                    print(f"    {line}")

        if adp is not None:
            print(f"  ADP score: {adp}")
        else:
            print("  ADP score: FAILED TO PARSE")

    except subprocess.TimeoutExpired:
        print("  [ERROR] Timed out after 2 hours")
        adp = None
        output = ""

    return {
        "run_id": run_id,
        "adp_score": adp,
        "params": dict(params),
        "timestamp": datetime.now().isoformat(),
    }


def write_csv_header(csv_path, param_names):
    """Write CSV header if file doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "stage", "adp_score", "timestamp"] + param_names)


def append_csv_row(csv_path, run_result, stage, param_names):
    """Append a result row to CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [
            run_result["run_id"],
            stage,
            run_result["adp_score"],
            run_result["timestamp"],
        ]
        row += [run_result["params"].get(p, "") for p in param_names]
        writer.writerow(row)


def main():
    all_param_names = sorted(PARAM_PATHS.keys())
    write_csv_header(RESULTS_CSV, all_param_names)

    # Start with defaults
    best_params = dict(DEFAULTS)
    best_adp = None
    run_id = 0
    all_results = []

    print("Hyperparameter Sweep for ADP Optimization")
    print(f"Dataset: {DATASET}")
    print(f"Results: {RESULTS_CSV}")
    print(f"Stages: {len(STAGES)}")
    print()

    for stage_idx, stage_spec in enumerate(STAGES, 1):
        param_names = [name for name, _ in stage_spec]
        candidate_lists = [candidates for _, candidates in stage_spec]
        combos = list(itertools.product(*candidate_lists))

        print(f"\n{'#'*70}")
        print(f"Stage {stage_idx}/{len(STAGES)}: Sweeping {', '.join(param_names)}")
        print(f"  Combinations: {len(combos)}")
        print(f"  Current best params: { {k: v for k, v in best_params.items() if v != DEFAULTS[k]} or '(defaults)' }")
        if best_adp is not None:
            print(f"  Current best ADP: {best_adp}")
        print(f"{'#'*70}")

        stage_best_adp = None
        stage_best_params = None

        for combo in combos:
            run_id += 1
            # Build params: best so far + this combo's overrides
            params = dict(best_params)
            for name, value in zip(param_names, combo):
                params[name] = value

            result = run_experiment(params, run_id)
            all_results.append(result)
            append_csv_row(RESULTS_CSV, result, stage_idx, all_param_names)

            if result["adp_score"] is not None:
                if stage_best_adp is None or result["adp_score"] > stage_best_adp:
                    stage_best_adp = result["adp_score"]
                    stage_best_params = dict(params)

        # Update best params from this stage
        if stage_best_params is not None:
            best_params = stage_best_params
            best_adp = stage_best_adp
            print(f"\n  Stage {stage_idx} winner: ADP={best_adp}")
            for name in param_names:
                print(f"    {name} = {best_params[name]}")
        else:
            print(f"\n  [WARN] Stage {stage_idx}: All runs failed. Keeping previous best.")

    # Final summary
    print(f"\n{'='*70}")
    print("SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Best ADP score: {best_adp}")
    print("Best parameters:")
    for k in sorted(best_params.keys()):
        marker = " *" if best_params[k] != DEFAULTS[k] else ""
        print(f"  {k}: {best_params[k]}{marker}")
    print(f"\nTotal runs: {run_id}")
    print(f"Successful: {sum(1 for r in all_results if r['adp_score'] is not None)}")
    print(f"Failed: {sum(1 for r in all_results if r['adp_score'] is None)}")
    print(f"\nFull results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
