"""
Runs every experiment in the paper with the current code, then the final evaluations and the
step-time benchmark. Resumable: finished jobs (existing result files) are skipped.

    python run_paper_experiments.py --root paper_runs --seeds 0,1,2 --max-parallel 10

Layout:
    <root>/<exp>/config.cfg
    <root>/<exp>/results/<method>_t<seed>.npz        training reward history (one trial)
    <root>/<exp>/weights/<agent>/trial_<seed>.pt     checkpoints of the main methods
    <root>/<exp>/weights_<method>/...                checkpoints of the averaging ablation
    <root>/<exp>/logs/<method>_t<seed>.log
    <root>/<exp>/eval/evaluation_summary.json        final evaluation (10 episodes, 100 steps)
    <root>/benchmark.json                            per-step time (PointMaze-UMaze), pooled over the seeds
    <root>/benchmark/times_s<seed>.npz               raw per-step times of each benchmark seed
"""
import argparse
import configparser
import json
import os
import subprocess
import sys
import time

import numpy as np

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters of the paper (hyperparameter table).
COMMON = {
    "GENERAL": {"seed": "0", "n_trials": "1", "buffer_size": "200000", "n_workers": "1"},
    "AGENT": {"gamma": "0.95", "epsilon": "0.60", "T": "200", "print_ev": "5000", "save_ev": "200",
              "save_checkpoints": "True"},
    "SFQL": {"learning_rate": "0.001", "learning_rate_prior": "0.00001", "learning_rate_w": "0.5",
             "use_true_reward": "False", "reward_model": "linear", "successor_representation": "sf"},
    "DQN": {"learning_rate": "0.007"},
    "FGSF": {"n_averaging": "5"},
}
MAZE_TASK = {"env": "maze", "maze_goal_cells": "[(1, 1)]", "maze_reset_cell": "(3, 3)",
             "maze_include_goal_in_state": "True", "continuous_actions": "True",
             "maze_similarity_radius": "1", "maze_goal_jitter": "0"}
EXPERIMENTS = {
    "fourrooms": {"GENERAL": {"n_samples": "10000", "n_tasks": "6", "n_batch": "512"},
                  "TASK": {"env": "gridworld"}},
    # Reacher: 40k steps per task and exploration rate 0.55
    "reacher": {"GENERAL": {"n_samples": "40000", "n_tasks": "4", "n_batch": "512"}, "AGENT": {"epsilon": "0.55"},
                "TASK": {"env": "reacher",
                         "reacher_train_targets": "[(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]",
                         "reacher_include_target_in_state": "False"}},
    "pointmaze_umaze": {"GENERAL": {"n_samples": "30000", "n_tasks": "8", "n_batch": "512"},
                        "TASK": {**MAZE_TASK, "maze_env_id": "PointMaze_UMaze-v3"}},
    "pointmaze_medium": {"GENERAL": {"n_samples": "30000", "n_tasks": "8", "n_batch": "512"},
                         "TASK": {**MAZE_TASK, "maze_env_id": "PointMaze_Medium-v3"}},
    "pointmaze_large": {"GENERAL": {"n_samples": "30000", "n_tasks": "8", "n_batch": "512"},
                        "TASK": {**MAZE_TASK, "maze_env_id": "PointMaze_Large-v3"}},
}
MAIN_METHODS = ["DQN", "FGDQN", "SFDQN", "FG-SFDQN", "alg2", "alg3"]
EVAL_MODELS = ["DQN", "FGDQN", "SFDQN", "FGSFDQN", "alg2", "alg3"]   # checkpoint directory names
ABLATION = {"fourrooms": ["AvgFGSFDQN", "AvgFGSFDQN-N10", "AvgFGSFDQN-N20"],
            "reacher": ["AvgFGSFDQN", "AvgFGSFDQN-N10", "AvgFGSFDQN-N20"]}
# Rough relative cost, used to start long jobs first within a seed.
COST = {"fourrooms": 1, "reacher": 8, "pointmaze_umaze": 6, "pointmaze_medium": 6, "pointmaze_large": 6}
METHOD_COST = {"DQN": 0.5, "FGDQN": 0.5, "SFDQN": 1.5, "FG-SFDQN": 1.0, "alg2": 1.0, "alg3": 1.5,
               "AvgFGSFDQN": 1.5, "AvgFGSFDQN-N10": 2.0, "AvgFGSFDQN-N20": 2.5}


def base_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(HERE, "configs", "config.cfg"))  # final Four Rooms config; provides the maze layout
    return cfg


def write_configs(root, smoke=False, dqn_lr=None):
    paths = {}
    for exp, over in EXPERIMENTS.items():
        cfg = base_config()
        for section in list(COMMON) + list(over):
            if not cfg.has_section(section):
                cfg.add_section(section)
        for section, vals in list(COMMON.items()) + list(over.items()):
            for k, v in vals.items():
                cfg[section][k] = v
        if dqn_lr is not None:  # DQN / FG-DQN learning rate
            cfg["DQN"]["learning_rate"] = str(dqn_lr)
        if smoke:  # tiny budgets to test the pipeline end to end
            cfg["GENERAL"]["n_samples"], cfg["GENERAL"]["n_batch"] = "300", "32"
            cfg["GENERAL"]["n_tasks"] = "2"
            cfg["AGENT"]["print_ev"], cfg["AGENT"]["save_ev"] = "100", "50"
        d = os.path.join(root, exp)
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        os.makedirs(os.path.join(d, "logs"), exist_ok=True)
        path = os.path.join(d, "config.cfg")
        with open(path, "w") as f:
            cfg.write(f)
        paths[exp] = path
    return paths


def build_jobs(root, seeds, experiments):
    jobs = []
    for seed in seeds:
        seed_jobs = []
        for exp in experiments:
            for m in MAIN_METHODS + ABLATION.get(exp, []):
                seed_jobs.append((exp, m, seed))
        # Four Rooms first (quick feedback), then the longest jobs first.
        seed_jobs.sort(key=lambda j: (j[0] != "fourrooms", -COST[j[0]] * METHOD_COST[j[1]]))
        jobs += seed_jobs
    return jobs


def job_paths(root, exp, method, seed):
    d = os.path.join(root, exp)
    weights = os.path.join(d, "weights") if method in MAIN_METHODS else os.path.join(d, f"weights_{method}")
    return (os.path.join(d, "results", f"{method}_t{seed}.npz"), weights,
            os.path.join(d, "logs", f"{method}_t{seed}.log"))


def launch(root, cfg_path, exp, method, seed, project):
    out, weights, log = job_paths(root, exp, method, seed)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_SILENT="true",
               FGSF_LOG_FILE=log)
    cmd = [PY, os.path.join(HERE, "train_parallel.py"), "--config", cfg_path, "--methods", method,
           "--trial-ids", str(seed), "--out", out, "--weights-dir", weights,
           "--wandb-project", project, "--wandb-group", exp]
    stdout = open(log + ".stdout", "w")
    return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=stdout, stderr=subprocess.STDOUT)


def status(msg, root):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(os.path.join(root, "orchestrator.log"), "a") as f:
        f.write(line + "\n")


def run_training(root, cfg_paths, jobs, max_parallel, project, retries=1):
    pending = [j for j in jobs if not os.path.exists(job_paths(root, *j)[0])]
    status(f"training: {len(jobs)} jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run", root)
    attempts = {j: 0 for j in pending}
    running = {}
    while pending or running:
        while pending and len(running) < max_parallel:
            j = pending.pop(0)
            attempts[j] += 1
            running[j] = (launch(root, cfg_paths[j[0]], *j, project), time.time())
            status(f"start  {j[0]:17s} {j[1]:15s} seed {j[2]} (attempt {attempts[j]})", root)
        time.sleep(20)
        for j, (proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[j]
            ok = rc == 0 and os.path.exists(job_paths(root, *j)[0])
            status(f"{'done  ' if ok else 'FAILED'} {j[0]:17s} {j[1]:15s} seed {j[2]} "
                   f"rc={rc} in {(time.time() - t0) / 60:.1f} min", root)
            if not ok and attempts[j] <= retries:
                pending.append(j)
    failed = [j for j in jobs if not os.path.exists(job_paths(root, *j)[0])]
    status(f"training finished; {len(failed)} job(s) without results: {failed}", root)
    return failed


def run_evaluation(root, cfg_paths, experiments):
    for exp in experiments:
        out_dir = os.path.join(root, exp, "eval")
        if os.path.exists(os.path.join(out_dir, "evaluation_summary.json")):
            continue
        status(f"evaluating {exp}", root)
        env = dict(os.environ, OMP_NUM_THREADS="1", WANDB_MODE="disabled",
                   FGSF_LOG_FILE=os.path.join(root, exp, "logs", "eval.log"))
        cmd = [PY, os.path.join(HERE, "evaluate_parallel.py"), "--config", cfg_paths[exp],
               "--weight-root", os.path.join(root, exp, "weights"), "--out-root", out_dir,
               "--models", *EVAL_MODELS, "--n-episodes", "10", "--n-eval-runs", "1", "--max-steps", "100",
               "--checkpoint-mode", "all", "--video-checkpoint", "none", "--n-workers", "6"]
        with open(os.path.join(root, exp, "logs", "eval.stdout"), "w") as f:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        status(f"evaluation {exp} rc={rc}", root)


def run_benchmark(root, cfg_paths, seeds, warmup=512, timed=1500):
    """Per-step training time on PointMaze-UMaze: one run per seed, `timed` steps after `warmup` steps,
    pooled over the seeds into <root>/benchmark.json (mean, median and variance in ms)."""
    out = os.path.join(root, "benchmark.json")
    if os.path.exists(out):
        return
    status("benchmarking per-step time on PointMaze-UMaze", root)
    bdir = os.path.join(root, "benchmark")
    os.makedirs(bdir, exist_ok=True)
    smoke = json.load(open(os.path.join(root, "plan.json"))).get("smoke", False)
    if smoke:
        warmup, timed = 10, 50
    raw = {}
    for seed in seeds:
        times = os.path.join(bdir, f"times_s{seed}.npz")
        env = dict(os.environ, WANDB_MODE="disabled", FGSF_LOG_FILE=os.path.join(bdir, f"bench_s{seed}.log"))
        cmd = [PY, os.path.join(HERE, "benchmark_step_time.py"), "--config", cfg_paths["pointmaze_umaze"],
               "--n-tasks", "2" if smoke else "8", "--steps", str(warmup + timed), "--batch-size", "512",
               "--seed", str(seed), "--methods", "DQN,SFDQN,FGSFDQN,alg2,alg3,FGDQN",
               "--out", os.path.join(bdir, f"bench_s{seed}.json"), "--times-out", times]
        with open(os.path.join(bdir, f"bench_s{seed}.stdout"), "w") as f:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        status(f"benchmark seed {seed} rc={rc}", root)
        for m, t in np.load(times).items():
            raw.setdefault(m, []).append(np.asarray(t, dtype=float)[warmup:])
    summary = {}
    for m, ts in raw.items():
        t = np.concatenate(ts)
        summary[m] = {"mean_ms": float(t.mean()), "median_ms": float(np.median(t)), "var_ms": float(t.var())}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="paper_runs")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--experiments", default=",".join(EXPERIMENTS))
    p.add_argument("--max-parallel", type=int, default=10)
    p.add_argument("--wandb-project", default="FG-SFRQL-paper-v1")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny budgets, for testing the pipeline")
    p.add_argument("--dqn-lr", type=float, default=None, help="DQN/FG-DQN learning rate (default 0.007)")
    p.add_argument("--methods", default=None, help="comma-separated subset of methods to train (default: all)")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg_paths = write_configs(root, smoke=args.smoke, dqn_lr=args.dqn_lr)
    with open(os.path.join(root, "plan.json"), "w") as f:
        json.dump({"seeds": seeds, "experiments": experiments, "main_methods": MAIN_METHODS,
                   "ablation": ABLATION, "wandb_project": args.wandb_project, "smoke": args.smoke}, f, indent=2)

    jobs = build_jobs(root, seeds, experiments)
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",")}
        jobs = [j for j in jobs if j[1] in keep]
    run_training(root, cfg_paths, jobs, args.max_parallel, args.wandb_project)
    if not args.skip_eval:
        run_evaluation(root, cfg_paths, experiments)
        run_benchmark(root, cfg_paths, seeds)
    status("all done", root)


if __name__ == "__main__":
    main()
