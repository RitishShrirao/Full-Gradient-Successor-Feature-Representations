"""
Slippery domains (the action is replaced by a random one with probability p), p in {0.2, 0.3} (--ps): trains every
method and runs its final evaluation. Resumable: finished jobs are skipped.
  - Four Rooms (--env fourrooms): 6 tasks x 30k steps, otherwise the Four Rooms settings.
  - Reacher (--env reacher): the Reacher settings of the paper (4 tasks x 40k steps). The averaged methods
    sample their N next states of each (s, a) from the simulator (FGSF.simulator_resampling).
  - PointMaze (--env pointmaze_large, ...): the PointMaze settings of the paper, with simulator resampling
    as for Reacher (MuJoCo state restore).

    python run_slippery.py --root results/slippery_fourrooms --seeds 0,1,2 --max-parallel 11
    python run_slippery.py --env reacher --root results/slippery_reacher --seeds 0,1,2 --max-parallel 14
    python run_slippery.py --env pointmaze_large --ps 0.3 --root results/slippery_pointmaze_large --seeds 0,1,2

Layout:
    <root>/p<p>/config.cfg
    <root>/p<p>/<method>_s<seed>.npz                         training reward history
    <root>/p<p>/eval_<method>_s<seed>/evaluation_summary.json  final evaluation (10 episodes, 100 steps)
    <root>/p<p>/weights/<method>_s<seed>/                    checkpoints
    <root>/p<p>/logs/
"""
import argparse
import configparser
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
SLIP_PS = ["0.2", "0.3"]
# method -> checkpoint directory name used by evaluate_parallel.py
METHODS = {"DQN": "DQN", "FGDQN": "FGDQN", "SFDQN": "SFDQN", "FG-SFDQN": "FGSFDQN",
           "AvgFGSFDQN": "AvgFGSFDQN", "alg2": "alg2", "alg3": "alg3"}


def write_config(d, p, env="fourrooms"):
    if env != "fourrooms":
        import run_paper_experiments as rpe
        cfg = rpe.base_config()
        for section, vals in list(rpe.COMMON.items()) + list(rpe.EXPERIMENTS[env].items()):
            if not cfg.has_section(section):
                cfg.add_section(section)
            for k, v in vals.items():
                cfg[section][k] = v
        cfg["FGSF"]["simulator_resampling"] = "True"
    else:
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(HERE, "configs", "config.cfg"))
        cfg["GENERAL"]["n_samples"] = "30000"
    cfg["TASK"]["slip_prob"] = p
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    path = os.path.join(d, "config.cfg")
    with open(path, "w") as f:
        cfg.write(f)
    return path


def job(d, cfg, method, seed):
    name, weights = f"{method}_s{seed}", os.path.join(d, "weights", f"{method}_s{seed}")
    out, ev = os.path.join(d, f"{name}.npz"), os.path.join(d, f"eval_{name}")
    env = dict(os.environ, OMP_NUM_THREADS="1", WANDB_MODE="disabled")
    if not os.path.exists(out):
        log = os.path.join(d, "logs", name)
        with open(log + ".stdout", "w") as f:
            subprocess.call([PY, os.path.join(HERE, "train_parallel.py"), "--config", cfg, "--methods", method,
                             "--trial-ids", str(seed), "--out", out, "--weights-dir", weights],
                            cwd=HERE, env=dict(env, FGSF_LOG_FILE=log + ".log"), stdout=f, stderr=subprocess.STDOUT)
    if os.path.exists(out) and not os.path.exists(os.path.join(ev, "evaluation_summary.json")):
        log = os.path.join(d, "logs", f"eval_{name}")
        with open(log + ".stdout", "w") as f:
            subprocess.call([PY, os.path.join(HERE, "evaluate_parallel.py"), "--config", cfg, "--weight-root", weights,
                             "--out-root", ev, "--models", METHODS[method], "--n-episodes", "10", "--n-eval-runs", "1",
                             "--max-steps", "100", "--checkpoint-mode", "all", "--video-checkpoint", "none",
                             "--n-workers", "1"],
                            cwd=HERE, env=dict(env, FGSF_LOG_FILE=log + ".log"), stdout=f, stderr=subprocess.STDOUT)
    done = os.path.exists(os.path.join(ev, "evaluation_summary.json"))
    print(f"{d} {name}: {'done' if done else 'FAILED'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="fourrooms", choices=["fourrooms", "reacher", "pointmaze_umaze", "pointmaze_medium",
                                                         "pointmaze_large"])
    p.add_argument("--ps", default=",".join(SLIP_PS), help="slip probabilities")
    p.add_argument("--root", default="slippery_runs")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--max-parallel", type=int, default=11)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    jobs = []
    for sp in args.ps.split(","):
        d = os.path.join(os.path.abspath(args.root), f"p{sp}")
        cfg = write_config(d, sp, args.env)
        jobs += [(d, cfg, m, s) for s in seeds for m in methods]
    with ThreadPoolExecutor(args.max_parallel) as ex:
        list(ex.map(lambda j: job(*j), jobs))


if __name__ == "__main__":
    main()
