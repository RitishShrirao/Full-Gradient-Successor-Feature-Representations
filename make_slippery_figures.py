"""
Slippery Four Rooms figures of the paper, from the runs of run_slippery.py.

    python make_slippery_figures.py --root results/slippery_fourrooms --out figures
    python make_slippery_figures.py --env reacher --root results/slippery_reacher --out figures
    python make_slippery_figures.py --env pointmaze_large --ps 0.3 --root results/slippery_pointmaze_large --out figures

Outputs (in --out):
    slippery_fourrooms.pdf   total training reward and final evaluation for p = 0.2, 0.3,
                             (mean +- std over seeds)
    slippery_total.pdf       running total of the training reward (mean +- std over seeds)
    slippery_train_tasks.pdf cumulative reward on the active task, sequential methods (mean +- std over seeds)
    slippery_eval_tasks.pdf  final evaluation per task, all methods (mean +- std over seeds); *_wide.pdf: README version
With --env reacher the same figures are slippery_reacher.pdf, slippery_reacher_total.pdf,
slippery_reacher_train_tasks.pdf and slippery_reacher_eval_tasks.pdf.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from make_paper_figures import COLORS, OURS

SLIP_PS, SLIP_SEEDS, SLIP_TASKS, SLIP_SAVE_EV = ["0.2", "0.3"], [0, 1, 2], 6, 200
# figure names per environment (set by --env, together with SLIP_TASKS)
ENVS = {"fourrooms": (6, {"bars": "slippery_fourrooms", "total": "slippery_total",
                          "train_tasks": "slippery_train_tasks", "eval_tasks": "slippery_eval_tasks"}),
        "reacher": (4, {"bars": "slippery_reacher", "total": "slippery_reacher_total",
                        "train_tasks": "slippery_reacher_train_tasks", "eval_tasks": "slippery_reacher_eval_tasks"}),
        "pointmaze_large": (8, {"bars": "slippery_pointmaze_large", "total": "slippery_pointmaze_large_total",
                                "train_tasks": "slippery_pointmaze_large_train_tasks",
                                "eval_tasks": "slippery_pointmaze_large_eval_tasks"})}
NAMES = ENVS["fourrooms"][1]
# npz key -> (eval key, label, color)
METHODS = {"DQN": ("DQN", "DQN", COLORS["DQN"]),
           "FGDQN": ("FGDQN", "FG-DQN", COLORS["FGDQN"]),
           "SFDQN": ("SFDQN", "SFDQN", COLORS["SFDQN"]),
           "FG-SFDQN": ("FGSFDQN", "FG-SFDQN (Alg. 1)", COLORS["FG-SFDQN"]),
           "AvgFGSFDQN": ("AvgFGSFDQN", "FG-SFDQN (Alg. 1), averaged N=5", COLORS["AvgFGSFDQN"]),
           "alg2": ("alg2", "FG-SFDQN (Alg. 2)", COLORS["alg2"]),
           "alg3": ("alg3", "FG-SFDQN (Alg. 3)", COLORS["alg3"])}
RANDOMIZED = {"alg2", "alg3"}   # their history is already one running total over all tasks
SEQUENTIAL = [k for k in METHODS if k not in RANDOMIZED]


def slip_history(root, p, key, s):
    return np.asarray(np.load(os.path.join(root, f"p{p}", f"{key}_s{s}.npz"))[key], dtype=float).reshape(-1)


def slip_eval(root, p, key, s):
    return json.load(open(os.path.join(root, f"p{p}", f"eval_{key}_s{s}", "evaluation_summary.json")))


def running_total(h, n_tasks):
    ppt = h.size // n_tasks
    segs, offset = [], 0.0
    for t in range(n_tasks):
        seg = h[t * ppt:(t + 1) * ppt]
        segs.append(seg + offset)
        offset += seg[-1]
    return np.concatenate(segs)


def slippery(root, out):
    res = {p: {k: {"total": [], "eval": [], "curve": []} for k in METHODS} for p in SLIP_PS}
    for p in SLIP_PS:
        for s in SLIP_SEEDS:
            for k, (ek, _, _) in METHODS.items():
                ev = slip_eval(root, p, k, s)
                h = slip_history(root, p, k, s)
                c = h if k in RANDOMIZED else running_total(h, SLIP_TASKS)
                res[p][k]["curve"].append(c)
                res[p][k]["total"].append(c[-1])
                res[p][k]["eval"].append(ev[ek]["overall_mean"])

    rows = (len(METHODS) + 1) // 2   # legend rows
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85 + 0.15 * rows))
    width = 0.76 / len(METHODS)
    for ax, metric, ylab in [(axes[0], "total", "total training reward"), (axes[1], "eval", "final evaluation")]:
        for j, (k, (_, lab, col)) in enumerate(METHODS.items()):
            x = np.arange(len(SLIP_PS)) + (j - (len(METHODS) - 1) / 2) * width
            v = np.array([res[p][k][metric] for p in SLIP_PS])   # [ps, seeds]
            ax.bar(x, v.mean(1), width, yerr=v.std(1), color=col, label=lab, linewidth=0.5,
                   edgecolor="black" if k in OURS or k == "FG-SFDQN" else "none", capsize=1, error_kw={"lw": 0.5})
        ax.set_xticks(np.arange(len(SLIP_PS))); ax.set_xticklabels([f"p={p}" for p in SLIP_PS])
        ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.04 + 0.06 * rows, 1, 1)); fig.savefig(os.path.join(out, NAMES["bars"] + ".pdf")); plt.close(fig)

    fig, axes = plt.subplots(1, len(SLIP_PS), figsize=(3.4, 1.85 + 0.15 * rows), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, SLIP_PS):
        for k, (_, lab, col) in METHODS.items():
            arr = np.stack(res[p][k]["curve"])
            x = np.arange(1, arr.shape[1] + 1) * SLIP_SAVE_EV
            ours = k in OURS or k == "FG-SFDQN"
            ax.plot(x, arr.mean(0), color=col, label=lab, lw=1.7 if ours else 1.1, ls="-" if ours else "--")
            ax.fill_between(x, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=col, alpha=0.18, lw=0)
        for t in range(1, SLIP_TASKS):   # task boundaries of the sequential methods
            ax.axvline(t * x[-1] / SLIP_TASKS, color="grey", ls=":", lw=0.6)
        ax.set_title(f"slip p = {p}"); ax.set_xlabel("env. steps"); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[0].set_ylabel("total cumulative reward")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.04 + 0.06 * rows, 1, 1)); fig.savefig(os.path.join(out, NAMES["total"] + ".pdf")); plt.close(fig)
    per_task(root, out)
    return {p: {k: {f"{m}_{st}": float(fn(v[m])) for m in ("total", "eval") for st, fn in (("mean", np.mean), ("std", np.std))}
                for k, v in r.items()} for p, r in res.items()}


def per_task(root, out):
    """Per-task figures of the sequential methods, as for the deterministic domains."""
    figsize = (3.4, 1.85 + 0.15 * ((len(SEQUENTIAL) + 1) // 2))
    fig, axes = plt.subplots(1, len(SLIP_PS), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, SLIP_PS):
        for k in SEQUENTIAL:
            _, lab, col = METHODS[k]
            arr = np.stack([slip_history(root, p, k, s) for s in SLIP_SEEDS])
            x = np.arange(1, arr.shape[1] + 1) * SLIP_SAVE_EV
            ours = k in OURS or k == "FG-SFDQN"
            ax.plot(x, arr.mean(0), color=col, label=lab, lw=1.7 if ours else 1.1, ls="-" if ours else "--")
            ax.fill_between(x, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=col, alpha=0.18, lw=0)
        for t in range(1, SLIP_TASKS):
            ax.axvline(t * x[-1] / SLIP_TASKS, color="grey", ls=":", lw=0.6)
        ax.set_title(f"slip p = {p}"); ax.set_xlabel("env. steps"); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[0].set_ylabel("cumulative reward\n(active task)")
    h, l = axes[0].get_legend_handles_labels()
    rows = (len(SEQUENTIAL) + 1) // 2
    fig.legend(h, l, ncol=2, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.04 + 0.06 * rows, 1, 1)); fig.savefig(os.path.join(out, NAMES["train_tasks"] + ".pdf")); plt.close(fig)

    # final evaluation per task: all methods (the randomized ones are evaluated per task as well);
    # a one-column version for the paper and a wide version (suffix _wide) for the README
    eval_tasks(root, out, wide=False)
    eval_tasks(root, out, wide=True)


def eval_tasks(root, out, wide):
    ncol = 4 if wide else 3
    rows = -(-len(METHODS) // ncol)
    size = (6.8, 2.0 + 0.15 * rows) if wide else (4.8, 1.55 + 0.15 * rows)
    fig, axes = plt.subplots(1, len(SLIP_PS), figsize=size, sharey=True, squeeze=False)
    axes = axes[0]
    width = 0.8 / len(METHODS)
    for ax, p in zip(axes, SLIP_PS):
        for j, k in enumerate(METHODS):
            ek, lab, col = METHODS[k]
            v = np.array([slip_eval(root, p, k, s)[ek]["task_mean"] for s in SLIP_SEEDS])   # [seeds, tasks]
            x = np.arange(SLIP_TASKS) + (j - (len(METHODS) - 1) / 2) * width
            ax.bar(x, v.mean(0), width, yerr=v.std(0), color=col, label=lab, linewidth=0.5,
                   edgecolor="black" if k in OURS or k == "FG-SFDQN" else "none", capsize=1, error_kw={"lw": 0.5})
        ax.set_xticks(np.arange(SLIP_TASKS)); ax.set_xticklabels([f"T{t + 1}" for t in range(SLIP_TASKS)])
        ax.set_title(f"slip p = {p}"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("mean return")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=ncol, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.05 + 0.075 * rows, 1, 1))
    fig.savefig(os.path.join(out, NAMES["eval_tasks"] + ("_wide" if wide else "") + ".pdf")); plt.close(fig)


def main():
    global SLIP_TASKS, NAMES, SLIP_PS
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="fourrooms", choices=list(ENVS))
    p.add_argument("--root", default="results/slippery_fourrooms")
    p.add_argument("--out", default="figures")
    p.add_argument("--ps", default=",".join(SLIP_PS), help="slip probabilities")
    args = p.parse_args()
    SLIP_TASKS, NAMES = ENVS[args.env]
    SLIP_PS = args.ps.split(",")
    os.makedirs(args.out, exist_ok=True)
    # TrueType (Type 42) fonts: PDF checkers of submission portals reject Type 3 fonts
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    print(json.dumps(slippery(args.root, args.out), indent=1))


if __name__ == "__main__":
    main()
