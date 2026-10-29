"""
Aggregates the runs of run_paper_experiments.py into the paper's figures and tables.

    python make_paper_figures.py --root results/paper_runs --out figures

Outputs (in --out):
    train_<exp>.pdf        cumulative training reward per task, sequential methods (mean +- std over seeds)
    total_<exp>.pdf        total cumulative reward over all steps, all methods incl. Alg. 2/3
    ablation_<exp>.pdf     FG-SFDQN vs averaging with N = 5, 10, 20
    eval_<exp>.pdf         final evaluation per task (mean +- std over seeds)
    results_grid.pdf       figure of the main text: training curves above final evaluation, per domain
    diagnostics.pdf        Bellman residual and GPI tie rates (SFDQN vs full-gradient variants)
    table_{train,eval,overhead}.tex   tables of the paper
    results.json           all numbers used in the paper text and tables
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

EXPS = ["fourrooms", "reacher", "pointmaze_umaze", "pointmaze_medium", "pointmaze_large"]
TITLES = {"fourrooms": "Four Rooms", "reacher": "Reacher", "pointmaze_umaze": "PointMaze-UMaze",
          "pointmaze_medium": "PointMaze-Medium", "pointmaze_large": "PointMaze-Large"}
LABELS = {"DQN": "DQN", "FGDQN": "FG-DQN", "SFDQN": "SFDQN", "FG-SFDQN": "FG-SFDQN (Alg. 1)",
          "alg2": "FG-SFDQN (Alg. 2)", "alg3": "FG-SFDQN (Alg. 3)",
          "AvgFGSFDQN": "Averaging, N=5", "AvgFGSFDQN-N10": "Averaging, N=10", "AvgFGSFDQN-N20": "Averaging, N=20"}
EVAL_NAME = {"FGSFDQN": "FG-SFDQN"}  # checkpoint dir name -> method name
# Baselines in light-to-dark greys, our full-gradient SF methods in saturated colours (drawn bolder, see OURS).
COLORS = {"DQN": "#c8c8c8", "FGDQN": "#969696", "SFDQN": "#5a5a5a",
          "FG-SFDQN": "#1a9850", "alg2": "#2166ac", "alg3": "#c51b7d",
          "AvgFGSFDQN": "#e66101", "AvgFGSFDQN-N10": "#fdb863", "AvgFGSFDQN-N20": "#8c510a"}
OURS = {"FG-SFDQN", "alg2", "alg3", "AvgFGSFDQN", "AvgFGSFDQN-N10", "AvgFGSFDQN-N20"}
SEQUENTIAL = ["DQN", "FGDQN", "SFDQN", "FG-SFDQN"]
RANDOMIZED = ["alg2", "alg3"]
ABLATION = ["FG-SFDQN", "AvgFGSFDQN", "AvgFGSFDQN-N10", "AvgFGSFDQN-N20"]


def load_histories(root, exp):
    """method -> array [n_seeds, n_points] of logged cumulative reward (save_ev apart)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, exp, "results", "*_t*.npz"))):
        method = re.sub(r"_t\d+\.npz$", "", os.path.basename(f))
        data = np.load(f)
        arr = np.asarray(data[method], dtype=float).reshape(-1)
        out.setdefault(method, []).append(arr)
    return {m: np.stack(v) for m, v in out.items() if len({a.size for a in v}) == 1}


def running_total(h, n_tasks, randomized):
    """Sequential methods restart at 0 for every task; chain them into one running total."""
    if randomized:
        return h
    ppt = h.shape[-1] // n_tasks
    segs, offset = [], np.zeros(h.shape[0])
    for t in range(n_tasks):
        seg = h[:, t * ppt:(t + 1) * ppt]
        segs.append(seg + offset[:, None])
        offset = offset + seg[:, -1]
    return np.concatenate(segs, axis=1)


def band(ax, x, arr, method, label=None):
    m, s = arr.mean(0), arr.std(0)
    ours = method in OURS
    ax.plot(x, m, color=COLORS.get(method), label=label or LABELS.get(method, method), lw=1.7 if ours else 1.1,
            ls="-" if ours else "--", zorder=3 if ours else 2)
    if arr.shape[0] > 1:
        ax.fill_between(x, m - s, m + s, color=COLORS.get(method), alpha=0.18, lw=0)


def cfg_value(root, exp, section, key):
    import configparser
    c = configparser.ConfigParser()
    c.read(os.path.join(root, exp, "config.cfg"))
    return c[section][key]


TABLE_METHODS = ["DQN", "FGDQN", "SFDQN", "FG-SFDQN", "alg2", "alg3"]
TABLE_LABELS = {"DQN": "DQN", "FGDQN": "FG-DQN", "SFDQN": "SFDQN", "FG-SFDQN": "FG-SFDQN (Alg.~1)",
                "alg2": "FG-SFDQN (Alg.~2)", "alg3": "FG-SFDQN (Alg.~3)"}
SHORT = {"fourrooms": "4R", "reacher": "Reacher", "pointmaze_umaze": "PM-U",
         "pointmaze_medium": "PM-M", "pointmaze_large": "PM-L"}


def _fmt(mean, std, bold, digits):
    txt = f"{mean:.{digits}f}\\,$\\pm$\\,{std:.{digits}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def write_tables(results, out):
    """LaTeX tables: training totals, final evaluation (mean +- std over seeds), step times."""
    exps = [e for e in EXPS if e in results]
    for kind, key, digits, caption, label in [
        ("train", "total", 0, "Total cumulative training reward over all tasks (mean\\,$\\pm$\\,std over seeds).",
         "tab:train_totals"),
        ("eval", "overall", 2, "Final evaluation: mean return per episode, averaged over tasks "
         "(10 episodes per task, 100-step cap; mean\\,$\\pm$\\,std over seeds).", "tab:final_eval")]:
        lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                 "\\setlength{\\tabcolsep}{4pt}",
                 "\\begin{tabular}{l" + "c" * len(exps) + "}", "\\hline",
                 "\\textbf{Method} & " + " & ".join(f"\\textbf{{{TITLES[e]}}}" for e in exps) + " \\\\", "\\hline"]
        best = {}
        for e in exps:
            vals = {m: v[f"{key}_mean"] for m, v in results[e][kind].items() if m in TABLE_METHODS}
            best[e] = max(vals, key=vals.get) if vals else None
        for m in TABLE_METHODS:
            cells = []
            for e in exps:
                v = results[e][kind].get(m)
                cells.append("--" if v is None else _fmt(v[f"{key}_mean"], v[f"{key}_std"], best[e] == m, digits))
            lines.append(f"{TABLE_LABELS[m]} & " + " & ".join(cells) + " \\\\")
            if m == "SFDQN":   # rule between the baselines and our methods
                lines.append("\\hline")
        lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
        with open(os.path.join(out, f"table_{kind}.tex"), "w") as f:
            f.write("\n".join(lines) + "\n")
    if "benchmark" in results:
        names = {"DQN": "DQN", "FGDQN": "FG-DQN", "SFDQN": "SFDQN", "FGSFDQN": "FG-SFDQN (Alg.~1)",
                 "alg1_avg": "FG-SFDQN (Alg.~1, $N=5$)", "alg2": "FG-SFDQN (Alg.~2)", "alg3": "FG-SFDQN (Alg.~3)"}
        med = all("median_ms" in results["benchmark"][k] for k in names if k in results["benchmark"])
        lines = ["\\begin{tabular}{lccc}" if med else "\\begin{tabular}{lcc}", "\\hline",
                 "\\textbf{Algorithm} & \\textbf{Mean (ms)} & " + ("\\textbf{Median (ms)} & " if med else "")
                 + "\\textbf{Std (ms)} \\\\", "\\hline"]
        for k in ["DQN", "FGDQN", "SFDQN", "FGSFDQN", "alg1_avg", "alg2", "alg3"]:
            if k in results["benchmark"]:
                b = results["benchmark"][k]
                cells = [f"{b['mean_ms']:.2f}"] + ([f"{b['median_ms']:.2f}"] if med else []) + [f"{b['var_ms'] ** 0.5:.2f}"]
                lines.append(f"{names[k]} & " + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        with open(os.path.join(out, "table_overhead.tex"), "w") as f:
            f.write("\n".join(lines) + "\n")


def results_grid(root, out):
    """Main-text figure: for each domain, the per-task cumulative training reward of the sequential methods
    (upper panel) above the final evaluation per task of all methods (lower panel); three domains per row,
    column widths growing with the number of tasks, the shared legend in the last slot."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    exps = [e for e in ["fourrooms", "reacher", "pointmaze_large", "pointmaze_medium", "pointmaze_umaze"]
            if os.path.isdir(os.path.join(root, e))]
    ncol = 3
    npair = -(-len(exps) // ncol)
    # column widths grow with the number of tasks, so that the bars of the 8-task domains are not squeezed;
    # the legend slot of an incomplete row gets a fixed width
    n_tasks_of = {e: int(cfg_value(root, e, "GENERAL", "n_tasks")) for e in exps}
    # rows: (training, evaluation) per row of domains, with a thin spacer between the pairs
    fig = plt.figure(figsize=(7.16, 0.82 * 2 * npair + 0.75))
    outer = fig.add_gridspec(2 * npair - 1, 1, height_ratios=([1, 0.42] * npair)[:-1], hspace=0.0,
                             left=0.09, right=0.995, top=0.955, bottom=0.07)
    grid = np.empty((2 * npair, ncol), dtype=object)
    for r in range(npair):
        row = exps[r * ncol:(r + 1) * ncol]
        widths = [n_tasks_of[e] + 3 for e in row] + [6] * (ncol - len(row))
        sub = outer[2 * r].subgridspec(2, ncol, width_ratios=widths, hspace=0.55, wspace=0.22)
        for i in range(2):
            for c in range(ncol):
                grid[2 * r + i, c] = fig.add_subplot(sub[i, c])
    nrow = 2 * npair
    axes = np.empty((2, len(exps)), dtype=object)
    for c in range(len(exps)):
        axes[0, c], axes[1, c] = grid[2 * (c // ncol), c % ncol], grid[2 * (c // ncol) + 1, c % ncol]
    for c, exp in enumerate(exps):
        hist = load_histories(root, exp)
        n_tasks = int(cfg_value(root, exp, "GENERAL", "n_tasks"))
        n_samples = int(cfg_value(root, exp, "GENERAL", "n_samples"))
        save_ev = int(cfg_value(root, exp, "AGENT", "save_ev"))
        ax = axes[0, c]
        for m in SEQUENTIAL:
            if m in hist:
                band(ax, np.arange(1, hist[m].shape[1] + 1) * save_ev, hist[m], m)
        for t in range(1, n_tasks):
            ax.axvline(t * n_samples, color="grey", ls=":", lw=0.5)
        ax.set_title(TITLES[exp]); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"))
        ev = json.load(open(os.path.join(root, exp, "eval", "evaluation_summary.json")))
        methods = [k for k in ["DQN", "FGDQN", "SFDQN", "FGSFDQN", "alg2", "alg3"] if k in ev]
        ax = axes[1, c]
        width = 0.88 / len(methods)
        for j, k in enumerate(methods):
            m = EVAL_NAME.get(k, k)
            per_seed = np.array([t["task_mean"] for t in ev[k]["per_trial"]])
            x = np.arange(per_seed.shape[1]) + (j - (len(methods) - 1) / 2) * width
            ax.bar(x, per_seed.mean(0), width, yerr=per_seed.std(0), color=COLORS.get(m),
                   edgecolor="black" if m in OURS else "none", linewidth=0.4, capsize=0.8, error_kw={"lw": 0.4})
        ax.set_xticks(np.arange(per_seed.shape[1]))
        ax.set_xticklabels([f"{t + 1}" for t in range(per_seed.shape[1])])
        ax.set_xlabel("task", labelpad=1); ax.grid(alpha=0.3, axis="y")
    for c in range(0, len(exps), ncol):
        axes[0, c].set_ylabel("training\nreward")
        axes[1, c].set_ylabel("final\nreturn")
    handles = [Line2D([], [], color=COLORS[m], lw=1.7 if m in OURS else 1.1, ls="-" if m in OURS else "--",
                      label=LABELS[m]) for m in SEQUENTIAL]
    handles += [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.4, label=LABELS[m]) for m in RANDOMIZED]
    free = [grid[r, c] for r in range(nrow) for c in range(ncol) if grid[r, c] not in axes.ravel()]
    for ax in free:
        ax.axis("off")
    if free:
        x0 = min(ax.get_position().x0 for ax in free)
        y1 = max(ax.get_position().y1 for ax in free)
        fig.legend(handles=handles, ncol=1, loc="upper left", bbox_to_anchor=(x0 + 0.005, y1), frameon=False,
                   handlelength=2.0, fontsize=7.5)
    else:
        fig.legend(handles=handles, ncol=6, loc="lower center", frameon=False, columnspacing=1.0, handlelength=1.8)
    fig.savefig(os.path.join(out, "results_grid.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/paper_runs")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # TrueType (Type 42) fonts: PDF checkers of submission portals reject Type 3 fonts
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    results = {}

    for exp in EXPS:
        if not os.path.isdir(os.path.join(args.root, exp)):
            continue
        hist = load_histories(args.root, exp)
        if not hist:
            continue
        n_tasks = int(cfg_value(args.root, exp, "GENERAL", "n_tasks"))
        n_samples = int(cfg_value(args.root, exp, "GENERAL", "n_samples"))
        save_ev = int(cfg_value(args.root, exp, "AGENT", "save_ev"))
        R = results.setdefault(exp, {"n_tasks": n_tasks, "n_samples": n_samples, "train": {}, "eval": {}})

        # training totals
        for m, h in hist.items():
            tot = running_total(h, n_tasks, m in RANDOMIZED)[:, -1]
            R["train"][m] = {"seeds": int(h.shape[0]), "total_mean": float(tot.mean()), "total_std": float(tot.std())}
            if m not in RANDOMIZED:
                ppt = h.shape[1] // n_tasks
                per_task = np.stack([h[:, (t + 1) * ppt - 1] for t in range(n_tasks)], 1)
                R["train"][m]["per_task_mean"] = per_task.mean(0).tolist()

        # Fig: per-task cumulative reward, sequential methods
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        for m in SEQUENTIAL:
            if m in hist:
                band(ax, np.arange(1, hist[m].shape[1] + 1) * save_ev, hist[m], m)
        for t in range(1, n_tasks):
            ax.axvline(t * n_samples, color="grey", ls=":", lw=0.6)
        ax.set_xlabel("training steps"); ax.set_ylabel("cumulative reward (active task)")
        ax.set_title(TITLES[exp]); ax.grid(alpha=0.3); ax.legend(loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"train_{exp}.pdf")); plt.close(fig)

        # Fig: total cumulative reward, all methods
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        for m in SEQUENTIAL + RANDOMIZED:
            if m in hist:
                tot = running_total(hist[m], n_tasks, m in RANDOMIZED)
                band(ax, np.arange(1, tot.shape[1] + 1) * save_ev, tot, m)
        ax.set_xlabel("total environment steps"); ax.set_ylabel("total cumulative reward")
        ax.set_title(TITLES[exp]); ax.grid(alpha=0.3); ax.legend(loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"total_{exp}.pdf")); plt.close(fig)

        # Fig: averaging ablation
        if any(m in hist for m in ABLATION[1:]):
            fig, ax = plt.subplots(figsize=(3.4, 2.2))
            for m in ABLATION:
                if m in hist:
                    tot = running_total(hist[m], n_tasks, False)
                    band(ax, np.arange(1, tot.shape[1] + 1) * save_ev, tot, m)
            ax.set_xlabel("training steps"); ax.set_ylabel("total cumulative reward")
            ax.set_title(f"{TITLES[exp]}: averaging ablation"); ax.grid(alpha=0.3); ax.legend(loc="upper left")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, f"ablation_{exp}.pdf")); plt.close(fig)

        # Fig: final evaluation per task
        ev_path = os.path.join(args.root, exp, "eval", "evaluation_summary.json")
        if os.path.exists(ev_path):
            ev = json.load(open(ev_path))
            methods = [k for k in ["DQN", "FGDQN", "SFDQN", "FGSFDQN", "alg2", "alg3"] if k in ev]
            fig, ax = plt.subplots(figsize=(3.4, 2.2))
            width = 0.8 / max(1, len(methods))
            for j, k in enumerate(methods):
                m = EVAL_NAME.get(k, k)
                per_seed = np.array([t["task_mean"] for t in ev[k]["per_trial"]])  # [seeds, tasks]
                mean, std = per_seed.mean(0), per_seed.std(0)
                x = np.arange(per_seed.shape[1]) + (j - (len(methods) - 1) / 2) * width
                ours = m in OURS
                ax.bar(x, mean, width, yerr=std if per_seed.shape[0] > 1 else None, color=COLORS.get(m),
                       edgecolor="black" if ours else "none", linewidth=0.5,
                       label=LABELS.get(m, m), capsize=1.5, error_kw={"lw": 0.6})
                overall = per_seed.mean(1)
                R["eval"][m] = {"seeds": int(per_seed.shape[0]), "per_task_mean": mean.tolist(),
                                "per_task_std": std.tolist(), "overall_mean": float(overall.mean()),
                                "overall_std": float(overall.std())}
            ax.set_xticks(np.arange(per_seed.shape[1])); ax.set_xticklabels([f"T{t+1}" for t in range(per_seed.shape[1])])
            ax.set_ylabel("mean return"); ax.set_title(f"{TITLES[exp]}: final evaluation")
            ax.grid(alpha=0.3, axis="y"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False,
                                               columnspacing=0.8, handlelength=1.4)
            fig.tight_layout(); fig.savefig(os.path.join(args.out, f"eval_{exp}.pdf")); plt.close(fig)

        # diagnostics numbers
        ddir = os.path.join(args.root, exp, "diagnostics")   # copied out of weights/ in results/paper_runs
        if not os.path.isdir(ddir):
            ddir = os.path.join(args.root, exp, "weights", "diagnostics")   # as written by run_paper_experiments.py
        diag = {}
        for f in glob.glob(os.path.join(ddir, "*_trial_*.json")):
            m = re.sub(r"_trial_\d+\.json$", "", os.path.basename(f))
            h = json.load(open(f))
            if h:
                diag.setdefault(m, []).append(h)
        R["diagnostics"] = {}
        for m, runs in diag.items():
            last = [r[-1] for r in runs]
            R["diagnostics"][m] = {
                "final_bellman_residual": float(np.mean([e["bellman_residual"] for e in last])),
                "final_target_tie_rate": float(np.mean([e["target_tie_rate"] for e in last])),
                "final_prior_tie_rate": float(np.mean([e.get("prior_tie_rate", np.nan) for e in last])),
                "median_bellman_residual": float(np.mean([np.median([e["bellman_residual"] for e in r]) for r in runs])),
            }
        results[exp]["_diag_runs"] = diag

    # Fig: diagnostics (columns = experiments; rows = Bellman residual, GPI target-action ties)
    exps_d = [e for e in EXPS if e in results and results[e]["_diag_runs"]]
    if exps_d:
        fig, axes = plt.subplots(2, len(exps_d), figsize=(7.0, 3.3), squeeze=False)
        for col, exp in enumerate(exps_d):
            for m in ["SFDQN", "FG-SFDQN", "alg2", "alg3"]:
                runs = results[exp]["_diag_runs"].get(m)
                if not runs:
                    continue
                L = min(len(r) for r in runs)
                # entries are logged at every progress print; place them on the global step axis
                steps = np.arange(1, L + 1) * (results[exp]["n_samples"] * results[exp]["n_tasks"] / L)
                for row, key in enumerate(["bellman_residual", "target_tie_rate"]):
                    vals = np.array([[e.get(key, np.nan) for e in r[:L]] for r in runs], dtype=float)
                    band(axes[row, col], steps, vals if row == 0 else 100 * vals, m)
            axes[0, col].set_title(TITLES[exp].replace("PointMaze", "PM")); axes[0, col].set_yscale("log")
            for row in range(2):
                axes[row, col].grid(alpha=0.3)
                axes[row, col].xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
            axes[1, col].set_xlabel("env. steps")
        axes[0, 0].set_ylabel("Bellman residual"); axes[1, 0].set_ylabel("GPI action ties (%)")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, ncol=4, loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(args.out, "diagnostics.pdf")); plt.close(fig)
    for exp in results:
        results[exp].pop("_diag_runs", None)

    bench = os.path.join(args.root, "benchmark.json")
    if os.path.exists(bench):
        results["benchmark"] = json.load(open(bench))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_tables(results, args.out)
    results_grid(args.root, args.out)
    print(json.dumps({e: {"train": {m: round(v["total_mean"], 1) for m, v in r["train"].items()},
                          "eval": {m: round(v["overall_mean"], 2) for m, v in r["eval"].items()}}
                      for e, r in results.items() if e != "benchmark"}, indent=1))


if __name__ == "__main__":
    main()
