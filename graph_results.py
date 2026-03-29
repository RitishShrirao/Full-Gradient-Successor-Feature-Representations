#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: np.array(data[k]) for k in data.keys()}


def series_stats(arr):
    # Handle (trials, T), (T,), and empty/object inputs.
    if arr is None:
        return np.array([]), np.array([]), 0
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([]), np.array([]), 0

    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception:
            logger.warning(f"series_stats: object array with non-numeric entries."
                           f" shape={arr.shape} contents={arr}")
            return np.array([]), np.array([]), 0

    if arr.ndim == 1:
        mean = arr
        std = np.zeros_like(arr)
        n = 1
    elif arr.ndim == 2:
        n = arr.shape[0]
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
    else:
        arr2 = arr.reshape(arr.shape[0], -1)
        n = arr2.shape[0]
        mean = np.mean(arr2, axis=0)
        std = np.std(arr2, axis=0)

    if mean is None or np.any([x is None for x in np.atleast_1d(mean)]):
        mean = np.array([])
    else:
        mean = np.asarray(mean, dtype=float)

    if std is None or np.any([x is None for x in np.atleast_1d(std)]):
        std = np.zeros_like(mean)
    else:
        std = np.asarray(std, dtype=float)

    return mean, std, n


def plot_combined(data_dict, out_path, title=None):
    plt.figure(figsize=(10, 6))
    for name, arr in data_dict.items():
        mean, std, n = series_stats(arr)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=f"{name} (n={n})")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Save step')
    plt.ylabel('Cumulative reward (per save)')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_individual(name, arr, out_dir):
    mean, std, n = series_stats(arr)
    x = np.arange(len(mean))
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label=f"mean (n={n})")
    if n > 1:
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel('Save step')
    plt.ylabel('Cumulative reward (per save)')
    plt.title(name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{name.replace(' ', '_').replace('/', '_')}.png")
    plt.savefig(fname)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='?', default='updated_results.npz', help='NPZ file to read')
    p.add_argument('--out', '-o', default='plots', help='Output directory for plots')
    p.add_argument('--combined', default='combined.png', help='Filename for combined plot')
    args = p.parse_args()

    data = load_npz(args.npz)
    ensure_dir(args.out)

    for remove in ['alg2', 'alg3']:
        if remove in data:
            data.pop(remove, None)

    combined_path = os.path.join(args.out, args.combined)
    plot_combined(data, combined_path, title=f"Results from {os.path.basename(args.npz)}")

    for name, arr in data.items():
        plot_individual(name, arr, args.out)

    print(f"Saved combined plot to {combined_path}")
    print(f"Saved individual plots to {args.out}/")


if __name__ == '__main__':
    main()
