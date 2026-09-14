from __future__ import annotations

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# --------------------------------------------------
# UTIL
# --------------------------------------------------

def _load(save_dir: str):
    true = torch.load(f"{save_dir}/trajectories/true.pt")
    pred = torch.load(f"{save_dir}/trajectories/pred.pt")
    return true.detach().cpu(), pred.detach().cpu()


def _prep_dirs(save_dir: str):
    os.makedirs(f"{save_dir}/plots", exist_ok=True)


def _flatten_time(x: torch.Tensor):
    return x.reshape(-1, x.shape[-1])


# --------------------------------------------------
# ERROR CURVE (LOG + STD BAND)
# --------------------------------------------------

def plot_error_curve(save_dir="results"):
    true, pred = _load(save_dir)

    error = ((true[:, 1:] - pred) ** 2).mean(dim=-1)  # [B, T]
    mean = error.mean(dim=0)
    std = error.std(dim=0)

    plt.figure(figsize=(6, 4))
    t = np.arange(len(mean))

    plt.plot(t, mean, label="Mean MSE")
    plt.fill_between(t, mean - std, mean + std, alpha=0.3)

    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("MSE (log)")
    plt.title("Error Growth (Log Scale)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{save_dir}/plots/error_curve.png", dpi=200)
    plt.close()


# --------------------------------------------------
# HEATMAP (BETTER NORMALIZATION)
# --------------------------------------------------

def plot_heatmap(save_dir="results"):
    true, pred = _load(save_dir)

    error = ((true[:, 1:] - pred) ** 2).mean(dim=0)  # [T, D]

    plt.figure(figsize=(7, 4))
    plt.imshow(error, aspect="auto", interpolation="nearest")
    plt.colorbar(label="MSE")

    plt.xlabel("Dimension")
    plt.ylabel("Time")
    plt.title("Error Heatmap")
    plt.tight_layout()

    plt.savefig(f"{save_dir}/plots/heatmap.png", dpi=200)
    plt.close()


# --------------------------------------------------
# TRAJECTORY COMPARISON (2D + 3D SAFE)
# --------------------------------------------------

def plot_trajectory(save_dir="results"):
    true, pred = _load(save_dir)

    true = true[0]
    pred = pred[0]

    dim = true.shape[-1]

    if dim >= 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")

        ax.plot(true[:, 0], true[:, 1], true[:, 2], label="True", alpha=0.8)
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], label="Pred", alpha=0.8)

        ax.set_title("3D Trajectory")
        ax.legend()

        plt.savefig(f"{save_dir}/plots/trajectory_3d.png", dpi=200)
        plt.close()

    else:
        plt.figure(figsize=(6, 4))
        plt.plot(true[:, 0], label="True")
        plt.plot(pred[:, 0], label="Pred")
        plt.legend()
        plt.title("Trajectory (1D)")
        plt.tight_layout()

        plt.savefig(f"{save_dir}/plots/trajectory_1d.png", dpi=200)
        plt.close()


# --------------------------------------------------
# PCA (TRUE vs PRED)
# --------------------------------------------------

def plot_pca(save_dir="results"):
    true, pred = _load(save_dir)

    true_flat = _flatten_time(true).numpy()
    pred_flat = _flatten_time(pred).numpy()

    pca = PCA(n_components=2)
    z_true = pca.fit_transform(true_flat)
    z_pred = pca.transform(pred_flat)

    plt.figure(figsize=(5, 5))
    plt.scatter(z_true[:, 0], z_true[:, 1], s=1, label="True", alpha=0.5)
    plt.scatter(z_pred[:, 0], z_pred[:, 1], s=1, label="Pred", alpha=0.5)

    plt.legend()
    plt.title("PCA: True vs Pred")
    plt.tight_layout()

    plt.savefig(f"{save_dir}/plots/pca.png", dpi=200)
    plt.close()


# --------------------------------------------------
# DISTRIBUTION SHIFT
# --------------------------------------------------

def plot_distribution(save_dir="results"):
    true, pred = _load(save_dir)

    true_flat = true.reshape(-1).numpy()
    pred_flat = pred.reshape(-1).numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(true_flat, bins=100, alpha=0.5, density=True, label="True")
    plt.hist(pred_flat, bins=100, alpha=0.5, density=True, label="Pred")

    plt.legend()
    plt.title("Value Distribution")
    plt.tight_layout()

    plt.savefig(f"{save_dir}/plots/distribution.png", dpi=200)
    plt.close()


# --------------------------------------------------
# SPECTRAL ANALYSIS (VERY IMPORTANT)
# --------------------------------------------------

def plot_spectrum(save_dir="results"):
    true, pred = _load(save_dir)

    true_fft = torch.fft.rfft(true, dim=1).abs().mean(dim=0)
    pred_fft = torch.fft.rfft(pred, dim=1).abs().mean(dim=0)

    plt.figure(figsize=(6, 4))
    plt.plot(true_fft.mean(dim=-1), label="True")
    plt.plot(pred_fft.mean(dim=-1), label="Pred")

    plt.yscale("log")
    plt.title("Frequency Spectrum")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{save_dir}/plots/spectrum.png", dpi=200)
    plt.close()


# --------------------------------------------------
# LONG-HORIZON DRIFT
# --------------------------------------------------

def plot_drift(save_dir="results"):
    true, pred = _load(save_dir)

    drift = (pred - true[:, 1:]).abs().mean(dim=[0, -1])

    plt.figure(figsize=(6, 4))
    plt.plot(drift)
    plt.title("Absolute Drift Over Time")
    plt.xlabel("Time")
    plt.ylabel("L1 Error")
    plt.tight_layout()

    plt.savefig(f"{save_dir}/plots/drift.png", dpi=200)
    plt.close()


# --------------------------------------------------
# MASTER
# --------------------------------------------------

def visualize_all(save_dir="results"):
    _prep_dirs(save_dir)

    plot_error_curve(save_dir)
    plot_heatmap(save_dir)
    plot_trajectory(save_dir)
    plot_pca(save_dir)
    plot_distribution(save_dir)
    plot_spectrum(save_dir)
    plot_drift(save_dir)

    print(f"All plots saved to {save_dir}/plots")