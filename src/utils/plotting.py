"""Plotting utilities for training curves."""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless back-end – no display required
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curve(
    csv_path: str,
    x_col: str = "step",
    y_col: str = "eval_mean_reward",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    window: int = 1,
) -> None:
    """Plot a single training metric from a CSV log file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file produced by :class:`~utils.logger.Logger`.
    x_col:
        Column name used for the x-axis.
    y_col:
        Column name to plot on the y-axis.
    title:
        Optional figure title.
    save_path:
        If provided the figure is saved here; otherwise it is shown
        interactively.
    window:
        Rolling-average window size (smoothing).
    """
    df = pd.read_csv(csv_path)
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in {csv_path}.")

    y = df[y_col].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(df[x_col], y, linewidth=1.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or y_col)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_multiple_runs(
    csv_paths: List[str],
    labels: List[str],
    x_col: str = "step",
    y_col: str = "eval_mean_reward",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    window: int = 1,
) -> None:
    """Overlay multiple training curves on the same axes."""
    plt.figure(figsize=(8, 4))
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        if y_col not in df.columns:
            continue
        y = df[y_col].rolling(window=window, min_periods=1).mean()
        plt.plot(df[x_col], y, linewidth=1.5, label=label)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or y_col)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
