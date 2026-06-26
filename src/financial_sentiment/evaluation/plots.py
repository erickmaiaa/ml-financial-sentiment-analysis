"""Visualisation of model evaluation results.

Kept separate from the modelling code so that *processing* and *presentation*
remain decoupled (Separation of Concerns). Plotting is the only place that
imports matplotlib/seaborn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from financial_sentiment.models.ml import ModelResult
from financial_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def results_to_frame(results: Iterable[ModelResult]) -> pd.DataFrame:
    """Convert model results to a tidy dataframe sorted by accuracy."""
    frame = pd.DataFrame(
        [{"Model": r.name, "Accuracy": r.accuracy} for r in results]
    )
    return frame.sort_values("Accuracy", ascending=False).reset_index(drop=True)


def plot_model_accuracies(
    frame: pd.DataFrame,
    *,
    title: str = "Accuracy of Different Classification Models",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Render a bar chart comparing model accuracies.

    Args:
        frame: Dataframe with ``Model`` and ``Accuracy`` columns
            (see :func:`results_to_frame`).
        title: Chart title.
        save_path: If given, the figure is written to this path.
        show: Whether to display the figure interactively.

    Returns:
        The matplotlib :class:`~matplotlib.axes.Axes` for further tweaking.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.barplot(
        data=frame, x="Model", y="Accuracy", hue="Model",
        palette="coolwarm", legend=False, ax=ax,
    )
    ax.set_xlabel("Classification Models", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="x", labelrotation=8, labelsize=11)
    ax.tick_params(axis="y", labelsize=13)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2%}",
            (patch.get_x() + patch.get_width() / 2, height * 1.02),
            ha="center",
            fontsize="x-large",
        )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Saved accuracy plot to %s", save_path)
    if show:
        plt.show()
    return ax


__all__ = ["results_to_frame", "plot_model_accuracies"]
