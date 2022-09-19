"""
Plotting module.
"""

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

def plot3d_iris_dataset(
    feature_matrix: np.ndarray,
    title: str,
) -> None:

    """Generates seaborn pairplot from dataset."""

    feature_df = pd.DataFrame(
        feature_matrix,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "target"],
    )

    fig = px.scatter_3d(
        feature_df,
        x="sepal_length",
        y="sepal_width",
        z="petal_width",
        color="target",
        size_max=18,
        opacity=0.7,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    fig.update_traces(showlegend=False)
    fig.show()


def plot_target_classes(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, stock: str) -> None:

    """Plot target classes distribution"""

    target_classes, train_occurrences = np.unique(y_train, return_counts=True)
    _, val_occurrences = np.unique(y_val, return_counts=True)
    _, test_occurrences = np.unique(y_test, return_counts=True)
    total_occurrences = train_occurrences + test_occurrences + val_occurrences

    assert (
        train_occurrences.shape[0] >= 2 and test_occurrences.shape[0] >= 2
    ), "Not enough target classes to plot"

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(16, 4))

    width = 1
    color = ["cornflowerblue", "forestgreen", "maroon"]

    ax0.bar(target_classes, train_occurrences, color=color, width=width, align="center")
    ax0.set_title("Train Absolute\nOccurrences")

    ax1.bar(target_classes, val_occurrences, color=color, width=width)
    ax1.set_title("Validation Absolute\nOccurrences")

    ax2.bar(target_classes, test_occurrences, color=color, width=width)
    ax2.set_title("Test Absolute\nOccurrences")

    ax3.bar(
        target_classes, train_occurrences / total_occurrences, color=color, width=width
    )
    ax3.set_title("Train Relative\nOccurrences")

    ax4.bar(
        target_classes, test_occurrences / total_occurrences, color=color, width=width
    )
    ax4.set_title("Train Relative\nOccurrences")

    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.set_xticks(target_classes)

    fig.suptitle(stock)
    plt.tight_layout()
    plt.show()
