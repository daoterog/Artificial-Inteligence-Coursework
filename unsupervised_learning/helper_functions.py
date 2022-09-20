"""
Helper Functions module.
"""

import pandas as pd
import numpy as np

import plotly.express as px


def plot3d_iris_dataset(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    title: str,
) -> None:

    """Generates seaborn pairplot from dataset."""

    feature_df = pd.DataFrame(
        np.concatenate([feature_matrix, target.reshape(-1, 1)], axis=1),
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "target",
        ],
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
