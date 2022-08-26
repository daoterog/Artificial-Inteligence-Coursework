"""
Random Lines Classifier Module.
"""

import pandas as pd
import numpy as np

from helper_functions import binary_search_percentile


class RandomLinesClassifier:

    """Model that linearly separates two classes by drawing random lines between to points."""

    def __init__(self, first_dim: str, second_dim: str, n_iter: int = 1000):

        """Assign model parameters."""

        self.n_inter = n_iter
        self.first_dim = first_dim
        self.second_dim = second_dim

        self.slope = None
        self.intercept = None
        self.first_class_up = None

    def _sample_point(self, x_features: pd.DataFrame) -> pd.Series:

        """Samples a point from the dataset."""

        random_number = np.random.normal(0, 1)
        percentile = binary_search_percentile(random_number, "normal", 0.0, 1.0, 10e-3)
        sample_index = int(percentile * x_features.shape[0])
        sample = x_features.iloc[sample_index, :]

        return sample

    def _make_line(self, points: list) -> tuple[float, float]:

        """Makes a line from the data points."""

        first_point, second_point = points

        # Make line
        slope = (first_point[self.second_dim] - second_point[self.second_dim]) / (
            first_point[self.first_dim] - second_point[self.first_dim]
        )
        intercept = first_point[self.second_dim] - first_point[self.first_dim] * slope

        return slope, intercept

    def _evaluate(
        self,
        prop_slope: float,
        prop_intercept: float,
        x_features: pd.DataFrame,
        y_target: pd.Series,
    ) -> float:

        """Evaluates performance of the proposed model."""

        in_1st_class = (
            x_features[self.second_dim]
            >= prop_slope * x_features[self.first_dim] + prop_intercept
        )
        correct = y_target[in_1st_class] == "Setosa"
        performance = correct.mean()

        self.first_class_up = performance > 0.5

        return performance if self.first_class_up else 1 - performance

    def fit(self, x_features: pd.DataFrame, y_target: pd.Series) -> None:

        """Trains the model."""

        x_features_copy = x_features.copy()
        points = [self._sample_point(x_features_copy) for _ in range(2)]
        best_slope, best_intercept = self._make_line(points)
        best_performance = self._evaluate(
            best_slope, best_intercept, x_features_copy, y_target
        )

        for _ in range(self.n_inter):

            new_point = self._sample_point(x_features_copy)

            while new_point.equals(points[0]) and new_point.equals(points[1]):
                new_point = self._sample_point(x_features_copy)

            points_combinations = [[points[0], new_point], [points[1], new_point]]

            for combination in points_combinations:

                new_slope, new_intercept = self._make_line(combination)
                new_performance = self._evaluate(
                    new_slope, new_intercept, x_features_copy, y_target
                )

                if new_performance > best_performance:
                    best_performance = new_performance
                    best_slope = new_slope
                    best_intercept = new_intercept
                    points = combination

        self.slope = best_slope
        self.intercept = best_intercept

    def predict(self, x_features: pd.DataFrame) -> pd.Series:

        """Predicts the classes of the given data points."""

        preds = (
            x_features[self.second_dim]
            >= self.slope * x_features[self.first_dim] + self.intercept
        )
        preds = ~preds if not self.first_class_up else preds

        return 1 * preds
