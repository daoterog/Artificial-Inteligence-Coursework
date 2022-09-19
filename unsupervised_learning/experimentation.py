"""
Experimentation module.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from metrics import get_distance_matrix

CRITERION = 'manhattan'

if __name__ == "__main__":
    iris = load_iris()
    feature_matrix = iris.data
    features_min = feature_matrix.min(axis=0)
    features_max = feature_matrix.max(axis=0)
    feature_matrix = (feature_matrix - features_min) / (features_max - features_min)

    distance_matrix = get_distance_matrix(feature_matrix, CRITERION)

    plt.figure(figsize=(7, 7))
    sns.heatmap(distance_matrix)
    plt.title(f'{CRITERION} distance matrix')
    plt.tight_layout()
    plt.show()
