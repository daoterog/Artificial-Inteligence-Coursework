"""
Clustering experimentation module.
"""

from sklearn.datasets import load_iris

from clustering import KMeans, FuzzCMeans, MountainClustering, SubstractiveClustering

def load_dataset():
    """Loads the iris dataset.
    Returns:
        np.ndarray: Feature matrix.
        np.ndarray: Target vector.
    """
    iris = load_iris()
    return iris['data'], iris['target']

if __name__ == "__main__":
    # Load the dataset
    feature_matrix, target_vector = load_dataset()
    # Fit Algorithms
    fuzzycmeans = FuzzCMeans(number_of_clusters=5,
                        fuzzines_parameter=2,
                        distance_metric='euclidean',
                        n_iter=1000,
                        verbose=False,)

    fuzzycmeans.fit(feature_matrix)
