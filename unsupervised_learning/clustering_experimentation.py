"""
Clustering experimentation module.
"""

from sklearn.datasets import load_iris

from clustering import KMeans, FuzzCMeans, MountainClustering, SubstractiveClustering, GaussianMixture

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
    # Scale Feature Matrix
    feature_min = feature_matrix.min(axis=0)
    feature_max = feature_matrix.max(axis=0)
    feature_matrix = (feature_matrix - feature_min) / (feature_max - feature_min)
    # Fit Algorithms
    mountain_clustering = MountainClustering(number_of_partitions=10,
                                        distance_metric='manhattan',
                                        sigma_squared=1,
                                        beta_squared=1.5,)
    mountain_clustering.fit(feature_matrix)
