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
    mountain_clustering = MountainClustering(number_of_clusters=2,
                                        number_of_partitions=10,
                                        distance_metric='euclidean',
                                        sigma_squared=1,
                                        beta_squared=1.5)

    mountain_clustering.fit(feature_matrix)
