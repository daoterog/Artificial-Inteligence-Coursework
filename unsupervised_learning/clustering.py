"""
Contains mountain, substractive, k-means and fuzzy c-means clustering algorithms.
"""

import numpy as np

from metrics import manhattan_distance, euclidean_distance, cosine_distance


class KMeans:

    """K-Means algorithm"""

    def __init__(
        self,
        number_of_clusters: int,
        distance_metric: str,
        max_iters: int,
        verbose: bool,
    ) -> None:
        """Initializes the KMeans class.
        Args:
            number_of_clusters (int): Number of clusters.
            distance_metric (str): Distance metric.
        """
        self.number_of_clusters = number_of_clusters
        self.max_iters = max_iters
        self.verbose = verbose
        if distance_metric == "cosine":
            self.distance_criterion = cosine_distance
        elif distance_metric == "euclidean":
            self.distance_criterion = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_criterion = manhattan_distance
        else:
            raise ValueError(
                "Invalid distance metric. Please choose from: cosine, euclidean,"
                + " manhattan."
            )

        self.assignments = None

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fits the KMeans algorithm.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Initialize assignments matrix
        self.assignments = np.zeros(feature_matrix.shape[0])
        # Initialize k cluster centers
        centroids = np.random.permutation(feature_matrix)[: self.number_of_clusters]
        # Repeat for max_iter iterations
        for _ in range(self.max_iters):
            # Assign data points to clusters
            for idx, datapoint in enumerate(feature_matrix):
                # Get distance to clusters
                centroids_distance = self.distance_criterion(centroids, datapoint)
                # Assign data point to closest cluster
                self.assignments[idx] = np.argmin(centroids_distance)
            # Update cluster centers
            for cluster in range(self.number_of_clusters):
                centroids[cluster] = np.mean(
                    feature_matrix[self.assignments == cluster]
                )
            # Compute cost function
            for cluster in range(self.number_of_clusters):
                cost = np.sum(
                    self.distance_criterion(
                        feature_matrix[self.assignments == cluster], centroids[cluster]
                    )
                )
            if self.verbose:
                print(f"Cost: {cost}")

    def transform(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments


class FuzzCMeans:

    """Fuzzy C-Means algorithm."""

    def __init__(
        self,
        number_of_clusters: int,
        fuzzines_parameter: float,
        distance_metric: str,
        max_iters: int,
        verbose: bool,
    ) -> None:
        """Initializes the FuzzCMeans class.
        Args:
            number_of_clusters (int): Number of clusters.
            max_iters (int): Maximum number of iterations.
        """
        self.number_of_clusters = number_of_clusters
        self.max_iters = max_iters
        self.verbose = verbose
        self.fuzzines_parameter = fuzzines_parameter
        if distance_metric == "cosine":
            self.distance_criterion = cosine_distance
        elif distance_metric == "euclidean":
            self.distance_criterion = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_criterion = manhattan_distance
        else:
            raise ValueError(
                "Invalid distance metric. Please choose from: cosine, euclidean,"
                + " manhattan."
            )
        self.assignments = None

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fits the FuzzCMeans algorithm.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Initialize membership matrix
        self.assignments = (
            np.ones((feature_matrix.shape[0], self.number_of_clusters))
            / self.number_of_clusters
        )
        # Repeat for max_iter iterations
        for _ in range(self.max_iters):
            # Get cluster centers
            membership_coefficients = np.power(
                self.assignments, self.fuzzines_parameter
            )
            centers = np.dot(membership_coefficients.T, feature_matrix) / np.sum(
                membership_coefficients, axis=0
            ).reshape(-1, 1)
            # Update membership matrix
            for idx, datapoint in enumerate(feature_matrix):
                # Get distance to clusters
                centroids_distance = self.distance_criterion(centers, datapoint)
                # Elevate to fuzzines parameter power
                centroids_distance = np.power(
                    centroids_distance, 2 / (self.fuzzines_parameter - 1)
                )
                # Update membership matrix
                self.assignments[idx] = np.sum(centroids_distance) / centroids_distance
            # Compute cost
            cost = 0
            for cluster in range(self.number_of_clusters):
                cost += np.sum(
                    np.power(self.assignments[:, cluster], self.fuzzines_parameter)
                    * np.power(
                        self.distance_criterion(feature_matrix, centers[cluster]), 2
                    )
                )
            if self.verbose:
                print(f"Cost: {cost}")

    def transform(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments


class MountainClustering:

    """Mountain clustering algorithm."""

    def __init__(
        self,
        number_of_clusters: int,
        number_of_partitions: int,
        distance_metric: str,
        sigma_squared: float,
        beta_squared: float,
        verbose: bool,
    ) -> None:
        """Initializes the MountainClustering class.
        Args:
            number_of_clusters (int): Number of clusters.
            number_of_partitions (int): Number of partitions.
            verbose (bool): Whether to print cost function.
        """
        self.number_of_clusters = number_of_clusters
        self.number_of_partitions = number_of_partitions
        self.sigma_squared = sigma_squared
        self.beta_squared = beta_squared
        self.verbose = verbose
        if distance_metric == "cosine":
            self.distance_criterion = cosine_distance
        elif distance_metric == "euclidean":
            self.distance_criterion = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_criterion = manhattan_distance
        else:
            raise ValueError(
                "Invalid distance metric. Please choose from: cosine, euclidean,"
                + " manhattan."
            )

        self.cluster_centers = None
        self.cluster_mountain_functions = None
        self.assignments = None

    def _get_mountains(
        self, prototype: np.ndarray, feature_matrix: np.ndarray
    ) -> float:
        """Returns mountain function of certain prototype.
        Args:
            prototype (np.ndarray): Prototype.
            feature_matrix (np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Mountain function.
        """
        squared_distance_to_datapoints = np.power(
            self.distance_criterion(feature_matrix, prototype), 2
        )
        exponential_exponent = -squared_distance_to_datapoints / (
            2 * self.sigma_squared
        )
        return np.sum(np.exp(exponential_exponent))

    def _update_mountains(
        self, prototype: np.ndarray, mountain_function: float
    ) -> float:
        """Updates mountain function of certain prototype.
        Args:
            prototype (np.ndarray): Prototype.
            mountain_function (float): Mountain function.
        Returns:
            float: Updated mountain function.
        """
        scaling_factor = np.exp(
            -np.power(
                self.distance_criterion(prototype, self.cluster_mountain_functions[-1]),
                2,
            )
            / (2 * self.beta_squared)
        )
        return mountain_function - self.cluster_mountain_functions[-1] * scaling_factor

    def _update_centers(
        self, mountain_functions: np.ndarray, prototypes: np.ndarray
    ) -> None:
        """Updates cluster centers.
        Args:
            mountain_functions (np.ndarray): Mountain functions.
            prototypes (np.ndarray): Prototypes.
        """
        maximum_index = np.argmax(mountain_functions)
        if self.cluster_centers is None:
            self.cluster_centers = np.array(prototypes[maximum_index, :])
            self.cluster_mountain_functions = np.array(
                mountain_functions[maximum_index]
            )
        else:
            self.cluster_centers = np.append(
                self.cluster_centers, prototypes[maximum_index, :]
            )
            self.cluster_mountain_functions = np.append(
                self.cluster_mountain_functions, mountain_functions[maximum_index]
            )

    def _assign_datapoints(self, feature_matrix: np.ndarray) -> None:
        """Assigns datapoints to clusters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        self.assignments = np.zeros((feature_matrix.shape[0], 1))
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            centroids_distance = self.distance_criterion(
                self.cluster_centers, datapoint
            )
            # Update membership matrix
            self.assignments[idx, 0] = np.argmin(centroids_distance)

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fits the algithm to the data.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Git minumum and maximum values for each feature
        features_min = np.min(feature_matrix, axis=0)
        features_max = np.max(feature_matrix, axis=0)
        # Create grid of evenly spaced points
        prototypes = np.linspace(features_min, features_max, self.number_of_partitions)
        mountain_functions = [
            self._get_mountains(prototype, feature_matrix) for prototype in prototypes
        ]
        # Get first cluster by finding the maximum of the mountain function
        self._update_centers(mountain_functions, prototypes)
        # Repeat number_of_clusters - 1 times
        for _ in range(self.number_of_clusters - 1):
            # Update mountain function
            mountain_functions = [
                self._update_mountains(prototype, mountain_function)
                for prototype, mountain_function in zip(prototypes, mountain_functions)
            ]
            # Get next cluster by finding the maximum of the mountain function
            self._update_centers(mountain_functions, prototypes)
        # Assign datapoints to clusters
        self._assign_datapoints(feature_matrix)

    def transform(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments


class SubstractiveClustering:

    """Substractive clustering algorithm."""

    def __init__(
        self, number_of_clusters: int, r_a: float, r_b: float, distance_metric: str
    ) -> None:
        """Initializes the SubstractiveClustering class.
        Args:
            number_of_clusters (int): Number of clusters.
        """
        self.number_of_clusters = number_of_clusters
        self.r_a = r_a
        self.r_b = r_b
        if distance_metric == "cosine":
            self.distance_criterion = cosine_distance
        elif distance_metric == "euclidean":
            self.distance_criterion = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_criterion = manhattan_distance
        else:
            raise ValueError(
                "Invalid distance metric. Please choose from: cosine, euclidean,"
                + " manhattan."
            )

        self.cluster_centers = None
        self.centers_densities = None
        self.assignments = None

    def _get_density(self, prototype: np.ndarray, feature_matrix: np.ndarray) -> float:
        """Returns the density of a datapoint.
        Args:
            datapoint (np.ndarray): Datapoint.
            feature_matrix (np.ndarray): Feature matrix.
        Returns:
            float: Density.
        """
        # Get distance to datapoints
        squared_distance = np.power(
            self.distance_criterion(feature_matrix, prototype), 2
        )
        return np.sum(np.exp(-squared_distance / ((self.r_a**2) / 4)))

    def _update_centers(
        self, feature_matrix: np.ndarray, densities: np.ndarray
    ) -> None:
        """Updates cluster centers.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            densities (np.ndarray): Densities.
        """
        maximum_index = np.argmax(densities)
        if self.cluster_centers is None:
            self.cluster_centers = np.array(feature_matrix[maximum_index, :])
            self.centers_densities = np.array(densities[maximum_index])
        else:
            self.cluster_centers = np.append(
                self.cluster_centers, feature_matrix[maximum_index, :]
            )
            self.centers_densities = np.append(
                self.centers_densities, [densities[maximum_index]]
            )

    def _update_densities(self, prototype: np.ndarray, density: np.ndarray) -> None:
        """Returns updated density.
        Args:
            prototype (np.ndarray): Prototype.
            density (np.ndarray): Density.
        Returns:
            float: Updated density.
        """
        scaling_factor = np.exp(
            -np.power(self.distance_criterion(prototype, self.cluster_centers[-1]), 2)
            / ((self.r_b**2) / 4)
        )
        return density - self.centers_densities[-1] * scaling_factor

    def _assign_datapoints(self, feature_matrix: np.ndarray) -> None:
        """Assigns datapoints to clusters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        self.assignments = np.zeros((feature_matrix.shape[0], 1))
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            centroids_distance = self.distance_criterion(
                self.cluster_centers, datapoint
            )
            # Update membership matrix
            self.assignments[idx, 0] = np.argmin(centroids_distance)

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fits the algorithm to the data.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Get density of datapoints
        densities = [
            self._get_density(datapoint, feature_matrix) for datapoint in feature_matrix
        ]
        # Get first center with highest density
        self._update_centers(feature_matrix, densities)
        # Repeat for number_of_clusters - 1 times
        for _ in range(self.number_of_clusters - 1):
            # Update densities
            densities = [
                self._update_densities(datapoint, density)
                for datapoint, density in zip(feature_matrix, densities)
            ]
            # Get next center with highest density
            self._update_centers(feature_matrix, densities)
        # Assign datapoints to clusters
        self._assign_datapoints(feature_matrix)

    def transform(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments
