"""
Contains mountain, substractive, k-means and fuzzy c-means clustering algorithms.
"""

import numpy as np

from metrics import (
    manhattan_distance,
    euclidean_distance,
    cosine_distance,
    mahalanobis_distance,
)


class DistanceMetric:

    """Distance metric class."""

    def __init__(self, distance_metric: str):
        """Initializes the DistanceMetric class.
        Args:
            distance_metric (str): Distance metric.
        """
        self.distance_metric = distance_metric
        if distance_metric == "cosine":
            self.distance_criterion = cosine_distance
        elif distance_metric == "euclidean":
            self.distance_criterion = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_criterion = manhattan_distance
        elif distance_metric == "mahalanobis":
            self.distance_criterion = mahalanobis_distance
        else:
            raise ValueError(
                "Invalid distance metric. Please choose from: cosine, euclidean,"
                + " manhattan, mahalanobis."
            )

        self.inverse_covariance_matrix = None

    def get_distance(self, feature_matrix: np.ndarray, datapoint: np.ndarray) -> float:
        """Returns the distance between a feature_matrix (or datapoint 1) and a datapoint.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            datapoint (np.ndarray): Data point.
        Returns
        """
        if self.distance_metric == "mahalanobis":
            if self.inverse_covariance_matrix is None:
                covariance_matrix = np.cov(feature_matrix.T, ddof=1)
                self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
            if feature_matrix.shape[0] == 1:
                return self.distance_criterion(
                    feature_matrix, datapoint, self.inverse_covariance_matrix
                )
            distances = [
                self.distance_criterion(
                    first_datapoint, datapoint, self.inverse_covariance_matrix
                )
                for first_datapoint in feature_matrix
            ]
            return np.array(distances).squeeze()
        else:
            return self.distance_criterion(feature_matrix, datapoint)


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
        self.distance_criterion = DistanceMetric(distance_metric).get_distance

        self.assignments = None
        self.centers = None

    def _initialize_variables(self, feature_matrix: np.ndarray) -> None:
        """Initializes the variables.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Initialize assignments matrix
        self.assignments = np.zeros(feature_matrix.shape[0])
        # Initialize k cluster centers with random points
        self.centers = np.random.permutation(feature_matrix)[: self.number_of_clusters]

    def _assign_datapoints(self, feature_matrix: np.ndarray) -> None:
        """Assigns datapoints to clusters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            distance_to_centers = self.distance_criterion(self.centers, datapoint)
            # Assign data point to closest cluster
            self.assignments[idx] = np.argmin(distance_to_centers)

    def _update_cluster_centers(self, feature_matrix: np.ndarray) -> None:
        """Updates the cluster centers.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        for cluster in range(self.number_of_clusters):
            self.centers[cluster] = np.mean(feature_matrix[self.assignments == cluster])

    def _compute_cost(self, feature_matrix: np.ndarray) -> None:
        """Computes the cost function.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        Outputs:
            Prints cost function.
        """
        cost = 0
        for cluster in range(self.number_of_clusters):
            cost += np.sum(
                self.distance_criterion(
                    feature_matrix[self.assignments == cluster], self.centers[cluster]
                )
            )
        if self.verbose:
            print(f"Cost: {cost}")

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fits the KMeans algorithm.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Initialize variables
        self._initialize_variables(feature_matrix)
        # Repeat for max_iter iterations
        for _ in range(self.max_iters):
            # Assign data points to clusters
            self._assign_datapoints(feature_matrix)
            # Update cluster centers
            self._update_cluster_centers(feature_matrix)
            # Compute cost function
            self._compute_cost(feature_matrix)

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
        fuzzines_parameter: int,
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
        self.distance_criterion = DistanceMetric(distance_metric).get_distance
        self.fuzzines_parameter = fuzzines_parameter

        self.assignments = None
        self.centers = None

    def _initialize_variables(self, feature_matrix: np.ndarray) -> None:
        """Initializes the variables.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Initialize membership matrix and normalize it
        self.assignments = np.random.rand(
            feature_matrix.shape[0], self.number_of_clusters
        )
        self.assignments = self.assignments / np.sum(self.assignments, axis=1)[:, None]
        # Initialize cluster centers
        self.centers = np.zeros((self.number_of_clusters, feature_matrix.shape[1]))

    def _update_cluster_centers(self, feature_matrix: np.ndarray) -> None:
        """Updates the cluster centers.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Elevate membership matrix to fuzzines parameter
        membership_coefficients = np.power(self.assignments, self.fuzzines_parameter)
        # Update cluster centers
        for i in range(self.number_of_clusters):
            self.centers[i, :] = np.sum(
                feature_matrix * membership_coefficients[:, i].reshape(-1, 1),
                axis=0,
                keepdims=True,
            ) / np.sum(membership_coefficients[:, i], axis=0, keepdims=True)

    def _update_membership_matrix(self, feature_matrix: np.ndarray) -> None:
        """Updates membership matrix.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            distance_to_centers = self.distance_criterion(self.centers, datapoint)
            # Elevate to fuzzines parameter power
            sum_centers_inverse_distance = np.sum(
                np.power(1 / distance_to_centers, 2 / (self.fuzzines_parameter - 1))
            )
            # Elevate centers distance to fuzzines parameter power
            elevated_centers_distance = np.power(
                distance_to_centers, 2 / (self.fuzzines_parameter - 1)
            )
            # Update membership matrix
            self.assignments[idx, :] = (
                1 / (elevated_centers_distance * sum_centers_inverse_distance)
            ).squeeze()

    def _compute_cost(self, feature_matrix: np.ndarray) -> None:
        """Computes the cost function.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        Outputs:
            Prints cost function.
        """
        cost = 0
        for cluster in range(self.number_of_clusters):
            powered_membership_coefficients = np.power(
                self.assignments[:, cluster].reshape(-1, 1), self.fuzzines_parameter
            ).T
            distance_to_centers = self.distance_criterion(
                feature_matrix, self.centers[cluster, :]
            )
            cost += np.dot(
                powered_membership_coefficients,
                np.power(distance_to_centers, 2),
            ).item()
        if self.verbose:
            print(f"Cost: {cost}")

    def fit(self, feature_matrix: np.ndarray) -> None:
        """Fits the FuzzCMeans algorithm.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Initialize parameters
        self._initialize_variables(feature_matrix)
        # Repeat for max_iter iterations
        for _ in range(self.max_iters):
            # Get cluster centers
            self._update_cluster_centers(feature_matrix)
            # Update membership matrix
            self._update_membership_matrix(feature_matrix)
            # Compute cost
            self._compute_cost(feature_matrix)

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
    ) -> None:
        """Initializes the MountainClustering class.
        Args:
            number_of_clusters (int): Number of clusters.
            number_of_partitions (int): Number of partitions.
        """
        self.number_of_clusters = number_of_clusters
        self.number_of_partitions = number_of_partitions
        self.sigma_squared = sigma_squared
        self.beta_squared = beta_squared
        self.distance_criterion = DistanceMetric(distance_metric).get_distance

        self.cluster_centers = None
        self.cluster_mountain_functions = None
        self.assignments = None

    def _create_grid(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Creates a grid of points.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Grid of points.
        """
        # Get min and max values for each feature
        mins = np.min(feature_matrix, axis=0)
        maxs = np.max(feature_matrix, axis=0)
        # Create grid
        grid = [[]]
        for i in range(feature_matrix.shape[1]):
            new_grid = np.linspace(mins[i], maxs[i], self.number_of_partitions).tolist()
            grid = [
                combination + [new_combination]
                for combination in grid
                for new_combination in new_grid
            ]
        return np.array(grid)

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
        scaling_factor = (
            np.exp(
                -np.power(
                    self.distance_criterion(prototype, self.cluster_centers[-1]),
                    2,
                )
                / (2 * self.beta_squared)
            )
            .squeeze()
            .item()
        )
        return (
            mountain_function
            - self.cluster_mountain_functions[-1].item() * scaling_factor
        )

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
            ).reshape(1, -1)
        else:
            self.cluster_centers = np.vstack(
                [self.cluster_centers, prototypes[maximum_index, :]]
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
        # Create prototype grid
        prototypes = self._create_grid(feature_matrix)
        # Evaluate prototypes mountain functions
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
        self.distance_criterion = DistanceMetric(distance_metric).get_distance

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
            self.centers_densities = np.array(densities[maximum_index]).reshape(1, -1)
        else:
            self.cluster_centers = np.vstack(
                [self.cluster_centers, feature_matrix[maximum_index, :]]
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
        scaling_factor = (
            np.exp(
                -np.power(
                    self.distance_criterion(prototype, self.cluster_centers[-1]), 2
                )
                / ((self.r_b**2) / 4)
            )
            .squeeze()
            .item()
        )
        return density - self.centers_densities[-1].item() * scaling_factor

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
