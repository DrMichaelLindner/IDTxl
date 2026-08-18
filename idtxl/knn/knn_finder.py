import numpy as np


class KnnFinder:

    def __init__(self, data, num_threads='USE_ALL', metric='chebyshev'):
        """Initialise the KnnFinder with settings.

        Args:
            data (np.ndarray): The data to find neighbors in. Shape is (n_points, n_dimensions).
            num_threads (int): The number of threads to use. If -1 or "USE_ALL", use all available threads.
            metric (str): The metric to use for finding neighbors."""

        if num_threads == 'USE_ALL':
            num_threads = -1

        self._data = data
        self._num_threads = num_threads
        self._metric = metric

    def find_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        """Find the k nearest neighbors to each point in x.
        May include x itself if it is in the data.

        Args:
            x : np.ndarray
                The points to find neighbors for.
            k : int
                The number of neighbors to find.

        Returns:
            np.ndarray
                Array of lists of distances to the k nearest neighbors for
                each point in x
            np.ndarray
                Array of lists of indices of the k nearest neighbors for each
                point in x.
        """
        raise NotImplementedError

    def find_neighbors_within(self, x: np.array, r: float) -> np.ndarray:
        """Find the neighbors strictly within (<) a given radius for each point in x.
        May include x itself if it is in the data.

        Args:
            x : np.ndarray
                The points to find neighbors for.
            r : float
                The radius to find neighbors within.

        Returns:
            np.ndarray
                Array of lists of indices of the neighbors within the given
                radius for each point in x.
        """
        raise NotImplementedError

    def find_all_dist_to_kth_neighbor(self, k: int) -> np.ndarray: 
        """Find the distance to the kth nearest neighbor for each point in the data.
        Does not include the point itself.

        Args:
            k (int): The kth nearest neighbor to find.
            
        Returns:
            np.ndarray: The distance to the kth neighbor for each point in the data.
        """
        return self.find_kth_neighbor(k + 1)
        
    def find_all_neighbors(self, k: int, return_index=False) -> np.ndarray:
        """Find the distance to the kth nearest neighbor for each point in x.

        May include x itself if it is in the data.

        Default implementation uses find_neighbors and returns the distance to
        the kth neighbor.

        Args:
            x : np.ndarray
                The points to find the kth nearest neighbor for.
            k : int
                The kth nearest neighbor to find.
        """
        if return_index:
            return self.find_neighbors(self._data, k + 1)[1]
        else:
            return self.find_neighbors(self._data, k + 1)[0]

    def find_kth_neighbor(self, k: int) -> np.ndarray:
        """Find the distance to the kth nearest neighbor for each point in x.

        May include x itself if it is in the data.

        Default implementation uses find_neighbors and returns the distance to
        the kth neighbor.

        Args:
            x : np.ndarray
                The points to find the kth nearest neighbor for.
            k : int
                The kth nearest neighbor to find.
        """

        
        """
        if theiler_t > 0:
            neighbors = self.find_neighbors(self._data, k + 2*theiler_t)[0][:, k ]
            neighbors = self.theiler_correction(neighbors, theiler_t)

        else:
        """
        return self.find_neighbors(self._data, k + 1)[0][:, k - 1]

    def count_all_neighbors(self, r: float) -> np.ndarray:
        """Count the number of neighbors strictly within (<=) a given radius for each point in the data.
        Does not include the point itself.

        Args:
            r (float): The radius to count neighbors within.
            
        Returns:
            np.ndarray: The number of neighbors within the given radius for each point in the data.
        """

        return self.count_neighbors(self._data, r) - 1
    
    def count_all_neighbors_within(self, r: float) -> np.ndarray:
        """Count the number of neighbors strictly within (<) a given radius for each point in the data.
        Does not include the point itself.

        Args:
            r (float): The radius to count neighbors within.
            theiler (int): no. next temporal neighbours ignored in range searches 
            
        Returns:
            np.ndarray: The number of neighbors within the given radius for each point in the data.
        """
        return self.count_neighbors_within(self._data, r) - 1

