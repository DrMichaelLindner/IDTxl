import numpy as np
from scipy.spatial import KDTree, cKDTree

from idtxl.knn.tree_knn_finder import TreeKnnFinder


class ScipyKDTreeKnnFinder(TreeKnnFinder):
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)

        if self._metric == "chebyshev":
            self._p = np.inf
        elif self._metric == "euclidean":
            self._p = 2
        else:
            raise ValueError(f"Unsupported metric {self._metric}")

        self._tree = KDTree(data, leafsize=self._leaf_size)

    def find_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        return self._tree.query(x, k=k, p=np.inf, workers=self._num_threads)

    def find_neighbors_within(self, x: np.array, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x, r=np.nextafter(r, 0), p=self._p, workers=self._num_threads
        )

    def count_neighbors(self, x: np.ndarray, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x,
            r=np.nextafter(r, 0),
            p=self._p,
            return_length=True,
            workers=self._num_threads,
        )

    def count_neighbors_theiler(self, x: np.ndarray, r: float, theiler) -> np.ndarray:
        counts = np.empty(len(x), dtype=int)
        #### round eps slightly smaller to mimic "strictly within" behavior
        ###eps_shrunk = r - 1e-12 if r > 0 else eps
        for i, p in enumerate(x):
            neighbors = self._tree.query_ball_point(p, r)
            if  theiler <= 0:
                # exclude self if present
                cnt = len(neighbors) - (1 if i in neighbors else 0)
            else:
                # exclude indices j with |i-j| <= theiler, and exclude self
                cnt = 0
                low = i - theiler
                high = i + theiler
                for j in neighbors:
                    if j < low or j > high:
                        if j != i:
                            cnt += 1
            counts[i] = cnt
        return counts



class ScipycKDTreeKnnFinder(TreeKnnFinder):
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)

        if self._metric == "chebyshev":
            self._p = np.inf
        elif self._metric == "euclidean":
            self._p = 2
        else:
            raise ValueError(f"Unsupported metric {self._metric}")

        self._tree = cKDTree(data, leafsize=self._leaf_size)

    def find_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        return self._tree.query(x, k=k, p=np.inf, workers=self._num_threads)

    def find_neighbors_within(self, x: np.array, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x, r=np.nextafter(r, 0), p=self._p, workers=self._num_threads
        )

    def count_neighbors(self, x: np.ndarray, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x,
            r=np.nextafter(r, 0),
            p=self._p,
            return_length=True,
            workers=self._num_threads,
        )
    """    
    def count_neighbors_theiler(self, x: np.ndarray, r: float, theiler) -> np.ndarray:
        counts = np.empty(len(x), dtype=int)
        #### round eps slightly smaller to mimic "strictly within" behavior
        ###eps_shrunk = r - 1e-12 if r > 0 else eps
        for i, p in enumerate(x):
            neighbors = self._tree.query_ball_point(p=p, r=r)
            if  theiler <= 0:
                # exclude self if present
                cnt = len(neighbors) - (1 if i in neighbors else 0)
            else:
                # exclude indices j with |i-j| <= theiler, and exclude self
                cnt = 0
                low = i - theiler
                high = i + theiler
                for j in neighbors:
                    if j < low or j > high:
                        if j != i:
                            cnt += 1
            counts[i] = cnt
        return counts
    """
    
