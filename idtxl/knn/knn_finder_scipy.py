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

    def find_neighbors_theiler(self, x: np.ndarray, k: int, theiler_t) -> np.ndarray:
        
        epsilon = np.zeros(len(x))
        for i in range(len(x)):
            dists, idxs  = self._tree.query(x[i], k=k+20, p=np.inf, workers=self._num_threads)

            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            valid_dists = []
            for d, j in zip(dists, idxs):
                if j == i:
                    continue
                if abs(i - j) < theiler_t:
                    continue
                valid_dists.append(d)
                if len(valid_dists) == k:
                    break

            if len(valid_dists) < k:
                # fallback: could re-query with larger k or adapt radius here
                raise RuntimeError("Not enough neighbors outside Theiler window")

            epsilon[i] = valid_dists[-1]  # distance to k-th valid neighbor

        return epsilon


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
    
    def find_neighbors_theiler(self, x: np.ndarray, k: int, theiler_t) -> np.ndarray:
        
        epsilon = np.zeros(len(x))
        for i in range(len(x)):
            dists, idxs  = self._tree.query(x[i], k=k+20, p=np.inf, workers=self._num_threads)

            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            valid_dists = []
            for d, j in zip(dists, idxs):
                if j == i:
                    continue
                if abs(i - j) < theiler_t:
                    continue
                valid_dists.append(d)
                if len(valid_dists) == k:
                    break

            if len(valid_dists) < k:
                # fallback: could re-query with larger k or adapt radius here
                raise RuntimeError("Not enough neighbors outside Theiler window")

            epsilon[i] = valid_dists[-1]  # distance to k-th valid neighbor

        return epsilon
