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
        return self._tree.query(x, k=k, p=self._p, workers=self._num_threads)

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
    
    def count_neighbors_within(self, x: np.ndarray, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x,
            r=r,
            p=self._p,
            return_length=True,
            workers=self._num_threads,
        )
    
    def find_neighbors_theiler(self, x: np.ndarray, k: int, theiler_t) -> np.ndarray:
        
        epsilon = np.zeros(len(x))
        for i in range(len(x)):
            dists, idxs  = self._tree.query(x[i], k=k+20, p=self._p, workers=self._num_threads)

            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            valid_dists = []
            for d, j in zip(dists, idxs):
                if j == i:
                    continue
                if abs(i - j) <= theiler_t:
                    continue
                valid_dists.append(d)
                if len(valid_dists) == k:
                    break

            if len(valid_dists) < k:
                # fallback: could re-query with larger k or adapt radius here
                raise RuntimeError("Not enough neighbors outside Theiler window")

            epsilon[i] = valid_dists[-1]  # distance to k-th valid neighbor

        return epsilon

    def count_neighbors_theiler(self, x: np.ndarray, y: np.ndarray, k: int, theiler_t: int) -> np.ndarray:
        
        n = len(x)
        xy = np.hstack([x, y])
        
        tree_xy = cKDTree(xy)
        tree_x = cKDTree(x)
        tree_y = cKDTree(y)
        
        eps = np.empty(n, dtype=int)
        nx = np.empty(n, dtype=int)
        ny = np.empty(n, dtype=int)
        
        k_query = k + theiler_t + 5

        for i in range(n):

            dists, idxs = tree_xy.query(xy[i], k=k_query, p=self._p, workers=self._num_threads)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            valid = idxs[np.abs(idxs - i) > theiler_t]
            valid_dists = dists[np.abs(idxs - i) > theiler_t]

            if len(valid_dists) < k:
                raise ValueError("Not enough valid neighbors; may reduce theiler_t")

            eps_i = valid_dists[k - 1]
            eps[i] = eps_i
            
            #cor = 1e-15
            cor = 0

            bx = tree_x.query_ball_point(x[i], r=eps_i - cor, p=self._p, workers=self._num_threads)
            by = tree_y.query_ball_point(y[i], r=eps_i - cor, p=self._p, workers=self._num_threads)

            nx[i] = sum(abs(j - i) > theiler_t for j in bx)
            ny[i] = sum(abs(j - i) > theiler_t for j in by)


        return nx, ny
    ######################################################################################## TODO remove?
    def count_neighbors_theiler_old(self, x: np.ndarray, r: float, theiler_t: int) -> np.ndarray:
        
        n = len(x)
        nx = np.empty(n, dtype=int)
        a = self._tree.query_ball_point(
                x=x,
                r=np.nextafter(r, 0),
                p=self._p,
                #return_length=True,
                workers=self._num_threads)

        for i in range(n):
            idx = [j for j in a[i] if abs(j - i) > theiler_t]
            nx[i] = len(idx)

        return nx

    def count_neighbors_n(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        
        n = len(x)
        nx = np.empty(n, dtype=int)
        
        for i in range(n):
        
            #dx_i = np.linalg.norm(x[r[i]] - x[i], axis=1)
        
            #eps_x = dx_i.max()
        
            nx[i] = len(self._tree.query_ball_point(
                x=x[i],
                r=r[i],
                #r=eps_x,
                p=self._p,
                workers=self._num_threads))

        return nx

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
        return self._tree.query(x, k=k, p=self._p, workers=self._num_threads)

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

    #########################################################
    def count_neighbors_within(self, x: np.ndarray, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x,
            r=r,
            p=self._p,
            return_length=True,
            workers=self._num_threads,
        )
    
    def find_neighbors_theiler(self, x: np.ndarray, k: int, theiler_t) -> np.ndarray:
        
        n = len(x)

        k_query = k+theiler_t+5
        dists, idxs  = self._tree.query(x, k=k_query, p=self._p, workers=self._num_threads)
        
        epsilon = np.empty(n, dtype=float)

        for i in range(n):
            # Filter indices: remove self and Theiler window indices.
            valid_mask = np.ones(k_query, dtype=bool)
            # remove self
            valid_mask[idxs[i] == i] = False
            if theiler_t > 0:
                # remove indices with |i-j| <= theiler
                time_diff = np.abs(idxs[i] - i)
                valid_mask[time_diff <= theiler_t] = False

            valid_dists = dists[i][valid_mask]
            valid_idxs = idxs[i][valid_mask]

            if len(valid_dists) < k:
                # Not enough valid neighbors; you may want to handle this
                # more gracefully (e.g., reduce k or skip this sample).
                raise RuntimeError(
                    f"Not enough valid neighbors for sample {i}; "
                    f"got {len(valid_dists)}, need {k}"
                )

            # eps_i = distance to k-th valid neighbor
            # sort to be safe
            order = np.argsort(valid_dists)
            epsilon[i] = valid_dists[order[k - 1]]

        
            """
            epsilon = np.zeros(len(x))
            for i in range(len(x)):
                dist, idx  = self._tree.query(x[i], k=k+20, p=self._p, workers=self._num_threads)

                dist = np.atleast_1d(dist)
                idx = np.atleast_1d(idx)

                valid = np.abs(idx - i) > theiler_t
                idx_valid = idx[valid]
                dist_valid = dist[valid]
            """

            """
            valid_dists = []
            for d, j in zip(dists, idxs):
                if j == i:
                    continue
                if abs(i - j) <= theiler_t:
                    continue
                valid_dists.append(d)
                if len(valid_dists) == k:
                    break

            if len(valid_dists) < k:
                # fallback: could re-query with larger k or adapt radius here
                raise RuntimeError("Not enough neighbors outside Theiler window")

            epsilon[i] = valid_dists[-1]  # distance to k-th valid neighbor
            """
        return epsilon


    def count_neighbors_theiler(self, x: np.ndarray, y: np.ndarray, k: int, theiler_t: int) -> np.ndarray:
        
        n = len(x)
        xy = np.hstack([x, y])
        
        tree_xy = cKDTree(xy)
        tree_x = cKDTree(x)
        tree_y = cKDTree(y)
        
        eps = np.empty(n, dtype=int)
        nx = np.empty(n, dtype=int)
        ny = np.empty(n, dtype=int)
        
        k_query = k + theiler_t + 5

        for i in range(n):

            dists, idxs = tree_xy.query(xy[i], k=k_query, p=self._p, workers=self._num_threads)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            valid = idxs[np.abs(idxs - i) > theiler_t]
            valid_dists = dists[np.abs(idxs - i) > theiler_t]

            if len(valid_dists) < k:
                raise ValueError("Not enough valid neighbors; may reduce theiler_t")

            eps_i = valid_dists[k - 1]
            eps[i] = eps_i

            bx = tree_x.query_ball_point(x[i], r=eps_i - 1e-15, p=self._p, workers=self._num_threads)
            by = tree_y.query_ball_point(y[i], r=eps_i - 1e-15, p=self._p, workers=self._num_threads)

            nx[i] = sum(abs(j - i) > theiler_t for j in bx)
            ny[i] = sum(abs(j - i) > theiler_t for j in by)


        return nx, ny
        """
            idx = len(self._tree.query_ball_point(
                x=x[i],
                r=r[i],
                p=self._p,
                workers=self._num_threads))
                
            print(idx.shape)
            
            idx = [j for j in idx if j != i and (theiler_t == 0 or abs(j - i) > theiler_t)]
            counts[i] = len(idx)

        return counts
        """

    def count_neighbors_theiler_o2(self, x: np.ndarray, r: np.array, theiler_t: int) -> np.ndarray:
        
        n = len(x)
        xy = np.hstack([x, y])
        

        counts = np.empty(n, dtype=int)
        
        for i in range(n):

            idx = len(self._tree.query_ball_point(
                x=x[i],
                r=r[i],
                p=self._p,
                workers=self._num_threads))
                
            print(idx.shape)
            
            idx = [j for j in idx if j != i and (theiler_t == 0 or abs(j - i) > theiler_t)]
            counts[i] = len(idx)

        return counts


    ######################################################################################## TODO remove?
    def count_neighbors_theiler_old(self, x: np.ndarray, r: float, theiler_t: int) -> np.ndarray:
        
        n = len(x)
        
        neighbours = self._tree.query_ball_point(
                x=x,
                r=r,#np.nextafter(r, 0),
                p=self._p,
                workers=self._num_threads)
        
        counts = np.empty(n, dtype=int)
        
        for i in range(n):
            idxs = neighbours[i]
            
            idxs = [j for j in idxs if abs(j - i) > theiler_t]

            counts[i] = len(idxs)

        return counts

        ######################################################################## TODO
        """
        for i in range(n):

            nx[i] = len(self._tree.query_ball_point(
                x=x[i],
                r=r[i],
                p=self._p,
                workers=self._num_threads))

            idx = [j for j in a[i] if abs(j - i) > theiler_t]
            nx[i] = len(idx)

        """
        #return nx

    def count_neighbors_n(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        
        n = len(x)
        nx = np.empty(n, dtype=int)
        
        for i in range(n):
        
            #dx_i = np.linalg.norm(x[r[i]] - x[i], axis=1)
        
            #eps_x = dx_i.max()
        
            nx[i] = len(self._tree.query_ball_point(
                x=x[i],
                r=r[i],
                #r=eps_x,
                p=self._p,
                workers=self._num_threads))

        return nx
