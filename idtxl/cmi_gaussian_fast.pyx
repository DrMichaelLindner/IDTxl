# cmi_gaussian_fast.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt

ctypedef np.float64_t DTYPE_t

cdef inline double _cholesky_logdet(double[:, ::1] A) except? -1:
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i, j, k
    cdef np.ndarray[DTYPE_t, ndim=2] Lnp = np.empty((n, n), dtype=np.float64)
    cdef double[:, ::1] L = Lnp
    cdef double s

    for i in range(n):
        for j in range(n):
            L[i, j] = 0.0

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    return -1.0
                L[i, j] = sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    s = 0.0
    for i in range(n):
        s += log(L[i, i])
    return 2.0 * s

cdef void _mean_and_center(double[:, ::1] X, double[::1] mean, double[:, ::1] Xc) nogil:
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t i, j
    cdef double s

    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j]
        mean[j] = s / n
        for i in range(n):
            Xc[i, j] = X[i, j] - mean[j]

cdef void _cov_block(double[:, ::1] X, Py_ssize_t a0, Py_ssize_t a1,
                     Py_ssize_t b0, Py_ssize_t b1,
                     double[:, ::1] C) nogil:
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t i, j, k
    cdef double s
    for i in range(a1 - a0):
        for j in range(b1 - b0):
            s = 0.0
            for k in range(n):
                s += X[k, a0 + i] * X[k, b0 + j]
            C[i, j] = s / (n - 1)

cdef void _copy_sym_block(double[:, ::1] src, Py_ssize_t r0, Py_ssize_t c0, double[:, ::1] dst) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nr = dst.shape[0]
    cdef Py_ssize_t nc = dst.shape[1]
    for i in range(nr):
        for j in range(nc):
            dst[i, j] = src[r0 + i, c0 + j]

cpdef double conditional_mutual_information(np.ndarray[DTYPE_t, ndim=2] data,
                                             Py_ssize_t dx,
                                             Py_ssize_t dy):
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t d = data.shape[1]
    cdef Py_ssize_t dz = d - dx - dy
    cdef np.ndarray[DTYPE_t, ndim=2] Xc_np = np.empty((n, d), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mean_np = np.empty(d, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] cov_np = np.empty((d, d), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] xz_np, yz_np, z_np
    cdef double[:, ::1] Xc = Xc_np
    cdef double[::1] mean = mean_np
    cdef double[:, ::1] cov = cov_np
    cdef double[:, ::1] xz, yz, z
    cdef double ld_xz, ld_yz, ld_z, ld_xyz

    if dz <= 0:
        raise ValueError("Need at least one conditioning variable.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")

    _mean_and_center(data, mean, Xc)

    _cov_block(Xc, 0, dx + dz, 0, dx + dz, cov[:dx + dz, :dx + dz])
    _cov_block(Xc, dx, dx + dy + dz, dx, dx + dy + dz, cov[dx:dx + dy + dz, dx:dx + dy + dz])
    _cov_block(Xc, dx + dy, d, dx + dy, d, cov[dx + dy:, dx + dy:])
    _cov_block(Xc, 0, d, 0, d, cov)

    xz_np = np.empty((dx + dz, dx + dz), dtype=np.float64)
    yz_np = np.empty((dy + dz, dy + dz), dtype=np.float64)
    z_np = np.empty((dz, dz), dtype=np.float64)

    _copy_sym_block(cov, 0, 0, <double[:, ::1]>xz_np)
    _copy_sym_block(cov, dx, dx, <double[:, ::1]>yz_np)
    _copy_sym_block(cov, dx + dy, dx + dy, <double[:, ::1]>z_np)

    ld_xz = _cholesky_logdet(<double[:, ::1]>xz_np)
    ld_yz = _cholesky_logdet(<double[:, ::1]>yz_np)
    ld_z = _cholesky_logdet(<double[:, ::1]>z_np)
    ld_xyz = _cholesky_logdet(cov)

    return 0.5 * (ld_xz + ld_yz - ld_z - ld_xyz)