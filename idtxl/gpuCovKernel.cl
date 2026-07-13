// covariance.cl
__kernel void cov_mat(
    __global const float *Z,  // input data: T x D, row-major
    const int T,
    const int D,
    __global float *C         // output covariance: D x D, row-major
) {
    int i = get_global_id(0); // row index in covariance
    int j = get_global_id(1); // col index in covariance
    
    if (i >= D || j >= D) return;
    
    double acc = 0.0;
    for (int t = 0; t < T; ++t) {
        float zi = Z[t*D + i];
        float zj = Z[t*D + j];
        acc += (double)zi * (double)zj;
    }
    // For mean-zero data, covariance = E[zi*zj] ≈ acc / T
    C[i*D + j] = (float)(acc / (double)T);
}