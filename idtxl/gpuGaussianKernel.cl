#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*
 * Compute means of X and Y.

 * X: n x dx
 * Y: n x dy
 */
__kernel void means_xy(
    __global const double *x,
    __global const double *y,
    __global double *mean_x,
    __global double *mean_y,
    const int n,
    const int dx,
    const int dy)
{
    const int j = get_global_id(0);

    if (j < dx) {
        double sum_x = 0.0;

        for (int i = 0; i < n; ++i)
            sum_x += x[i * dx + j];

        mean_x[j] = sum_x / (double)n;
    }

    if (j < dy) {
        double sum_y = 0.0;

        for (int i = 0; i < n; ++i)
            sum_y += y[i * dy + j];

        mean_y[j] = sum_y / (double)n;
    }
}


/*
 * Center X and Y independently.

 * X input:  n x dx
 * Y input:  n x dy

 * CX output:  n_pad x px
 * CY output:  n_pad x py

 * All padded rows and columns are initialized to zero.
 */
__kernel void center_x(
    __global const double *x,
    __global const double *mean_x,
    __global double *cx,
    const int n,
    const int n_pad,
    const int dx,
    const int px)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= n_pad || j >= px)
        return;

    if (j < dx)
        cx[i * px + j] = x[i * dx + j] - mean_x[j];
    else
        cx[i * px + j] = 0.0;
}


__kernel void center_y(
    __global const double *y,
    __global const double *mean_y,
    __global double *cy,
    const int n,
    const int n_pad,
    const int dy,
    const int py)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= n_pad || j >= py)
        return;

    if (j < dy)
        cy[i * py + j] = y[i * dy + j] - mean_y[j];
    else
        cy[i * py + j] = 0.0;
}


/*
 * Construct centered XY.

 * CXY output: n_pad x pxy
 * First dx columns are centered X.
 * Next dy columns are centered Y.
 * Remaining columns are zero.
 */
__kernel void center_xy(
    __global const double *cx,
    __global const double *cy,
    __global double *cxy,
    const int n,
    const int dx,
    const int dy,
    const int px,
    const int py,
    const int pxy,
    const int n_pad)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= n_pad || j >= pxy)
        return;

    if (j < dx) {
        cxy[i * pxy + j] = cx[i * px + j];
    }
    else if (j < dx + dy) {
        const int jy = j - dx;
        cxy[i * pxy + j] = cy[i * py + jy];
    }
    else {
        cxy[i * pxy + j] = 0.0;
    }
}


/*
 * Compute one covariance matrix.

 * Input:
 *     centered: n x dp

 * Output:
 *     covariance: dp x dp

 * Only the first d x d part is statistically meaningful.
 */
__kernel void covariance_one(
    __global const double *centered,
    __global double *covariance,
    const int n,
    const int d,
    const int dp)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= dp || col >= dp)
        return;

    if (row >= d || col >= d) {
        covariance[row * dp + col] = 0.0;
        return;
    }

    double sum = 0.0;

    for (int i = 0; i < n; ++i)
        sum += centered[i * dp + row]
             * centered[i * dp + col];

    covariance[row * dp + col] = sum / (double)n;
}


/*
 * Calculate qx, qy, and qxy in one kernel.

 * qx[i]  = || inv(Lx)  cx[i]  ||^2
 * qy[i]  = || inv(Ly)  cy[i]  ||^2
 * qxy[i] = || inv(Lxy) cxy[i] ||^2
 */
__kernel void quadratic_forms_three(
    __global const double *cx,
    __global const double *cy,
    __global const double *cxy,
    __global const double *inv_lx,
    __global const double *inv_ly,
    __global const double *inv_lxy,
    __global double *qx,
    __global double *qy,
    __global double *qxy,
    const int n,
    const int dx,
    const int dy,
    const int dxy,
    const int px,
    const int py,
    const int pxy)
{
    const int i = get_global_id(0);

    if (i >= n)
        return;

    double result_x = 0.0;
    double result_y = 0.0;
    double result_xy = 0.0;

    for (int k = 0; k < dx; ++k) {
        double z = 0.0;

        for (int j = 0; j < dx; ++j)
            z += inv_lx[k * px + j]
               * cx[i * px + j];

        result_x += z * z;
    }

    for (int k = 0; k < dy; ++k) {
        double z = 0.0;

        for (int j = 0; j < dy; ++j)
            z += inv_ly[k * py + j]
               * cy[i * py + j];

        result_y += z * z;
    }

    for (int k = 0; k < dxy; ++k) {
        double z = 0.0;

        for (int j = 0; j < dxy; ++j)
            z += inv_lxy[k * pxy + j]
               * cxy[i * pxy + j];

        result_xy += z * z;
    }

    qx[i] = result_x;
    qy[i] = result_y;
    qxy[i] = result_xy;
}





/*
 * Compute means of X, Y, and Z.

 * X: n x dx
 * Y: n x dy
 * Z: n x dz
 */
__kernel void means_xyz(
    __global const double *x,
    __global const double *y,
    __global const double *z,
    __global double *mean_x,
    __global double *mean_y,
    __global double *mean_z,
    const int n,
    const int dx,
    const int dy,
    const int dz)
{
    const int j = get_global_id(0);

    if (j < dx) {
        double sum_x = 0.0;

        for (int i = 0; i < n; ++i)
            sum_x += x[i * dx + j];

        mean_x[j] = sum_x / (double)n;
    }

    if (j < dy) {
        double sum_y = 0.0;

        for (int i = 0; i < n; ++i)
            sum_y += y[i * dy + j];

        mean_y[j] = sum_y / (double)n;
    }

    if (j < dz) {
        double sum_z = 0.0;

        for (int i = 0; i < n; ++i)
            sum_z += z[i * dz + j];

        mean_z[j] = sum_z / (double)n;
    }
}


/*
 * Compute four quadratic forms for each observation.

 * q_z[i]   = || inv(Lz)   * z[i]   ||^2
 * q_xz[i]  = || inv(Lxz)  * xz[i]  ||^2
 * q_yz[i]  = || inv(Lyz)  * yz[i]  ||^2
 * q_xyz[i] = || inv(Lxyz) * xyz[i] ||^2

 * Each inverse Cholesky matrix uses its own padded row stride.
 */
__kernel void quadratic_forms_four(
    __global const double *cz,
    __global const double *cxz,
    __global const double *cyz,
    __global const double *cxyz,
    __global const double *inv_lz,
    __global const double *inv_lxz,
    __global const double *inv_lyz,
    __global const double *inv_lxyz,
    __global double *qz,
    __global double *qxz,
    __global double *qyz,
    __global double *qxyz,
    const int n,
    const int dz,
    const int dxz,
    const int dyz,
    const int dxyz,
    const int pz,
    const int pxz,
    const int pyz,
    const int pxyz)
{
    const int i = get_global_id(0);

    if (i >= n)
        return;

    double result_z = 0.0;
    double result_xz = 0.0;
    double result_yz = 0.0;
    double result_xyz = 0.0;

    for (int k = 0; k < dz; ++k) {
        double value = 0.0;

        for (int j = 0; j < dz; ++j) {
            value += inv_lz[k * pz + j]
                    * cz[i * pz + j];
        }

        result_z += value * value;
    }

    for (int k = 0; k < dxz; ++k) {
        double value = 0.0;

        for (int j = 0; j < dxz; ++j) {
            value += inv_lxz[k * pxz + j]
                    * cxz[i * pxz + j];
        }

        result_xz += value * value;
    }

    for (int k = 0; k < dyz; ++k) {
        double value = 0.0;

        for (int j = 0; j < dyz; ++j) {
            value += inv_lyz[k * pyz + j]
                    * cyz[i * pyz + j];
        }

        result_yz += value * value;
    }

    for (int k = 0; k < dxyz; ++k) {
        double value = 0.0;

        for (int j = 0; j < dxyz; ++j) {
            value += inv_lxyz[k * pxyz + j]
                    * cxyz[i * pxyz + j];
        }

        result_xyz += value * value;
    }

    qz[i] = result_z;
    qxz[i] = result_xz;
    qyz[i] = result_yz;
    qxyz[i] = result_xyz;
}

/*
 * Center one input matrix.

 * The output has padded row stride dp.
 */
__kernel void center_one(
    __global const double *input,
    __global const double *mean,
    __global double *centered,
    const int n,
    const int d,
    const int dp)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= n || j >= dp)
        return;

    if (j < d)
        centered[i * dp + j] =
            input[i * d + j] - mean[j];
    else
        centered[i * dp + j] = 0.0;
}


/*
 * Construct a centered concatenation.

 * Output layout:
 *
 *     [X | Y]
 *     [X | Z]
 *     [Y | Z]
 *     [X | Y | Z]
 */
__kernel void concat_two(
    __global const double *a,
    __global const double *b,
    __global double *ab,
    const int n,
    const int da,
    const int db,
    const int d_ab,
    const int pa,
    const int pb,
    const int p_ab)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= n || j >= p_ab)
        return;

    if (j < da) {
        ab[i * p_ab + j] =
            a[i * pa + j];
    }
    else if (j < da + db) {
        const int jb = j - da;

        ab[i * p_ab + j] =
            b[i * pb + jb];
    }
    else {
        ab[i * p_ab + j] = 0.0;
    }
}


__kernel void concat_three(
    __global const double *a,
    __global const double *b,
    __global const double *c,
    __global double *abc,
    const int n,
    const int da,
    const int db,
    const int dc,
    const int pa,
    const int pb,
    const int pc,
    const int pabc)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= n || j >= pabc)
        return;

    if (j < da) {
        abc[i * pabc + j] =
            a[i * pa + j];
    }
    else if (j < da + db) {
        const int jb = j - da;

        abc[i * pabc + j] =
            b[i * pb + jb];
    }
    else if (j < da + db + dc) {
        const int jc = j - da - db;

        abc[i * pabc + j] =
            c[i * pc + jc];
    }
    else {
        abc[i * pabc + j] = 0.0;
    }
}



/*
 * Compute one Mahalanobis quadratic form per observation.

 * q[i] = || inv(L) @ centered[i] ||^2
 */
__kernel void quadratic_form(
    __global const double *centered,
    __global const double *inv_l,
    __global double *q,
    const int n,
    const int d,
    const int dp)
{
    const int i = get_global_id(0);

    if (i >= n)
        return;

    double result = 0.0;

    for (int k = 0; k < d; ++k) {
        double value = 0.0;

        for (int j = 0; j < d; ++j)
            value += inv_l[k * dp + j]
                   * centered[i * dp + j];

        result += value * value;
    }

    q[i] = result;
}
