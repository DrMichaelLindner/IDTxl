#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline ulong u64(long v)
{
    return (ulong)v;
}

// One work item per observation.
// joint_counts is zeroed before this kernel is launched.
__kernel void histogram_joint(
    __global const int *x_idx,
    __global const int *y_idx,
    __global uint *joint_counts,
    const int ny,
    const int n)
{
    const int i = get_global_id(0);

    if (i < n) {
        const int x = x_idx[i];
        const int y = y_idx[i];
        const int index = x * ny + y;

        atomic_inc(&joint_counts[index]);
    }
}


// Compute local MI for each sample:
//
// log2(pxy / (px * py))
//
// counts are used directly, so the factors of n cancel:
//
// log2(joint_count * n / (x_count * y_count))
__kernel void local_mi(
    __global const int *x_idx,
    __global const int *y_idx,
    __global const uint *joint_counts,
    __global const uint *x_counts,
    __global const uint *y_counts,
    __global float *result,
    const int ny,
    const float n,
    const int sample_count)
{
    const int i = get_global_id(0);

    if (i < sample_count) {
        const int x = x_idx[i];
        const int y = y_idx[i];
        const uint joint = joint_counts[x * ny + y];

        if (joint == 0) {
            result[i] = 0.0f;
        } else {
            const float ratio =
                ((float) joint * n) /
                ((float) x_counts[x] * (float) y_counts[y]);

            result[i] = log2(ratio);
        }
    }
}

__kernel void mi_terms(
    __global const uint *joint,
    __global const uint *px,
    __global const uint *py,
    __global float *terms,
    const int nx,
    const int ny,
    const float n)
{
    int k = get_global_id(0);

    if (k < nx * ny) {
        uint cxy = joint[k];

        if (cxy == 0) {
            terms[k] = 0.0f;
        } else {
            int x = k / ny;
            int y = k - x * ny;

            float ratio =
                ((float)cxy * n) /
                ((float)px[x] * (float)py[y]);

            terms[k] =
                ((float)cxy / n) * log2(ratio);
        }
    }
}


__kernel void count_cmi(
    __global const long *x,
    __global const long *y,
    __global const long *z,
    __global int *c_xyz,
    __global int *c_xz,
    __global int *c_yz,
    __global int *c_z,
    const long ny,
    const long nz,
    const ulong n)
{
    ulong i = (ulong)get_global_id(0);

    if (i >= n)
        return;

    ulong xv = u64(x[i]);
    ulong yv = u64(y[i]);
    ulong zv = u64(z[i]);

    ulong xz  = xv * u64(nz) + zv;
    ulong yz  = yv * u64(nz) + zv;
    ulong xyz = (xv * u64(ny) + yv) * u64(nz) + zv;

    atomic_add(c_xyz + xyz, 1);
    atomic_add(c_xz  + xz,  1);
    atomic_add(c_yz  + yz,  1);
    atomic_add(c_z   + zv,  1);
}


__kernel void local_cmi(
    __global const long *x,
    __global const long *y,
    __global const long *z,
    __global const int *c_xyz,
    __global const int *c_xz,
    __global const int *c_yz,
    __global const int *c_z,
    __global double *out,
    const long ny,
    const long nz,
    const ulong n)
{
    ulong i = (ulong)get_global_id(0);

    if (i >= n)
        return;

    ulong xv = u64(x[i]);
    ulong yv = u64(y[i]);
    ulong zv = u64(z[i]);

    ulong xz  = xv * u64(nz) + zv;
    ulong yz  = yv * u64(nz) + zv;
    ulong xyz = (xv * u64(ny) + yv) * u64(nz) + zv;

    int a = c_xyz[xyz];
    int b = c_z[zv];
    int c = c_xz[xz];
    int d = c_yz[yz];

    if (a > 0 && b > 0 && c > 0 && d > 0)
        out[i] = log2(
            ((double)a * (double)b) /
            ((double)c * (double)d)
        );
    else
        out[i] = 0.0;
}