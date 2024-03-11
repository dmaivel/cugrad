#include <cugrad.hpp>
#include <cugrad_kernels.hpp>

__global__ void kernel_add(int N, float *dst, float *src0, float *src1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = src0[i] + src1[i];
}

void cugrad::kernel::add(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1)
{
    kernel_add<<<blk_in_grid, thr_per_blk>>>(N, dst, src0, src1);
}

__global__ void kernel_add_grad(int N, float *grad, float *path)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    grad[i] += path[i];
}

void cugrad::kernel::add_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path)
{
    kernel_add_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path);
}

__global__ void kernel_sub(int N, float *dst, float *src0, float *src1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = src0[i] - src1[i];
}

void cugrad::kernel::sub(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1)
{
    kernel_sub<<<blk_in_grid, thr_per_blk>>>(N, dst, src0, src1);
}

__global__ void kernel_sub_grad(int N, float *grad, float *path, bool subtracter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    grad[i] += path[i] * (subtracter ? -1.f : 1.f);
}

void cugrad::kernel::sub_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, bool subtracter)
{
    kernel_sub_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path, subtracter);
}

__global__ void kernel_mul(int N, float *dst, float *src0, float *src1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = src0[i] * src1[i];
}

void cugrad::kernel::mul(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1)
{
    kernel_mul<<<blk_in_grid, thr_per_blk>>>(N, dst, src0, src1);
}

__global__ void kernel_mul_grad(int N, float *grad, float *path, float *local)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    grad[i] += path[i] * local[i];
}

void cugrad::kernel::mul_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local)
{
    kernel_mul_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path, local);
}

__global__ void kernel_div(int N, float *dst, float *src0, float *src1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = src0[i] / src1[i];
}

void cugrad::kernel::div(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1)
{
    kernel_div<<<blk_in_grid, thr_per_blk>>>(N, dst, src0, src1);
}

__global__ void kernel_div_grad(int N, float *grad, float *path, float *local0, float *local1, bool divisor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    grad[i] += path[i] * (
        !divisor ? 1.f / local1[i] : (-1.f * local0[i]) / (local1[i] * local1[i])
    );
}

void cugrad::kernel::div_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local0, float *local1, bool divisor)
{
    kernel_div_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path, local0, local1, divisor);
}

__global__ void kernel_matmul(int N, int M, float *dst, float *src0, float *src1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    float temp = 0.f;
    for (int j = 0; j < M; j++)
        temp += src0[j] * src1[j * N + i];

    dst[i] = temp;
}

void cugrad::kernel::matmul(int blk_in_grid, int thr_per_blk, int N, int M, float *dst, float *src0, float *src1)
{
    kernel_matmul<<<blk_in_grid, thr_per_blk>>>(N, M, dst, src0, src1);
}

__global__ void kernel_matmul_grad(int N, int M, float *grad, float *path, float *local0, float *local1, bool wrt_l0)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M)
        return;

    if (wrt_l0) {
        float temp = 0.f;
        for (int j = 0; j < N; j++)
            temp += local1[i * N + j];

        grad[i] += path[i] * temp;
    }
    else {
        for (int j = 0; j < N; j++)
            atomicAdd(&grad[i * N + j], path[j] * local0[i]);
    }
}

void cugrad::kernel::matmul_grad(int blk_in_grid, int thr_per_blk, int N, int M, float *grad, float *path, float *local0, float *local1, bool wrt_l0)
{
    kernel_matmul_grad<<<blk_in_grid, thr_per_blk>>>(N, M, grad, path, local0, local1, wrt_l0);
}

__global__ void kernel_sum(int N, float *dst, float *src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    atomicAdd(dst, src[i]);
}

void cugrad::kernel::sum(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src)
{
    kernel_sum<<<blk_in_grid, thr_per_blk>>>(N, dst, src);
}

__global__ void kernel_sum_grad(int N, float *grad, float *path)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    grad[i] += path[0] * 1;
}

void cugrad::kernel::sum_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path)
{
    kernel_sum_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path);
}

__global__ void kernel_exp(int N, float *dst, float *src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = exp(src[i]);
}

void cugrad::kernel::exp(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src)
{
    kernel_exp<<<blk_in_grid, thr_per_blk>>>(N, dst, src);
}

void cugrad::kernel::exp_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local)
{
    // the gradients are the same, except for local which in this case is just src (instead of being swapped between 2 src)
    kernel_mul_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path, local);
}

__global__ void kernel_log(int N, float *dst, float *src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = log(src[i]);
}

void cugrad::kernel::log(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src)
{
    kernel_log<<<blk_in_grid, thr_per_blk>>>(N, dst, src);
}

__global__ void kernel_log_grad(int N, float *grad, float *path, float *local)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    grad[i] += path[i] * (1.f / local[i]);
}

void cugrad::kernel::log_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local)
{
    kernel_log_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path, local);
}

__global__ void kernel_sqrt(int N, float *dst, float *src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    dst[i] = sqrt(src[i]);
}

void cugrad::kernel::sqrt(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src)
{
    kernel_sqrt<<<blk_in_grid, thr_per_blk>>>(N, dst, src);
}

__global__ void kernel_sqrt_grad(int N, float *grad, float *path, float *local)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    
    grad[i] += path[i] * (0.5f / sqrt(local[i]));
}

void cugrad::kernel::sqrt_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local)
{
    kernel_sqrt_grad<<<blk_in_grid, thr_per_blk>>>(N, grad, path, local);
}