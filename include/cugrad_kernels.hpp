#pragma once

namespace cugrad {
    namespace kernel {
        void add(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1);
        void add_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path);

        void sub(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1);
        void sub_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, bool subtracter);

        void mul(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1);
        void mul_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local);

        void div(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src0, float *src1);
        void div_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local0, float *local1, bool divisor);

        /* (n) * (n, m) */
        void matmul(int blk_in_grid, int thr_per_blk, int N, int M, float *dst, float *src0, float *src1);
        void matmul_grad(int blk_in_grid, int thr_per_blk, int N,  int M, float *grad, float *path, float *local0, float *local1, bool wrt_l0);

        void sum(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src);
        void sum_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path);

        void exp(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src);
        void exp_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local);

        void log(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src);
        void log_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local);

        void sqrt(int blk_in_grid, int thr_per_blk, int N, float *dst, float *src);
        void sqrt_grad(int blk_in_grid, int thr_per_blk, int N, float *grad, float *path, float *local);
    };
}