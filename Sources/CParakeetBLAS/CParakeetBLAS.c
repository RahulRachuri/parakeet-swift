#include "include/CParakeetBLAS.h"

// Read with ACCELERATE_NEW_LAPACK and ACCELERATE_LAPACK_ILP64 set, which Package.swift
// declares on this target. That is the only reason this file exists; see the header.
#include <Accelerate/Accelerate.h>

void pk_sgemm_row_major(long m, long n, long k,
                        const float *a, long lda,
                        const float *b, long ldb,
                        float *c, long ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
}

void pk_sgemv_row_major_accumulate(long m, long n,
                                   const float *a, long lda,
                                   const float *x, float *y) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, lda, x, 1, 1.0f, y, 1);
}
