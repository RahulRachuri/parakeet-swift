//
//  CParakeetBLAS.h — the two Accelerate CBLAS calls this port makes, behind plain C.
//
//  Accelerate ships two CBLAS interfaces. The current one is selected by defining
//  ACCELERATE_NEW_LAPACK and ACCELERATE_LAPACK_ILP64 before its headers are read; without
//  them you get the interface Apple deprecated in macOS 13.3, which is a different symbol
//  (cblas_sgemm against cblas_sgemm$NEWLAPACK$ILP64), not just a different warning.
//
//  Those defines used to be passed as raw compiler flags from Package.swift, which lands
//  them in SwiftPM's `unsafeFlags`, and a package carrying those cannot be depended on by
//  version — only by branch or local path. Declaring them as `cSettings: [.define(...)]`
//  on a C target is a setting SwiftPM recognises, so the restriction does not apply.
//
//  Hence this target. It exists so the defines have a C target to live on.
//
//  This header deliberately does NOT include Accelerate. It is what Swift imports, and
//  anything it includes would be parsed on the Swift side, where the defines are not set —
//  which would put the deprecated declarations back in front of the compiler and undo the
//  whole arrangement. Accelerate is included only in the .c file, where they are set.
//
//  Integer widths are `long` here and converted at the call: the current interface takes
//  64-bit indices, the iOS SDK's takes 32-bit, and doing the narrowing in C keeps one
//  spelling on the Swift side instead of the `numericCast` that used to be at every site.
//

#ifndef CPARAKEET_BLAS_H
#define CPARAKEET_BLAS_H

/// c = a·b, row-major, neither operand transposed. a is m×k, b is k×n, c is m×n.
/// Overwrites c (alpha = 1, beta = 0).
void pk_sgemm_row_major(long m, long n, long k,
                        const float *a, long lda,
                        const float *b, long ldb,
                        float *c, long ldc);

/// y += A·x, row-major, A not transposed, A is m×n, x and y unit-strided.
/// Accumulates into y (alpha = 1, beta = 1), so y must already hold the bias.
void pk_sgemv_row_major_accumulate(long m, long n,
                                   const float *a, long lda,
                                   const float *x, float *y);

#endif /* CPARAKEET_BLAS_H */
