/// Minimal linear algebra helpers for the normal equations (no external deps).
///
/// Provides matrix transpose-multiplication and Gauss-Jordan inversion
/// needed for ordinary least squares fitting.

/// Multiply A^T * A where A is stored row-major with `rows` rows and `cols` cols.
/// Returns a flat `cols * cols` vector in row-major order.
pub fn mul_transpose_a(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..rows {
                sum += a[k * cols + i] * a[k * cols + j];
            }
            out[i * cols + j] = sum;
        }
    }
    out
}

/// Multiply A^T * y where A is stored row-major.
/// Returns a flat `cols`-element vector.
pub fn mul_transpose_a_vec(a: &[f64], rows: usize, cols: usize, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; cols];
    for i in 0..cols {
        let mut sum = 0.0;
        for k in 0..rows {
            sum += a[k * cols + i] * y[k];
        }
        out[i] = sum;
    }
    out
}

/// Multiply matrix (flat, row-major, `m x n`) by vector (length `n`).
/// Returns length `m`.
pub fn mul_vec(mat: &[f64], vec: &[f64], n: usize) -> Vec<f64> {
    let m = mat.len() / n;
    let mut out = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += mat[i * n + j] * vec[j];
        }
        out[i] = sum;
    }
    out
}

/// Compute the inverse (or pseudoinverse) of a square matrix using
/// Gauss-Jordan elimination.
///
/// Matrix is stored flat row-major, size `n x n`.
/// Returns a flat `n * n` matrix (zero matrix if singular).
pub fn pseudoinverse(mat: &[f64], n: usize) -> Vec<f64> {
    // eprintln!("pseudoinverse called with n={}, mat={:?}", n, mat);
    let tolerance = 1e-10;

    if n == 0 {
        return Vec::new();
    }

    // Augmented matrix [A | I]
    let mut aug = vec![0.0; n * (2 * n)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = mat[i * n + j];
        }
        aug[i * (2 * n) + n + i] = 1.0;
    }

    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut max_val = aug[col * (2 * n) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (2 * n) + col].abs();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        if max_val < tolerance {
            // Singular: return zero matrix
            return vec![0.0; n * n];
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * (2 * n) + j, pivot_row * (2 * n) + j);
            }
        }

        // Normalize pivot row
        let pivot = aug[col * (2 * n) + col];
        for j in 0..(2 * n) {
            aug[col * (2 * n) + j] /= pivot;
        }

        // Eliminate other rows
        for row in 0..n {
            if row != col {
                let factor = aug[row * (2 * n) + col];
                if factor.abs() > tolerance {
                    for j in 0..(2 * n) {
                        aug[row * (2 * n) + j] -= factor * aug[col * (2 * n) + j];
                    }
                }
            }
        }
    }

    // Extract inverse
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * (2 * n) + n + j];
        }
    }

    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_transpose_small() {
        // A = [[1, 2], [3, 4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let result = mul_transpose_a(&a, 2, 2);
        // A^T A = [[10, 14], [14, 20]]
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 14.0).abs() < 1e-10);
        assert!((result[2] - 14.0).abs() < 1e-10);
        assert!((result[3] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul_transpose_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0];
        let result = mul_transpose_a_vec(&a, 2, 2, &y);
        // A^T y = [1*5 + 3*6, 2*5 + 4*6] = [23, 34]
        assert!((result[0] - 23.0).abs() < 1e-10);
        assert!((result[1] - 34.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul_vec_2x2() {
        let mat = vec![1.0, 2.0, 3.0, 4.0];
        let vec = vec![5.0, 6.0];
        let result = mul_vec(&mat, &vec, 2);
        // [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        assert!((result[0] - 17.0).abs() < 1e-10);
        assert!((result[1] - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_2x2_identity() {
        let mat = vec![1.0, 0.0, 0.0, 1.0];
        let inv = pseudoinverse(&mat, 2);
        assert!((inv[0] - 1.0).abs() < 1e-10);
        assert!((inv[1] - 0.0).abs() < 1e-10);
        assert!((inv[2] - 0.0).abs() < 1e-10);
        assert!((inv[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_2x2_general() {
        let mat = vec![4.0, 7.0, 2.0, 6.0];
        let inv = pseudoinverse(&mat, 2);
        assert!((inv[0] - 0.6).abs() < 1e-10, "got {}", inv[0]);
        assert!((inv[1] + 0.7).abs() < 1e-10, "got {}", inv[1]);
        assert!((inv[2] + 0.2).abs() < 1e-10, "got {}", inv[2]);
        assert!((inv[3] - 0.4).abs() < 1e-10, "got {}", inv[3]);

        // Verify A * A^{-1} ≈ I
        let identity_00 = mat[0] * inv[0] + mat[1] * inv[2];
        let identity_01 = mat[0] * inv[1] + mat[1] * inv[3];
        let identity_10 = mat[2] * inv[0] + mat[3] * inv[2];
        let identity_11 = mat[2] * inv[1] + mat[3] * inv[3];
        assert!((identity_00 - 1.0).abs() < 1e-10, "A*A^-1 [0,0] = {}", identity_00);
        assert!(identity_01.abs() < 1e-10);
        assert!(identity_10.abs() < 1e-10);
        assert!((identity_11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_singular_returns_zero() {
        let mat = vec![1.0, 2.0, 2.0, 4.0];
        let result = pseudoinverse(&mat, 2);
        // Should return zero matrix for singular
        assert!(result.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_inverse_3x3() {
        let mat = vec![
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0,
        ];
        let inv = pseudoinverse(&mat, 3);

        // Verify A * A^{-1} ≈ I
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += mat[i * 3 + k] * inv[k * 3 + j];
                }
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-10, "A*A^-1[{},{}] = {}", i, j, dot);
                } else {
                    assert!(dot.abs() < 1e-10, "A*A^-1[{},{}] = {}", i, j, dot);
                }
            }
        }
    }
}
