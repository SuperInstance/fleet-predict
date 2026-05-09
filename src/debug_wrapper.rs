// temporary debug wrapper
include!("matrix.rs");

pub fn debug_pseudoinverse(mat: &[f64], n: usize) -> Vec<f64> {
    pseudoinverse(mat, n)
}
