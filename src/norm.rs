use nalgebra::{Dim, Matrix, Storage};

// We need real::Real in no-std but GitHub CI raises some warning without this
// attribute
#[allow(unused_imports)]
use num_traits::real::Real;

pub fn norm_squared<R: Dim, C: Dim, S: Storage<f64, R, C>>(matrix: &Matrix<f64, R, C, S>) -> f64 {
    let mut res = 0f64;

    for i in 0..matrix.ncols() {
        let col = matrix.column(i);
        res += col.dot(&col);
    }

    res
}

pub fn norm<R: Dim, C: Dim, S: Storage<f64, R, C>>(matrix: &Matrix<f64, R, C, S>) -> f64 {
    f64::sqrt(norm_squared(matrix))
}
