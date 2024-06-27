pub use crate::types::{Matrix2, Matrix3, Rotation2, Vector2, Vector3};

// We need real::Real in no-std but GitHub CI raises some warning without this
// attribute
#[allow(unused_imports)]
use num_traits::real::Real;

pub fn new_rotation2(theta: f64) -> Rotation2 {
    // In Rotation2::new is not supported in the no-std environment
    #[rustfmt::skip]
    Rotation2::from_matrix_unchecked(
        Matrix2::new(
            f64::cos(theta), -f64::sin(theta),
            f64::sin(theta), f64::cos(theta)
        )
    )
}

pub fn log(rotation: &Matrix2) -> f64 {
    f64::atan2(rotation[(1, 0)], rotation[(0, 0)])
}

pub fn exp(theta: f64) -> Matrix2 {
    let cos = f64::cos(theta);
    let sin = f64::sin(theta);
    #[rustfmt::skip]
    Matrix2::new(
        cos, -sin,
        sin, cos
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts;

    #[test]
    fn test_exp() {
        let theta = 0.3;
        let rot = exp(theta);
        assert_eq!(rot.nrows(), 2);
        assert_eq!(rot.ncols(), 2);
        assert_eq!(rot[(0, 0)], f64::cos(theta));
        assert_eq!(rot[(0, 1)], -f64::sin(theta));
        assert_eq!(rot[(1, 0)], f64::sin(theta));
        assert_eq!(rot[(1, 1)], f64::cos(theta));
    }

    #[test]
    fn test_log() {
        let theta = 0.3 * consts::PI;
        let rot = exp(theta);
        assert!((log(&rot) - theta).abs() < 1e-6);

        let theta = 0.8 * consts::PI;
        let rot = exp(theta);
        assert!((log(&rot) - theta).abs() < 1e-6);

        let theta = -0.7 * consts::PI;
        let rot = exp(theta);
        assert!((log(&rot) - theta).abs() < 1e-6);

        let theta = -0.1 * consts::PI;
        let rot = exp(theta);
        assert!((log(&rot) - theta).abs() < 1e-6);
    }
}
