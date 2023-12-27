use nalgebra::Matrix2;

pub fn log_so2(rotation: Matrix2<f64>) -> f64 {
    f64::atan2(rotation[(1, 0)], rotation[(0, 0)])
}

pub fn exp_so2(theta: f64) -> Matrix2<f64> {
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

    #[test]
    fn test_exp_so2() {
        let theta = 0.3;
        let rot = exp_so2(theta);
        assert_eq!(rot.nrows(), 2);
        assert_eq!(rot.ncols(), 2);
        assert_eq!(rot[(0, 0)], f64::cos(theta));
        assert_eq!(rot[(0, 1)], -f64::sin(theta));
        assert_eq!(rot[(1, 0)], f64::sin(theta));
        assert_eq!(rot[(1, 1)], f64::cos(theta));
    }

    #[test]
    fn test_log_so2() {
        let theta = 0.3 * std::f64::consts::PI;
        let rot = exp_so2(theta);
        assert!((log_so2(rot) - theta).abs() < 1e-6);

        let theta = 0.8 * std::f64::consts::PI;
        let rot = exp_so2(theta);
        assert!((log_so2(rot) - theta).abs() < 1e-6);

        let theta = -0.7 * std::f64::consts::PI;
        let rot = exp_so2(theta);
        assert!((log_so2(rot) - theta).abs() < 1e-6);

        let theta = -0.1 * std::f64::consts::PI;
        let rot = exp_so2(theta);
        assert!((log_so2(rot) - theta).abs() < 1e-6);
    }
}
