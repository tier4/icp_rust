use num_traits::real::Real;

pub fn rho(e: f64, k: f64) -> f64 {
    debug_assert!(e >= 0.);
    debug_assert!(k >= 0.);
    let k_squared = k * k;
    if e <= k_squared {
        e
    } else {
        2. * k * e.sqrt() - k_squared
    }
}

pub fn drho(e: f64, k: f64) -> f64 {
    debug_assert!(e >= 0.);
    debug_assert!(k >= 0.);
    let k_squared = k * k;
    if e <= k_squared {
        1.
    } else {
        k / e.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rho() {
        assert_eq!(rho(0.1 * 0.1, 0.1), 0.1 * 0.1);
        assert_eq!(rho(0.101 * 0.101, 0.1), 2. * 0.1 * 0.101 - 0.1 * 0.1);
        assert_eq!(rho(0.09 * 0.09, 0.1), 0.09 * 0.09);
    }

    #[test]
    fn test_drho() {
        let e1 = (4.000_f64 + 0.001_f64).powi(2);
        let e0 = 4.000_f64.powi(2);
        let k = 4.0_f64;
        let expected = (rho(e1, k) - rho(e0, k)) / (e1 - e0);
        assert!((drho(e0, k) - expected).abs() < 1e-3);

        let e1 = (0.10_f64 + 0.01_f64).powi(2);
        let e0 = 0.10_f64.powi(2);
        let k = 4.0_f64;
        let expected = (rho(e1, k) - rho(e0, k)) / (e1 - e0);
        assert_eq!(expected, drho(e0, k));

        let e1 = (0.10_f64 + 0.0001_f64).powi(2);
        let e0 = 0.10_f64.powi(2);
        let k = 0.10_f64;
        let expected = (rho(e1, k) - rho(e0, k)) / (e1 - e0);
        assert!((drho(e0, k) - expected).abs() < 1e-3);

        let e1 = (5.000_f64 + 0.001_f64).powi(2);
        let e0 = 5.000_f64.powi(2);
        let k = 4.0_f64;
        let expected = (rho(e1, k) - rho(e0, k)) / (e1 - e0);
        assert!((drho(e0, k) - expected).abs() < 1e-3);

        let e1 = (10.000_f64 + 0.001_f64).powi(2);
        let e0 = 10.000_f64.powi(2);
        let k = 4.0_f64;
        let expected = (rho(e1, k) - rho(e0, k)) / (e1 - e0);
        assert!((drho(e0, k) - expected).abs() < 1e-3);
    }
}
