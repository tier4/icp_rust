#![feature(stmt_expr_attributes)]

use nalgebra::Cholesky;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

mod median;

type Param = nalgebra::Vector3<f64>;
type Measurement = nalgebra::Vector2<f64>;
type Jacobian = nalgebra::Matrix2x3<f64>;
type Hessian = nalgebra::Matrix3<f64>;

const huber_k: f64 = 1.345;

fn calc_rt(param: &Vector3<f64>) -> (Matrix2<f64>, Vector2<f64>) {
    let theta = param[2];
    let cos = f64::cos(theta);
    let sin = f64::sin(theta);

    #[rustfmt::skip]
    let R = Matrix2::new(
        cos, -sin,
        sin, cos,
    );

    let vx = param[0];
    let vy = param[1];

    let t = if theta == 0. {
        Vector2::new(vx, vy)
    } else {
        Vector2::new(
            (sin * vx - (1. - cos) * vy) / theta,
            ((1. - cos) * vx + sin * vy) / theta,
        )
    };
    (R, t)
}

fn exp_se2(param: &Vector3<f64>) -> Matrix3<f64> {
    let (R, t) = calc_rt(param);

    #[rustfmt::skip]
    Matrix3::new(
        R[(0, 0)], R[(0, 1)], t[0],
        R[(1, 0)], R[(1, 1)], t[1],
        0., 0., 1.,
    )
}

fn transform(param: &Param, landmark: &Measurement) -> Measurement {
    let (R, t) = calc_rt(param);
    R * landmark + t
}

pub fn residual(param: &Param, src: &Measurement, dst: &Measurement) -> Measurement {
    transform(param, src) - dst
}

pub fn error(param: &Param, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> f64 {
    src.iter().zip(dst.iter()).fold(0f64, |sum, (s, d)| {
        let r = residual(param, s, d);
        sum + r.dot(&r)
    })
}

fn inverse_3x3(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let m00 = matrix[(0, 0)];
    let m01 = matrix[(0, 1)];
    let m02 = matrix[(0, 2)];
    let m10 = matrix[(1, 0)];
    let m11 = matrix[(1, 1)];
    let m12 = matrix[(1, 2)];
    let m20 = matrix[(2, 0)];
    let m21 = matrix[(2, 1)];
    let m22 = matrix[(2, 2)];

    #[rustfmt::skip]
    let det = m00 * (m22 * m11 - m21 * m12)
            - m10 * (m22 * m01 - m21 * m02)
            + m20 * (m12 * m01 - m11 * m02);
    if det == 0f64 {
        return None;
    }

    #[rustfmt::skip]
    let mat = Matrix3::new(
        m22 * m11 - m21 * m12, -(m22 * m01 - m21 * m02), m12 * m01 - m11 * m02,
        -(m22 * m10 - m20 * m12), m22 * m00 - m20 * m02, -(m12 * m00 - m10 * m02),
        m21 * m10 - m20 * m11, -(m21 * m00 - m20 * m01), m11 * m00 - m10 * m01,
    );
    Some(mat / det)
}

fn jacobian(param: &Param, landmark: &Measurement) -> Jacobian {
    let a = Vector2::new(-landmark[1], landmark[0]);
    let (R, t) = calc_rt(param);
    let b = R * a;
    Jacobian::new(R[(0, 0)], R[(0, 1)], b[0], R[(1, 0)], R[(1, 1)], b[1])
}

fn check_input_size(input: &Vec<Measurement>) -> bool {
    // Check if the input does not have sufficient samples to estimate the update
    input.len() > 0 && input.len() >= input[0].len()
}

fn gauss_newton_update(param: &Param, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> Option<Param> {
    if !check_input_size(&src) {
        // The input does not have sufficient samples to estimate the update
        return None;
    }

    let (jtr, jtj) = src.iter().zip(dst.iter()).fold(
        (Param::zeros(), Hessian::zeros()),
        |(jtr, jtj), (s, d)| {
            let j = jacobian(param, s);
            let r = transform(param, s) - d;
            let jtr_: Param = j.transpose() * r;
            let jtj_: Hessian = j.transpose() * j;
            (jtr + jtr_, jtj + jtj_)
        },
    );
    let update = Cholesky::new_unchecked(jtj).solve(&jtr);
    Some(-update)
}

fn calc_mads(residuals: &Vec<Measurement>) -> Option<Vec<f64>> {
    let dimension = residuals[0].len();
    let mut mads = vec![0f64; dimension];
    for j in 0..dimension {
        let jth_dim = residuals.iter().map(|r| r[j]).collect::<Vec<_>>();
        mads[j] = match mad(&jth_dim) {
            Some(s) => s,
            None => return None,
        };
    }
    Some(mads)
}

fn weighted_gauss_newton_update(
    param: &Param,
    src: &Vec<Measurement>,
    dst: &Vec<Measurement>,
) -> Option<Param> {
    debug_assert_eq!(src.len(), dst.len());

    if !check_input_size(&src) {
        // The input does not have sufficient samples to estimate the update
        return None;
    }

    let residuals = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| residual(param, s, d))
        .collect::<Vec<_>>();

    let mads = match calc_mads(&residuals) {
        Some(m) => m,
        None => return None,
    };

    let mut jtr = Param::zeros();
    let mut jtj = Hessian::zeros();
    for (s, r) in src.iter().zip(residuals.iter()) {
        let jacobian_i = jacobian(param, s);
        for (j, jacobian_ij) in jacobian_i.row_iter().enumerate() {
            let r_ij = r[j];
            let w_ij = drho(r_ij * r_ij, huber_k);
            let g = 1. / mads[j];
            jtr += w_ij * g * jacobian_ij.transpose() * r_ij;
            jtj += w_ij * g * jacobian_ij.transpose() * jacobian_ij;
        }
    }

    let update = Cholesky::new_unchecked(jtj).solve(&jtr);
    Some(-update)
}

fn rho(e: f64, k: f64) -> f64 {
    debug_assert!(e >= 0.);
    debug_assert!(k >= 0.);
    let k_squared = k * k;
    if e <= k_squared {
        e
    } else {
        2. * k * e.sqrt() - k_squared
    }
}

fn drho(e: f64, k: f64) -> f64 {
    debug_assert!(e >= 0.);
    debug_assert!(k >= 0.);
    let k_squared = k * k;
    if e <= k_squared {
        1.
    } else {
        k / e.sqrt()
    }
}

fn mad(input: &Vec<f64>) -> Option<f64> {
    let m = match median::median(&input) {
        None => return None,
        Some(m) => m,
    };
    let a = input.iter().map(|e| (e - m).abs()).collect::<Vec<f64>>();
    return median::median(&a);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_se2() {
        // In python,
        // >>> import numpy as np
        // >>> from scipy.linalg import expm
        // >>> def skew_se2(v):
        // ...     return np.array([
        // ...         [0, -v[2], v[0]],
        // ...         [v[2], 0, v[1]],
        // ...         [0, 0, 0]])
        // ...
        // >>> a = np.array([-0.29638466, -0.15797957, -0.89885138])
        // >>> expm(skew_se2(a))
        // array([[ 0.6225093 ,  0.7826124 , -0.32440305],
        //        [-0.7826124 ,  0.6225093 , -0.01307704],
        //        [ 0.        ,  0.        ,  1.        ]])

        let transform = exp_se2(&Vector3::new(-0.29638466, -0.15797957, -0.89885138));

        #[rustfmt::skip]
        let expected = Matrix3::new(
            0.6225093, 0.7826124, -0.32440305,
            -0.7826124, 0.6225093, -0.01307704,
            0., 0., 1.,
        );
        assert!((transform - expected).norm() < 1e-6);

        // >>> a = np.array([-0.24295876,  0.95847196,  0.91052553])
        // >>> expm(skew_se2(a))
        // array([[ 0.61333076, -0.78982617, -0.61778258],
        //        [ 0.78982617,  0.61333076,  0.72824049],
        //        [ 0.        ,  0.        ,  1.        ]])

        let transform = exp_se2(&Vector3::new(-0.24295876, 0.95847196, 0.91052553));

        #[rustfmt::skip]
        let expected = Matrix3::new(
            0.61333076, -0.78982617, -0.61778258,
            0.78982617, 0.61333076, 0.72824049,
            0., 0., 1.,
        );
        assert!((transform - expected).norm() < 1e-6);

        // >>> a = np.array([10., -20., 0.])
        // >>> expm(skew_se2(a))
        // array([[  1.,   0.,  10.],
        //        [  0.,   1., -20.],
        //        [  0.,   0.,   1.]])

        let transform = exp_se2(&Vector3::new(10., -20., 0.));

        #[rustfmt::skip]
        let expected = Matrix3::new(
            1., 0., 10.,
            0., 1., -20.,
            0., 0., 1.,
        );
        assert!((transform - expected).norm() < 1e-6);
    }

    #[test]
    fn test_residual() {
        let param: Param = Vector3::new(-10., 20., 0.01);
        let src = Vector2::new(7f64, 8f64);
        let dst = transform(&param, &src);
        assert_eq!(residual(&param, &src, &dst), Vector2::zeros());
    }

    #[test]
    fn test_error() {
        let src = vec![
            Measurement::new(-6., 9.),
            Measurement::new(-1., 9.),
            Measurement::new(-4., -4.),
        ];

        let dst = vec![
            Measurement::new(-4., 4.),
            Measurement::new(0., 3.),
            Measurement::new(-3., -8.),
        ];

        let param: Param = Vector3::new(10., 20., 0.01);
        let r0 = residual(&param, &src[0], &dst[0]);
        let r1 = residual(&param, &src[1], &dst[1]);
        let r2 = residual(&param, &src[2], &dst[2]);
        let expected = r0.norm_squared() + r1.norm_squared() + r2.norm_squared();
        assert_eq!(error(&param, &src, &dst), expected);
    }

    #[test]
    fn test_inverse_3x3() {
        let identity = Matrix3::identity();

        #[rustfmt::skip]
        let matrix = Matrix3::new(
            -3.64867356, 0.11236464, -7.60555263,
            -3.56881707, -9.77855129, 0.50475873,
            -9.34728378, 0.25373179, -7.55422161,
        );
        let inverse = match inverse_3x3(&matrix) {
            Some(inverse) => inverse,
            None => panic!("Should return Some(inverse_matrix)"),
        };
        assert!((inverse * matrix - identity).norm() < 1e-14);

        assert!(inverse_3x3(&Matrix3::zeros()).is_none());

        #[rustfmt::skip]
        let matrix = Matrix3::new(
            3.0, 1.0, 2.0,
            6.0, 2.0, 4.0,
            9.0, 9.0, 7.0,
        );
        assert!(inverse_3x3(&matrix).is_none());

        #[rustfmt::skip]
        let matrix = Matrix3::new(
            3.00792510e-38, -1.97985750e-45, 3.61627897e-44,
            7.09699991e-49, -3.08764937e-49, -8.31427092e-41,
            2.03723891e-42, -3.84594910e-42, 1.00872600e-40,
        );
        let inverse = match inverse_3x3(&matrix) {
            Some(inverse) => inverse,
            None => panic!("Should return Some(inverse_matrix)"),
        };
        assert!((inverse * matrix - identity).norm() < 1e-14);
    }

    #[test]
    fn test_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

        let src = vec![];
        let dst = vec![];
        assert!(gauss_newton_update(&initial_param, &src, &dst).is_none());

        let src = vec![Measurement::new(-8.89304516, 0.54202289)];
        let dst = vec![transform(&true_param, &src[0])];
        assert!(gauss_newton_update(&initial_param, &src, &dst).is_none());

        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![
            transform(&true_param, &src[0]),
            transform(&true_param, &src[1])
        ];
        assert!(gauss_newton_update(&initial_param, &src, &dst).is_some());

        let src = vec![
            Measurement::new(-8.76116663, 3.50338231),
            Measurement::new(-5.21184804, -1.91561705),
            Measurement::new(6.63141168, 4.8915293),
            Measurement::new(-2.29215281, -4.72658399),
            Measurement::new(6.81352587, -0.81624617),
        ];
        let dst = src
            .iter()
            .map(|p| transform(&true_param, p))
            .collect::<Vec<_>>();

        let update = match gauss_newton_update(&initial_param, &src, &dst) {
            Some(s) => s,
            None => panic!("Return value cannot be None"),
        };
        let updated_param = initial_param + update;

        let e0 = error(&initial_param, &src, &dst);
        let e1 = error(&updated_param, &src, &dst);
        assert!(e1 < e0 * 0.01);
    }

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

    #[test]
    fn test_mad() {
        let a = vec![16., -16., -1., 8., -9., 4., -3., 17., 3., -7., 11., -1.];
        assert_eq!(mad(&a), Some(7.5));

        let a = vec![22., 1., -9., -35., -29., -40., -50., -45., 4.];
        assert_eq!(mad(&a), Some(20.0));

        let a = vec![-53., -36.];
        assert_eq!(mad(&a), Some(8.5));
    }

    #[test]
    fn test_weighted_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

        let src = vec![];
        let dst = vec![];
        assert!(weighted_gauss_newton_update(&initial_param, &src, &dst).is_none());

        let src = vec![Measurement::new(-8.89304516, 0.54202289)];
        let dst = vec![transform(&true_param, &src[0])];
        assert!(weighted_gauss_newton_update(&initial_param, &src, &dst).is_none());

        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![
            transform(&true_param, &src[0]),
            transform(&true_param, &src[1])
        ];
        assert!(weighted_gauss_newton_update(&initial_param, &src, &dst).is_some());

        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
            Measurement::new(-5.9267953, 9.62339266),
            Measurement::new(-4.04966218, -4.44595403),
            Measurement::new(-2.8636942, -9.13843999),
            Measurement::new(-6.97749644, -8.90180581),
            Measurement::new(-9.66454985, 6.32282424),
            Measurement::new(7.02264007, -0.88684585),
            Measurement::new(4.1970011, -1.42366424),
            Measurement::new(-1.98903219, -0.96437383),
            Measurement::new(-0.68034875, -0.48699014),
            Measurement::new(1.89645382, 1.861194),
            Measurement::new(7.09550743, 2.18289525),
            Measurement::new(-7.95383118, -5.16650913),
            Measurement::new(-5.40235599, 2.70675665),
            Measurement::new(-5.38909696, -5.48180288),
            Measurement::new(-9.00498232, -5.12191142),
            Measurement::new(-8.54899319, -3.25752055),
            Measurement::new(6.89969814, 3.53276123),
            Measurement::new(5.06875729, -0.2891854),
        ];

        // noise follows the normal distribution of
        // mean 0.0 and standard deviation 0.01
        let noise = [
            Measurement::new(0.0105879, 0.01302535),
            Measurement::new(0.01392508, 0.0083586),
            Measurement::new(0.01113885, -0.00693269),
            Measurement::new(0.01673124, -0.01735564),
            Measurement::new(-0.01219263, 0.00080933),
            Measurement::new(-0.00396817, 0.00111582),
            Measurement::new(-0.00444043, 0.00658505),
            Measurement::new(-0.01576271, -0.00701065),
            Measurement::new(0.00464, -0.0040679),
            Measurement::new(-0.32268585, 0.49653010), // but add much larger noise here
            Measurement::new(0.00269374, -0.00787015),
            Measurement::new(-0.00494243, 0.00350137),
            Measurement::new(0.00343766, -0.00039311),
            Measurement::new(0.00661565, -0.00341112),
            Measurement::new(-0.00936695, -0.00673899),
            Measurement::new(-0.00240039, -0.00314409),
            Measurement::new(-0.01434128, -0.0058539),
            Measurement::new(0.00874225, 0.00295633),
            Measurement::new(0.00736213, -0.00328875),
            Measurement::new(0.00585082, -0.01232619),
        ];

        assert_eq!(src.len(), noise.len());
        let dst = src
            .iter()
            .zip(noise.iter())
            .map(|(p, n)| transform(&true_param, p) + n)
            .collect::<Vec<_>>();
        let update = match weighted_gauss_newton_update(&initial_param, &src, &dst) {
            Some(u) => u,
            None => panic!("Return value cannot be None"),
        };
        let updated_param = initial_param + update;

        let e0 = error(&initial_param, &src, &dst);
        let e1 = error(&updated_param, &src, &dst);
        assert!(e1 < e0 * 0.1);
    }
}
