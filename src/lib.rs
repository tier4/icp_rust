#![feature(stmt_expr_attributes)]
#![feature(test)]
extern crate test;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use nalgebra::Cholesky;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use std::time::Instant;

mod median;

pub type Param = nalgebra::Vector3<f64>;
pub type Transform = nalgebra::Matrix3<f64>;
pub type Rotation = Matrix2<f64>;
pub type Translation = Vector2<f64>;
pub type Measurement = nalgebra::Vector2<f64>;
type Jacobian = nalgebra::Matrix2x3<f64>;
type Hessian = nalgebra::Matrix3<f64>;

const HUBER_K: f64 = 1.345;

pub fn get_rt(transform: &Transform) -> (Rotation, Translation) {
    #[rustfmt::skip]
    let rot = Rotation::new(
        transform[(0, 0)], transform[(0, 1)],
        transform[(1, 0)], transform[(1, 1)],
    );
    let t = Translation::new(transform[(0, 2)], transform[(1, 2)]);
    (rot, t)
}

pub fn calc_rt(param: &Param) -> (Rotation, Translation) {
    let theta = param[2];
    let cos = f64::cos(theta);
    let sin = f64::sin(theta);

    #[rustfmt::skip]
    let rot = Matrix2::new(
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
    (rot, t)
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

pub fn exp_se2(param: &Param) -> Transform {
    let (rot, t) = calc_rt(param);

    #[rustfmt::skip]
    Matrix3::new(
        rot[(0, 0)], rot[(0, 1)], t[0],
        rot[(1, 0)], rot[(1, 1)], t[1],
        0., 0., 1.,
    )
}

fn make_kdtree(landmarks: &Vec<Measurement>) -> KdTree<f64, usize, [f64; 2]> {
    let mut kdtree = KdTree::new(2);
    for i in 0..landmarks.len() {
        let array: [f64; 2] = landmarks[i].into();
        kdtree.add(array, i).unwrap();
    }
    kdtree
}

fn associate(kdtree: &KdTree<f64, usize, [f64; 2]>, src: &Vec<Measurement>) -> Vec<(usize, usize)> {
    let mut correspondence = vec![];
    for (query_index, query) in src.iter().enumerate() {
        let (_distance, nearest_index) = match kdtree.nearest(query.into(), 1, &squared_euclidean) {
            Ok(p) => p[0],
            Err(e) => {
                eprintln!("Error: {:?}", e);
                continue;
            }
        };
        correspondence.push((query_index, *nearest_index));
    }
    correspondence
}

pub fn transform(param: &Param, landmark: &Measurement) -> Measurement {
    let (rot, t) = calc_rt(param);
    rot * landmark + t
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

pub fn huber_error(param: &Param, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> f64 {
    src.iter().zip(dst.iter()).fold(0f64, |sum, (s, d)| {
        let r = residual(param, s, d);
        sum + rho(r.dot(&r), HUBER_K)
    })
}

pub fn estimate_transform(
    initial_param: &Param,
    src: &Vec<Measurement>,
    dst: &Vec<Measurement>,
) -> Param {
    let delta_norm_threshold: f64 = 1e-6;
    let max_iter: usize = 200;

    let mut prev_error: f64 = f64::MAX;

    let mut param = *initial_param;
    for _ in 0..max_iter {
        let delta = match weighted_gauss_newton_update(&param, &src, &dst) {
            Some(d) => d,
            None => break,
        };

        if delta.norm_squared() < delta_norm_threshold {
            break;
        }

        let error = huber_error(&param, src, dst);
        if error > prev_error {
            break;
        }
        prev_error = error;

        // TODO better to compute T <- T * exp_se2(delta), T in SE(2)
        param = param + delta;
    }
    param
}

fn log_so2(rotation: Matrix2<f64>) -> f64 {
    f64::atan2(rotation[(1, 0)], rotation[(0, 0)])
}

fn log_se2(transform: &Transform) -> Param {
    let (rot, t) = get_rt(transform);
    let theta = log_so2(rot);
    let v_inv = if theta == 0. {
        Matrix2::identity()
    } else if theta == std::f64::consts::PI {
        #[rustfmt::skip]
        Matrix2::new(
            0., 0.5 * theta,
            -0.5 * theta, 0.
        )
    } else {
        let k = f64::sin(theta) / (1. - f64::cos(theta));

        #[rustfmt::skip]
        let m = Matrix2::new(
            k, 1.,
            -1., k
        );
        0.5 * theta * m
    };
    let u = v_inv * t;
    Param::new(u[0], u[1], theta)
}

fn get_corresponding_points(
    correspondence: &Vec<(usize, usize)>,
    src: &Vec<Measurement>,
    dst: &Vec<Measurement>,
) -> (Vec<Measurement>, Vec<Measurement>) {
    let src_points = correspondence
        .iter()
        .map(|(src_index, _)| src[*src_index])
        .collect::<Vec<_>>();
    let dst_points = correspondence
        .iter()
        .map(|(_, dst_index)| dst[*dst_index])
        .collect::<Vec<_>>();
    (src_points, dst_points)
}

pub fn icp(initial_param: &Param, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> Param {
    let kdtree = make_kdtree(dst);
    let max_iter: usize = 500;

    let mut param: Param = *initial_param;
    for _ in 0..max_iter {
        let src_tranformed = src
            .iter()
            .map(|sp| transform(&param, &sp))
            .collect::<Vec<Measurement>>();

        let correspondence = associate(&kdtree, &src_tranformed);

        let (sp, dp) = get_corresponding_points(&correspondence, &src_tranformed, dst);

        let dparam = estimate_transform(&Param::zeros(), &sp, &dp);

        param = dparam + param;
    }
    param
}

pub fn inverse_3x3(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
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
    let (rot, _t) = calc_rt(param);
    let b = rot * a;
    #[rustfmt::skip]
    Jacobian::new(
        rot[(0, 0)], rot[(0, 1)], b[0],
        rot[(1, 0)], rot[(1, 1)], b[1])
}

fn check_input_size(input: &Vec<Measurement>) -> bool {
    // Check if the input does not have sufficient samples to estimate the update
    input.len() > 0 && input.len() >= input[0].len()
}

pub fn gauss_newton_update(
    param: &Param,
    src: &Vec<Measurement>,
    dst: &Vec<Measurement>,
) -> Option<Param> {
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
    // TODO Check matrix rank before solving linear equation
    let update = Cholesky::new_unchecked(jtj).solve(&jtr);
    Some(-update)
}

fn calc_stddevs(residuals: &Vec<Measurement>) -> Option<Vec<f64>> {
    debug_assert!(residuals.len() > 0);
    let dimension = residuals[0].nrows();
    let mut stddevs = vec![0f64; dimension];
    for j in 0..dimension {
        let mut jth_dim = residuals.iter().map(|r| r[j]).collect::<Vec<_>>();
        let stddev = median::mutable_standard_deviation(&mut jth_dim);

        stddevs[j] = match stddev {
            Some(s) => s,
            None => return None,
        };
    }
    Some(stddevs)
}

pub fn weighted_gauss_newton_update(
    param: &Param,
    src: &Vec<Measurement>,
    dst: &Vec<Measurement>,
) -> Option<Param> {
    debug_assert_eq!(src.len(), dst.len());

    if !check_input_size(&src) {
        // The input does not have sufficient samples to estimate the update
        return None;
    }

    // let t1 = Instant::now();

    let residuals = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| residual(param, s, d))
        .collect::<Vec<_>>();

    // println!("t1 = {:.4?}", t1.elapsed());
    // let t2 = Instant::now();

    let stddevs = match calc_stddevs(&residuals) {
        Some(m) => m,
        None => return None,
    };

    // println!("t2 = {:.4?}", t2.elapsed());
    // let t3 = Instant::now();

    let mut jtr = Param::zeros();
    let mut jtj = Hessian::zeros();
    for (s, r) in src.iter().zip(residuals.iter()) {
        let jacobian_i = jacobian(param, s);
        for (j, jacobian_ij) in jacobian_i.row_iter().enumerate() {
            if stddevs[j] == 0. {
                continue;
            }
            let g = 1. / stddevs[j];
            let r_ij = r[j];
            let w_ij = drho(r_ij * r_ij, HUBER_K);

            jtr += w_ij * g * jacobian_ij.transpose() * r_ij;
            jtj += w_ij * g * jacobian_ij.transpose() * jacobian_ij;
        }
    }

    // println!("t3 = {:.4?}", t3.elapsed());
    // let t4 = Instant::now();

    match inverse_3x3(&jtj) {
        Some(jtj_inv) => {
            // println!("t4 = {:.4?}", t4.elapsed());
            // println!("\n");
            return Some(-jtj_inv * jtr);
        }
        None => return None,
    }
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

    #[test]
    fn test_log_se2() {
        // >>> def skew_se2(v):
        // ...     return np.array([
        // ...         [0, -v[2], v[0]],
        // ...         [v[2], 0, v[1]],
        // ...         [0, 0, 0]])
        // ...
        // >>> a = np.random.uniform(-3, 3, 3)
        // >>> a
        // array([ 2.89271776,  0.34275002, -1.6427056 ])
        // >>> expm(skew_se2(a))
        // array([[-7.18473159e-02,  9.97415642e-01,  1.98003686e+00],
        //        [-9.97415642e-01, -7.18473159e-02, -1.67935601e+00],
        //        [ 0.00000000e+00,  1.11022302e-16,  1.00000000e+00]])

        #[rustfmt::skip]
        let transform = Matrix3::new(
            -7.18473159e-02,  9.97415642e-01,  1.98003686e+00,
            -9.97415642e-01, -7.18473159e-02, -1.67935601e+00,
             0.00000000e+00,  1.11022302e-16,  1.00000000e+00
        );
        let expected = Param::new(2.89271776, 0.34275002, -1.6427056);
        let param = log_se2(&transform);
        assert!((param - expected).norm() < 1e-6);

        // >>> a = np.array([-1., 3., np.pi])
        // >>> expm(skew_se2(a))
        // array([[-1.00000000e+00, -1.52695104e-16, -1.90985932e+00],
        //        [ 1.52695104e-16, -1.00000000e+00, -6.36619772e-01],
        //        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        #[rustfmt::skip]
        let transform = Matrix3::new(
            -1.00000000e+00, 0.00000000e+00, -1.90985932e+00,
            0.00000000e+00, -1.00000000e+00, -6.36619772e-01,
            0.00000000e+00,  0.00000000e+00,  1.00000000e+00
        );
        let expected = Param::new(-1., 3., std::f64::consts::PI);
        let param = log_se2(&transform);
        assert!((param - expected).norm() < 1e-6);

        // >>> a = np.array([-1., 3., 0.])
        // >>> expm(skew_se2(a))
        // array([[ 1.,  0., -1.],
        //        [ 0.,  1.,  3.],
        //        [ 0.,  0.,  1.]])
        #[rustfmt::skip]
        let transform = Matrix3::new(
            1.,  0., -1.,
            0.,  1.,  3.,
            0.,  0.,  1.
        );
        let expected = Param::new(-1., 3., 0.);
        let param = log_se2(&transform);
        assert!((param - expected).norm() < 1e-6);
    }

    #[test]
    fn test_get_rt() {
        #[rustfmt::skip]
        let transform = Transform::new(
            0.6225093, 0.7826124, -0.32440305,
            -0.7826124, 0.6225093, -0.01307704,
            0., 0., 1.,
        );
        let (rot, t) = get_rt(&transform);

        #[rustfmt::skip]
        let expected_rot = Rotation::new(
            0.6225093, 0.7826124,
            -0.7826124, 0.6225093,
        );
        let expected_t = Translation::new(-0.32440305, -0.01307704);

        assert_eq!(rot, expected_rot);
        assert_eq!(t, expected_t);
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
    fn test_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);

        let src = vec![];
        let dst = vec![];
        assert!(gauss_newton_update(&param, &src, &dst).is_none());

        let src = vec![Measurement::new(-8.89304516, 0.54202289)];
        let dst = vec![transform(&param, &src[0])];
        assert!(gauss_newton_update(&param, &src, &dst).is_none());

        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform(&param, &src[0]), transform(&param, &src[1])];
        assert!(gauss_newton_update(&param, &src, &dst).is_some());
    }

    #[test]
    fn test_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

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
    fn test_weighted_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);

        // insufficient input size
        let src = vec![];
        let dst = vec![];
        assert!(weighted_gauss_newton_update(&param, &src, &dst).is_none());

        // insufficient input size
        let src = vec![Measurement::new(-8.89304516, 0.54202289)];
        let dst = vec![transform(&param, &src[0])];
        assert!(weighted_gauss_newton_update(&param, &src, &dst).is_none());

        // insufficient input size
        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform(&param, &src[0]), transform(&param, &src[1])];
        assert!(weighted_gauss_newton_update(&param, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![
            transform(&param, &src[0]),
            transform(&param, &src[1]),
            transform(&param, &src[2]),
        ];
        assert!(weighted_gauss_newton_update(&param, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
            Measurement::new(4.40356349, -9.43358563),
        ];
        let dst = vec![
            transform(&param, &src[0]),
            transform(&param, &src[1]),
            transform(&param, &src[2]),
        ];
        assert!(weighted_gauss_newton_update(&param, &src, &dst).is_none());
    }

    #[test]
    fn test_weighted_gauss_newton_update_zero_x_diff() {
        let src = vec![
            Measurement::new(0.0, 0.0),
            Measurement::new(0.0, 0.1),
            Measurement::new(0.0, 0.2),
            Measurement::new(0.0, 0.3),
            Measurement::new(0.0, 0.4),
            Measurement::new(0.0, 0.5),
        ];

        let param_true = Param::new(0.00, 0.00, 0.00);

        let dst = src
            .iter()
            .map(|p| transform(&param_true, p))
            .collect::<Vec<Measurement>>();

        assert!(weighted_gauss_newton_update(&param_true, &src, &dst).is_some());
    }

    #[test]
    fn test_weighted_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

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
            // Measurement::new(-1.98903219, -0.96437383),  // corresponing to the large noise
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
            // Measurement::new(-0.32268585, 0.49653010), // but add much larger noise here
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

        let updated_param = estimate_transform(&initial_param, &src, &dst);

        let e0 = error(&initial_param, &src, &dst);
        let e1 = error(&updated_param, &src, &dst);
        assert!(e1 < e0 * 0.001);
    }

    #[test]
    fn test_calc_stddevs() {
        #[rustfmt::skip]
        let measurements = vec![
            Measurement::new(53.72201757, 52.99126564),
            Measurement::new(47.10884813, 53.59975516),
            Measurement::new(39.39661665, 61.08762518),
            Measurement::new(62.81692917, 54.56765183),
            Measurement::new(39.26208329, 45.65102341),
            Measurement::new(50.86473295, 44.72763481),
            Measurement::new(39.28791948, 34.88506328),
            Measurement::new(55.25576933, 39.59323902),
            Measurement::new(36.75721579, 57.17795218),
            Measurement::new(30.13909168, 64.76416708),
            Measurement::new(44.81493956, 54.94041174),
            Measurement::new(53.88324537, 60.4374775 ),
            Measurement::new(47.88396982, 66.59441293),
            Measurement::new(64.42865488, 40.9932948 ),
            Measurement::new(44.81265264, 50.45413795),
            Measurement::new(53.19558104, 28.24225202),
            Measurement::new(55.95984582, 65.33672375),
            Measurement::new(59.05920996, 27.61279324),
            Measurement::new(46.8073715 , 30.79477285),
            Measurement::new(39.59866249, 45.6226116 ),
            Measurement::new(49.15739909, 55.53557656),
            Measurement::new(43.24838042, 43.95231977),
            Measurement::new(54.78299967, 40.5593425 ),
            Measurement::new(41.9153867 , 55.54639181),
            Measurement::new(52.18015184, 46.38912455),
            Measurement::new(29.59992903, 46.32180761),
            Measurement::new(75.51275641, 57.73265648),
            Measurement::new(61.78180837, 54.48655747),
            Measurement::new(72.17828583, 66.37805296),
            Measurement::new(41.72995451, 50.9864875 )
        ];
        let stddevs = match calc_stddevs(&measurements) {
            Some(stddevs) => stddevs,
            None => panic!(),
        };

        // compare to stddevs calced by numpy
        assert!((stddevs[0] - 10.88547151).abs() < 1.0);
        assert!((stddevs[1] - 10.75361579).abs() < 1.0);
    }

    #[test]
    fn test_association() {
        let src = vec![
            Measurement::new(-8.30289767, 8.47750876),
            Measurement::new(-6.45751825, -1.34801312),
            Measurement::new(-8.66777369, -9.77914636),
            Measurement::new(-8.36130159, -2.39500161),
            Measurement::new(-9.64529718, -7.23686057),
        ];

        let dst = vec![src[3], src[2], src[0], src[1], src[4]];

        let kdtree = make_kdtree(&dst);
        let correspondence = associate(&kdtree, &src);

        assert_eq!(src.len(), correspondence.len());

        let (sp, dp) = get_corresponding_points(&correspondence, &src, &dst);

        for (s, d) in sp.iter().zip(dp.iter()) {
            assert_eq!(s, d);
        }
    }

    #[test]
    fn test_icp() {
        let src = vec![
            Measurement::new(0.0, 0.0),
            Measurement::new(0.0, 0.1),
            Measurement::new(0.0, 0.2),
            Measurement::new(0.0, 0.3),
            Measurement::new(0.0, 0.4),
            Measurement::new(0.0, 0.5),
            Measurement::new(0.0, 0.6),
            Measurement::new(0.0, 0.7),
            Measurement::new(0.0, 0.8),
            Measurement::new(0.0, 0.9),
            Measurement::new(0.0, 1.0),
            Measurement::new(0.1, 0.0),
            Measurement::new(0.2, 0.0),
            Measurement::new(0.3, 0.0),
            Measurement::new(0.4, 0.0),
            Measurement::new(0.5, 0.0),
            Measurement::new(0.6, 0.0),
            Measurement::new(0.7, 0.0),
            Measurement::new(0.8, 0.0),
            Measurement::new(0.9, 0.0),
            Measurement::new(1.0, 0.0),
        ];

        let param_true = Param::new(0.01, 0.01, -0.02);

        let dst = src
            .iter()
            .map(|p| transform(&param_true, p))
            .collect::<Vec<Measurement>>();

        let diff = Param::new(0.05, 0.010, 0.010);
        let initial_param = param_true + diff;
        let param_pred = icp(&initial_param, &src, &dst);

        assert!((param_pred - param_true).norm() < 1e-3);
    }

    use test::Bencher;

    #[bench]
    fn bench_gauss_newton_update(b: &mut Bencher) {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

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
            // Measurement::new(-1.98903219, -0.96437383),  // corresponing to the large noise
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
            // Measurement::new(-0.32268585, 0.49653010), // but add much larger noise here
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

        b.iter(|| weighted_gauss_newton_update(&initial_param, &src, &dst));
    }

    #[bench]
    fn bench_icp(b: &mut Bencher) {
        let src = vec![
            Measurement::new(0.0, 0.0),
            Measurement::new(0.0, 0.1),
            Measurement::new(0.0, 0.2),
            Measurement::new(0.0, 0.3),
            Measurement::new(0.0, 0.4),
            Measurement::new(0.0, 0.5),
            Measurement::new(0.0, 0.6),
            Measurement::new(0.0, 0.7),
            Measurement::new(0.0, 0.8),
            Measurement::new(0.0, 0.9),
            Measurement::new(0.0, 1.0),
            Measurement::new(0.1, 0.0),
            Measurement::new(0.2, 0.0),
            Measurement::new(0.3, 0.0),
            Measurement::new(0.4, 0.0),
            Measurement::new(0.5, 0.0),
            Measurement::new(0.6, 0.0),
            Measurement::new(0.7, 0.0),
            Measurement::new(0.8, 0.0),
            Measurement::new(0.9, 0.0),
            Measurement::new(1.0, 0.0),
        ];

        let param_true = Param::new(0.01, 0.01, -0.02);

        let dst = src
            .iter()
            .map(|p| transform(&param_true, p))
            .collect::<Vec<Measurement>>();

        let diff = Param::new(0.000, 0.010, 0.010);
        let initial_param = param_true + diff;

        b.iter(|| icp(&initial_param, &src, &dst));
    }
}
