#![cfg_attr(not(feature = "std"), no_std)]
#![feature(stmt_expr_attributes)]
#![feature(test)]

#[macro_use]
extern crate alloc;
extern crate test;

use alloc::vec::Vec;

use nalgebra::Cholesky;

pub mod se2;
pub mod so2;
pub mod transform;

mod geometry;
mod huber;
mod kdtree;
mod linalg;
mod stats;
mod types;

use crate::geometry::Rotation;
pub use crate::transform::Transform;

pub type Param = nalgebra::Vector3<f64>;
pub type Measurement = nalgebra::Vector2<f64>;
type Jacobian = nalgebra::Matrix2x3<f64>;
type Hessian = nalgebra::Matrix3<f64>;

const HUBER_K: f64 = 1.345;

pub fn residual(transform: &Transform, src: &Measurement, dst: &Measurement) -> Measurement {
    transform.transform(src) - dst
}

pub fn error(transform: &Transform, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> f64 {
    src.iter().zip(dst.iter()).fold(0f64, |sum, (s, d)| {
        let r = residual(transform, s, d);
        sum + r.dot(&r)
    })
}

pub fn huber_error(transform: &Transform, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> f64 {
    src.iter().zip(dst.iter()).fold(0f64, |sum, (s, d)| {
        let r = residual(transform, s, d);
        sum + huber::rho(r.dot(&r), HUBER_K)
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
        let transform = Transform::new(&param);
        let Some(delta) = weighted_gauss_newton_update(&transform, &src, &dst) else {
            break;
        };

        if delta.norm_squared() < delta_norm_threshold {
            break;
        }

        let error = huber_error(&transform, src, dst);
        if error > prev_error {
            break;
        }
        prev_error = error;

        // TODO better to compute T <- T * exp_se2(delta), T in SE(2)
        param = param + delta;
    }
    param
}

pub fn icp(initial_param: &Param, src: &Vec<Measurement>, dst: &Vec<Measurement>) -> Param {
    let kdtree = kdtree::KdTree::new(dst);
    let max_iter: usize = 20;

    let mut param: Param = *initial_param;
    for _ in 0..max_iter {
        let transform = Transform::new(&param);
        let src_tranformed = src
            .iter()
            .map(|sp| transform.transform(&sp))
            .collect::<Vec<Measurement>>();

        let correspondence = kdtree::associate(&kdtree, &src_tranformed);
        let (sp, dp) = kdtree::get_corresponding_points(&correspondence, &src_tranformed, dst);
        let dparam = estimate_transform(&Param::zeros(), &sp, &dp);

        param = dparam + param;
    }
    param
}

fn jacobian(rot: &Rotation<2>, landmark: &Measurement) -> Jacobian {
    let a = nalgebra::Vector2::new(-landmark[1], landmark[0]);
    let rot = rot;
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
    transform: &Transform,
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
            let j = jacobian(&transform.rot, s);
            let r = transform.transform(s) - d;
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
        let Some(s) = stats::mutable_standard_deviation(&mut jth_dim) else {
            return None;
        };
        stddevs[j] = s;
    }
    Some(stddevs)
}

pub fn weighted_gauss_newton_update(
    transform: &Transform,
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
        .map(|(s, d)| residual(transform, s, d))
        .collect::<Vec<_>>();

    let Some(stddevs) = calc_stddevs(&residuals) else {
        return None;
    };

    let mut jtr = Param::zeros();
    let mut jtj = Hessian::zeros();
    for (s, r) in src.iter().zip(residuals.iter()) {
        let jacobian_i = jacobian(&transform.rot, s);
        for (j, jacobian_ij) in jacobian_i.row_iter().enumerate() {
            if stddevs[j] == 0. {
                continue;
            }
            let g = 1. / stddevs[j];
            let r_ij = r[j];
            let w_ij = huber::drho(r_ij * r_ij, HUBER_K);

            jtr += w_ij * g * jacobian_ij.transpose() * r_ij;
            jtj += w_ij * g * jacobian_ij.transpose() * jacobian_ij;
        }
    }

    match linalg::inverse3x3(&jtj) {
        Some(jtj_inv) => return Some(-jtj_inv * jtr),
        None => return None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual() {
        let param: Param = nalgebra::Vector3::new(-10., 20., 0.01);
        let transform = Transform::new(&param);
        let src = nalgebra::Vector2::new(7f64, 8f64);
        let dst = transform.transform(&src);
        assert_eq!(residual(&transform, &src, &dst), nalgebra::Vector2::zeros());
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

        let param: Param = nalgebra::Vector3::new(10., 20., 0.01);
        let transform = Transform::new(&param);
        let r0 = residual(&transform, &src[0], &dst[0]);
        let r1 = residual(&transform, &src[1], &dst[1]);
        let r2 = residual(&transform, &src[2], &dst[2]);
        let expected = r0.norm_squared() + r1.norm_squared() + r2.norm_squared();
        assert_eq!(error(&transform, &src, &dst), expected);
    }

    #[test]
    fn test_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);
        let transform = Transform::new(&param);

        let src = vec![];
        let dst = vec![];
        assert!(gauss_newton_update(&transform, &src, &dst).is_none());

        let src = vec![Measurement::new(-8.89304516, 0.54202289)];
        let dst = vec![transform.transform(&src[0])];
        assert!(gauss_newton_update(&transform, &src, &dst).is_none());

        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform.transform(&src[0]), transform.transform(&src[1])];
        assert!(gauss_newton_update(&transform, &src, &dst).is_some());
    }

    #[test]
    fn test_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;
        let true_transform = Transform::new(&true_param);
        let initial_transform = Transform::new(&initial_param);

        let src = vec![
            Measurement::new(-8.76116663, 3.50338231),
            Measurement::new(-5.21184804, -1.91561705),
            Measurement::new(6.63141168, 4.8915293),
            Measurement::new(-2.29215281, -4.72658399),
            Measurement::new(6.81352587, -0.81624617),
        ];
        let dst = src
            .iter()
            .map(|p| true_transform.transform(&p))
            .collect::<Vec<_>>();

        let Some(update) = gauss_newton_update(&initial_transform, &src, &dst) else {
            panic!("Return value cannot be None");
        };
        let updated_param = initial_param + update;

        let initial_transform = Transform::new(&initial_param);
        let updated_transform = Transform::new(&updated_param);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
        assert!(e1 < e0 * 0.01);
    }

    #[test]
    fn test_weighted_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);
        let transform = Transform::new(&param);

        // insufficient input size
        let src = vec![];
        let dst = vec![];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // insufficient input size
        let src = vec![Measurement::new(-8.89304516, 0.54202289)];
        let dst = vec![transform.transform(&src[0])];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // insufficient input size
        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform.transform(&src[0]), transform.transform(&src[1])];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
            Measurement::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![
            transform.transform(&src[0]),
            transform.transform(&src[1]),
            transform.transform(&src[2]),
        ];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Measurement::new(-8.89304516, 0.54202289),
            Measurement::new(-4.03198385, -2.81807802),
            Measurement::new(4.40356349, -9.43358563),
        ];
        let dst = vec![
            transform.transform(&src[0]),
            transform.transform(&src[1]),
            transform.transform(&src[2]),
        ];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());
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

        let true_param = Param::new(0.00, 0.01, 0.00);
        let true_transform = Transform::new(&true_param);

        let dst = src
            .iter()
            .map(|p| true_transform.transform(p))
            .collect::<Vec<Measurement>>();

        let initial_param = Param::new(0.00, 0.00, 0.00);
        let initial_transform = Transform::new(&initial_param);
        // TODO Actually there is some error, but Hessian is not invertible so
        // the update cannot be calculated
        assert!(weighted_gauss_newton_update(&initial_transform, &src, &dst).is_none());
    }

    #[test]
    fn test_weighted_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

        let true_transform = Transform::new(&true_param);
        let initial_transform = Transform::new(&initial_param);

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
            .map(|(p, n)| true_transform.transform(&p) + n)
            .collect::<Vec<_>>();
        let Some(update) = weighted_gauss_newton_update(&initial_transform, &src, &dst) else {
            panic!("Return value cannot be None");
        };
        let updated_param = initial_param + update;
        let updated_transform = Transform::new(&updated_param);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
        assert!(e1 < e0 * 0.1);

        let updated_param = estimate_transform(&initial_param, &src, &dst);
        let updated_transform = Transform::new(&updated_param);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
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
        let Some(stddevs) = calc_stddevs(&measurements) else {
            panic!();
        };

        // compare to stddevs calced by numpy
        assert!((stddevs[0] - 10.88547151).abs() < 1.0);
        assert!((stddevs[1] - 10.75361579).abs() < 1.0);
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

        let true_param = Param::new(0.01, 0.01, -0.02);
        let true_transform = Transform::new(&true_param);

        let dst = src
            .iter()
            .map(|p| true_transform.transform(&p))
            .collect::<Vec<Measurement>>();

        let diff = Param::new(0.05, 0.010, 0.010);
        let initial_param = true_param + diff;
        let pred_param = icp(&initial_param, &src, &dst);

        assert!((pred_param - true_param).norm() < 1e-3);
    }

    use test::Bencher;

    #[bench]
    fn bench_gauss_newton_update(b: &mut Bencher) {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let true_transform = Transform::new(&true_param);
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
            .map(|(p, n)| true_transform.transform(&p) + n)
            .collect::<Vec<_>>();

        let initial_transform = Transform::new(&initial_param);
        b.iter(|| weighted_gauss_newton_update(&initial_transform, &src, &dst));
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

        let true_param = Param::new(0.01, 0.01, -0.02);
        let true_transform = Transform::new(&true_param);

        let dst = src
            .iter()
            .map(|p| true_transform.transform(&p))
            .collect::<Vec<Measurement>>();

        let diff = Param::new(0.000, 0.010, 0.010);
        let initial_param = true_param + diff;

        b.iter(|| icp(&initial_param, &src, &dst));
    }
}
