#![cfg_attr(not(feature = "std"), no_std)]
#![feature(stmt_expr_attributes)]

//!
//! The detailed jacobian derivation process is at [`doc::jacobian`].
#[macro_use]
extern crate alloc;

use alloc::vec::Vec;

pub mod doc;
pub mod se2;
pub mod so2;
pub mod transform;

mod huber;
mod kdtree;
mod linalg;
mod norm;
mod stats;
mod types;

pub use crate::norm::norm;
pub use crate::transform::Transform;
pub use crate::types::{Rotation2, Vector2, Vector3};

pub type Param = nalgebra::Vector3<f64>;
type Jacobian = nalgebra::Matrix2x3<f64>;
type Hessian = nalgebra::Matrix3<f64>;

const HUBER_K: f64 = 1.345;

pub fn residual(transform: &Transform, src: &Vector2, dst: &Vector2) -> Vector2 {
    transform.transform(src) - dst
}

pub fn error(transform: &Transform, src: &Vec<Vector2>, dst: &Vec<Vector2>) -> f64 {
    src.iter().zip(dst.iter()).fold(0f64, |sum, (s, d)| {
        let r = residual(transform, s, d);
        sum + r.dot(&r)
    })
}

pub fn huber_error(transform: &Transform, src: &Vec<Vector2>, dst: &Vec<Vector2>) -> f64 {
    src.iter().zip(dst.iter()).fold(0f64, |sum, (s, d)| {
        let r = residual(transform, s, d);
        sum + huber::rho(r.dot(&r), HUBER_K)
    })
}

fn transform_xy(transform: &Transform, sp: &Vector3) -> Vector3 {
    let z = sp[2];
    let sxy = Vector2::new(sp[0], sp[1]);
    let dxy = transform.transform(&sxy);
    Vector3::new(dxy[0], dxy[1], z)
}

pub fn estimate_transform(src: &Vec<Vector2>, dst: &Vec<Vector2>) -> Transform {
    let delta_norm_threshold: f64 = 1e-6;
    let max_iter: usize = 200;

    let mut prev_error: f64 = f64::MAX;

    let mut transform = Transform::identity();
    for _ in 0..max_iter {
        let Some(delta) = weighted_gauss_newton_update(&transform, &src, &dst) else {
            break;
        };

        if delta.dot(&delta) < delta_norm_threshold {
            break;
        }

        let error = huber_error(&transform, src, dst);
        if error > prev_error {
            break;
        }
        prev_error = error;

        transform = Transform::new(&delta) * transform;
    }
    transform
}

fn get_xy(xyz: &Vec<Vector3>) -> Vec<Vector2> {
    let f = |p: &Vector3| Vector2::new(p[0], p[1]);
    xyz.iter().map(f).collect::<Vec<Vector2>>()
}

/// Estimates the transform that converts the `src` points to `dst`.
///
pub fn icp_2dscan(
    initial_transform: &Transform,
    src: &Vec<Vector2>,
    dst: &Vec<Vector2>,
) -> Transform {
    let kdtree = kdtree::KdTree::new(dst);
    let max_iter: usize = 20;

    let mut transform = *initial_transform;
    for _ in 0..max_iter {
        let src_tranformed = src
            .iter()
            .map(|sp| transform.transform(&sp))
            .collect::<Vec<Vector2>>();

        let nearest_dsts = kdtree.nearest_ones(&src_tranformed);
        let dtransform = estimate_transform(&src_tranformed, &nearest_dsts);

        transform = dtransform * transform;
    }
    transform
}

/// Estimates the transform on the xy-plane that converts the `src` points to `dst`.
/// This function assumes that the vehicle, LiDAR or other point cloud scanner is moving on the xy-plane.
pub fn icp_3dscan(
    initial_transform: &Transform,
    src: &Vec<Vector3>,
    dst: &Vec<Vector3>,
) -> Transform {
    let kdtree = kdtree::KdTree::new(dst);
    let max_iter: usize = 20;

    let mut transform = *initial_transform;
    for _ in 0..max_iter {
        let src_tranformed = src
            .iter()
            .map(|sp| transform_xy(&transform, &sp))
            .collect::<Vec<Vector3>>();

        let nearest_dsts = kdtree.nearest_ones(&src_tranformed);
        let dtransform = estimate_transform(&get_xy(&src_tranformed), &get_xy(&nearest_dsts));

        transform = dtransform * transform;
    }
    transform
}

fn jacobian(rot: &Rotation2, landmark: &Vector2) -> Jacobian {
    let a = Vector2::new(-landmark[1], landmark[0]);
    let r = rot.matrix();
    let b = rot * a;
    #[rustfmt::skip]
    Jacobian::new(
        r[(0, 0)], r[(0, 1)], b[0],
        r[(1, 0)], r[(1, 1)], b[1])
}

fn check_input_size(input: &Vec<Vector2>) -> bool {
    // Check if the input does not have sufficient samples to estimate the update
    input.len() > 0 && input.len() >= input[0].len()
}

pub fn gauss_newton_update(
    transform: &Transform,
    src: &Vec<Vector2>,
    dst: &Vec<Vector2>,
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
    match linalg::inverse3x3(&jtj) {
        Some(jtj_inv) => return Some(-jtj_inv * jtr),
        None => return None,
    }
}

pub fn weighted_gauss_newton_update(
    transform: &Transform,
    src: &Vec<Vector2>,
    dst: &Vec<Vector2>,
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

    let Some(stddevs) = stats::calc_stddevs(&residuals) else {
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
        let param: Param = Vector3::new(-10., 20., 0.01);
        let transform = Transform::new(&param);
        let src = Vector2::new(7f64, 8f64);
        let dst = transform.transform(&src);
        assert_eq!(residual(&transform, &src, &dst), Vector2::zeros());
    }

    #[test]
    fn test_error() {
        let src = vec![
            Vector2::new(-6., 9.),
            Vector2::new(-1., 9.),
            Vector2::new(-4., -4.),
        ];

        let dst = vec![
            Vector2::new(-4., 4.),
            Vector2::new(0., 3.),
            Vector2::new(-3., -8.),
        ];

        let param: Param = Vector3::new(10., 20., 0.01);
        let transform = Transform::new(&param);
        let r0 = residual(&transform, &src[0], &dst[0]);
        let r1 = residual(&transform, &src[1], &dst[1]);
        let r2 = residual(&transform, &src[2], &dst[2]);
        let expected = r0.dot(&r0) + r1.dot(&r1) + r2.dot(&r2);
        assert_eq!(error(&transform, &src, &dst), expected);
    }

    #[test]
    fn test_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);
        let transform = Transform::new(&param);

        let src = vec![];
        let dst = vec![];
        assert!(gauss_newton_update(&transform, &src, &dst).is_none());

        let src = vec![Vector2::new(-8.89304516, 0.54202289)];
        let dst = vec![transform.transform(&src[0])];
        assert!(gauss_newton_update(&transform, &src, &dst).is_none());

        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
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
            Vector2::new(-8.76116663, 3.50338231),
            Vector2::new(-5.21184804, -1.91561705),
            Vector2::new(6.63141168, 4.8915293),
            Vector2::new(-2.29215281, -4.72658399),
            Vector2::new(6.81352587, -0.81624617),
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
        let src = vec![Vector2::new(-8.89304516, 0.54202289)];
        let dst = vec![transform.transform(&src[0])];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // insufficient input size
        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform.transform(&src[0]), transform.transform(&src[1])];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
            Vector2::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![
            transform.transform(&src[0]),
            transform.transform(&src[1]),
            transform.transform(&src[2]),
        ];
        assert!(weighted_gauss_newton_update(&transform, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
            Vector2::new(4.40356349, -9.43358563),
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
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.1),
            Vector2::new(0.0, 0.2),
            Vector2::new(0.0, 0.3),
            Vector2::new(0.0, 0.4),
            Vector2::new(0.0, 0.5),
        ];

        let true_param = Param::new(0.00, 0.01, 0.00);
        let true_transform = Transform::new(&true_param);

        let dst = src
            .iter()
            .map(|p| true_transform.transform(p))
            .collect::<Vec<Vector2>>();

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
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
            Vector2::new(-5.92679530, 9.62339266),
            Vector2::new(-4.04966218, -4.44595403),
            Vector2::new(-2.86369420, -9.13843999),
            Vector2::new(-6.97749644, -8.90180581),
            Vector2::new(-9.66454985, 6.32282424),
            Vector2::new(7.02264007, -0.88684585),
            Vector2::new(4.19700110, -1.42366424),
            // Vector2::new(-1.98903219, -0.96437383),  // corresponds to the large noise
            Vector2::new(-0.68034875, -0.48699014),
            Vector2::new(1.89645382, 1.86119400),
            Vector2::new(7.09550743, 2.18289525),
            Vector2::new(-7.95383118, -5.16650913),
            Vector2::new(-5.40235599, 2.70675665),
            Vector2::new(-5.38909696, -5.48180288),
            Vector2::new(-9.00498232, -5.12191142),
            Vector2::new(-8.54899319, -3.25752055),
            Vector2::new(6.89969814, 3.53276123),
            Vector2::new(5.06875729, -0.28918540),
        ];

        // noise follow the normal distribution with
        // mean 0.0 and standard deviation 0.01
        let noise = [
            Vector2::new(0.01058790, 0.01302535),
            Vector2::new(0.01392508, 0.00835860),
            Vector2::new(0.01113885, -0.00693269),
            Vector2::new(0.01673124, -0.01735564),
            Vector2::new(-0.01219263, 0.00080933),
            Vector2::new(-0.00396817, 0.00111582),
            Vector2::new(-0.00444043, 0.00658505),
            Vector2::new(-0.01576271, -0.00701065),
            Vector2::new(0.00464000, -0.00406790),
            // Vector2::new(-0.32268585,  0.49653010),  // but add much larger noise here
            Vector2::new(0.00269374, -0.00787015),
            Vector2::new(-0.00494243, 0.00350137),
            Vector2::new(0.00343766, -0.00039311),
            Vector2::new(0.00661565, -0.00341112),
            Vector2::new(-0.00936695, -0.00673899),
            Vector2::new(-0.00240039, -0.00314409),
            Vector2::new(-0.01434128, -0.00585390),
            Vector2::new(0.00874225, 0.00295633),
            Vector2::new(0.00736213, -0.00328875),
            Vector2::new(0.00585082, -0.01232619),
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

        let updated_transform = estimate_transform(&src, &dst);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
        assert!(e1 < e0 * 0.001);
    }

    #[test]
    fn test_icp_3dscan() {
        let src = vec![
            Vector3::new(0.0, 0.0, 2.0),
            Vector3::new(0.0, 0.1, 2.0),
            Vector3::new(0.0, 0.2, 2.0),
            Vector3::new(0.0, 0.3, 2.0),
            Vector3::new(0.0, 0.4, 2.0),
            Vector3::new(0.0, 0.5, 2.0),
            Vector3::new(0.0, 0.6, 2.0),
            Vector3::new(0.0, 0.7, 2.0),
            Vector3::new(0.0, 0.8, 2.0),
            Vector3::new(0.0, 0.9, 2.0),
            Vector3::new(0.0, 1.0, 2.0),
            Vector3::new(0.1, 0.0, 1.0),
            Vector3::new(0.2, 0.0, 1.0),
            Vector3::new(0.3, 0.0, 1.0),
            Vector3::new(0.4, 0.0, 1.0),
            Vector3::new(0.5, 0.0, 1.0),
            Vector3::new(0.6, 0.0, 1.0),
            Vector3::new(0.7, 0.0, 1.0),
            Vector3::new(0.8, 0.0, 1.0),
            Vector3::new(0.9, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ];

        let true_transform = Transform::new(&Param::new(0.01, 0.01, -0.02));

        let dst = src
            .iter()
            .map(|p| transform_xy(&true_transform, &p))
            .collect::<Vec<Vector3>>();

        let noise = Transform::new(&Param::new(0.05, 0.010, 0.010));
        let initial_transform = noise * true_transform;
        let pred_transform = icp_3dscan(&initial_transform, &src, &dst);

        for (sp, dp_true) in src.iter().zip(dst.iter()) {
            let dp_pred = transform_xy(&pred_transform, &sp);
            assert!(norm(&(dp_pred - dp_true)) < 1e-3);
        }
    }

    #[test]
    fn test_icp_2dscan() {
        let src = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.1),
            Vector2::new(0.0, 0.2),
            Vector2::new(0.0, 0.3),
            Vector2::new(0.0, 0.4),
            Vector2::new(0.0, 0.5),
            Vector2::new(0.0, 0.6),
            Vector2::new(0.0, 0.7),
            Vector2::new(0.0, 0.8),
            Vector2::new(0.0, 0.9),
            Vector2::new(0.0, 1.0),
            Vector2::new(0.1, 0.0),
            Vector2::new(0.2, 0.0),
            Vector2::new(0.3, 0.0),
            Vector2::new(0.4, 0.0),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.6, 0.0),
            Vector2::new(0.7, 0.0),
            Vector2::new(0.8, 0.0),
            Vector2::new(0.9, 0.0),
            Vector2::new(1.0, 0.0),
        ];

        let true_transform = Transform::new(&Param::new(0.01, 0.01, -0.02));

        let dst = src
            .iter()
            .map(|p| true_transform.transform(p))
            .collect::<Vec<Vector2>>();

        let noise = Transform::new(&Param::new(0.05, 0.010, 0.010));
        let initial_transform = noise * true_transform;
        let pred_transform = icp_2dscan(&initial_transform, &src, &dst);

        for (sp, dp_true) in src.iter().zip(dst.iter()) {
            let dp_pred = pred_transform.transform(&sp);
            assert!(norm(&(dp_pred - dp_true)) < 1e-3);
        }
    }
}
