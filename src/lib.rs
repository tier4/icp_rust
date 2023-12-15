// use kdtree::distance::squared_equclidean;
use nalgebra::{Vector2, Vector3, Matrix2, Matrix3};

type Param = nalgebra::Vector3<f64>;
type Measurement = nalgebra::Vector2<f64>;
type Jacobian = nalgebra::Matrix2x3<f64>;
type Hessian = nalgebra::Matrix3<f64>;
type Output = nalgebra::Vector2<f64>;

fn calc_rt(param: &Vector3<f64>) -> (Matrix2<f64>, Vector2<f64>) {
    let theta = param[2];
    let cos = f64::cos(theta);
    let sin = f64::sin(theta);

    let R = Matrix2::new(
        cos, -sin,
        sin, cos);

    let vx = param[0];
    let vy = param[1];

    let t = if theta == 0. {
        Vector2::new(vx, vy)
    } else {
        Vector2::new(
            (sin * vx - (1. - cos) * vy) / theta,
            ((1. - cos) * vx + sin * vy) / theta)
    };
    (R, t)
}

fn exp_se2(param: &Vector3<f64>) -> Matrix3<f64> {
    let (R, t) = calc_rt(param);

    Matrix3::new(
        R[(0, 0)], R[(0, 1)], t[0],
        R[(1, 0)], R[(1, 1)], t[1],
        0., 0., 1.)
}

fn transform(param: &Param, landmark: &Measurement) -> Measurement {
    let (R, t) = calc_rt(param);
    R * landmark + t
}

pub fn residual(param: &Param, src: &Measurement, dst: &Measurement) -> Output {
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

    let det = m00 * (m22*m11-m21*m12) - m10 * (m22*m01-m21*m02) + m20 * (m12*m01-m11*m02);
    if det == 0f64 {
        return None
    }
    let mat = Matrix3::new(
        m22*m11-m21*m12, -(m22*m01-m21*m02), m12*m01-m11*m02,
        -(m22*m10-m20*m12), m22*m00-m20*m02, -(m12*m00-m10*m02),
        m21*m10-m20*m11, -(m21*m00-m20*m01), m11*m00-m10*m01);
    Some(mat / det)
}

fn jacobian(param: &Param, landmark: &Measurement) -> Jacobian {
    let a = Vector2::new(-landmark[1], landmark[0]);
    let (R, t) = calc_rt(param);
    let b = R * a;
    Jacobian::new(
        R[(0, 0)], R[(0, 1)], b[0],
        R[(1, 0)], R[(1, 1)], b[1])
}

// fn jacobian(param: &Param, landmark: &Measurement) -> Jacobian {
//     let theta = param[0];
//     let x = landmark[0];
//     let y = landmark[1];
//     let cos = f64::cos(theta);
//     let sin = f64::sin(theta);
//
//     Jacobian::new(
//         -sin * x - cos * y, 1., 0.,
//         cos * x - sin * y, 0., 1.)
// }

fn gauss_newton_update(param: &Param, src: &Vec<Measurement>, dst: &Vec<Measurement>) {
    let (jtr, jtj) = src.iter().zip(dst.iter()).fold(
        (Param::zeros(), Hessian::zeros()),
        |(jtr, jtj), (s, d)| {
        let j = jacobian(param, s);
        let r = transform(param, s) - d;
        let jtr_: Param = j.transpose() * r;
        let jtj_: Hessian = j.transpose() * j;
        (jtr + jtr_, jtj + jtj_)
    });
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
        let expected = Matrix3::new(
             0.6225093,  0.7826124, -0.32440305,
            -0.7826124,  0.6225093, -0.01307704,
             0.       ,  0.       ,  1.        );
        assert!((transform - expected).norm() < 1e-6);

        // >>> a = np.array([-0.24295876,  0.95847196,  0.91052553])
        // >>> expm(skew_se2(a))
        // array([[ 0.61333076, -0.78982617, -0.61778258],
        //        [ 0.78982617,  0.61333076,  0.72824049],
        //        [ 0.        ,  0.        ,  1.        ]])

        let transform = exp_se2(&Vector3::new(-0.24295876, 0.95847196, 0.91052553));
        let expected = Matrix3::new(
            0.61333076, -0.78982617, -0.61778258,
            0.78982617,  0.61333076,  0.72824049,
            0.        ,  0.        ,  1.        );
        assert!((transform - expected).norm() < 1e-6);

        // >>> a = np.array([10., -20., 0.])
        // >>> expm(skew_se2(a))
        // array([[  1.,   0.,  10.],
        //        [  0.,   1., -20.],
        //        [  0.,   0.,   1.]])

        let transform = exp_se2(&Vector3::new(10., -20., 0.));
        let expected = Matrix3::new(
             1., 0., 10.,
             0., 1., -20.,
             0., 0., 1.);
        assert!((transform - expected).norm() < 1e-6);
    }

    #[test]
    fn test_residual() {
        let param: Param = Vector3::new(std::f64::consts::FRAC_PI_2, 10f64, 20f64);
        let src = Vector2::new(7f64, 8f64);
        let dst = transform(&param, &src);
        assert_eq!(residual(&param, &src, &dst), Vector2::zeros());
    }

    #[test]
    fn test_inverse_3x3() {
        let identity = Matrix3::identity();

        let matrix = Matrix3::new(
            -3.64867356,  0.11236464, -7.60555263,
            -3.56881707, -9.77855129,  0.50475873,
            -9.34728378,  0.25373179, -7.55422161);
        let inverse = match inverse_3x3(&matrix) {
            Some(inverse) => inverse,
            None => panic!("Should return Some(inverse_matrix)"),
        };
        assert!((inverse * matrix - identity).norm() < 1e-14);

        assert!(inverse_3x3(&Matrix3::zeros()).is_none());

        let matrix = Matrix3::new(
            3.0, 1.0, 2.0,
            6.0, 2.0, 4.0,
            9.0, 9.0, 7.0,
        );
        assert!(inverse_3x3(&matrix).is_none());

        let matrix = Matrix3::new(
            3.00792510e-38, -1.97985750e-45,  3.61627897e-44,
            7.09699991e-49, -3.08764937e-49, -8.31427092e-41,
            2.03723891e-42, -3.84594910e-42,  1.00872600e-40);
        let inverse = match inverse_3x3(&matrix) {
            Some(inverse) => inverse,
            None => panic!("Should return Some(inverse_matrix)"),
        };
        assert!((inverse * matrix - identity).norm() < 1e-14);
    }

    #[test]
    fn test_jacobian() {
        let src = Measurement::new(100.0, 200.0);
        let dst = Measurement::new(180.0, 250.0);

        let param = Param::new(10.0, 10.0, 3.14);
        let dparam = Param::new(-0.3, -0.5, 0.01);
        let r0 = residual(&param, &src, &dst);
        let r1 = residual(&(param + dparam), &src, &dst);
        let j = jacobian(&param, &src);
        let d = (r1 - r0) - j * dparam;
        assert!(d.norm() < 1e-2);
    }
}
