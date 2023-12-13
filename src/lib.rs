// use kdtree::distance::squared_equclidean;
use nalgebra::{Vector2, Vector3, Matrix3};

type Param = nalgebra::Vector3<f64>;
type Output = nalgebra::Vector2<f64>;
type Measurement = nalgebra::Vector2<f64>;

fn transform(param: &Param, landmark: &Measurement) -> Vector2<f64> {
    let theta = param[0];
    let tx = param[1];
    let ty = param[2];
    let cos = f64::cos(theta);
    let sin = f64::sin(theta);
    let x = landmark[0];
    let y = landmark[0];
    Vector2::new(
        cos * x - sin * y + tx,
        sin * x + cos * y + ty)
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

fn jacobian() {
}

// fn gauss_newton_update() {
//     let (jtr, jtj) = src.iter().zip(dst.iter()).fold(
//         (Jacobian::zero(), Hessian::Zero()),
//         |jtr, jtj, (s, d)| {
//         let j = jacobian(param, s);
//         let r = transform(param, s) - d;
//         let jtr_ = j.transpose() * r;
//         let jtj_ = j.transpose() * j;
//         (jtr + jtr_, jtj + jtj_)
//     });
//
//     inverse_3x3(jtj)
//     jtr ;
// }

#[cfg(test)]
mod tests {
    use super::*;

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
}
