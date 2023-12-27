use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use crate::so2;

pub fn get_rt(transform: &Matrix3<f64>) -> (Matrix2<f64>, Vector2<f64>) {
    #[rustfmt::skip]
    let rot = Matrix2::new(
        transform[(0, 0)], transform[(0, 1)],
        transform[(1, 0)], transform[(1, 1)],
    );
    let t = Vector2::new(transform[(0, 2)], transform[(1, 2)]);
    (rot, t)
}

pub fn calc_rt(param: &Vector3<f64>) -> (Matrix2<f64>, Vector2<f64>) {
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

pub fn exp(param: &Vector3<f64>) -> Matrix3<f64> {
    let (rot, t) = calc_rt(param);

    #[rustfmt::skip]
    Matrix3::new(
        rot[(0, 0)], rot[(0, 1)], t[0],
        rot[(1, 0)], rot[(1, 1)], t[1],
        0., 0., 1.,
    )
}

pub fn log(transform: &Matrix3<f64>) -> Vector3<f64> {
    let (rot, t) = get_rt(transform);
    let theta = so2::log(rot);
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
    Vector3::new(u[0], u[1], theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp() {
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

        let transform = exp(&Vector3::new(-0.29638466, -0.15797957, -0.89885138));

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

        let transform = exp(&Vector3::new(-0.24295876, 0.95847196, 0.91052553));

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

        let transform = exp(&Vector3::new(10., -20., 0.));

        #[rustfmt::skip]
        let expected = Matrix3::new(
            1., 0., 10.,
            0., 1., -20.,
            0., 0., 1.,
        );
        assert!((transform - expected).norm() < 1e-6);
    }

    #[test]
    fn test_log() {
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
        let expected = Vector3::new(2.89271776, 0.34275002, -1.6427056);
        let param = log(&transform);
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
        let expected = Vector3::new(-1., 3., std::f64::consts::PI);
        let param = log(&transform);
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
        let expected = Vector3::new(-1., 3., 0.);
        let param = log(&transform);
        assert!((param - expected).norm() < 1e-6);
    }

    #[test]
    fn test_get_rt() {
        #[rustfmt::skip]
        let transform = Matrix3::new(
            0.6225093, 0.7826124, -0.32440305,
            -0.7826124, 0.6225093, -0.01307704,
            0., 0., 1.,
        );
        let (rot, t) = get_rt(&transform);

        #[rustfmt::skip]
        let expected_rot = Matrix2::new(
            0.6225093, 0.7826124,
            -0.7826124, 0.6225093,
        );
        let expected_t = Vector2::new(-0.32440305, -0.01307704);

        assert_eq!(rot, expected_rot);
        assert_eq!(t, expected_t);
    }
}
