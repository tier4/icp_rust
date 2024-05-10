use crate::types::Matrix3;

pub fn inverse3x3(matrix: &Matrix3) -> Option<Matrix3> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norm::norm;

    #[test]
    fn test_inverse3x3() {
        let identity = Matrix3::identity();

        #[rustfmt::skip]
        let matrix = Matrix3::new(
            -3.64867356, 0.11236464, -7.60555263,
            -3.56881707, -9.77855129, 0.50475873,
            -9.34728378, 0.25373179, -7.55422161,
        );
        let inverse = match inverse3x3(&matrix) {
            Some(inverse) => inverse,
            None => panic!("Should return Some(inverse_matrix)"),
        };
        assert!(norm(&(inverse * matrix - identity)) < 1e-14);

        assert!(inverse3x3(&Matrix3::zeros()).is_none());

        #[rustfmt::skip]
        let matrix = Matrix3::new(
            3.0, 1.0, 2.0,
            6.0, 2.0, 4.0,
            9.0, 9.0, 7.0,
        );
        assert!(inverse3x3(&matrix).is_none());

        #[rustfmt::skip]
        let matrix = Matrix3::new(
            3.00792510e-38, -1.97985750e-45, 3.61627897e-44,
            7.09699991e-49, -3.08764937e-49, -8.31427092e-41,
            2.03723891e-42, -3.84594910e-42, 1.00872600e-40,
        );
        let Some(inverse) = inverse3x3(&matrix) else {
            panic!("Should return Some(inverse_matrix)");
        };
        assert!(norm(&(inverse * matrix - identity)) < 1e-14);
    }
}
