use crate::se2;
use crate::types::{Rotation2, Vector2, Vector3};

pub struct Transform {
    pub rot: Rotation2,
    pub t: Vector2,
}

impl Transform {
    pub fn new(param: &Vector3) -> Self {
        let (rot, t) = se2::calc_rt(param);
        Transform { rot, t }
    }

    pub fn from_rt(rot: &Rotation2, t: &Vector2) -> Self {
        Transform { rot: *rot, t: *t }
    }

    pub fn transform(&self, landmark: &Vector2) -> Vector2 {
        self.rot * landmark + self.t
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rot.inverse();
        Transform {
            rot: inv_rot,
            t: -(inv_rot * self.t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::f64::consts::FRAC_PI_2;

    #[test]
    fn test_transform() {
        let r = Rotation2::new(FRAC_PI_2);
        let t = Vector2::new(3., 6.);
        let transform = Transform::from_rt(&r, &t);

        let x = Vector2::new(4., 2.);
        let expected = Vector2::new(-2. + 3., 4. + 6.);
        assert!((transform.transform(&x) - expected).norm() < 1e-8);
    }
}
