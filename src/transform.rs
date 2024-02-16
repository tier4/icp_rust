use crate::geometry::{Rotation, Translation};
use crate::se2;
use crate::types::Vector;

pub struct Transform {
    pub rot: Rotation<2>,
    pub t: Translation<2>,
    pub param: Vector<3>,
}

impl Transform {
    pub fn new(param: &Vector<3>) -> Self {
        let (rot, t) = se2::calc_rt(param);
        Transform {
            rot,
            t,
            param: *param,
        }
    }

    pub fn transform(&self, landmark: &Vector<2>) -> Vector<2> {
        self.rot * landmark + self.t
    }
}
