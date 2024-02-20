use nalgebra::base::dimension::Const;
use nalgebra::{ArrayStorage, Matrix, U1};

pub type Vector<const D: usize> = Matrix<f64, Const<D>, U1, ArrayStorage<f64, D, 1>>;
