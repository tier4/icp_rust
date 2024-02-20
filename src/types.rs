use nalgebra::base::dimension::Const;
use nalgebra::{ArrayStorage, Matrix, U1};

pub type Vector<const D: usize> = Matrix<f64, Const<D>, U1, ArrayStorage<f64, D, 1>>;
pub type MatrixNxN<const N: usize> = Matrix<f64, Const<N>, Const<N>, ArrayStorage<f64, N, N>>;
pub type Rotation<const D: usize> = nalgebra::Rotation<f64, D>;

pub type Matrix2 = MatrixNxN<2>;
pub type Matrix3 = MatrixNxN<3>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Rotation2 = Rotation<2>;
