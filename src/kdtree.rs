use alloc::vec::Vec;

use crate::types::Vector;
use kdtree::distance::squared_euclidean;

pub struct KdTree<'a, const D: usize> {
    tree: kdtree::KdTree<f64, usize, [f64; D]>,
    landmarks: &'a [Vector<D>],
}

fn as_array<const D: usize>(v: &Vector<D>) -> [f64; D] {
    v.data.0[0]
}

impl<'a, const D: usize> KdTree<'a, D> {
    pub fn new(landmarks: &'a [Vector<D>]) -> KdTree<'a, D> {
        let mut tree = kdtree::KdTree::new(D);
        for (i, landmark) in landmarks.iter().enumerate() {
            let p: [f64; D] = as_array(landmark);
            tree.add(p, i).unwrap();
        }
        KdTree { tree, landmarks }
    }

    pub fn nearest_one(&self, query: &Vector<D>) -> Vector<D> {
        let p: [f64; D] = as_array(query);
        let items = self.tree.nearest(&p, 1, &squared_euclidean).unwrap();
        let (_distance, index) = items[0];
        self.landmarks[*index]
    }

    pub fn nearest_ones(&self, src: &Vec<Vector<D>>) -> Vec<Vector<D>> {
        let mut dst = vec![];
        for query in src.iter() {
            let neighbor = self.nearest_one(&query);
            dst.push(neighbor);
        }
        dst
    }
}

#[test]
fn test_association() {
    type Measurement = Vector<3>;

    #[rustfmt::skip]
    let src = vec![
        Measurement::new( 5.08169369,  3.68767137,  0.76520543),
        Measurement::new(-2.04122854, -5.16303848, -1.82852499),
        Measurement::new( 9.48919697,  8.51366532,  4.42767643),
        Measurement::new(-9.88950231, -4.30358176,  0.75194542),
        Measurement::new(-4.03923337,  4.24277134, -1.73619704),
        Measurement::new( 1.65107471,  1.60232318,  6.52893714),
    ];

    let dst = vec![src[3], src[2], src[5], src[0], src[1], src[4]];

    let kdtree = KdTree::new(&dst);
    let nearest_dsts = kdtree.nearest_ones(&src);

    assert_eq!(src.len(), nearest_dsts.len());

    for (s, d) in src.iter().zip(nearest_dsts.iter()) {
        assert_eq!(s, d);
    }
}
