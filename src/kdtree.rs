use alloc::vec::Vec;

use acap::NearestNeighbors;
use acap::euclid::Euclidean;
use acap::vp::VpTree;

use crate::types::Vector;

pub struct KdTree<const D: usize> {
    tree: VpTree<Euclidean<Vec<f64>>>,
}

impl<const D: usize> KdTree<D> {
    pub fn new(landmarks: &Vec<Vector<D>>) -> Self {
        let mut points = vec![];
        for landmark in landmarks.iter() {
            let array: [f64; D] = (*landmark).into();
            let p = Euclidean(array.to_vec());
            points.push(p);
        }
        KdTree { tree: VpTree::balanced(points) }
    }

    pub fn nearest_one(&self, query: &Vector<D>) -> Vector<D> {
        let array: [f64; D] = (*query).into();
        let p = Euclidean(array.to_vec());
        let nearest = self.tree.nearest(&p).unwrap();
        let item: [f64; D] = nearest.item.inner().clone().try_into().unwrap();
        item.into()
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
