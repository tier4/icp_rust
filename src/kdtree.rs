use alloc::vec::Vec;
use kiddo;
use kiddo::float::distance::SquaredEuclidean;
use kiddo::float::kdtree::KdTree as KiddoTree;

use crate::types::Vector;

type Tree<const D: usize> = KiddoTree<f64, usize, D, 128, u32>;

pub struct KdTree<const D: usize> {
    tree: Tree<D>,
}

impl<const D: usize> KdTree<D> {
    pub fn new(landmarks: &Vec<Vector<D>>) -> Self {
        let mut tree: Tree<D> = KiddoTree::with_capacity(landmarks.len());
        for (i, landmark) in landmarks.iter().enumerate() {
            let array: [f64; D] = (*landmark).into();
            tree.add(&array, i);
        }
        KdTree { tree }
    }

    pub fn nearest_one(&self, query: &Vector<D>) -> usize {
        let p: [f64; D] = (*query).into();
        let nearest = self.tree.nearest_one::<SquaredEuclidean>(&p);
        nearest.item
    }
}

pub fn associate<const D: usize>(kdtree: &KdTree<D>, src: &Vec<Vector<D>>) -> Vec<(usize, usize)> {
    let mut correspondence = vec![];
    for (query_index, query) in src.iter().enumerate() {
        let nearest_index = kdtree.nearest_one(&query);
        correspondence.push((query_index, nearest_index));
    }
    correspondence
}

pub fn get_corresponding_points<const D: usize>(
    correspondence: &Vec<(usize, usize)>,
    src: &Vec<Vector<D>>,
    dst: &Vec<Vector<D>>,
) -> (Vec<Vector<D>>, Vec<Vector<D>>) {
    let src_points = correspondence
        .iter()
        .map(|(src_index, _)| src[*src_index])
        .collect::<Vec<_>>();
    let dst_points = correspondence
        .iter()
        .map(|(_, dst_index)| dst[*dst_index])
        .collect::<Vec<_>>();
    (src_points, dst_points)
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
    let correspondence = associate(&kdtree, &src);

    assert_eq!(src.len(), correspondence.len());

    let (sp, dp) = get_corresponding_points(&correspondence, &src, &dst);

    for (s, d) in sp.iter().zip(dp.iter()) {
        assert_eq!(s, d);
    }
}
