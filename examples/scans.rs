use std::fs::File;
use std::path::Path;
use std::io::BufRead;
use nalgebra::{Vector2, Vector3, Matrix1x2};
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{ChartBuilder, Circle, LineSeries, GREEN, YELLOW, WHITE, BLUE, RED, BLACK, MAGENTA, RGBColor};
use plotters::style::Color;
use plotters_piston::{draw_piston_window, PistonBackend};

use icp;

type Measurement = icp::Measurement;

fn make_kdtree(landmarks: &Vec<Measurement>) -> KdTree<f64, usize, [f64; 2]> {
    let mut kdtree = KdTree::new(2);
    for i in 0..landmarks.len() {
        let array: [f64; 2] = landmarks[i].into();
        kdtree.add(array, i).unwrap();
    }
    kdtree
}

fn associate(src: &Vec<Measurement>, dst: &Vec<Measurement>) -> Vec<(usize, usize)> {
    let kdtree = make_kdtree(dst);

    let mut correspondence = vec![];
    for (query_index, query) in src.iter().enumerate() {
        let (distance, nearest_index) = match kdtree.nearest(query.into(), 1, &squared_euclidean) {
            Ok(p) => p[0],
            Err(e) => { eprintln!("Error: {:?}", e); continue; }
        };
        correspondence.push((query_index, *nearest_index));
        // let nearest = dst[*index];
    }
    correspondence
}

fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(std::io::BufReader::new(file).lines())
}

fn load_scan(lines: std::io::Lines<std::io::BufReader<File>>) -> Vec<Measurement> {
    let mut scan_landmarks = vec![];
    for line in lines {
        let s = match line {
            Ok(s) => s,
            Err(e) => { eprintln!("{:?}", e); continue; }
        };
        let xy = s.split(" ").collect::<Vec<_>>();
        let x = xy[0].parse().unwrap();
        let y = xy[1].parse().unwrap();
        scan_landmarks.push(Vector2::new(x, y));
    }
    return scan_landmarks;
}

fn calc_update(src_points: &Vec<Measurement>, dst_points: &Vec<Measurement>) -> icp::Param {
    match icp::weighted_gauss_newton_update(&icp::Param::zeros(), &src_points, &dst_points) {
        Some(update) => return update,
        None => return icp::Param::zeros(),
    };
}

fn estimate_transform(
        src: &Vec<Measurement>, dst: &Vec<Measurement>, correspondence: &Vec<(usize, usize)>) -> icp::Transform {
    let mut transform = icp::Transform::identity();
    let src_points = correspondence.iter().map(|(src_index, _)| src[*src_index]).collect::<Vec<_>>();
    let dst_points = correspondence.iter().map(|(_, dst_index)| dst[*dst_index]).collect::<Vec<_>>();
    for _ in 0..5 {
        let dparam = calc_update(&src_points, &dst_points);
        transform = transform * icp::exp_se2(&dparam);
    }
    transform
}

fn to_point(p: &Measurement, color: &RGBColor) -> Circle<(f64, f64), u32> {
    Circle::new((p[0], p[1]), 2, color.mix(0.7).filled())
}

const WINDOW_RANGE: f64 = 4000.;
const FPS: u64 = 60;
fn main() {
    let mut window: PistonWindow = WindowSettings::new("LiDAR scan", [800, 800])
       .build()
       .unwrap();
    window.set_max_fps(FPS);

    let mut src = vec![];
    let mut index = 0;

    let mut draw = |b: PistonBackend| -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!("scan/{}.txt", index);
        index += 1;
        let lines = match read_lines(filename) {
            Ok(lines) => lines,
            Err(e) => { println!("{:?}", e); return Ok(()); },
        };
        let dst = load_scan(lines);
        let root = b.into_drawing_area();
        root.fill(&WHITE).unwrap();

        if src.len() == 0 {
            src = dst;
            return Ok(());
        }

        let correspondence = associate(&src, &dst);

        let mut cc = ChartBuilder::on(&root)
            .build_cartesian_2d(-WINDOW_RANGE..WINDOW_RANGE, -WINDOW_RANGE..WINDOW_RANGE).unwrap();
        // cc.draw_series(src.iter().map(|p| { to_point(&p, &YELLOW) })).unwrap();
        // cc.draw_series(dst.iter().map(|p| { to_point(&p, &GREEN) })).unwrap();

        let transform = estimate_transform(&src, &dst, &correspondence);
        cc.draw_series(dst.iter().map(|p| { to_point(&p, &RED) })).unwrap();
        cc.draw_series(src.iter().map(|p| { to_point(&p, &BLUE) })).unwrap();
        cc.draw_series(src.iter().map(|p| {
            let dp = transform * Vector3::new(p[0], p[1], 1.);
            to_point(&Vector2::new(dp[0], dp[1]), &GREEN)
        })).unwrap();

        src = dst;
        Ok(())
    };

    while let Some(_) = draw_piston_window(&mut window, &mut draw) {}
}
