use std::fs::File;
use std::path::Path;
use std::io::BufRead;
use nalgebra::Matrix1x2;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;

use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{ChartBuilder, Circle, LineSeries, GREEN, YELLOW, WHITE, BLUE, RED, BLACK, MAGENTA, RGBColor};
use plotters::style::Color;
use plotters_piston::{draw_piston_window, PistonBackend};

fn make_kdtree(landmarks: &Vec<[f64; 2]>) -> KdTree<f64, usize, &[f64; 2]> {
    let mut kdtree = KdTree::new(2);
    for i in 0..landmarks.len() {
        kdtree.add(&landmarks[i], i).unwrap();
    }
    kdtree
}

fn associate(src: &Vec<[f64; 2]>, dst: &Vec<[f64; 2]>) -> Vec<(usize, usize)> {
    let kdtree = make_kdtree(dst);

    let mut correspondence = vec![];
    for query_index in 0..src.len() {
        let query = &src[query_index];
        let (distance, nearest_index) = match kdtree.nearest(query, 1, &squared_euclidean) {
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

fn load_scan(lines: std::io::Lines<std::io::BufReader<File>>) -> Vec<[f64; 2]> {
    let mut scan_landmarks = vec![];
    for line in lines {
        let s = match line {
            Ok(s) => s,
            Err(e) => { eprintln!("{:?}", e); continue; }
        };
        let xy = s.split(" ").collect::<Vec<_>>();
        let x = xy[0].parse().unwrap();
        let y = xy[1].parse().unwrap();
        scan_landmarks.push([x, y]);
    }
    return scan_landmarks;
}

fn to_point(p: &[f64; 2], color: RGBColor) -> Circle<(f64, f64), u32> {
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


    let mut draw = |b: PistonBackend| {
        let filename = format!("scan/{}.txt", index);
        index += 1;
        let lines = read_lines(filename).unwrap();
        let dst = load_scan(lines);
        let root = b.into_drawing_area();
        root.fill(&WHITE).unwrap();

        println!("dst size = {}", dst.len());
        if src.len() == 0 {
            src = dst;
            return Ok(());
        }

        println!("src size = {}", src.len());
        let correspondence = associate(&src, &dst);

        let mut cc = ChartBuilder::on(&root)
            .build_cartesian_2d(-WINDOW_RANGE..WINDOW_RANGE, -WINDOW_RANGE..WINDOW_RANGE).unwrap();
        cc.draw_series(src.iter().map(|p| { to_point(p, YELLOW) })).unwrap();
        cc.draw_series(dst.iter().map(|p| { to_point(p, GREEN) })).unwrap();
        for (src_index, dst_index) in correspondence {
            let sp = src[src_index];
            let dp = dst[dst_index];
            let line = LineSeries::new(vec![(sp[0], sp[1]), (dp[0], dp[1])], RED);
            cc.draw_series(line).unwrap();
        }

        src = dst;
        Ok(())
    };

    while let Some(_) = draw_piston_window(&mut window, &mut draw) {}
}
