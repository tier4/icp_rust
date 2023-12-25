use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use nalgebra::{Vector2, Vector3};
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{
    ChartBuilder, Circle, LineSeries, RGBColor, BLACK, BLUE, GREEN, MAGENTA, RED, WHITE, YELLOW,
};
use plotters::style::Color;
use plotters_piston::{draw_piston_window, PistonBackend};
use std::fs::File;
use std::io::BufRead;
use std::path::Path;

use icp;

fn make_kdtree(landmarks: &Vec<icp::Measurement>) -> KdTree<f64, usize, [f64; 2]> {
    let mut kdtree = KdTree::new(2);
    for i in 0..landmarks.len() {
        let array: [f64; 2] = landmarks[i].into();
        kdtree.add(array, i).unwrap();
    }
    kdtree
}

fn associate(src: &Vec<icp::Measurement>, dst: &Vec<icp::Measurement>) -> Vec<(usize, usize)> {
    // TODO not necessary to make kdtree for each iteration in ICP
    let kdtree = make_kdtree(dst);

    let mut correspondence = vec![];
    for (query_index, query) in src.iter().enumerate() {
        let (_distance, nearest_index) = match kdtree.nearest(query.into(), 1, &squared_euclidean) {
            Ok(p) => p[0],
            Err(e) => {
                eprintln!("Error: {:?}", e);
                continue;
            }
        };
        correspondence.push((query_index, *nearest_index));
        // let nearest = dst[*index];
    }
    correspondence
}

fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(std::io::BufReader::new(file).lines())
}

fn load_scan(lines: std::io::Lines<std::io::BufReader<File>>) -> Vec<icp::Measurement> {
    let mut scan_landmarks = vec![];
    for line in lines {
        let s = match line {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{:?}", e);
                continue;
            }
        };
        let xy = s.split(" ").collect::<Vec<_>>();
        let x = xy[0].parse().unwrap();
        let y = xy[1].parse().unwrap();
        scan_landmarks.push(Vector2::new(x, y));
    }
    return scan_landmarks;
}

fn estimate_transform(
    src: &Vec<icp::Measurement>,
    dst: &Vec<icp::Measurement>,
    correspondence: &Vec<(usize, usize)>,
    maybe_initial_param: &Option<icp::Param>,
) -> icp::Param {
    let initial_param = match maybe_initial_param {
        Some(param) => *param,
        None => icp::Param::zeros(),
    };
    let src_points = correspondence
        .iter()
        .map(|(src_index, _)| src[*src_index])
        .collect::<Vec<_>>();
    let dst_points = correspondence
        .iter()
        .map(|(_, dst_index)| dst[*dst_index])
        .collect::<Vec<_>>();
    icp::estimate_transform(&initial_param, &src_points, &dst_points)
}

fn to_point(p: &icp::Measurement, color: &RGBColor) -> Circle<(f64, f64), u32> {
    Circle::new((p[0], p[1]), 2, color.mix(0.7).filled())
}

fn axis_lines(
    transform: &icp::Transform,
    length: f64,
) -> (Vector2<f64>, Vector2<f64>, Vector2<f64>) {
    // Vec<LineSeries<PistonBackend, (f64, f64)>> {
    let x = Vector2::new(length, 0.);
    let y = Vector2::new(0., length);

    let (rot, t) = icp::get_rt(transform);

    let xp = rot * x + t;
    let yp = rot * y + t;
    (t, xp, yp)
    // vec![LineSeries::new(vec![(t[0], t[1]), (xp[0], xp[1])], RED),
    //      LineSeries::new(vec![(t[0], t[1]), (yp[0], yp[1])], GREEN)]
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

    let mut param = icp::Param::zeros();

    let mut path: Vec<Vector2<f64>> = vec![];
    let mut draw = |b: PistonBackend| -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!("scan/{}.txt", index);
        index += 1;

        let lines = match read_lines(filename) {
            Ok(lines) => lines,
            Err(e) => {
                println!("{:?}", e);
                return Ok(());
            }
        };

        let dst = load_scan(lines);
        if src.len() == 0 {
            src = dst;
            return Ok(());
        }

        if dst.len() == 0 {
            return Ok(());
        }

        let root = b.into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut cc = ChartBuilder::on(&root)
            .build_cartesian_2d(-WINDOW_RANGE..WINDOW_RANGE, -WINDOW_RANGE..WINDOW_RANGE)
            .unwrap();

        println!("initial error = {}", icp::huber_error(&param, &src, &dst));

        for _ in 0..10 {
            let correspondence = associate(&src, &dst);

            // for (src_index, dst_index) in &correspondence {
            //     let sp = src[*src_index];
            //     let dp = dst[*dst_index];
            //     let line = LineSeries::new(vec![(sp[0], sp[1]), (dp[0], dp[1])], RED);
            //     cc.draw_series(line).unwrap();
            // }

            param = estimate_transform(&src, &dst, &correspondence, &Some(param));
            src = src.iter().map(|sp| icp::transform(&param, sp)).collect();
        }

        println!("updated error = {}", icp::huber_error(&param, &src, &dst));

        let transform = icp::exp_se2(&param);

        // cc.draw_series(dst.iter().map(|p| { to_point(&p, &GREEN) })).unwrap();
        cc.draw_series(src.iter().map(|p| to_point(&p, &BLUE)))
            .unwrap();
        // cc.draw_series(src.iter().map(|p| {
        //     let dp = dtransform * Vector3::new(p[0], p[1], 1.);
        //     to_point(&Vector2::new(dp[0], dp[1]), &GREEN)
        // })).unwrap();

        let inv_transform = icp::exp_se2(&(-param));

        cc.draw_series(dst.iter().map(|dp| {
            let sp = inv_transform * Vector3::new(dp[0], dp[1], 1.);
            to_point(&Vector2::new(sp[0], sp[1]), &GREEN)
        }))
        .unwrap();

        // let (t, xp, yp) = axis_lines(&inv_transform, 100.);
        // cc.draw_series(LineSeries::new(vec![(t[0], t[1]), (xp[0], xp[1])], RED)).unwrap();
        // cc.draw_series(LineSeries::new(vec![(t[0], t[1]), (yp[0], yp[1])], BLUE)).unwrap();

        path.push(Vector2::new(inv_transform[(0, 2)], inv_transform[(1, 2)]));

        // for i in 0..(path.len()-1) {
        //     let p0 = path[i+0];
        //     let p1 = path[i+1];
        //     let line = LineSeries::new(vec![(p0[0], p0[1]), (p1[0], p1[1])], GREEN);
        //     cc.draw_series(line).unwrap();
        // }

        src = dst;
        Ok(())
    };

    while let Some(_) = draw_piston_window(&mut window, &mut draw) {}
}
