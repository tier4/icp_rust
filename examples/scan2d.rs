use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{ChartBuilder, Circle, LineSeries, RGBColor, BLUE, GREEN, RED, WHITE};
use plotters::style::Color;
use plotters_piston::{draw_piston_window, PistonBackend};
use std::fs::File;
use std::io::BufRead;
use std::path::Path;

fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(std::io::BufReader::new(file).lines())
}

fn load_scan(lines: std::io::Lines<std::io::BufReader<File>>) -> Vec<icp::Vector2> {
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
        scan_landmarks.push(icp::Vector2::new(x, y));
    }
    return scan_landmarks;
}

fn to_point(p: &icp::Vector2, color: &RGBColor) -> Circle<(f64, f64), u32> {
    Circle::new((p[0], p[1]), 2, color.mix(0.7).filled())
}

fn axis_lines(
    transform: &icp::Transform,
    length: f64,
) -> (icp::Vector2, icp::Vector2, icp::Vector2) {
    let rot = transform.rot;
    let t = transform.t;
    let x = icp::Vector2::new(length, 0.);
    let y = icp::Vector2::new(0., length);

    let xp = rot * x + t;
    let yp = rot * y + t;
    (t, xp, yp)
}

const WINDOW_RANGE: f64 = 3000.;
const FPS: u64 = 100;
fn main() {
    let mut window: PistonWindow = WindowSettings::new("LiDAR scan", [800, 800])
        .build()
        .unwrap();
    window.set_max_fps(FPS);

    let mut src = vec![];
    let mut index = 0;

    let mut param = icp::Param::zeros();

    let mut path: Vec<icp::Vector2> = vec![];
    let mut draw = |b: PistonBackend| -> Result<(), Box<dyn std::error::Error>> {
        index += 1;
        let filename = format!("scans/2d/{:03}.txt", index);

        let lines = read_lines(filename)?;

        if index == 1 {
            src = load_scan(lines);
            return Ok(());
        }

        let root = b.into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut cc = ChartBuilder::on(&root)
            .build_cartesian_2d(-WINDOW_RANGE..WINDOW_RANGE, -WINDOW_RANGE..WINDOW_RANGE)
            .unwrap();

        let dst = load_scan(lines);

        param = icp::icp_2dscan(&param, &src, &dst);
        let inv_transform = icp::Transform::new(&(-param));

        cc.draw_series(src.iter().map(|p| to_point(&p, &BLUE)))
            .unwrap();
        cc.draw_series(dst.iter().map(|p| {
            let b = inv_transform.transform(&p);
            to_point(&b, &GREEN)
        }))
        .unwrap();

        let (t, xp, yp) = axis_lines(&inv_transform, 200.);
        cc.draw_series(LineSeries::new(vec![(t[0], t[1]), (xp[0], xp[1])], RED))
            .unwrap();
        cc.draw_series(LineSeries::new(vec![(t[0], t[1]), (yp[0], yp[1])], BLUE))
            .unwrap();
        path.push(icp::Vector2::new(inv_transform.t[0], inv_transform.t[1]));

        for i in 0..(path.len() - 1) {
            let p0 = path[i + 0];
            let p1 = path[i + 1];
            let line = LineSeries::new(vec![(p0[0], p0[1]), (p1[0], p1[1])], GREEN);
            cc.draw_series(line).unwrap();
        }

        Ok(())
    };

    while let Some(_) = draw_piston_window(&mut window, &mut draw) {}
}
