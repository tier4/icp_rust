use ndarray::{Array2, Array3};

use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{ChartBuilder, LineSeries, RGBColor, Rectangle, BLUE, RED};
use plotters::style::Color;
use plotters_piston::{draw_piston_window, PistonBackend};

const N_POINTS_IN_PACKET: usize = 24 * 16;

fn to_point(p: &(f64, f64), color: &RGBColor) -> Rectangle<(f64, f64)> {
    let (x, y) = p;
    let s = 4e-3;
    Rectangle::new([(x - s, y - s), (x + s, y + s)], color.filled())
}

struct Scan {
    datasets: Vec<hdf5::Dataset>,
}

fn reshape(array: Array3<f64>) -> Array2<f64> {
    array.into_shape((N_POINTS_IN_PACKET, 3)).unwrap()
}

fn to_vectors(array: Array2<f64>) -> Vec<icp::Vector3> {
    (0..array.shape()[0])
        .map(|i| {
            let p = array.slice(ndarray::s![i, ..]);
            icp::Vector3::new(p[0], p[1], p[2])
        })
        .collect::<Vec<icp::Vector3>>()
}

impl Scan {
    fn new(path: &str) -> hdf5::Result<Self> {
        let file = hdf5::File::open(path)?;
        let datasets = file.datasets()?;
        Ok(Scan { datasets })
    }

    fn size(&self) -> usize {
        self.datasets.len()
    }

    fn get(&self, index: usize) -> hdf5::Result<Vec<icp::Vector3>> {
        let array = self.datasets[index].read()?;
        let array = reshape(array);
        Ok(to_vectors(array))
    }

    fn get_range(&self, start: usize, end: usize) -> hdf5::Result<Vec<icp::Vector3>> {
        let size = end - start;

        let mut vectors = vec![];
        for i in 0..size {
            let a = self.get(start + i)?;
            vectors.extend(a);
        }
        Ok(vectors)
    }
}

fn remove_invalid_values(points: &Vec<icp::Vector3>) -> Vec<icp::Vector3> {
    points
        .iter()
        .cloned()
        .filter(|&p| p.norm() > 0.2)
        .collect::<Vec<icp::Vector3>>()
}

fn get_xy(p: &icp::Vector3) -> icp::Vector2 {
    icp::Vector2::new(p[0], p[1])
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

const FPS: u64 = 30;
const WINDOW_SIZE: u32 = 1200;
fn main() -> hdf5::Result<()> {
    let background_color = RGBColor(0xfa, 0xfd, 0xff);
    let path_color = RGBColor(0xff, 0xdc, 0x00);
    let src_color = RGBColor(0xf1, 0x9c, 0xa7);
    let dst_color = RGBColor(0x00, 0x6e, 0xb0);

    let scan = Scan::new("scans/3d/scans.hdf5")?;

    let mut window: PistonWindow = WindowSettings::new("LiDAR scan", [WINDOW_SIZE, WINDOW_SIZE])
        .build()
        .unwrap();
    window.set_max_fps(FPS);

    let step = 75;

    let src = scan.get_range(0, step).unwrap();
    let src = remove_invalid_values(&src);

    let mut transform = icp::Transform::identity();
    let mut path: Vec<icp::Vector2> = vec![];

    let mut index = 0;
    let mut draw = |b: PistonBackend| -> Result<(), Box<dyn std::error::Error>> {
        if index + step > scan.size() {
            return Ok(());
        }

        let dst = scan.get_range(index, index + step).unwrap();
        let dst = remove_invalid_values(&dst);

        index += step;

        let root = b.into_drawing_area();
        root.fill(&background_color).unwrap();

        let mut cc = ChartBuilder::on(&root)
            .build_cartesian_2d(-3.0..3.0, -3.0..3.0)
            .unwrap();

        transform = icp::icp_3dscan(&transform, &src, &dst);
        let inv_transform = transform.inverse();

        cc.draw_series(src.iter().map(|p| to_point(&(p[0], p[1]), &src_color)))
            .unwrap();
        cc.draw_series(dst.iter().map(|p| {
            let b = inv_transform.transform(&get_xy(p));
            to_point(&(b[0], b[1]), &dst_color)
        }))
        .unwrap();

        let (t, xp, yp) = axis_lines(&inv_transform, 0.2);
        cc.draw_series(LineSeries::new(vec![(t[0], t[1]), (xp[0], xp[1])], RED))
            .unwrap();
        cc.draw_series(LineSeries::new(vec![(t[0], t[1]), (yp[0], yp[1])], BLUE))
            .unwrap();
        path.push(inv_transform.t);

        for i in 0..(path.len() - 1) {
            let p0 = path[i + 0];
            let p1 = path[i + 1];
            let line = LineSeries::new(vec![(p0[0], p0[1]), (p1[0], p1[1])], path_color);
            cc.draw_series(line).unwrap();
        }

        Ok(())
    };

    while let Some(_) = draw_piston_window(&mut window, &mut draw) {}
    Ok(())
}
