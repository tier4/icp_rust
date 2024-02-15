use core::ops::Range;

use hdf5;
use ndarray;
use ndarray::{Array1, Array2, Array3};

use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{ChartBuilder, RGBColor, Rectangle, BLUE, WHITE};
use plotters::style::Color;
use plotters_piston::{draw_piston_window, PistonBackend};

const N_POINTS_IN_PACKET: usize = 24 * 16;

fn to_point(p: &(f64, f64), color: &RGBColor) -> Rectangle<(f64, f64)> {
    let (x, y) = p;
    let s = 1e-2;
    Rectangle::new([(x - s, y - s), (x + s, y + s)], color.filled())
}

fn squared_norm(p: &Array1<f64>) -> f64 {
    (0..p.len()).fold(0., |sum, i| sum + p[i] * p[i])
}

struct Scan {
    datasets: Vec<hdf5::Dataset>,
}

fn reshape(array: Array3<f64>) -> Array2<f64> {
    array.into_shape((N_POINTS_IN_PACKET, 3)).unwrap()
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

    fn get(&self, index: usize) -> hdf5::Result<Array2<f64>> {
        let array = self.datasets[index].read()?;
        Ok(reshape(array))
    }

    fn get_range(&self, start: usize, end: usize) -> hdf5::Result<Array2<f64>> {
        let size = end - start;

        let mut array = Array2::zeros((size * N_POINTS_IN_PACKET, 3));
        for i in 0..size {
            let a = self.get(start + i)?;
            let s = N_POINTS_IN_PACKET * i;
            let e = N_POINTS_IN_PACKET * (i + 1);
            array.slice_mut(ndarray::s![s..e, ..]).assign(&a);
        }
        Ok(array)
    }
}

const FPS: u64 = 120;
const WINDOW_RANGE: f64 = 5.0;
fn main() -> hdf5::Result<()> {
    let scan = Scan::new("points/points1.hdf5")?;

    let mut window: PistonWindow = WindowSettings::new("LiDAR scan", [800, 800])
        .build()
        .unwrap();
    window.set_max_fps(FPS);

    let step = 75;
    let mut index = 0;

    let mut draw = |b: PistonBackend| -> Result<(), Box<dyn std::error::Error>> {
        if index + step > scan.size() {
            return Ok(());
        }

        let array = scan.get_range(index, index + step).unwrap();
        index += step;

        let root = b.into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut cc = ChartBuilder::on(&root)
            .build_cartesian_2d(-WINDOW_RANGE..WINDOW_RANGE, -WINDOW_RANGE..WINDOW_RANGE)
            .unwrap();

        cc.draw_series((0..array.shape()[0]).map(|i| {
            let p = array.slice(ndarray::s![i, ..]);
            to_point(&(p[0], p[1]), &BLUE)
        }))
        .unwrap();

        Ok(())
    };

    while let Some(_) = draw_piston_window(&mut window, &mut draw) {}
    Ok(())
}
