mod gp;
extern crate nalgebra as na;

use plotters::prelude::*;
use rand::Rng;
use std::f64::consts::PI;
const NUM: usize = 100;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("test.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.titled("Gaussian Process", ("sans-serif", 100).into_font())?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1f32..1f32, 0f32..1f32)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .label_style(("sans_serif", 30).into_font())
        .draw()?;

    let mut rng = rand::thread_rng();
    let (x1, x2): (Vec<f64>, Vec<f64>) = (0..NUM)
        .map(|_| rng.gen_range(-PI..PI))
        .map(|x| (x.cos(), x.sin()))
        .unzip();
    let x = x1.iter().chain(x2.iter()).cloned();
    let x = na::MatrixXx2::<f64>::from_iterator(NUM, x);
    let mut y = x.clone();
    for mut r in y.row_iter_mut() {
        r[1] *= r[1];
    }

    let gp =
        gp::GaussianProcess::<f64>::new(x, y.clone(), 2.0, 0.001f64).expect("Unable to create gp");

    let (x1, x2): (Vec<f64>, Vec<f64>) = (-100..100)
        .map(|x| x as f64 * PI / 100f64)
        .map(|x| (x.cos(), x.sin()))
        .unzip();
    let x = na::MatrixXx2::<f64>::from_iterator(200, x1.iter().chain(x2.iter()).cloned());
    let mut pred_y = x.clone();

    for mut r in pred_y.row_iter_mut() {
        r.set_row(0, &gp.predict_f(&r).expect("unable to predict").0);
    }

    chart
        .draw_series(
            pred_y
                .row_iter()
                .map(|r| Circle::new((r[0] as f32, r[1] as f32), 10, RED.filled())),
        )?
        .label("gp")
        .legend(|(x, y)| Circle::new((x + 5, y), 10, RED.filled()));

    chart
        .draw_series(
            y.row_iter()
                .map(|r| Circle::new((r[0] as f32, r[1] as f32), 10, GREEN.filled())),
        )?
        .label("true")
        .legend(|(x, y)| Circle::new((x + 5, y), 10, GREEN.filled()));
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 50).into_font())
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
