use ndarray::{Array, Array1};
use charming::{component::Axis, series::Scatter, Chart};
use gnuplot::{Figure, Caption, Color, Fix, AxesCommon};

fn lennard_jones(eps: f64, sig: f64, r: &Array1<f64>) -> Array1<f64> {
    let mut ir = sig / r;
    let mut ir6 = ir.view_mut();
    ir6.par_mapv_inplace(|a| a.powf(6.0));
    let mut ir12 = ir6.to_owned();
    ir12.par_mapv_inplace(|a| a.powf(2.0));

    4.0 * eps * (ir12.to_owned() - ir6.to_owned())
}

fn weeks_chandler_andersen(eps: f64, sig: f64, r: &Array1<f64>) -> Array1<f64> {
    todo!()
}

fn main() {
    let (epsilon, sigma) = (1.0, 1.0);
    let (npts, radius) = (1024, 10.24);
    let dr = radius / npts as f64;
    let r = Array::range(0.5, npts as f64, 1.0) * dr;
    let potential = lennard_jones(epsilon, sigma, &r);

    let r_vec = r.to_vec();
    let lj_vec = potential.to_vec();
    let mut fg = Figure::new();
    fg.axes2d().lines(&r_vec, &lj_vec, &[Caption("Lennard-Jones"), Color("black")]).set_y_range(Fix(-1.5), Fix(1.0)).set_x_range(Fix(0.0), Fix(5.0));
    fg.show();
}
