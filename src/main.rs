use gnuplot::{AxesCommon, Caption, Color, Figure, Fix, LineWidth};
use ndarray::{Array, Array1};

fn lennard_jones(eps: f64, sig: f64, r: &Array1<f64>) -> Array1<f64> {
    let mut ir = sig / r;
    let mut ir6 = ir.view_mut();
    ir6.par_mapv_inplace(|a| a.powf(6.0));
    let mut ir12 = ir6.to_owned();
    ir12.par_mapv_inplace(|a| a.powf(2.0));

    4.0 * eps * (ir12.to_owned() - ir6.to_owned())
}

fn weeks_chandler_andersen(eps: f64, sig: f64, r: &Array1<f64>) -> Array1<f64> {
    let rmin = (2.0_f64).powf(1.0 / 6.0) * sig;
    r.mapv(|a| {
        if a > rmin {
            0.0
        } else {
            let ir = sig / a;
            (4.0 * eps * (ir.powf(12.0) - ir.powf(6.0))) + eps
        }
    })
}

fn j0(a: f64) -> f64 {
    a.sin() / a
}

fn main() {
    let (epsilon, sigma) = (1.0, 1.0);
    let (npts, radius) = (1024, 10.24);
    let dr = radius / npts as f64;
    let dk = 2.0 * std::f64::consts::PI / (2.0 * npts as f64 * dr);
    let r = Array::range(0.5, npts as f64, 1.0) * dr;
    let k = Array::range(0.5, npts as f64, 1.0) * dk;
    let lj_potential = lennard_jones(epsilon, sigma, &r);
    let wca_potential = weeks_chandler_andersen(epsilon, sigma, &r);

    let bond_length = 1.0;

    let j0_adjacent = 2.0 * k.mapv(|a| j0(a * bond_length));
    let j0_between = 2.0 * k.mapv(|a| j0(a * bond_length * (3.0_f64).sqrt()));
    let j0_opposite = k.mapv(|a| 2.0 * a * bond_length);

    let w = 1.0 + j0_adjacent + j0_between + j0_opposite;


    let r_vec = r.to_vec();
    let lj_vec = lj_potential.to_vec();
    let wca_vec = wca_potential.to_vec();
    let mut fg = Figure::new();
    fg.axes2d()
        .lines(&r_vec, &lj_vec, &[LineWidth(1.5), Caption("Lennard-Jones"), Color("black")])
        .set_y_range(Fix(-1.5), Fix(1.0))
        .set_x_range(Fix(0.0), Fix(5.0))
        .lines(&r_vec, &wca_vec, &[LineWidth(1.5), Caption("Weeks-Chandler-Andersen"), Color("red")]);
    fg.show();
}
