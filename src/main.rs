use ndarray::{Array, Array1};

fn lennard_jones(eps: f64, sig: f64, r: &Array1<f64>) -> Array1<f64> {
    let mut ir = sig / r;
    let mut ir6 = ir.view_mut();
    ir6.par_mapv_inplace(|a| a.powf(6.0));
    let mut ir12 = ir6.to_owned();
    ir12.par_mapv_inplace(|a| a.powf(2.0));

    4.0 * eps * (ir12.to_owned() - ir6.to_owned())
}

fn main() {
    let (epsilon, sigma) = (1.0, 1.0);
    let (npts, radius) = (1024, 10.24);
    let dr = radius / npts as f64;
    let r = Array::range(0.5, npts as f64, 1.0) * dr;
    let potential = lennard_jones(epsilon, sigma, &r);
    println!("Hello, world!");
}
