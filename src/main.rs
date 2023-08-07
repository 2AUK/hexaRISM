use gnuplot::{AxesCommon, Caption, Color, Figure, Fix, LineWidth};
use ndarray::{Array, Array1};
use rustdct::{DctPlanner, TransformType4};
use std::{f64::consts::PI, sync::Arc};

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

fn hankel_transform(
    prefac: f64,
    func: &Array1<f64>,
    grid_a: &Array1<f64>,
    grid_b: &Array1<f64>,
    plan: &Arc<dyn TransformType4<f64>>,
) -> Array1<f64> {
    let mut buffer = (func * grid_a).to_vec();

    plan.process_dst4(&mut buffer);

    prefac * Array1::from_vec(buffer) / grid_b
}

fn plot_potentials(r: &Array1<f64>, lj: &Array1<f64>, wca: &Array1<f64>) {
    let r_vec = r.to_vec();
    let lj_vec = lj.to_vec();
    let wca_vec = wca.to_vec();
    let mut fg = Figure::new();
    fg.axes2d()
        .lines(
            &r_vec,
            &lj_vec,
            &[LineWidth(1.5), Caption("Lennard-Jones"), Color("black")],
        )
        .set_y_range(Fix(-1.5), Fix(1.0))
        .set_x_range(Fix(0.0), Fix(5.0))
        .lines(
            &r_vec,
            &wca_vec,
            &[
                LineWidth(1.5),
                Caption("Weeks-Chandler-Andersen"),
                Color("red"),
            ],
        );
    fg.show().unwrap();
}

fn plot(x: &Array1<f64>, y: &Array1<f64>) {
    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut fg = Figure::new();
    fg.axes2d()
        .lines(&x_vec, &y_vec, &[LineWidth(1.5), Color("black")]);
    fg.show().unwrap();
}

fn rism(ck: &Array1<f64>, wk: &Array1<f64>, p: f64) -> Array1<f64> {
    ((ck * wk * wk) / (1.0 - 6.0 * p * wk * ck)) - ck
}

fn hnc(tr: &Array1<f64>, ur: &Array1<f64>, beta: f64) -> Array1<f64> {
    (-beta * ur + tr).mapv(f64::exp) - 1.0 - tr
}

fn main() {
    let (epsilon, sigma) = (1.0, 1.0);
    let (npts, radius) = (10, 10.24);
    let k_b = 1.0;
    let (T, p) = (1.6, 0.02);
    let beta = 1.0 / T / k_b;
    let dr = radius / npts as f64;
    let dk = 2.0 * PI / (2.0 * npts as f64 * dr);

    let plan: Arc<dyn TransformType4<f64>> = DctPlanner::new().plan_dst4(npts);

    let rtok = 2.0 * PI * dr;
    let ktor = dk / (2.0 * PI).powf(2.0);

    let r = Array::range(0.5, npts as f64, 1.0) * dr;
    let k = Array::range(0.5, npts as f64, 1.0) * dk;
    let lj_potential = lennard_jones(epsilon, sigma, &r);
    let wca_potential = weeks_chandler_andersen(epsilon, sigma, &r);

    let mayer_f = -beta * &lj_potential.mapv(|a| f64::exp(-a)) - 1.0;

    let bond_length = 1.0;

    let j0_adjacent = 2.0 * k.mapv(|a| j0(a * bond_length));
    let j0_between = 2.0 * k.mapv(|a| j0(a * bond_length * (3.0_f64).sqrt()));
    let j0_opposite = k.mapv(|a| 2.0 * a * bond_length);

    let wk = 1.0 + j0_adjacent + j0_between + j0_opposite;

    let mut cr = Array1::<f64>::zeros(npts);
    let mut tr = Array1::<f64>::zeros(npts);

    let mayer_f_k = hankel_transform(rtok, &mayer_f, &r, &k, &plan);
    let mayer_f_r = hankel_transform(ktor, &mayer_f_k, &k, &r, &plan);
    println!("{}, {}", mayer_f, mayer_f_r);


    // plot_potentials(&r, &lj_potential, &wca_potential);
    // plot(&k, &intramolecular_correlation_rspace);

    let (itermax, tol) = (5, 1e-7);
    let damp = 0.217;

    for i in 0..itermax {
        println!("Iteration: {i}");
        let cr_prev = cr.clone();
        let ck = hankel_transform(rtok, &cr, &r, &k, &plan);
        println!("ck before: {}", ck);
        let tk = rism(&ck, &wk, p);
        println!("ck after: {}", ck);
        println!("tk: {}", tk);
        tr = hankel_transform(ktor, &tk, &k, &r, &plan);
        println!("tr: {}", tr);
        let cr_a = hnc(&tr, &lj_potential, beta);
        println!("cr_a: {}", cr_a);
        cr = &cr_prev + (damp * (&cr_a - &cr_prev));
    }
    println!("beta: {beta}");
    let gr = (&tr + &cr) + 1.0;
    println!("{}\n{}\n{}", cr, tr, gr);
    plot(&r, &gr);
}
