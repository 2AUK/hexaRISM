use approx::assert_relative_eq;
use fftw::plan::*;
use fftw::types::*;
use gnuplot::{AxesCommon, Caption, Color, Figure, Fix, LineWidth};
use ndarray::{Array, Array1};
use rustdct::algorithm::{Dst1Naive, Type4Naive};
use rustdct::{DctPlanner, Dst1, Dst4, TransformType4};
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
    grid1: &Array1<f64>,
    grid2: &Array1<f64>,
) -> Array1<f64> {
    // let mut buffer = (func * grid_a).to_vec();
    //
    // let npts = func.len();
    //
    // let dst = Type4Naive::new(npts);
    //
    // //let dst1 = Dst1Naive::new(npts);
    //
    // dst.process_dst4(&mut buffer);
    //
    // prefac * Array1::from_vec(buffer) / grid_b

    let arr = func * grid1;
    let mut r2r: R2RPlan64 =
        R2RPlan::aligned(&[grid1.len()], R2RKind::FFTW_RODFT11, Flag::ESTIMATE)
            .expect("could not execute FFTW plan");
    let mut input = arr.as_standard_layout();
    let mut output = Array1::zeros(input.raw_dim());
    r2r.r2r(
        input.as_slice_mut().unwrap(),
        output.as_slice_mut().unwrap(),
    )
    .expect("could not perform DST-IV operation");
    prefac * output / grid2
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
    let top = ck * wk * wk;
    let bottom = 1.0 - (6.0 * p * wk * ck);
    let hk = top / bottom;
    hk - ck
}

fn hnc(tr: &Array1<f64>, ur: &Array1<f64>, beta: f64) -> Array1<f64> {
    ((-beta * ur) + tr).mapv(f64::exp) - 1.0 - tr
}

fn main() {
    let (epsilon, sigma) = (1.0, 1.0);
    let (npts, radius) = (1024, 10.24);
    let k_b = 1.0;
    let (T, p) = (1.6, 0.02);
    let beta = 1.0 / T / k_b;
    let dr = radius / npts as f64;
    let dk = 2.0 * PI / (2.0 * npts as f64 * dr);

    let plan: Arc<dyn TransformType4<f64>> = DctPlanner::new().plan_dst4(npts);

    let rtok = 2.0 * PI * dr;
    let ktor = dk / 4.0 / PI / PI;

    let r = Array::range(0.5, npts as f64, 1.0) * dr;
    let k = Array::range(0.5, npts as f64, 1.0) * dk;
    let lj_potential = lennard_jones(epsilon, sigma, &r);
    let wca_potential = weeks_chandler_andersen(epsilon, sigma, &r);

    let mayer_f = (&lj_potential).mapv(|a| f64::exp(-beta * a)) - 1.0;

    let bond_length = 1.0;

    let j0_adjacent = 2.0 * k.mapv(|a| j0(a * bond_length));
    let j0_between = 2.0 * k.mapv(|a| j0(a * bond_length * (3.0_f64).sqrt()));
    let j0_opposite = k.mapv(|a| 2.0 * a * bond_length);

    let wk = 1.0 + j0_adjacent + j0_between + j0_opposite;

    let mayer_f_k = hankel_transform(rtok, &mayer_f, &r, &k);
    let mayer_f_r = hankel_transform(ktor, &mayer_f_k, &k, &r);
    plot(&r, &mayer_f);
    plot(&r, &mayer_f_r);
    //assert_relative_eq!(mayer_f, mayer_f_r, epsilon=1e-5);

    let mut cr = Array1::<f64>::zeros(npts);
    let mut tr = Array1::<f64>::zeros(npts);
    let ones = Array1::<f64>::ones(npts);
    let kfromr = hankel_transform(rtok, &ones, &r, &k);
    // println!("k: {}\nk from r: {}", k, kfromr);

    // plot_potentials(&r, &lj_potential, &wca_potential);
    // plot(&k, &intramolecular_correlation_rspace);

    let (itermax, tol) = (10, 1e-7);
    let damp = 0.1;

    for i in 0..itermax {
        println!("Iteration: {i}");
        let cr_prev = cr.clone();
        let ck = hankel_transform(rtok, &cr_prev.clone(), &r, &k);
        println!("cr before: {}", cr_prev);
        let tk = rism(&ck, &wk, p);
        let crfromck = hankel_transform(ktor, &ck.clone(), &k, &r);
        println!("cr after: {}", crfromck);
        println!("tk: {}", tk);
        tr = hankel_transform(ktor, &tk, &k, &r);
        println!("tr: {}", tr);
        let cr_a = hnc(&tr, &lj_potential, beta);
        println!("cr_a: {}", cr_a);
        println!("cr_prev: {}", cr_prev.clone());
        cr = cr_prev.clone() + damp * (cr_a.clone() - cr_prev.clone());
    }
    let gr = (&tr + &cr) + 1.0;
    println!("{}\n{}\n{}", cr, tr, gr);
    // plot(&r, &gr);
}
