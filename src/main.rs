#[macro_use(array)]
extern crate ndarray;
use ndarray::Array2;

fn split(m: &Array2<f64>, k: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let m_dim = m.raw_dim();
    let mut m11 = Array2::zeros((k, k));
    let mut m12 = Array2::zeros((k, k));
    let mut m21 = Array2::zeros((k, k));
    let mut m22 = Array2::zeros((k, k));
    for r in 0..m_dim[0] {
        for c in 0..m_dim[1] {
            let r_m = r % k;
            let c_m = c % k;
            if r < k && c < k {
                m11[(r_m, c_m)] = m[(r, c)];
            }
            if r < k && c >= k {
                m12[(r_m, c_m)] = m[(r, c)];
            }
            if r >= k && c < k {
                m21[(r_m, c_m)] = m[(r, c)];
            }
            if r >= k && c >= k {
                m22[(r_m, c_m)] = m[(r, c)];
            }
        }
    }
    (m11, m12, m21, m22)
}

fn strassen(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let a_dim = a.raw_dim();
    let k = a_dim[0] / 2;
    if k == 0 {
        return a * b;
    }

    let (a11, a12, a21, a22) = split(a, k);
    let (b11, b12, b21, b22) = split(b, k);
    let t1 = strassen(&(&a11 + &a22), &(&b11 + &b22));
    let t2 = strassen(&(&a21 + &a22), &b11);
    let t3 = strassen(&a11, &(&b12 - &b22));
    let t4 = strassen(&a22, &(&b21 - &b11));
    let t5 = strassen(&(&a11 + &a12), &b22);
    let t6 = strassen(&(&a21 - &a11), &(&b11 + &b12));
    let t7 = strassen(&(&a12 - &a22), &(&b21 + &b22));
    let m1 = &t1 + &t4 - &t5 + &t7;
    let m2 = &t3 + &t5;
    let m3 = &t2 + &t4;
    let m4 = &t1 - &t2 + &t3 + &t6;
    let mut result = Array2::zeros(a_dim);
    for r in 0..a_dim[0] {
        for c in 0..a_dim[1] {
            let r_m = r % k;
            let c_m = c % k;
            if r < k && c < k {
                result[(r, c)] = m1[(r_m, c_m)];
            }
            if r < k && c >= k {
                result[(r, c)] = m2[(r_m, c_m)];
            }
            if r >= k && c < k {
                result[(r, c)] = m3[(r_m, c_m)];
            }
            if r >= k && c >= k {
                result[(r, c)] = m4[(r_m, c_m)];
            }
        }
    }
    result
}

fn main() {
    let a = array![
        [8., 4., 2., 4., 0., 5., 6., 1.],
        [5., 6., 3., 1., 7., 6., 6., 3.],
        [6., 6., 8., 7., 3., 6., 2., 2.],
        [2., 1., 1., 5., 7., 9., 4., 6.],
        [8., 3., 3., 4., 3., 1., 3., 4.],
        [5., 6., 3., 3., 3., 9., 6., 8.],
        [5., 0., 3., 2., 7., 6., 9., 5.],
        [6., 2., 0., 7., 8., 1., 9., 4.]
    ];
    let b = array![
        [7., 9., 2., 3., 3., 1., 6., 1.],
        [1., 4., 0., 1., 0., 6., 8., 0.],
        [2., 7., 3., 6., 5., 7., 0., 4.],
        [6., 7., 4., 4., 5., 2., 2., 6.],
        [9., 5., 1., 6., 0., 0., 1., 8.],
        [0., 4., 0., 7., 3., 6., 9., 2.],
        [8., 5., 6., 9., 0., 3., 1., 7.],
        [9., 5., 1., 2., 8., 9., 5., 2.]
    ];
    let r = strassen(&a, &b);
    println!("{}", r);
}
