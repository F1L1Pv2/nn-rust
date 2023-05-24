use rand::Rng;

#[derive(Clone, Debug)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct NN {
    count: usize,
    weights: Vec<Mat>,
    biases: Vec<Mat>,
    activations: Vec<Mat>,
}

fn sigmoidf(x: f32) -> f32 {
    return 1.0 / (1.0 + -x.exp());
}

fn mat_dot(dst: &mut Mat, a: &Mat, b: &Mat) {
    assert_eq!(a.cols, b.rows);
    let n = a.cols;
    assert_eq!(dst.rows, a.rows);
    assert_eq!(dst.cols, b.cols);

    for i in 0..dst.rows {
        for j in 0..dst.cols {
            dst.data[i][j] = 0.0;
            for k in 0..n {
                dst.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
}

// do a jest dodawane b
fn mat_sum(a: &mut Mat, b: &Mat) {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);

    for i in 0..a.rows {
        for j in 0..a.cols {
            a.data[i][j] += b.data[i][j];
        }
    }
}

fn rand_float(min: f32, max: f32) -> f32 {
    rand::thread_rng().gen_range(min..max)
}

fn fill_mat(dst: &mut Mat, val: f32) {
    for i in 0..dst.rows {
        for j in 0..dst.cols {
            dst.data[i][j] = val;
        }
    }
}

fn mat_sig(dst: &mut Mat) {
    for i in 0..dst.rows {
        for j in 0..dst.cols {
            dst.data[i][j] = sigmoidf(dst.data[i][j]);
        }
    }
}

fn nn_forward(nn: &mut NN) {
    for i in 0..nn.count {
        let mut new_nn = nn.clone();
        mat_dot(
            &mut new_nn.activations[i + 1],
            &nn.activations[i],
            &nn.weights[i],
        );
        mat_sum(&mut nn.activations[i + 1], &nn.biases[i]);
        mat_sig(&mut nn.activations[i + 1]);
    }
}

fn mat_row(mat: &Mat, row: usize) -> Mat {
    return Mat {
        rows: 1,
        cols: mat.cols,
        data: vec![mat.data[row].clone()],
    };
}

fn mat_copy(dst: &mut Mat, src: &Mat) {
    assert_eq!(dst.rows, src.rows);
    assert_eq!(dst.cols, src.cols);
    for i in 0..dst.rows {
        for j in 0..dst.cols {
            dst.data[i][j] = src.data[i][j];
        }
    }
}

macro_rules! nn_input {
    ($nn:expr) => {
        $nn.activations[0]
    };
}

macro_rules! nn_output {
    ($nn:expr) => {
        $nn.activations[$nn.count]
    };
}

fn nn_cost(mut nn: NN, t_input: Mat, t_output: Mat) -> f32 {
    assert_eq!(t_input.rows, t_output.rows);
    assert_eq!(t_input.cols, nn.activations[nn.count].cols);
    let n = t_input.rows;

    let mut cost = 0.0;
    // to idzie przez kazdy training data (index training data)
    for i in 0..n {
        let x: Mat = mat_row(&t_input, i);
        let y: Mat = mat_row(&t_output, i);

        mat_copy(&mut nn_input!(nn), &x);
        nn_forward(&mut nn);
        let q = t_output.cols;
        for j in 0..q {
            let diff: f32 = nn_output!(nn).data[0][j] - y.data[0][j];
            // cost is magnified
            cost += diff * diff;
        }
    }

    cost
}

fn nn_learn(nn: &mut NN, g: &NN, rate: f32) {
    for i in 0..nn.count {
        for j in 0..nn.weights[i].rows {
            for k in 0..nn.weights[i].cols {
                nn.weights[i].data[j][k] -= rate * g.weights[i].data[j][k];
            }
        }

        for j in 0..nn.biases[i].rows {
            for k in 0..nn.biases[i].cols {
                nn.biases[i].data[j][k] -= rate * g.biases[i].data[j][k];
            }
        }
    }
}

fn nn_finite_diff(nn: &mut NN, g: &mut NN, eps: f32, ti: Mat, to: Mat) {
    let mut saved: f32;
    let c = nn_cost(nn.clone(), ti.clone(), to.clone());

    for i in 0..nn.count {
        for j in 0..nn.weights[i].rows {
            for k in 0..nn.weights[i].cols {
                saved = nn.weights[i].data[j][k];
                nn.weights[i].data[j][k] += eps;
                g.weights[i].data[j][k] = (nn_cost(nn.clone(), ti.clone(), to.clone()) - c) / eps;
                nn.weights[i].data[j][k] = saved;
            }
        }

        for j in 0..nn.biases[i].rows {
            for k in 0..nn.biases[i].cols {
                saved = nn.biases[i].data[j][k];
                nn.biases[i].data[j][k] += eps;
                g.biases[i].data[j][k] = (nn_cost(nn.clone(), ti.clone(), to.clone()) - c) / eps;
                nn.biases[i].data[j][k] = saved;
            }
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let number = rand::random::<f32>();
//         println!("Random number: {}", number);
//         assert_eq!(number, 69.0);
//     }
// }
