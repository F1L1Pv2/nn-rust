use rand::{seq::SliceRandom, Rng};
use std::f32::consts::PI;

#[macro_export]
macro_rules! nn_input {
    ($nn:expr) => {
        $nn.activations[0]
    };
}

#[macro_export]
macro_rules! nn_output {
    ($nn:expr) => {
        $nn.activations[$nn.count - 1]
    };
}

// macro_rules! mat_at {
//     ($m:expr, $i:expr, $j:expr) => {
//         $m.data[$i][$j]
//     };
// }

#[derive(Clone, Debug)]
pub struct NN {
    pub count: usize,
    pub weights: Vec<Mat>,
    pub biases: Vec<Mat>,
    pub activations: Vec<Mat>,
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub input: Mat,
    pub output: Mat,
}

impl NN {
    pub fn new(arch: &[usize]) -> NN {
        Self::alloc(arch)
    }

    pub fn forward(nn: &mut NN) {
        for i in 0..nn.count - 1 {
            let mut temp = Mat::new(&[&[0.0]]);
            temp.rows = nn.activations[i + 1].rows;
            temp.cols = nn.activations[i + 1].cols;
            temp.data = vec![vec![0.0; temp.cols]; temp.rows];
            Mat::dot(&mut temp, &nn.activations[i], &nn.weights[i]);
            nn.activations[i + 1] = temp;
            Mat::sum(&mut nn.activations[i + 1], &nn.biases[i]);
            // Mat::sig(&mut nn.activations[i + 1]);
            Mat::activation_function(&mut nn.activations[i + 1]);
        }
    }

    pub fn gen_batches(t_input: &Mat, t_output: &Mat, batch_size: usize) -> Vec<Batch> {
        let mut batches: Vec<Batch> = Vec::new();

        let batchcount = t_input.rows / batch_size;

        for i in 0..batchcount {
            let mut batch = Batch {
                input: Mat::new(&[&[0.]]),
                output: Mat::new(&[&[0.]]),
            };

            //set batch input and output to the right size
            batch.input.cols = t_input.cols;
            batch.output.cols = t_output.cols;
            batch.input.data = vec![];
            batch.output.data = vec![];

            for j in 0..batch_size {
                batch.input.push_row(t_input.get_row(i * batch_size + j));
                batch
                    .output
                    .data
                    .push(t_output.get_row(i * batch_size + j).to_vec());
            }

            //fix rows and cols for batch
            batch.input.rows = batch.input.data.len();
            batch.output.rows = batch.output.data.len();
            batch.input.cols = batch.input.data[0].len();
            batch.output.cols = batch.output.data[0].len();

            batches.push(batch);

            //shuffle batches without shuffle
        }

        batches.shuffle(&mut rand::thread_rng());

        batches
    }

    pub fn cost(nn: &mut NN, t_input: &Mat, t_output: &Mat) -> f32 {
        //let mut nn = nn.clone();
        assert_eq!(t_input.rows, t_output.rows);
        assert_eq!(t_output.cols, nn.activations[nn.count - 1].cols);
        let n = t_input.rows;

        let mut cost = 0.0;
        // to idzie przez kazdy training data (index training data)
        for i in 0..n {
            let x: Mat = Mat::row(t_input, i);
            let y: Mat = Mat::row(t_output, i);

            Mat::copy(&mut nn_input!(nn), &x);
            Self::forward(nn);
            let q = t_output.cols;
            for j in 0..q {
                let diff: f32 = nn_output!(nn).data[0][j] - y.data[0][j];
                // cost is magnified
                cost += diff * diff;
            }
        }

        cost
    }

    pub fn learn(nn: &mut NN, g: &NN, rate: f32) {
        for i in 0..nn.count - 1 {
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

    pub fn randomize(nn: &mut NN, min: f32, max: f32) {
        for i in 0..nn.count - 1 {
            for j in 0..nn.weights[i].rows {
                for k in 0..nn.weights[i].cols {
                    nn.weights[i].data[j][k] = rand_float(min, max);
                }
            }

            for j in 0..nn.biases[i].rows {
                for k in 0..nn.biases[i].cols {
                    nn.biases[i].data[j][k] = rand_float(min, max);
                }
            }
        }
    }

    pub fn zero(nn: &mut NN) {
        for i in 0..nn.count - 1 {
            for j in 0..nn.weights[i].rows {
                for k in 0..nn.weights[i].cols {
                    nn.weights[i].data[j][k] = 0.0;
                }
            }

            for j in 0..nn.biases[i].rows {
                for k in 0..nn.biases[i].cols {
                    nn.biases[i].data[j][k] = 0.0;
                }
            }
        }
    }

    pub fn finite_diff(nn: &mut NN, g: &mut NN, eps: f32, t_input: &Mat, t_output: &Mat) {
        let mut saved: f32;
        let c = Self::cost(nn, &t_input.clone(), &t_output.clone());

        for i in 0..nn.count - 1 {
            for j in 0..nn.weights[i].rows {
                for k in 0..nn.weights[i].cols {
                    saved = nn.weights[i].data[j][k];
                    nn.weights[i].data[j][k] += eps;
                    g.weights[i].data[j][k] =
                        (Self::cost(nn, &t_input.clone(), &t_output.clone()) - c) / eps;
                    nn.weights[i].data[j][k] = saved;
                }
            }

            for j in 0..nn.biases[i].rows - 1 {
                for k in 0..nn.biases[i].cols - 1 {
                    saved = nn.biases[i].data[j][k];
                    nn.biases[i].data[j][k] += eps;
                    g.biases[i].data[j][k] =
                        (Self::cost(nn, &t_input.clone(), &t_output.clone()) - c) / eps;
                    nn.biases[i].data[j][k] = saved;
                }
            }
        }
    }

    pub fn backprop(nn: &mut NN, g: &mut NN, t_input: &Mat, t_output: &Mat) {
        assert_eq!(t_input.rows, t_output.rows);
        let n = t_input.rows;
        assert_eq!(nn.activations[nn.count - 1].cols, t_output.cols);

        NN::zero(g);

        for i in 0..n {
            Mat::copy(&mut nn_input!(nn), &Mat::row(t_input, i));
            Self::forward(nn);

            for j in 0..nn.count {
                Mat::fill(&mut g.activations[j], 0.0);
            }

            for j in 0..t_output.cols {
                g.activations[nn.count - 1].data[0][j] =
                    (nn_output!(nn).data[0][j] - t_output.data[i][j]) * 2.0 / n as f32;
            }

            for l in (0..nn.count - 1).rev() {
                for j in 0..nn.activations[l + 1].cols {
                    let a = nn.activations[l + 1].data[0][j];
                    let da = g.activations[l + 1].data[0][j];
                    // g.biases[l].data[0][j] += da * a * (1.0 - a);
                    g.biases[l].data[0][j] += da * Mat::activation_derivative(a);
                    for k in 0..nn.activations[l].cols {
                        let pa = nn.activations[l].data[0][k];
                        let w = nn.weights[l].data[k][j];
                        // g.weights[l].data[k][j] += da * a * (1.0 - a) * pa;
                        // g.activations[l].data[0][k] += da * a * (1.0 - a) * w;
                        g.weights[l].data[k][j] += da * Mat::activation_derivative(a) * pa;
                        g.activations[l].data[0][k] += da * Mat::activation_derivative(a) * w;
                    }
                }
            }
        }
    }

    pub fn alloc(arch: &[usize]) -> NN {
        assert!(!arch.is_empty());

        let count = arch.len();

        let mut weights = Vec::with_capacity(count);
        let mut biases = Vec::with_capacity(count);
        let mut activations = Vec::with_capacity(count);

        activations.push(Mat {
            rows: 1,
            cols: arch[0],
            data: vec![vec![0.0; arch[0]]],
        });

        for i in 1..count {
            weights.push(Mat {
                rows: activations[i - 1].cols,
                cols: arch[i],
                data: vec![vec![0.0; arch[i]]; activations[i - 1].cols],
            });

            biases.push(Mat {
                rows: 1,
                cols: arch[i],
                data: vec![vec![0.0; arch[i]]],
            });

            activations.push(Mat {
                rows: 1,
                cols: arch[i],
                data: vec![vec![0.0; arch[i]]],
            });
        }

        NN {
            count,
            weights,
            biases,
            activations,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f32>>,
}

impl Mat {
    pub fn new(data: &[&[f32]]) -> Mat {
        let rows = data.len();
        let cols = data[0].len();

        let mut mat = Mat {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        };

        for (i, row) in data.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                mat.data[i][j] = *val;
            }
        }

        mat
    }

    pub fn get_row(&self, row: usize) -> &[f32] {
        &self.data[row]
    }

    pub fn push_row(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.cols);
        self.data.push(row.to_vec());
        self.rows += 1;
    }

    // do a jest dodawane b
    pub fn sum(a: &mut Mat, b: &Mat) {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);

        for (i, row) in a.data.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                *val += b.data[i][j];
            }
        }
    }

    pub fn dot(dst: &mut Mat, a: &Mat, b: &Mat) {
        assert_eq!(a.cols, b.rows);
        // let n = a.cols;
        assert_eq!(dst.rows, a.rows);
        assert_eq!(dst.cols, b.cols);

        Mat::fill(dst, 0.0);

        for (i, row) in dst.data.iter_mut().enumerate() {
            for (k, val) in a.data[i].iter().enumerate() {
                for (j, val2) in row.iter_mut().enumerate() {
                    *val2 += val * b.data[k][j];
                }
            }
        }
    }

    pub fn fill(dst: &mut Mat, val: f32) {
        for i in 0..dst.rows {
            for j in 0..dst.cols {
                dst.data[i][j] = val;
            }
        }
    }

    pub fn sig(dst: &mut Mat) {
        for row in &mut dst.data {
            for val in row.iter_mut() {
                *val = sigmoidf(*val);
            }
        }
    }

    pub fn hyperbolic(dst: &mut Mat) {
        for row in &mut dst.data {
            for val in row.iter_mut() {
                *val = tanhf(*val);
            }
        }
    }

    pub fn relu(dst: &mut Mat) {
        for row in &mut dst.data {
            for val in row.iter_mut() {
                *val = reluf(*val);
                clampf(*val, 0., 1.);
            }
        }
    }

    pub fn gelu(dst: &mut Mat) {
        for row in &mut dst.data {
            for val in row.iter_mut() {
                *val = geluf(*val);
                // clampf(*val, 0., 1.);
            }
        }
    }

    pub fn activation_function(dst: &mut Mat) {
        Mat::sig(dst)
    }

    pub fn activation_derivative(x: f32) -> f32 {
        Mat::activation_sigmoid_derivative(x)
    }

    pub fn activation_hyperbolic_derivative(x: f32) -> f32 {
        1.0 - x * x
    }

    pub fn activation_sigmoid_derivative(x: f32) -> f32 {
        // let s = sigmoidf(x);
        x * (1.0 - x)
    }

    pub fn activation_relu_derivative(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    /*
    def gelu_derivative(x):
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    pdf = np.exp(-np.power(x, 2) / 2) / np.sqrt(2 * np.pi)
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) + x * (1 - np.power(cdf, 2)) * np.sqrt(2 / np.pi) * 0.0356774 + pdf * (0.0356774 * np.power(x, 2) + 0.797885 * x + 0.0356774)
     */

    pub fn activation_gelu_derivative(x: f32) -> f32 {
        let cdf = 0.5 * (1.0 + (2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3)).tanh());
        let pdf = (-x.powi(2) / 2.0).exp() / (2.0 * PI).sqrt();
        0.5 * (1.0 + (2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3)).tanh())
            + x * (1.0 - cdf.powi(2)) * (2.0 / PI).sqrt() * 0.0356774
            + pdf * (0.0356774 * x.powi(2) + 0.797885 * x + 0.0356774)
    }

    pub fn row(mat: &Mat, row: usize) -> Mat {
        Mat {
            rows: 1,
            cols: mat.cols,
            data: vec![mat.data[row].clone()],
        }
    }

    pub fn copy(dst: &mut Mat, src: &Mat) {
        assert_eq!(dst.rows, src.rows);
        assert_eq!(dst.cols, src.cols);
        for i in 0..dst.rows {
            for j in 0..dst.cols {
                dst.data[i][j] = src.data[i][j];
            }
        }
    }
}

pub fn reluf(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/*
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

 */

pub fn geluf(x: f32) -> f32 {
    0.5 * x * (1.0 + (2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3)).tanh())
}

pub fn tanhf(x: f32) -> f32 {
    x.tanh()
}

pub fn clampf(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

pub fn rand_float(min: f32, max: f32) -> f32 {
    rand::thread_rng().gen_range(min..max)
}

#[cfg(test)]
mod test;
