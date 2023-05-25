use framework::*;
use macroquad::prelude::*;

// #[derive(Clone, Debug)]
// pub struct Mat {
//     pub rows: usize,
//     pub cols: usize,
//     pub data: Vec<Vec<f32>>,
// }

// #[derive(Clone, Debug)]
// pub struct NN {
//     pub count: usize,
//     pub weights: Vec<Mat>,
//     pub biases: Vec<Mat>,
//     pub activations: Vec<Mat>,
// }
// Commented because it's already in lib.rs

const NODE_RADIUS: f32 = 20.0;
const LAYER_GAP: f32 = 100.0;
const NODE_GAP: f32 = 50.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "Neural Network Visualization".to_owned(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

fn draw_nn(nn: &NN) {
    let mut x = 50.0;

    for i in 0..nn.count {
        let y_offset = (screen_height()
            - (nn.activations[i].rows as f32 * (NODE_RADIUS * 2.0 + NODE_GAP) - NODE_GAP))
            / 2.0;
        for j in 0..nn.activations[i].rows {
            let y = y_offset + j as f32 * (NODE_RADIUS * 2.0 + NODE_GAP);

            if i > 0 {
                let prev_y_offset = (screen_height()
                    - (nn.activations[i - 1].rows as f32 * (NODE_RADIUS * 2.0 + NODE_GAP)
                        - NODE_GAP))
                    / 2.0;
                for k in 0..nn.activations[i - 1].rows {
                    let prev_y = prev_y_offset + k as f32 * (NODE_RADIUS * 2.0 + NODE_GAP);
                    draw_line(x - LAYER_GAP - NODE_GAP, prev_y, x, y, 2.0, GRAY);
                }
            }

            draw_circle(x, y, NODE_RADIUS, BLUE);
        }

        x += NODE_RADIUS * 2.0 + LAYER_GAP;
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Use the NN struct to visualize the neural network
    let mut nn = NN {
        count: 2,
        weights: vec![
            Mat {
                rows: 2,
                cols: 1,
                data: vec![vec![1.0], vec![1.0]],
            },
        ],
        biases: vec![
            Mat {
                rows: 1,
                cols: 1,
                data: vec![vec![0.0]],
            },
        ],
        activations: vec![
            Mat {
                rows: 1,
                cols: 2,
                data: vec![vec![1.0, 1.0]],
            },
            Mat {
                rows: 1,
                cols: 1,
                data: vec![vec![0.3]],
            },
        ],
    };


    let ti = Mat {
        rows: 4,
        cols: 2,
        data: vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
    };

    let to = Mat {
        rows: 4,
        cols: 1,
        data: vec![
            vec![0.0],
            vec![0.0], 
            vec![0.0], 
            vec![1.0]
        ],
    };

    let mut g = nn.clone();

    println!("{:?}", nn);
    nn_forward(&mut nn);
    //nn_randomize(&mut nn, -1.0, 1.0);
    // println!("{:?}", nn);
    println!("{:?}", nn_cost(nn.clone(), &ti, &to));
    nn_finite_diff(&mut nn, &mut g, 1e-1, &ti, &to);
    nn_learn(&mut nn, &g, 1.0);
    println!("{:?}", nn_cost(nn.clone(), &ti, &to));
    nn_finite_diff(&mut nn, &mut g, 1e-1, &ti, &to);
    nn_learn(&mut nn, &g, 1.0);
    println!("{:?}", nn_cost(nn.clone(), &ti, &to));

    loop {
        clear_background(LIGHTGRAY);

        draw_nn(&nn);

        next_frame().await
    }
}
