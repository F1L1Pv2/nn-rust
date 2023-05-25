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
    let nn = NN {
        count: 3,
        weights: vec![
            Mat {
                rows: 2,
                cols: 3,
                data: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            },
            Mat {
                rows: 3,
                cols: 2,
                data: vec![vec![0.7, 0.8], vec![0.9, 1.0], vec![1.1, 1.2]],
            },
        ],
        biases: vec![
            Mat {
                rows: 3,
                cols: 1,
                data: vec![vec![0.1], vec![0.2], vec![0.3]],
            },
            Mat {
                rows: 2,
                cols: 1,
                data: vec![vec![0.4], vec![0.5]],
            },
        ],
        activations: vec![
            Mat {
                rows: 2,
                cols: 1,
                data: vec![vec![0.1], vec![0.2]],
            },
            Mat {
                rows: 3,
                cols: 1,
                data: vec![vec![0.3], vec![0.4], vec![0.5]],
            },
            Mat {
                rows: 2,
                cols: 1,
                data: vec![vec![0.6], vec![0.7]],
            },
        ],
    };
    loop {
        clear_background(LIGHTGRAY);

        draw_nn(&nn);

        next_frame().await
    }
}
