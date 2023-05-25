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

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

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
    for i in 0..nn.count - 1 {
        for j in 0..nn.activations[i].cols {
            for k in 0..nn.activations[i + 1].cols {
                let x1 = i as f32 * LAYER_GAP + NODE_RADIUS;
                let y1 = j as f32 * NODE_GAP + NODE_RADIUS;
                let x2 = (i + 1) as f32 * LAYER_GAP + NODE_RADIUS;
                let y2 = k as f32 * NODE_GAP + NODE_RADIUS;
                let weight = nn.weights[i].data[j][k];
                let color = Color {
                    r: lerp(1.0, 0.0, (weight + 1.) / 2.),
                    g: lerp(0.0, 1.0, (weight + 1.) / 2.),
                    b: lerp(1.0, 0.0, (weight + 1.) / 2.),
                    a: 1.0,
                };
                draw_line(x1, y1, x2, y2, 2.0, color);
            }
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Use the NN struct to visualize the neural network
    // let mut nn = NN {
    //     count: 2,
    //     weights: vec![Mat {
    //         rows: 2,
    //         cols: 1,
    //         data: vec![vec![1.0], vec![1.0]],
    //     }],
    //     biases: vec![Mat {
    //         rows: 1,
    //         cols: 1,
    //         data: vec![vec![0.0]],
    //     }],
    //     activations: vec![
    //         Mat {
    //             rows: 1,
    //             cols: 2,
    //             data: vec![vec![1.0, 1.0]],
    //         },
    //         Mat {
    //             rows: 1,
    //             cols: 1,
    //             data: vec![vec![0.3]],
    //         },
    //     ],
    // };

    let mut nn = NN::new(&[2, 2, 1]);

    let t_input: Mat = Mat {
        rows: 4,
        cols: 2,
        data: vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
    };

    let t_output = Mat {
        rows: 4,
        cols: 1,
        data: vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    };

    let mut g = nn.clone();

    println!("{:?}", nn);
    // nn_forward(&mut nn);
    NN::randomize(&mut nn, -1.0, 1.0);
    // println!("{:?}", nn);

    for i in 0..10000 {
        // NN::finite_diff(&mut nn, &mut g, 1e-1, &t_input, &t_output);
        NN::backprop(&mut nn, &mut g, &t_input, &t_output);
        NN::learn(&mut nn, &g, 1e-1);
        if i % 500 == 0 {
            clear_background(LIGHTGRAY);
            draw_nn(&nn);
            next_frame().await;

            println!(
                "i:{} cost:{:?}",
                i,
                NN::cost(nn.clone(), &t_input, &t_output)
            );
        }
    }

    for i in 0..t_input.rows {
        nn.activations[0].data[0][0] = t_input.data[i][0];
        nn.activations[0].data[0][1] = t_input.data[i][1];

        NN::forward(&mut nn);
        println!(
            "input:{:?} output:{:?}",
            t_input.data[i],
            nn.activations[nn.count - 1].data[0]
        );
    }

    loop {
        clear_background(LIGHTGRAY);

        draw_nn(&nn);

        next_frame().await
    }
}
