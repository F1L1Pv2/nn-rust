use std::{
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
    thread,
};

use framework::{sigmoidf, Mat, NN};
use macroquad::prelude::*;

mod draw;
use draw::draw_frame;

const EPOCH_MAX: i32 = 100_000_000;
const LEARNING_RATE: f32 = 1.;

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

const BACKGROUND_COLOR: Color = BLACK;
const TEXT_COLOR: Color = WHITE;
const LINE_COLOR: Color = RED;

#[derive(Clone, Debug)]
pub struct Renderinfo {
    epoch: i32,
    cost: f32,
    t_input: Mat,
    t_output: Mat,
    training_time: f32,
    cost_history: Vec<f32>,
}

impl Renderinfo {
    pub fn new(t_input: &Mat, t_output: &Mat) -> Renderinfo {
        Renderinfo {
            epoch: 0,
            cost: 0.,
            t_input: t_input.clone(),
            t_output: t_output.clone(),
            training_time: 0.,
            cost_history: Vec::new(),
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // XOR Example
    let nn_structure = &[2, 4, 1];
    let t_input = Mat::new(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
    let t_output = Mat::new(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

    // Shared NN and Renderinfo objects protected by Mutex
    let nn = Arc::new(Mutex::new(NN::new(nn_structure)));
    let info = Arc::new(Mutex::new(Renderinfo::new(&t_input, &t_output)));

    // Spawn a separate thread for training
    let training_thread = {
        let nn = Arc::clone(&nn);
        let info = Arc::clone(&info);

        let t_input = t_input.clone();
        let t_output = t_output.clone();

        thread::spawn(move || {
            let mut nn = nn.lock().unwrap();

            NN::randomize(&mut nn, -1.0, 1.0);
            let mut cost = NN::cost(&nn, &t_input, &t_output);
            println!("Initial cost: {:?}", cost);

            drop(nn);

            let time_elapsed = chrono::Utc::now().timestamp_millis();

            // Training loop
            for i in 0..=EPOCH_MAX {
                // Lock the nn and info objects separately to avoid deadlock
                let mut nn = nn.lock().unwrap();
                let mut info = info.lock().unwrap();

                info.epoch = i;
                info.cost = cost;
                info.t_input = t_input.clone();
                info.t_output = t_output.clone();
                info.training_time =
                    (chrono::Utc::now().timestamp_millis() - time_elapsed) as f32 / 1000.0;
                info.cost_history = info.cost_history.clone();

                let mut cloned_nn = nn.clone();
                NN::backprop(&mut nn, &mut cloned_nn, &t_input, &t_output);
                NN::learn(&mut nn, &cloned_nn, LEARNING_RATE);

                if i % 1000 == 0 {
                    cost = NN::cost(&nn, &t_input, &t_output);
                    println!("i:{:?} cost:{:?}", i, cost);
                    info.cost_history.push(cost);
                }

                // Unlock the mutexes explicitly to allow other threads to access them
                drop(info);
                drop(nn);
            }

            // TESTING
            for i in 0..t_input.rows {
                let mut nn = nn.lock().unwrap();
                nn.activations[0].data[0][0] = t_input.data[i][0];
                nn.activations[0].data[0][1] = t_input.data[i][1];

                NN::forward(&mut nn);
                println!(
                    "input:{:?} output:{:?}",
                    t_input.data[i],
                    nn.activations[nn.count - 1].data[0]
                );

                // Unlock the mutex explicitly to allow other threads to access it
                drop(nn);
            }

            let info = info.lock().unwrap();
            println!("training time:{}", info.training_time);
        })
    };

    'main: loop {
        // Get the nn and renderinfo objects
        let nn = nn.lock().unwrap();
        let info = info.lock().unwrap();

        println!("epoch:{}", info.epoch);

        clear_background(BACKGROUND_COLOR);
        draw_frame(&nn, screen_width(), screen_height() / 1.2, &info);

        // Unlock the mutexes explicitly to allow other threads to access them
        drop(info);
        drop(nn);

        next_frame().await;
    }

    // Join the training thread to ensure it finishes
    training_thread.join().unwrap();
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn color_lerp(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: lerp(a.r, b.r, t),
        g: lerp(a.g, b.g, t),
        b: lerp(a.b, b.b, t),
        a: lerp(a.a, b.a, t),
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "nn-rust".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        window_resizable: true,
        ..Default::default()
    }
}
