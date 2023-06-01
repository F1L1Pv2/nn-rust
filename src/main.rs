use std::{
    sync::{Arc, Mutex},
    thread,
};

use framework::{sigmoidf, Mat, NN};
use macroquad::prelude::*;

mod draw;
use draw::{draw_frame, Renderinfo};

const EPOCH_MAX: i32 = 100_000;
const LEARNING_RATE: f32 = 1.;

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

const BACKGROUND_COLOR: Color = BLACK;
const TEXT_COLOR: Color = WHITE;
const LINE_COLOR: Color = RED;

#[macroquad::main(window_conf)]
async fn main() {
    let nn_structure = &[1, 10, 10, 1];
    let nn = Arc::new(Mutex::new(NN::new(nn_structure)));
    let g = NN::new(nn_structure);

    'main: loop {
        // XOR Example
        // let t_input = Mat::new(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
        // let t_output = Mat::new(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

        // Opposite example
        let t_input = Mat::new(&[
            &[1.0],
            &[0.9],
            &[0.8],
            &[0.7],
            &[0.6],
            &[0.5],
            &[0.4],
            &[0.3],
            &[0.2],
            &[0.1],
            &[0.0],
        ]);

        let t_output = Mat::new(&[
            &[0.0],
            &[0.1],
            &[0.2],
            &[0.3],
            &[0.4],
            &[0.5],
            &[0.6],
            &[0.7],
            &[0.8],
            &[0.9],
            &[1.0],
        ]);

        let mut g = g.clone();

        let (tx, rx) = std::sync::mpsc::channel();

        {
            let mut nn = nn.lock().unwrap();
            NN::randomize(&mut nn, -1.0, 1.0);
            let cost = NN::cost(&nn, &t_input, &t_output);
            println!("Initial cost: {cost}");
        }

        // let mut paused = false;
        let time_elapsed = chrono::Utc::now().timestamp_millis();
        let info = Arc::new(Mutex::new(Renderinfo {
            epoch: 0,
            cost: 0.0,
            t_input: t_input.clone(),
            t_output: t_output.clone(),
            training_time: 0.0,
            cost_history: Vec::new(),
        }));

        clear_background(BACKGROUND_COLOR);
        {
            let mut info = info.lock().unwrap();
            draw_frame(
                &nn.lock().unwrap(),
                screen_width(),
                screen_height() / 1.2,
                &mut info,
            );
        }
        next_frame().await;

        // TRAINING
        let nn_clone = Arc::clone(&nn);
        let info_clone = Arc::clone(&info);

        let _training_thread = thread::spawn(move || {
            for i in 0..=EPOCH_MAX {
                // Check if we recieved a stop signal
                if rx.try_recv().is_ok() {
                    break;
                }

                {
                    let mut info = info_clone.lock().unwrap();
                    info.epoch = i;
                    info.t_input = t_input.clone();
                    info.t_output = t_output.clone();
                    info.training_time =
                        (chrono::Utc::now().timestamp_millis() - time_elapsed) as f32 / 1000.0;
                }

                {
                    let mut nn = nn_clone.lock().unwrap();
                    NN::backprop(&mut nn, &mut g, &t_input, &t_output);
                    NN::learn(&mut nn, &g, LEARNING_RATE);
                }
            }
            println!("training time:{}", info_clone.lock().unwrap().training_time);
        });

        loop {
            // Quit?
            if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
                std::process::exit(0);
            }

            // Reset?
            if is_key_pressed(KeyCode::R) {
                // Stop the training thread
                let _ = tx.send(());
                // Restart the program
                continue 'main;
            }

            // Pause?
            if is_key_pressed(KeyCode::P) {
                println!("WIP")
            }

            clear_background(BACKGROUND_COLOR);
            {
                let mut info = info.lock().unwrap();
                draw_frame(
                    &nn.lock().unwrap(),
                    screen_width(),
                    screen_height() / 1.2,
                    &mut info,
                );
            }
            next_frame().await;
        }
    }
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
