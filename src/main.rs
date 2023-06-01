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

const EPOCH_MAX: i32 = 100_000;
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
    let nn_structure = &[1, 5, 1];
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
                if let Ok(_) = rx.try_recv() {
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
                continue 'main;
            }

            // Pause?
            if is_key_pressed(KeyCode::P) {
                println!("WIP")
            }

            // Save?
            if is_key_pressed(KeyCode::S) {
                let nn = nn.lock().unwrap();
                let json = NN::to_json(&nn);
                let mut file = File::create("nn.json").unwrap();
                file.write_all(json.as_bytes()).unwrap();
                println!("Saved to nn.json");
            }

            // Load?
            if is_key_pressed(KeyCode::L) {
                println!("Loading is WIP")
            }

            // if is_key_pressed(KeyCode::L) {
            //     let mut file = File::open("nn.json").unwrap();
            //     let mut json = String::new();
            //     file.read_to_string(&mut json).unwrap();
            //     let new_nn = NN::from_json(&json);

            //     // Stop the training thread
            //     let _ = tx.send(());

            //     // Change the nn to the loaded one
            //     let mut nn = nn.lock().unwrap();
            //     *nn = new_nn;

            //     println!("Loaded from nn.json");
            //     println!("{:?}", nn);

            //     // Restart the training
            //     continue 'main;
            // }

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
