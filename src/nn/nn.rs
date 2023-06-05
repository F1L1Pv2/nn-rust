use std::{
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
};

use framework::{sigmoidf, Mat, NN};
use macroquad::prelude::*;
use macroquad::rand::ChooseRandom;

use crate::draw;
use crate::draw::{color_lerp, draw_frame, lerp, window_conf, Renderinfo, BACKGROUND_COLOR};

const EPOCH_MAX: i32 = 1000;
const LEARNING_RATE: f32 = 1.;

const OUT_IMG_WIDTH: u32 = 400;
const OUT_IMG_HEIGHT: u32 = 400;
const BATCH_SIZE: usize = 28;

// enum Mode {
//     Image,
//     Normal,
// }

#[derive(PartialEq)]
enum Signal {
    Pause,
    Resume,
    Stop,
}

#[derive(Debug, Clone)]
struct Batch {
    input: Mat,
    output: Mat,
}

#[macroquad::main(window_conf)]
async fn main() {
    let nn_structure = &[2, 4, 4, 1];
    let nn = Arc::new(Mutex::new(NN::new(nn_structure)));
    let gradient = NN::new(nn_structure);

    'reset: loop {
        // Set random seed
        rand::srand(chrono::Utc::now().timestamp_millis() as u64);

        // XOR Example
        let t_input = Mat::new(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
        let t_output = Mat::new(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

        // Batches
        let mut batches: Vec<Batch> = Vec::new();

        let batchcount = t_input.rows / BATCH_SIZE;

        for i in 0..batchcount {
            let mut batch = Batch {
                input: Mat::new(&[&[0.]]),
                output: Mat::new(&[&[0.]]),
            };

            // Set batch input and output to the right size
            batch.input.cols = t_input.cols;
            batch.output.cols = t_output.cols;
            batch.input.data = vec![];
            batch.output.data = vec![];

            for j in 0..BATCH_SIZE {
                batch.input.push_row(t_input.get_row(i * BATCH_SIZE + j));
                batch
                    .output
                    .data
                    .push(t_output.get_row(i * BATCH_SIZE + j).to_vec());
            }

            // Fix rows and cols for batch
            batch.input.rows = batch.input.data.len();
            batch.output.rows = batch.output.data.len();
            batch.input.cols = batch.input.data[0].len();
            batch.output.cols = batch.output.data[0].len();

            batches.push(batch);
        }

        batches.shuffle();

        println!("Batches: {:?}", batches);

        let mut gradient = gradient.clone();

        let (tx, rx): (Sender<Signal>, Receiver<Signal>) = channel();

        let mut paused = false;
        let time_elapsed = chrono::Utc::now().timestamp_millis();

        let info: Arc<Mutex<Renderinfo>>;
        {
            // Calculate first cost for creating the struct
            let mut nn = nn.lock().unwrap();
            NN::randomize(&mut nn, -1.0, 1.0);
            let cost = NN::cost(&mut nn, &t_input, &t_output);
            println!("Initial cost: {}", cost);
            info = Arc::new(Mutex::new(Renderinfo {
                epoch: 0,
                cost,
                t_input: t_input.clone(),
                t_output: t_output.clone(),
                training_time: 0.0,
                cost_history: vec![cost],
                paused,
                learning_rate: LEARNING_RATE,
            }));
        }

        clear_background(BACKGROUND_COLOR);
        {
            let mut info = info.lock().unwrap();
            draw_frame(&mut nn.lock().unwrap(), &mut info);
        }
        next_frame().await;

        // TRAINING
        let nn_clone = Arc::clone(&nn);
        let info_clone = Arc::clone(&info);

        let _training_thread = thread::spawn(move || {
            'training: for i in 0..=EPOCH_MAX {
                // Learning rate decay
                let lr = lerp(LEARNING_RATE, 0.0001, i as f32 / EPOCH_MAX as f32);

                if let Ok(signal) = rx.try_recv() {
                    match signal {
                        Signal::Pause => {
                            let mut info = info_clone.lock().unwrap();
                            info.paused = true;
                            drop(info);

                            while let Ok(signal) = rx.recv() {
                                if signal == Signal::Resume {
                                    let mut info = info_clone.lock().unwrap();
                                    info.paused = false;
                                    drop(info);
                                    break;
                                } else if signal == Signal::Stop {
                                    break 'training;
                                }
                            }
                        }
                        Signal::Stop => {
                            break 'training;
                        }
                        _ => {}
                    }
                }

                for batch in batches.iter() {
                    {
                        let mut info = info_clone.lock().unwrap();
                        info.epoch = i;
                        info.t_input = t_input.clone();
                        info.t_output = t_output.clone();
                        info.training_time =
                            (chrono::Utc::now().timestamp_millis() - time_elapsed) as f32 / 1000.0;
                        info.learning_rate = lr;
                    }

                    {
                        let mut nn = nn_clone.lock().unwrap();
                        NN::backprop(&mut nn, &mut gradient, &batch.input, &batch.output);
                        NN::learn(&mut nn, &gradient, lr);
                    }
                }
            }
        });

        loop {
            // Quit?
            if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
                std::process::exit(0);
            }

            // Reset?
            if is_key_pressed(KeyCode::R) {
                // Stop the training thread
                let _ = tx.send(Signal::Stop);
                println!("Reset");
                // Restart the program
                continue 'reset;
            }

            // Pause/Resume?
            if is_key_pressed(KeyCode::P) {
                if paused {
                    // Send a "resume" signal to the training thread
                    let _ = tx.send(Signal::Resume);
                    paused = false;
                    println!("Resumed");
                } else {
                    // Send a "pause" signal to the training thread
                    let _ = tx.send(Signal::Pause);
                    paused = true;
                    println!("Paused");
                }
            }

            clear_background(BACKGROUND_COLOR);
            {
                let mut info = info.lock().unwrap();
                draw_frame(&mut nn.lock().unwrap(), &mut info);
            }
            next_frame().await;
        }
    }
}
