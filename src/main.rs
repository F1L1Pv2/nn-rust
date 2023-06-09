use std::{
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
};

use framework::{sigmoidf, Mat, NN};
use macroquad::prelude::*;

mod draw;
use draw::{draw_frame, lerp, window_conf, Renderinfo, BACKGROUND_COLOR};

// mod img2nn;
// mod nn;

const EPOCH_MAX: i32 = 100_000;
const LEARNING_RATE: f32 = 1.;

const OUT_IMG_WIDTH: u32 = 256;
const OUT_IMG_HEIGHT: u32 = 256;
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
#[macroquad::main(window_conf)]
async fn main() {
    let mut argv = std::env::args();

    // let mode = match argv.nth(1) {
    //     Some(_) => Mode::Image,
    //     None => Mode::Normal,
    // };

    let image_path = argv.nth(1).expect("No image1 path given");

    let image = image::open(image_path).unwrap();

    let image_data = image.to_rgba8();

    let image_path2 = argv.nth(0).expect("No image2 path given");

    let image2 = image::open(image_path2).unwrap();

    let image_data2 = image2.to_rgba8();

    let mut value = 0.5;

    let nn_structure = &[3, 28,28,9, 3];
    let nn = Arc::new(Mutex::new(NN::new(nn_structure)));
    let gradient = NN::new(nn_structure);

    'reset: loop {
        // Set random seed
        rand::srand(chrono::Utc::now().timestamp_millis() as u64);

        // XOR Example
        // let t_input = Mat::new(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
        // let t_output = Mat::new(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

        // Image example
        let mut t_input = Mat::new(&[&[0.]]);
        let mut t_output = Mat::new(&[&[0.]]);

        t_input.data = vec![];
        t_output.data = vec![];
        t_input.cols = 3;
        t_output.cols = 3;

        for (x, y, pixel) in image_data.enumerate_pixels() {
            t_input.push_row(&[
                x as f32 / image_data.width() as f32,
                y as f32 / image_data.height() as f32,
                0.,
            ]);
            // t_output.push_row(&[pixel[0] as f32 / 255.]);
            t_output.push_row(&[
                pixel[0] as f32 / 255.,
                pixel[1] as f32 / 255.,
                pixel[2] as f32 / 255.,
            ]);
        }

        for (x, y, pixel) in image_data2.enumerate_pixels() {
            t_input.push_row(&[
                x as f32 / image_data.width() as f32,
                y as f32 / image_data.height() as f32,
                1.,
            ]);
            // t_output.push_row(&[pixel[0] as f32 / 255.]);
            t_output.push_row(&[
                pixel[0] as f32 / 255.,
                pixel[1] as f32 / 255.,
                pixel[2] as f32 / 255.,
            ]);
        }

        t_input.rows = t_input.data.len();
        t_output.rows = t_output.data.len();

        // Batches
        let batches = NN::gen_batches(&t_input, &t_output, BATCH_SIZE);

        // println!("batches: {:?}", batches);

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
            draw_frame(&mut nn.lock().unwrap(), &mut info, &mut value);
        }
        next_frame().await;

        // TRAINING
        let nn_clone = Arc::clone(&nn);
        let info_clone = Arc::clone(&info);

        let _training_thread = thread::spawn(move || {
            'training: for i in 0..=EPOCH_MAX {
                //learning rate decay
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
                        // NN::finite_diff(&mut nn, &mut gradient, 1e-4, &batch.input, &batch.output);
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

            //Save image
            if is_key_pressed(KeyCode::F) && paused {
                let nn = nn.lock().unwrap();
                let mut nn = nn.clone();
                let mut image = image::ImageBuffer::new(OUT_IMG_WIDTH, OUT_IMG_HEIGHT);

                for (x, y, pixel) in image.enumerate_pixels_mut() {
                    let input = Mat::new(&[&[
                        x as f32 / OUT_IMG_WIDTH as f32,
                        y as f32 / OUT_IMG_HEIGHT as f32,
                        value,
                    ]]);
                    nn.activations[0] = input.clone();
                    NN::forward(&mut nn);
                    let output = &nn.activations[nn.activations.len() - 1];
                    // let color = (output.data[0][0] * 255.) as u8;
                    let red = (output.data[0][0] * 255.) as u8;
                    let green = (output.data[0][1] * 255.) as u8;
                    let blue = (output.data[0][2] * 255.) as u8;
                    *pixel = image::Rgba([red, green, blue, 255]);
                }

                image.save("output.png").unwrap();
                println!("Saved image");
            }

            clear_background(BACKGROUND_COLOR);
            {
                let mut info = info.lock().unwrap();
                draw_frame(&mut nn.lock().unwrap(), &mut info, &mut value);
            }
            next_frame().await;
        }
    }
}
