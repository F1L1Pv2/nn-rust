use framework::{sigmoidf, Mat, NN};
use macroquad::prelude::*;

mod draw;
use draw::*;

const EPOCH_MAX: i32 = 100_000;
const LEARNING_RATE: f32 = 0.1;

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

const RENDER_X: f32 = 0.0;
const RENDER_Y: f32 = 0.0;

const BACKGROUND_COLOR: Color = BLACK;
const TEXT_COLOR: Color = WHITE;

#[derive(Clone, Debug)]
pub struct Renderinfo {
    epoch: i32,
    cost: f32,
    t_input: Mat,
    training_time: f32,
    cost_history: Vec<f32>,
}

#[macroquad::main(window_conf)]
async fn main() {
    'main: loop {
        let mut nn = NN::new(&[2, 10, 10, 1]);
        let mut g = nn.clone();

        let t_input = Mat::new(&[
            // Even numbers are one
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
        ]);

        let t_output = Mat::new(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

        let mut cost = 0.0;

        NN::randomize(&mut nn, -1.0, 1.0);

        let time_elapsed = chrono::Utc::now().timestamp_millis();
        let mut info = Renderinfo {
            epoch: 0,
            cost,
            t_input: t_input.clone(),
            training_time: 0.0,
            cost_history: Vec::new(),
        };

        clear_background(BACKGROUND_COLOR);
        draw_frame(&nn, screen_width(), screen_height() / 1.2, &info);
        next_frame().await;

        // TRAINING
        for i in 0..EPOCH_MAX {
            // NN::finite_diff(&mut nn, &mut g, 0.1, &t_input, &t_output);
            info = Renderinfo {
                epoch: i + 1,
                cost,
                t_input: t_input.clone(),
                training_time: (chrono::Utc::now().timestamp_millis() - time_elapsed) as f32
                    / 1000.0,
                cost_history: info.cost_history.clone(),
            };

            // Reset?
            if is_key_down(KeyCode::R) {
                continue 'main;
            }

            NN::backprop(&mut nn, &mut g, &t_input, &t_output);
            NN::learn(&mut nn, &g, LEARNING_RATE);

            if i % 1000 == 0 {
                cost = NN::cost(&nn, &t_input, &t_output);
                println!("i:{i} cost:{cost:?}");
                info.cost_history.push(cost);

                clear_background(BACKGROUND_COLOR);
                draw_frame(&nn, screen_width(), screen_height() / 1.2, &info);
                next_frame().await;
            }
        }

        // TESTING
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
        println!("training time:{}", info.training_time);

        loop {
            // Reset?
            if is_key_down(KeyCode::R) {
                continue 'main;
            }

            clear_background(BACKGROUND_COLOR);
            draw_frame(&nn, screen_width(), screen_height() / 1.2, &info);
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
        window_title: "Neural Network Visualization".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        window_resizable: true,
        ..Default::default()
    }
}
