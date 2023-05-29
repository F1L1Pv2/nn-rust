use framework::*;
use macroquad::prelude::*;

const EPOCH_MAX: i32 = 100000;
const LEARNING_RATE: f32 = 0.1;

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

const RENDER_X: f32 = 0.0;
const RENDER_Y: f32 = 0.0;

const BACKGROUND_COLOR: Color = BLACK;
const TEXT_COLOR: Color = WHITE;

#[derive(Clone, Debug)]
struct Renderinfo {
    epoch: i32,
    cost: f32,
    t_input: Mat,
    training_time: f32,
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
        };

        clear_background(BACKGROUND_COLOR);
        draw_nn(&nn, screen_width(), screen_height() / 1.2, &info);
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
            };

            // Reset?
            if is_key_down(KeyCode::R) {
                continue 'main;
            }

            NN::backprop(&mut nn, &mut g, &t_input, &t_output);
            NN::learn(&mut nn, &g, LEARNING_RATE);

            if i % 1000 == 0 {
                cost = NN::cost(&nn, &t_input, &t_output);
                println!("i:{} cost:{:?}", i, cost);

                clear_background(BACKGROUND_COLOR);
                draw_nn(&nn, screen_width(), screen_height() / 1.2, &info);
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
            draw_nn(&nn, screen_width(), screen_height() / 1.2, &info);
            next_frame().await
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
        ..Default::default()
    }
}

fn draw_nn(nn: &NN, width: f32, height: f32, info: &Renderinfo) {
    let mut nn = nn.clone();
    let low_color = Color {
        r: 1.,
        g: 0.,
        b: 1.,
        a: 1.,
    };
    let high_color = Color {
        r: 0.,
        g: 1.,
        b: 0.,
        a: 1.,
    };

    let x = RENDER_X;
    let y = RENDER_Y;

    let neuron_radius = height * 0.03;
    let layer_border_vpad = height * 0.08;
    let layer_border_hpad = width * 0.06;
    let nn_width = width - 2.0 * layer_border_hpad;
    let nn_height = height - 2.0 * layer_border_vpad;
    let nn_x = x + width / 2.0 - nn_width / 2.0;
    let nn_y = y + height / 2.0 - nn_height / 2.0;
    let arch_count = nn.count;
    let layer_hpad = nn_width / arch_count as f32;

    for l in 0..arch_count {
        let layer_vpad1 = nn_height / nn.activations[l].cols as f32;
        for i in 0..nn.activations[l].cols {
            let cx1 = nn_x + l as f32 * layer_hpad + layer_hpad / 2.0;
            let cy1 = nn_y + i as f32 * layer_vpad1 + layer_vpad1 / 2.0;
            if l + 1 < arch_count {
                let layer_vpad2 = nn_height / nn.activations[l + 1].cols as f32;
                for j in 0..nn.activations[l + 1].cols {
                    let cx2 = nn_x + (l + 1) as f32 * layer_hpad + layer_hpad / 2.0;
                    let cy2 = nn_y + j as f32 * layer_vpad2 + layer_vpad2 / 2.0;
                    let value = sigmoidf(nn.weights[l].data[i][j]);
                    let thick = height * 0.004;
                    draw_line(
                        cx1,
                        cy1,
                        cx2,
                        cy2,
                        thick,
                        color_lerp(low_color, high_color, value),
                    );
                }
            }
            if l > 0 {
                let value = sigmoidf(nn.biases[l - 1].data[0][i]);
                draw_circle(
                    cx1,
                    cy1,
                    neuron_radius,
                    color_lerp(low_color, high_color, value),
                );
            } else {
                draw_circle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }

    // Write the parameters at the bottom left
    draw_text(
        format!("Epoch: {}/{}", info.epoch, EPOCH_MAX,).as_str(),
        0.,
        15.,
        20.,
        TEXT_COLOR,
    );
    draw_text(
        format!("Cost: {}", info.cost,).as_str(),
        0.,
        30.,
        20.,
        TEXT_COLOR,
    );
    draw_text(
        format!("Learning Rate: {}", LEARNING_RATE).as_str(),
        0.,
        45.,
        20.,
        TEXT_COLOR,
    );
    draw_text(
        format!("Training time: {:.1}s", info.training_time).as_str(),
        0.,
        60.,
        20.,
        TEXT_COLOR,
    );

    // Write the testing results at the bottom right
    for i in 0..info.t_input.rows {
        nn.activations[0].data[0][0] = info.t_input.data[i][0];
        nn.activations[0].data[0][1] = info.t_input.data[i][1];

        NN::forward(&mut nn);
        draw_text(
            format!(
                "input:{:?} output:{:?}",
                info.t_input.data[i],
                nn.activations[nn.count - 1].data[0]
            )
            .as_str(),
            0.,
            screen_height() - 10. - i as f32 * 20.,
            20.,
            TEXT_COLOR,
        );
    }
}
