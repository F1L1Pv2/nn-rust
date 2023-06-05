use macroquad::window::screen_width;
use macroquad::prelude::*;

use super::{
    color_lerp, draw_circle, draw_line, draw_rectangle, draw_text, f32, screen_height, sigmoidf,
    Color, Mat, EPOCH_MAX, GRAY, NN, 
};

pub const WINDOW_WIDTH: i32 = 800;
pub const WINDOW_HEIGHT: i32 = 600;

pub const BACKGROUND_COLOR: Color = BLACK;
pub const TEXT_COLOR: Color = WHITE;
pub const LINE_COLOR: Color = RED;

const LOW_COLOR: Color = Color {
    r: 0.,
    g: 1.,
    b: 0.,
    a: 1.,
};

const HIGH_COLOR: Color = Color {
    r: 1.,
    g: 0.,
    b: 1.,
    a: 1.,
};

#[derive(Clone, Debug)]
pub struct Renderinfo {
    pub epoch: i32,
    pub cost: f32,
    pub t_input: Mat,
    pub t_output: Mat,
    pub training_time: f32,
    pub cost_history: Vec<f32>,
    pub paused: bool,
    pub learning_rate: f32,
}

pub fn draw_frame(nn: &NN, info: &mut Renderinfo) {
    let nn = nn.clone();
    let (width, height) = (screen_width(), screen_height());

    // Skip epoch 0 because the value is already in the cost history (from creating the struct)
    if info.epoch < EPOCH_MAX && !info.paused && info.epoch != 0 {
        let cost = NN::cost(&nn, &info.t_input, &info.t_output);

        info.cost = cost;
        info.cost_history.push(cost);
    }

    draw_nn(&nn, width, height * 0.8);
    draw_graph(width, height, info);
    draw_data(info, nn);

    draw_text("r - reset", width - 100., 20., 20., TEXT_COLOR);
    draw_text("p - pause", width - 100., 40., 20., TEXT_COLOR);
    draw_text("q - quit", width - 100., 60., 20., TEXT_COLOR);
    draw_text(
        "f - save image (while pausing)",
        width - 120.,
        80.,
        20.,
        TEXT_COLOR,
    );
    draw_text("(while pausing)", width - 120., 100., 20., TEXT_COLOR);
}

fn draw_nn(nn: &NN, width: f32, height: f32) {
    let x = 0.;
    let y = 0.;

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
                        color_lerp(LOW_COLOR, HIGH_COLOR, value),
                    );
                }
            }
            if l > 0 {
                let value = sigmoidf(nn.biases[l - 1].data[0][i]);
                draw_circle(
                    cx1,
                    cy1,
                    neuron_radius,
                    color_lerp(LOW_COLOR, HIGH_COLOR, value),
                );
            } else {
                draw_circle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

fn draw_graph(width: f32, height: f32, info: &Renderinfo) {
    let x = 0.;
    let y = 0.;

    // Draw a cost history graph in the bottom right
    let graph_width = width * 0.3;
    let graph_height = height * 0.3;
    let graph_x = x + width - graph_width;
    let graph_y = y + height - graph_height;

    draw_rectangle(
        graph_x,
        graph_y,
        graph_width,
        graph_height,
        Color {
            r: 0.2,
            g: 0.2,
            b: 0.2,
            a: 0.5,
        },
    );
    draw_text(
        format!("Cost: {}", info.cost).as_str(),
        graph_x,
        graph_y - 5.,
        20.,
        Color {
            r: 1.,
            g: 1.,
            b: 1.,
            a: 1.,
        },
    );

    let mut max_cost = 0.;
    // Check for max cost
    for i in 0..info.cost_history.len() {
        if info.cost_history[i] > max_cost {
            max_cost = info.cost_history[i];
        }
    }
    let mut last_x = graph_x;
    let mut last_y;
    last_y = graph_y + graph_height - info.cost_history[0] / max_cost * graph_height;

    for i in 0..info.cost_history.len() {
        let x = graph_x + i as f32 * graph_width / info.cost_history.len() as f32;
        let y = graph_y + graph_height - info.cost_history[i] / max_cost * graph_height;
        draw_line(last_x, last_y, x, y, 1., LINE_COLOR);
        last_x = x;
        last_y = y;
    }
}

fn draw_data(info: &Renderinfo, mut nn: NN) {
    // Top right parameters
    draw_text(
        format!(
            "Epoch: {}/{} | Learning Rate: {:.4}",
            info.epoch, EPOCH_MAX, info.learning_rate
        )
        .as_str(),
        0.,
        15.,
        20.,
        TEXT_COLOR,
    );

    draw_text(
        format!("Training time: {:.2}s", info.training_time).as_str(),
        0.,
        30.,
        20.,
        TEXT_COLOR,
    );

    // Write the testing results at the bottom left
    for i in 0..info.t_input.rows {
        for j in 0..nn.activations[0].data[0].len() {
            nn.activations[0].data[0][j] = info.t_input.data[i][j];
        }

        NN::forward(&mut nn);
        draw_text(
            format!(
                // Input | Output
                "{:?} -> {:?}",
                info.t_input.data[i],
                nn.activations[nn.count - 1].data[0] // -1 because the last activation is the output
            )
            .as_str(),
            0.,
            screen_height() - 20. - i as f32 * 10.,
            20.,
            TEXT_COLOR,
        );
    }
}

pub fn window_conf() -> Conf {
    Conf {
        window_title: "nn-rust".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        window_resizable: true,
        ..Default::default()
    }
}
