use macroquad::{prelude::WHITE, text::measure_text, window::screen_width};

use super::{
    color_lerp, draw_circle, draw_line, draw_rectangle, draw_text, f32, screen_height, sigmoidf,
    Color, Renderinfo, EPOCH_MAX, GRAY, LEARNING_RATE, LINE_COLOR, LOAD_BUTTON_COORDS, NN,
    PAUSE_BUTTON_COORDS, RESET_BUTTON_COORDS, SAVE_BUTTON_COORDS, TEXT_COLOR,
};

pub fn draw_frame(nn: &NN, width: f32, height: f32, info: &Renderinfo) {
    let nn = nn.clone();

    draw_nn(&nn, width, height);
    draw_graph(width, height, info);
    draw_data(info, nn);
    draw_button(
        RESET_BUTTON_COORDS.0 + screen_width() - RESET_BUTTON_COORDS.2 * 2.,
        RESET_BUTTON_COORDS.1,
        RESET_BUTTON_COORDS.2,
        RESET_BUTTON_COORDS.3,
        "Reset",
        Color::new(0.4, 0.2, 0.2, 0.5),
    );
    draw_button(
        PAUSE_BUTTON_COORDS.0 + screen_width() - RESET_BUTTON_COORDS.2 * 2.,
        PAUSE_BUTTON_COORDS.1,
        PAUSE_BUTTON_COORDS.2,
        PAUSE_BUTTON_COORDS.3,
        "Pause",
        Color::new(0.2, 0.2, 0.2, 0.5),
    );
    draw_button(
        SAVE_BUTTON_COORDS.0 + screen_width() - RESET_BUTTON_COORDS.2 * 2.,
        SAVE_BUTTON_COORDS.1,
        SAVE_BUTTON_COORDS.2,
        SAVE_BUTTON_COORDS.3,
        "Save",
        Color::new(0.2, 0.4, 0.2, 0.5),
    );
    draw_button(
        LOAD_BUTTON_COORDS.0 + screen_width() - RESET_BUTTON_COORDS.2 * 2.,
        LOAD_BUTTON_COORDS.1,
        LOAD_BUTTON_COORDS.2,
        LOAD_BUTTON_COORDS.3,
        "Load",
        Color::new(0.2, 0.2, 0.2, 0.5),
    );
}

fn draw_nn(nn: &NN, width: f32, height: f32) {
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
}

fn draw_graph(width: f32, height: f32, info: &Renderinfo) {
    let x = 0.;
    let y = 0.;

    // Draw a cost history graph in the bottom right
    let graph_width = width * 0.3;
    let graph_height = height * 0.3;
    let graph_x = x + width - graph_width;
    let graph_y = y + height - (graph_height * 0.4);
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
        "Cost History",
        graph_x,
        graph_y,
        20.,
        Color {
            r: 1.,
            g: 1.,
            b: 1.,
            a: 1.,
        },
    );

    let mut max_cost = 0.;
    for i in 0..info.cost_history.len() {
        if info.cost_history[i] > max_cost {
            max_cost = info.cost_history[i];
        }
    }
    let mut last_x = graph_x;
    let mut last_y;
    if info.cost_history.is_empty() {
        last_y = graph_y + graph_height;
    } else {
        last_y = graph_y + graph_height - info.cost_history[0] / max_cost * graph_height;
    }
    for i in 0..info.cost_history.len() {
        let x = graph_x + i as f32 * graph_width / info.cost_history.len() as f32;
        let y = graph_y + graph_height - info.cost_history[i] / max_cost * graph_height;
        draw_line(last_x, last_y, x, y, 1., LINE_COLOR);
        last_x = x;
        last_y = y;
    }
}

fn draw_data(info: &Renderinfo, mut nn: NN) {
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
        format!("Learning Rate: {LEARNING_RATE:.4}",).as_str(),
        0.,
        45.,
        20.,
        TEXT_COLOR,
    );
    draw_text(
        format!("Training time: {:.4}s", info.training_time).as_str(),
        0.,
        60.,
        20.,
        TEXT_COLOR,
    );

    // Write the testing results at the bottom left
    for i in 0..info.t_input.rows {
        nn.activations[0].data[0][0] = info.t_input.data[i][0];
        nn.activations[0].data[0][1] = info.t_input.data[i][1];

        NN::forward(&mut nn);
        draw_text(
            format!(
                // Input | Output | Expected
                "{:?} -> {:?} | {:?}",
                info.t_input.data[i],
                nn.activations[nn.count - 1].data[0],
                info.t_output.data[i]
            )
            .as_str(),
            0.,
            screen_height() - 10. - i as f32 * 20.,
            20.,
            TEXT_COLOR,
        );
    }
}

fn draw_button(x: f32, y: f32, width: f32, height: f32, text: &str, color: Color) {
    draw_rectangle(x, y, width, height, color);
    let text_width = measure_text(text, None, 20, 1.0).width;
    let text_height = measure_text(text, None, 20, 1.0).height;
    let font_size = 20.0;
    draw_text(
        text,
        x + width / 2.0 - text_width / 2.0,
        y + height / 2.0 - text_height / 2.0 + (font_size / 2.0),
        font_size,
        WHITE,
    );
}