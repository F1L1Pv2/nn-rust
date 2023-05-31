use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "Button Example".to_owned(),
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut button = Button::new(100.0, 100.0, 200.0, 50.0);

    loop {
        clear_background(Color::from_rgba(0, 0, 0, 255));

        if button.draw() {
            println!("Hello, World!");
        }

        next_frame().await
    }
}

struct Button {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    clicked: bool,
}

impl Button {
    fn new(x: f32, y: f32, width: f32, height: f32) -> Button {
        Button {
            x,
            y,
            width,
            height,
            clicked: false,
        }
    }

    fn draw(&mut self) -> bool {
        let mouse_position = mouse_position();

        let is_hovered = self.is_point_inside(mouse_position.0, mouse_position.1);

        let is_pressed = is_mouse_button_down(MouseButton::Left);

        let is_released = !is_pressed && self.clicked;

        if is_hovered && is_released {
            self.clicked = false;
            return true;
        }

        self.clicked = is_pressed && is_hovered;

        draw_rectangle(self.x, self.y, self.width, self.height, WHITE);

        let text = if self.clicked {
            "Click me!"
        } else {
            "Hello, World!"
        };

        let text_width = measure_text(text, None, 20, 1.0).width;

        let text_x = self.x + (self.width - text_width) / 2.0;
        let text_y = self.y + self.height / 2.0 - 10.0;

        draw_text(text, text_x, text_y, 20.0, BLACK);

        false
    }

    fn is_point_inside(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}
