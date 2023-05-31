use macroquad::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;

#[macroquad::main("Threaded UI Example")]
async fn main() {
    // Shared counter value protected by a Mutex
    let counter = Arc::new(Mutex::new(0));

    // Spawn a separate thread to increment the counter
    let counter_thread = {
        let counter = Arc::clone(&counter);
        thread::spawn(move || {
            loop {
                // Increment the counter
                {
                    let mut counter = counter.lock().unwrap();
                    *counter += 1;
                }
            }
        })
    };

    loop {
        clear_background(BLACK);

        // Display the counter value
        let counter_value = {
            let counter = counter.lock().unwrap();
            *counter
        };

        draw_text(
            &format!("Counter: {}", counter_value),
            10.0,
            30.0,
            30.0,
            WHITE,
        );

        next_frame().await;
    }

    // Join the counter thread to ensure it finishes
    counter_thread.join().unwrap();
}
