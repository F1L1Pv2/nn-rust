use bevy::prelude::*;
use framework::*;

fn main() {
    App::new()
        // .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .run();
}

fn setup() {
    let mut nn = nn_alloc(&[1, 1]);
    println!("{:?}", nn);
    nn_randomize(&mut nn, -1.0, 1.0);
    println!("{:?}", nn);
}
