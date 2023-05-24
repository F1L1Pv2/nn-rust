use bevy::prelude::*;
use framework::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn(SpriteBundle {
        texture: asset_server.load("static/image.jpeg"),
        ..default()
    });

    let nn = nn_alloc(&[1, 1]);
    println!("{:?}", nn);
}
