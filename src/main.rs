use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use framework::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup_bevy)
        // .add_system(update_frame)
        .add_system(bevy::window::close_on_esc)
        .run();
}

fn setup_bevy(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());

    // Get the struct
    let mut nn = nn_alloc(&[1, 1]);
    nn_randomize(&mut nn, -1.0, 1.0);

    // {
    //   "count": 2, // number of layers including input and output
    //   "weights": [
    //     {
    //       "rows": 1,
    //       "cols": 1,
    //       "data": [
    //         [0.0]
    //       ]
    //     }
    //   ],
    //   "biases": [
    //     {
    //       "rows": 1,
    //       "cols": 1,
    //       "data": [
    //         [0.0]
    //       ]
    //     }
    //   ],
    //   "activations": [
    //     {
    //       "rows": 1,
    //       "cols": 1,
    //       "data": [
    //         [0.0]
    //       ]
    //     },
    //     {
    //       "rows": 1,
    //       "cols": 1,
    //       "data": [
    //         [0.0]
    //       ]
    //     }
    //   ]
    // }

    // Spawn input circles at the left, hidden layers in the middle, and output circles on the right
    for i in 0..nn.count {
        let x = 0.0;
        let y = (i as f32 - nn.count as f32 / 2.0) * 100.0;
        commands.spawn(MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::default().into()).into(),
            material: materials.add(ColorMaterial::from(Color::WHITE)),
            transform: Transform::from_translation(Vec3::new(x, y, 0.0))
                .with_scale(Vec3::new(75.0, 75.0, 75.0)),
            ..default()
        });
    }
}

fn update_frame(mut commands: Commands) {
    let mut nn = nn_alloc(&[1, 1]);
    nn_randomize(&mut nn, -1.0, 1.0);
    println!("{:?}", nn);
}
