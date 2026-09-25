use bevy::{
    pbr::NotShadowCaster,
    prelude::*,
    render::{
        camera::{ClearColorConfig, RenderTarget, ScalingMode},
        mesh::VertexAttributeValues,
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        view::{NoFrustumCulling, RenderLayers},
    },
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};

#[derive(Clone, Copy)]
struct CameraAnglePreset {
    button_label: &'static str,
    label: &'static str,
    rotation: Vec3,
    key: KeyCode,
}

#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    Orthogonal,
    Perspective,
}

#[derive(PartialEq, Clone, Copy)]
enum SelectionMode {
    None,
    Point,
    Edge,
    Area,
}

struct AppState {
    camera_rotation: Vec3,
    camera_position: Vec3,
    zoom: f32,
    view_mode: ViewMode,
    selection_mode: SelectionMode,
    hovered_point: Option<Vec3>,
    hovered_edge: Option<(Vec3, Vec3)>,
    hovered_area: Option<[Vec3; 3]>,
    camera_angle_presets: [CameraAnglePreset; 6],
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            camera_rotation: Vec3::ZERO,
            camera_position: Vec3::ZERO,
            zoom: 5.0,
            view_mode: ViewMode::Orthogonal,
            selection_mode: SelectionMode::None,
            hovered_point: None,
            hovered_edge: None,
            hovered_area: None,
            hovered_circle: None,
            circle_points: Vec::new(),
            camera_angle_presets: [
                CameraAnglePreset {
                    button_label: "r",
                    label: "right",
                    rotation: Vec3::new(90.0, 0.0, 90.0),
                    key: KeyCode::Digit4,
                },
                CameraAnglePreset {
                    button_label: "t",
                    label: "top",
                    rotation: Vec3::new(0.0, 0.0, 0.0),
                    key: KeyCode::Digit2,
                },
                CameraAnglePreset {
                    button_label: "bo",
                    label: "bottom",
                    rotation: Vec3::new(180.0, 0.0, 0.0),
                    key: KeyCode::Digit3,
                },
                CameraAnglePreset {
                    button_label: "l",
                    label: "left",
                    rotation: Vec3::new(90.0, 0.0, 270.0),
                    key: KeyCode::Digit1,
                },
                CameraAnglePreset {
                    button_label: "f",
                    label: "front",
                    rotation: Vec3::new(90.0, 0.0, 0.0),
                    key: KeyCode::Digit5,
                },
                CameraAnglePreset {
                    button_label: "ba",
                    label: "back",
                    rotation: Vec3::new(90.0, 0.0, 180.0),
                    key: KeyCode::Digit6,
                },
            ],
        }
    }
}

#[derive(Resource)]
struct ViewportImage(Handle<Image>);

#[derive(Resource)]
struct CompassImage(Handle<Image>);

#[derive(Component)]
struct ImportedObject {
    name: String,
    euler_angles: Vec3,
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct CompassCamera;

#[derive(Component)]
struct CompassAxis {
    local_pos: Vec3,
    base_rotation: Quat,
}

#[derive(Component)]
struct CompassLabel {
    local_pos: Vec3,
}

fn ray_triangle_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<f32> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray_dir.cross(edge2);
    let a = edge1.dot(h);
    if a > -0.00001 && a < 0.00001 {
        return None;
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * ray_dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);
    if t > 0.00001 {
        Some(t)
    } else {
        None
    }
}

fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    a + t * ab
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // main camera configuration
    let main_camera_size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };
    let mut main_camera_image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: main_camera_size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    main_camera_image.resize(main_camera_size);
    let main_camera_handle = images.add(main_camera_image);
    commands.insert_resource(ViewportImage(main_camera_handle.clone()));

    // spawn main camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: RenderTarget::Image(main_camera_handle),
                ..default()
            },
            projection: Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical(5.0),
                near: -100000.0,
                far: 100000.0,
                ..default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        MainCamera,
    ));

    // compass camera configuration
    let compass_center = Vec3::new(0.0, 100.0, 0.0);
    let compass_camera_size = Extent3d {
        width: 128,
        height: 128,
        ..default()
    };
    let mut compass_camera_image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: compass_camera_size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    compass_camera_image.resize(compass_camera_size);
    let compass_camera_handle = images.add(compass_camera_image);
    commands.insert_resource(CompassImage(compass_camera_handle.clone()));

    // spawn compass camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: RenderTarget::Image(compass_camera_handle.clone()),
                order: 1,
                clear_color: ClearColorConfig::Custom(Color::srgba(0.0, 0.0, 0.0, 0.0)),
                ..default()
            },
            projection: Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical(2.5),
                ..default()
            }),
            transform: Transform::from_xyz(0.0, 100.0, 5.0).looking_at(compass_center, Vec3::Y),
            ..default()
        },
        RenderLayers::layer(1),
        CompassCamera,
    ));

    // spawn compass scene axis label
    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                target: RenderTarget::Image(compass_camera_handle.clone()),
                order: 2,
                clear_color: ClearColorConfig::None,
                ..default()
            },
            ..default()
        },
        RenderLayers::layer(1),
    ));

    // compass axis
    // ignores light
    // red x
    let compass_x_axis_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.0, 0.0),
        unlit: true,
        ..default()
    });
    // green y
    let compass_y_axis_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 1.0, 0.0),
        unlit: true,
        ..default()
    });
    // blue z
    let compass_z_axis_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.0, 1.0),
        unlit: true,
        ..default()
    });
    // form and dimensions of axis
    let compass_mesh = meshes.add(Cylinder::new(0.015, 0.8));

    // spawn compass x axis
    commands.spawn((
        PbrBundle {
            mesh: compass_mesh.clone(),
            material: compass_x_axis_material,
            transform: Transform::from_rotation(Quat::from_rotation_z(
                -std::f32::consts::FRAC_PI_2,
            ))
            .with_translation(compass_center + Vec3::X * 0.4),
            ..default()
        },
        CompassAxis {
            local_pos: Vec3::X * 0.4,
            base_rotation: Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
        },
        NotShadowCaster,
        RenderLayers::layer(1),
    ));
    // spawn compass y axis
    commands.spawn((
        PbrBundle {
            mesh: compass_mesh.clone(),
            material: compass_y_axis_material,
            transform: Transform::from_translation(compass_center + Vec3::Y * 0.4),
            ..default()
        },
        CompassAxis {
            local_pos: Vec3::Y * 0.4,
            base_rotation: Quat::IDENTITY,
        },
        NotShadowCaster,
        RenderLayers::layer(1),
    ));
    // spawn compass z axis
    commands.spawn((
        PbrBundle {
            mesh: compass_mesh.clone(),
            material: compass_z_axis_material,
            transform: Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
                .with_translation(compass_center + Vec3::Z * 0.4),
            ..default()
        },
        CompassAxis {
            local_pos: Vec3::Z * 0.4,
            base_rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        },
        NotShadowCaster,
        RenderLayers::layer(1),
    ));

    // spawn compass scene axis label
    for (axis_offset, label) in [(Vec3::X, "X"), (Vec3::Y, "Y"), (Vec3::Z, "Z")] {
        commands.spawn((
            Text2dBundle {
                text: Text::from_section(
                    label,
                    TextStyle {
                        font_size: 18.0,
                        color: Color::WHITE,
                        ..default()
                    },
                ),
                ..default()
            },
            CompassLabel {
                local_pos: axis_offset,
            },
            RenderLayers::layer(1),
        ));
    }
}

fn ui_system(
    mut commands: Commands,
    mut contexts: EguiContexts,
    (viewport, compass_res, time, keys, mouse_buttons): (
        Res<ViewportImage>,
        Res<CompassImage>,
        Res<Time>,
        Res<ButtonInput<KeyCode>>,
        Res<ButtonInput<MouseButton>>,
    ),
    (mut images, mut materials, mut meshes): (
        ResMut<Assets<Image>>,
        ResMut<Assets<StandardMaterial>>,
        ResMut<Assets<Mesh>>,
    ),
    mut state: Local<AppState>,
    mut mouse_wheel_events: EventReader<bevy::input::mouse::MouseWheel>,
    mut mouse_motion_events: EventReader<bevy::input::mouse::MouseMotion>,
    mut gizmos: Gizmos,
    mut object_query: Query<
        (
            Entity,
            &mut Transform,
            &mut Visibility,
            &mut ImportedObject,
            &Handle<Mesh>,
        ),
        (Without<Camera>, Without<CompassAxis>, Without<CompassLabel>),
    >,
    mut camera_query: Query<
        (&Camera, &mut Transform, &GlobalTransform, &mut Projection),
        (
            With<MainCamera>,
            Without<CompassAxis>,
            Without<CompassLabel>,
            Without<CompassCamera>,
        ),
    >,
    mut compass_cam_query: Query<
        &mut Transform,
        (
            With<Camera>,
            With<CompassCamera>,
            Without<MainCamera>,
            Without<CompassAxis>,
            Without<CompassLabel>,
        ),
    >,
    mut compass_query: Query<
        (&mut Transform, &CompassAxis),
        (
            Without<Camera>,
            Without<CompassLabel>,
            Without<ImportedObject>,
        ),
    >,
    mut compass_label_query: Query<
        (&mut Transform, &CompassLabel),
        (
            Without<Camera>,
            Without<CompassAxis>,
            Without<ImportedObject>,
        ),
    >,
) {
    let texture_id = contexts.add_image(viewport.0.clone());
    let compass_texture_id = contexts.add_image(compass_res.0.clone());
    let ctx = contexts.ctx_mut();

    let (camera, mut cam_transform, cam_global, mut projection) = camera_query.single_mut();

    let current_rotation = Quat::from_euler(
        EulerRot::XYZ,
        -state.camera_rotation.x.to_radians(),
        state.camera_rotation.y.to_radians(),
        -state.camera_rotation.z.to_radians(),
    );

    let rot_speed = 90.0 * time.delta_seconds();
    let pan_speed = state.zoom * time.delta_seconds();

    let mut delta_rot = Quat::IDENTITY;
    let mut view_pan_delta = Vec3::ZERO;
    let mut zoom_delta = 0.0;

    let mouse_zoom_speed = 0.2;
    let keyboard_zoom_speed = state.zoom;

    for event in mouse_wheel_events.read() {
        zoom_delta -= event.y * mouse_zoom_speed;
    }

    if keys.pressed(KeyCode::NumpadAdd) || keys.pressed(KeyCode::Equal) {
        zoom_delta -= 2.0 * keyboard_zoom_speed * time.delta_seconds();
    }
    if keys.pressed(KeyCode::NumpadSubtract) || keys.pressed(KeyCode::Minus) {
        zoom_delta += 2.0 * keyboard_zoom_speed * time.delta_seconds();
    }

    if zoom_delta != 0.0 {
        state.zoom = (state.zoom + zoom_delta).clamp(1.0, 5000.0);
    }

    let ctrl_pressed = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);

    let mut mouse_delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        mouse_delta += event.delta;
    }

    let egui_wants_pointer = ctx.wants_pointer_input() || ctx.is_pointer_over_area();

    if !egui_wants_pointer && mouse_buttons.pressed(MouseButton::Left) {
        let mouse_pan_speed = 0.01 * state.zoom;
        let mouse_rot_speed = 0.5;

        if ctrl_pressed {
            view_pan_delta.x += mouse_delta.x * mouse_pan_speed;
            view_pan_delta.y -= mouse_delta.y * mouse_pan_speed;
        } else {
            delta_rot =
                Quat::from_axis_angle(Vec3::Y, (mouse_delta.x * mouse_rot_speed).to_radians())
                    * delta_rot;
            delta_rot =
                Quat::from_axis_angle(Vec3::X, (mouse_delta.y * mouse_rot_speed).to_radians())
                    * delta_rot;
        }
    }

    if ctrl_pressed {
        if keys.pressed(KeyCode::Numpad1) {
            view_pan_delta.x += pan_speed;
        }
        if keys.pressed(KeyCode::Numpad3) {
            view_pan_delta.x -= pan_speed;
        }
        if keys.pressed(KeyCode::Numpad4) {
            view_pan_delta.y += pan_speed;
        }
        if keys.pressed(KeyCode::Numpad6) {
            view_pan_delta.y -= pan_speed;
        }
        if keys.pressed(KeyCode::Numpad7) {
            zoom_delta -= 2.0 * keyboard_zoom_speed * time.delta_seconds();
        }
        if keys.pressed(KeyCode::Numpad9) {
            zoom_delta += 2.0 * keyboard_zoom_speed * time.delta_seconds();
        }
    } else {
        for preset in state.camera_angle_presets {
            if keys.pressed(preset.key) {
                state.camera_rotation = preset.rotation;
            }
        }
        if keys.pressed(KeyCode::Numpad1) {
            delta_rot = Quat::from_axis_angle(Vec3::X, -rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad3) {
            delta_rot = Quat::from_axis_angle(Vec3::X, rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad4) {
            delta_rot = Quat::from_axis_angle(Vec3::Y, -rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad6) {
            delta_rot = Quat::from_axis_angle(Vec3::Y, rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad7) {
            delta_rot = Quat::from_axis_angle(Vec3::Z, -rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad9) {
            delta_rot = Quat::from_axis_angle(Vec3::Z, rot_speed.to_radians()) * delta_rot;
        }
    }

    if zoom_delta != 0.0 {
        state.zoom = (state.zoom + zoom_delta).clamp(1.0, 5000.0);
    }

    if view_pan_delta != Vec3::ZERO {
        let dp = current_rotation * -view_pan_delta;
        state.camera_position += dp;
    }

    if let Projection::Orthographic(ortho) = &mut *projection {
        ortho.scaling_mode = ScalingMode::FixedVertical(state.zoom);
    }

    if delta_rot != Quat::IDENTITY {
        let new_rot = delta_rot * current_rotation;
        let (ex, ey, ez) = new_rot.to_euler(EulerRot::XYZ);

        state.camera_rotation.x = -ex.to_degrees();
        state.camera_rotation.y = ey.to_degrees();
        state.camera_rotation.z = -ez.to_degrees();
    }

    let target_rotation = Quat::from_euler(
        EulerRot::XYZ,
        -state.camera_rotation.x.to_radians(),
        state.camera_rotation.y.to_radians(),
        -state.camera_rotation.z.to_radians(),
    );

    cam_transform.rotation = target_rotation;
    cam_transform.translation =
        state.camera_position + target_rotation * Vec3::new(0.0, 0.0, state.zoom);

    if let Ok(mut compass_cam_transform) = compass_cam_query.get_single_mut() {
        let compass_center = Vec3::new(0.0, 100.0, 0.0);
        compass_cam_transform.rotation = target_rotation;
        compass_cam_transform.translation =
            compass_center + target_rotation * Vec3::new(0.0, 0.0, 5.0);
    }

    egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            if ui.button("Import STL").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("STL files", &["stl", "STL"])
                    .pick_file()
                {
                    // Get the file name for the UI label
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    if let Ok(mut file) = std::fs::File::open(&path) {
                        if let Ok(stl) = stl_io::read_stl(&mut file) {
                            let mut new_mesh = Mesh::new(
                                bevy::render::render_resource::PrimitiveTopology::TriangleList,
                                bevy::render::render_asset::RenderAssetUsages::default(),
                            );

                            let mut positions = Vec::with_capacity(stl.faces.len() * 3);
                            let mut uvs = Vec::with_capacity(stl.faces.len() * 3);
                            let mut colors = Vec::with_capacity(stl.faces.len() * 3);

                            for face in stl.faces {
                                let p0 = stl.vertices[face.vertices[0]];
                                let p1 = stl.vertices[face.vertices[1]];
                                let p2 = stl.vertices[face.vertices[2]];

                                let v0 = Vec3::new(p0[0], p0[1], p0[2]);
                                let v1 = Vec3::new(p1[0], p1[1], p1[2]);
                                let v2 = Vec3::new(p2[0], p2[1], p2[2]);

                                let normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();

                                let r = 0.5 + (normal.x * 0.5);
                                let g = 0.5 + (normal.y * 0.5);
                                let b = 0.5 + (normal.z * 0.5);

                                for i in 0..3 {
                                    let vertex = stl.vertices[face.vertices[i]];
                                    positions.push([vertex[0], vertex[1], vertex[2]]);
                                    uvs.push([0.0, 0.0]);
                                    colors.push([r, g, b, 1.0]);
                                }
                            }

                            new_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                            new_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
                            new_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                            new_mesh.compute_flat_normals();

                            // ADDED: Spawn the new imported object dynamically
                            commands.spawn((
                                PbrBundle {
                                    mesh: meshes.add(new_mesh),
                                    material: materials.add(StandardMaterial {
                                        base_color: Color::WHITE,
                                        unlit: true,
                                        ..default()
                                    }),
                                    ..default()
                                },
                                ImportedObject {
                                    name: file_name,
                                    euler_angles: Vec3::ZERO,
                                },
                                NoFrustumCulling,
                            ));
                        } else {
                            println!("Failed to parse STL data.");
                        }
                    } else {
                        println!("Failed to open file.");
                    }
                }
            }
            if ui.button("view").clicked() {
                println!("view clicked!");
            }
            if ui.button("window").clicked() {
                println!("window clicked!");
            }
        });
        ui.add_space(3.0);
    });

    egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
        ui.add_space(3.0);

        ui.horizontal(|ui| {
            for preset in state.camera_angle_presets {
                let tooltip_text = format!("{} (Shortcut: {:?})", preset.label, preset.key);

                if ui
                    .button(preset.button_label)
                    .on_hover_text(tooltip_text)
                    .clicked()
                {
                    state.camera_rotation = preset.rotation;
                }
            }
            if ui.button("p").clicked() {
                state.view_mode = ViewMode::Perspective;
                *projection = Projection::Perspective(PerspectiveProjection {
                    far: 100000.0, // Ensure massive models aren't culled
                    ..default()
                });
            }
            if ui.button("o").clicked() {
                state.view_mode = ViewMode::Orthogonal;
                *projection = Projection::Orthographic(OrthographicProjection {
                    scaling_mode: ScalingMode::FixedVertical(state.zoom),
                    near: -100000.0, // Retain the negative clip plane
                    far: 100000.0,   // Retain the far clip plane
                    ..default()
                });
            }
        });

        ui.horizontal(|ui| {
            ui.label("Selection Mode:");
            ui.radio_value(&mut state.selection_mode, SelectionMode::None, "None");
            ui.radio_value(&mut state.selection_mode, SelectionMode::Point, "Point");
            ui.radio_value(&mut state.selection_mode, SelectionMode::Edge, "Edge");
            ui.radio_value(&mut state.selection_mode, SelectionMode::Area, "Area");
        });

        ui.horizontal(|ui| {
            ui.label(format!(
                "pos: [{:.1},{:.1},{:.1}]",
                state.camera_position.x, state.camera_position.y, state.camera_position.z
            ));
            ui.label(format!(
                "rot: [{:.0},{:.0},{:.0}]",
                state.camera_rotation.x.rem_euclid(360.0),
                state.camera_rotation.y.rem_euclid(360.0),
                state.camera_rotation.z.rem_euclid(360.0)
            ));
            ui.label(format!("zoom: {:.1}", state.zoom));
            ui.separator();

            match state.selection_mode {
                SelectionMode::Point => {
                    if let Some(p) = state.hovered_point {
                        ui.label(format!(
                            "Hovered Point: [{:.2}, {:.2}, {:.2}]",
                            p.x, p.y, p.z
                        ));
                    }
                }
                SelectionMode::Edge => {
                    if let Some((a, b)) = state.hovered_edge {
                        ui.label(format!(
                            "Hovered Edge: A[{:.2},{:.2},{:.2}] B[{:.2},{:.2},{:.2}]",
                            a.x, a.y, a.z, b.x, b.y, b.z
                        ));
                    }
                }
                SelectionMode::Area => {
                    if let Some([a, b, c]) = state.hovered_area {
                        let center_x = (a.x + b.x + c.x) / 3.0;
                        let center_y = (a.y + b.y + c.y) / 3.0;
                        let center_z = (a.z + b.z + c.z) / 3.0;
                        ui.label(format!(
                            "Hovered Area Center: [{:.2}, {:.2}, {:.2}]",
                            center_x, center_y, center_z
                        ));
                    }
                }
                SelectionMode::None => {}
            }
        });
        ui.add_space(3.0);
    });

    let mut view_ray = None;

    // ADDED: The object manager side panel
    egui::SidePanel::right("right_panel")
        .resizable(true)
        .default_width(200.0)
        .show(ctx, |ui| {
            ui.heading("Loaded Objects");
            ui.separator();

            let mut despawn_target = None;

            for (entity, mut transform, mut visibility, mut obj, _) in object_query.iter_mut() {
                let object_name = obj.name.clone();
                ui.collapsing(object_name, |ui| {
                    let mut is_visible = *visibility != Visibility::Hidden;
                    if ui.checkbox(&mut is_visible, "Visible").changed() {
                        *visibility = if is_visible {
                            Visibility::Inherited
                        } else {
                            Visibility::Hidden
                        };
                    }

                    ui.label("Position:");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut transform.translation.x)
                                .speed(0.1)
                                .prefix("X: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut transform.translation.y)
                                .speed(0.1)
                                .prefix("Y: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut transform.translation.z)
                                .speed(0.1)
                                .prefix("Z: "),
                        );
                    });

                    ui.label("Rotation (°):");
                    let mut euler = obj.euler_angles;
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::DragValue::new(&mut euler.x).speed(1.0).prefix("X: "))
                            .changed()
                            || ui
                                .add(egui::DragValue::new(&mut euler.y).speed(1.0).prefix("Y: "))
                                .changed()
                            || ui
                                .add(egui::DragValue::new(&mut euler.z).speed(1.0).prefix("Z: "))
                                .changed()
                        {
                            obj.euler_angles = euler;
                            transform.rotation = Quat::from_euler(
                                EulerRot::XYZ,
                                euler.x.to_radians(),
                                euler.y.to_radians(),
                                euler.z.to_radians(),
                            );
                        }
                    });

                    ui.add_space(5.0);
                    if ui.button("Delete Object").clicked() {
                        despawn_target = Some(entity);
                    }
                });
            }

            if let Some(entity) = despawn_target {
                commands.entity(entity).despawn_recursive();
            }
        });

    egui::CentralPanel::default().show(ctx, |ui| {
        let available_size = ui.available_size();
        let new_width = (available_size.x as u32).max(1);
        let new_height = (available_size.y as u32).max(1);

        if let Some(image) = images.get_mut(&viewport.0) {
            if image.texture_descriptor.size.width != new_width
                || image.texture_descriptor.size.height != new_height
            {
                let new_size = Extent3d {
                    width: new_width,
                    height: new_height,
                    ..image.texture_descriptor.size
                };
                image.texture_descriptor.size = new_size;
                image.resize(new_size);
            }
        }

        let image_response = ui.image(egui::load::SizedTexture::new(texture_id, available_size));
        let rect = image_response.rect;

        if state.selection_mode != SelectionMode::None {
            if let Some(hover_pos) = image_response.hover_pos() {
                let rel_x = hover_pos.x - rect.min.x;
                let rel_y = hover_pos.y - rect.min.y;
                let pixel_x = (rel_x / rect.width()) * new_width as f32;
                let pixel_y = (rel_y / rect.height()) * new_height as f32;

                if let Some(ray3d) =
                    camera.viewport_to_world(cam_global, Vec2::new(pixel_x, pixel_y))
                {
                    view_ray = Some((ray3d.origin, *ray3d.direction));
                }
            }
        }

        let compass_rect = egui::Rect::from_min_size(
            rect.left_bottom() + egui::vec2(15.0, -115.0),
            egui::vec2(100.0, 100.0),
        );
        ui.put(
            compass_rect,
            egui::Image::new(egui::load::SizedTexture::new(
                compass_texture_id,
                compass_rect.size(),
            )),
        );
    });

    let focal_point = state.camera_position;

    // --- RAYCASTING MESH INTERSECTION ---
    state.hovered_point = None;
    state.hovered_edge = None;
    state.hovered_area = None;

    if let Some((ray_origin, ray_dir)) = view_ray {
        let mut closest_hit: Option<(f32, Vec3, Vec3, Vec3)> = None;

        // Iterate through ALL loaded objects
        for (_, transform, visibility, _, mesh_handle) in object_query.iter() {
            // Ignore geometry if the user hid it in the right panel
            if *visibility == Visibility::Hidden {
                continue;
            }

            if let Some(mesh) = meshes.get(mesh_handle) {
                let pos_attr = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
                let positions: &[[f32; 3]] = match pos_attr {
                    VertexAttributeValues::Float32x3(p) => p,
                    _ => &[],
                };

                let indices: Vec<usize> = match mesh.indices() {
                    Some(bevy::render::mesh::Indices::U32(i)) => {
                        i.iter().map(|idx| *idx as usize).collect()
                    }
                    Some(bevy::render::mesh::Indices::U16(i)) => {
                        i.iter().map(|idx| *idx as usize).collect()
                    }
                    None => (0..positions.len()).collect(),
                };

                let obj_matrix = transform.compute_matrix();
                let inverse_matrix = obj_matrix.inverse();

                let ray_origin_local = inverse_matrix.transform_point3(ray_origin);
                let ray_dir_local = inverse_matrix.transform_vector3(ray_dir).normalize();

                for chunk in indices.chunks(3) {
                    let v0 = Vec3::from(positions[chunk[0]]);
                    let v1 = Vec3::from(positions[chunk[1]]);
                    let v2 = Vec3::from(positions[chunk[2]]);

                    if let Some(t) =
                        ray_triangle_intersect(ray_origin_local, ray_dir_local, v0, v1, v2)
                    {
                        // Check if this triangle is closer than any hit from previous objects
                        if closest_hit.is_none() || t < closest_hit.unwrap().0 {
                            let wv0 = obj_matrix.transform_point3(v0);
                            let wv1 = obj_matrix.transform_point3(v1);
                            let wv2 = obj_matrix.transform_point3(v2);
                            closest_hit = Some((t, wv0, wv1, wv2));
                        }
                    }
                }
            }
        }

        // Apply highlights only for the absolute closest geometry found
        if let Some((t, wv0, wv1, wv2)) = closest_hit {
            let hit_pt = ray_origin + ray_dir * t;

            match state.selection_mode {
                SelectionMode::Point => {
                    let d0 = hit_pt.distance_squared(wv0);
                    let d1 = hit_pt.distance_squared(wv1);
                    let d2 = hit_pt.distance_squared(wv2);
                    if d0 <= d1 && d0 <= d2 {
                        state.hovered_point = Some(wv0);
                    } else if d1 <= d0 && d1 <= d2 {
                        state.hovered_point = Some(wv1);
                    } else {
                        state.hovered_point = Some(wv2);
                    }
                }
                SelectionMode::Edge => {
                    let p0 = closest_point_on_segment(hit_pt, wv0, wv1);
                    let p1 = closest_point_on_segment(hit_pt, wv1, wv2);
                    let p2 = closest_point_on_segment(hit_pt, wv2, wv0);
                    let d0 = hit_pt.distance_squared(p0);
                    let d1 = hit_pt.distance_squared(p1);
                    let d2 = hit_pt.distance_squared(p2);
                    if d0 <= d1 && d0 <= d2 {
                        state.hovered_edge = Some((wv0, wv1));
                    } else if d1 <= d0 && d1 <= d2 {
                        state.hovered_edge = Some((wv1, wv2));
                    } else {
                        state.hovered_edge = Some((wv2, wv0));
                    }
                }
                SelectionMode::Area => {
                    state.hovered_area = Some([wv0, wv1, wv2]);
                }
                SelectionMode::None => {}
            }
        }
    }

    // The 0-point origin axes now scale infinitely with the zoom
    let axis_len = state.zoom * 3.0;
    let gray = Color::srgb(0.3, 0.3, 0.3);

    // Positive 0-point axes (White)
    gizmos.line(Vec3::ZERO, Vec3::X * axis_len, Color::WHITE);
    gizmos.line(Vec3::ZERO, Vec3::Y * axis_len, Color::WHITE);
    gizmos.line(Vec3::ZERO, Vec3::Z * axis_len, Color::WHITE);

    // Negative 0-point axes (Gray tone)
    gizmos.line(Vec3::ZERO, Vec3::X * -axis_len, gray);
    gizmos.line(Vec3::ZERO, Vec3::Y * -axis_len, gray);
    gizmos.line(Vec3::ZERO, Vec3::Z * -axis_len, gray);

    // Dynamically calculate coordinate ticks to match real absolute values
    // This snaps to base-10 intervals (ticks at 1s, 10s, 100s, 1000s) based on zoom
    let step_power = (state.zoom / 25.0).log10().floor();
    let step = 10_f32.powf(step_power + 1.0);
    let tick_size = step * 0.15;

    for i in 1..=20 {
        let val = i as f32 * step;

        // X-axis coordinate ticks
        gizmos.line(
            Vec3::new(val, -tick_size, 0.0),
            Vec3::new(val, tick_size, 0.0),
            Color::WHITE,
        );
        gizmos.line(
            Vec3::new(-val, -tick_size, 0.0),
            Vec3::new(-val, tick_size, 0.0),
            gray,
        );

        // Y-axis coordinate ticks
        gizmos.line(
            Vec3::new(-tick_size, val, 0.0),
            Vec3::new(tick_size, val, 0.0),
            Color::WHITE,
        );
        gizmos.line(
            Vec3::new(-tick_size, -val, 0.0),
            Vec3::new(tick_size, -val, 0.0),
            gray,
        );

        // Z-axis coordinate ticks
        gizmos.line(
            Vec3::new(0.0, -tick_size, val),
            Vec3::new(0.0, tick_size, val),
            Color::WHITE,
        );
        gizmos.line(
            Vec3::new(0.0, -tick_size, -val),
            Vec3::new(0.0, tick_size, -val),
            gray,
        );
    }

    // Camera Focal Center Crosshair
    let crosshair_length = state.zoom * 0.1;
    let diag1 = (target_rotation * Vec3::new(1.0, 1.0, 0.0)).normalize();
    let diag2 = (target_rotation * Vec3::new(1.0, -1.0, 0.0)).normalize();
    let cross_color = Color::srgb(0.8, 0.1, 0.1);

    gizmos.line(
        focal_point - diag1 * crosshair_length,
        focal_point + diag1 * crosshair_length,
        cross_color,
    );
    gizmos.line(
        focal_point - diag2 * crosshair_length,
        focal_point + diag2 * crosshair_length,
        cross_color,
    );

    let compass_center = Vec3::new(0.0, 100.0, 0.0);
    for (mut compass_transform, compass_axis) in compass_query.iter_mut() {
        compass_transform.translation = compass_center + compass_axis.local_pos;
        compass_transform.rotation = compass_axis.base_rotation;
    }

    for (mut label_transform, compass_label) in compass_label_query.iter_mut() {
        let view_space_pos = target_rotation.inverse() * compass_label.local_pos;

        let pixel_x = view_space_pos.x * 50.0;
        let pixel_y = view_space_pos.y * 50.0;

        label_transform.translation = Vec3::new(pixel_x, pixel_y, 0.0);
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, ui_system)
        .run();
}
