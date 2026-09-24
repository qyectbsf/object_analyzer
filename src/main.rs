use bevy::{
    pbr::NotShadowCaster,
    prelude::*,
    render::{
        camera::{ClearColorConfig, RenderTarget, ScalingMode},
        mesh::VertexAttributeValues,
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        view::RenderLayers,
    },
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};

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
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    zoom: f32,
    view_mode: ViewMode,
    selection_mode: SelectionMode,
    hovered_point: Option<Vec3>,
    hovered_edge: Option<(Vec3, Vec3)>,
    hovered_area: Option<[Vec3; 3]>,
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            rotation_x: 0.0,
            rotation_y: 0.0,
            rotation_z: 0.0,
            pos_x: 0.0,
            pos_y: 0.0,
            pos_z: 0.0,
            zoom: 5.0,
            view_mode: ViewMode::Orthogonal,
            selection_mode: SelectionMode::None,
            hovered_point: None,
            hovered_edge: None,
            hovered_area: None,
        }
    }
}

#[derive(Resource)]
struct ViewportImage(Handle<Image>);

#[derive(Resource)]
struct CompassImage(Handle<Image>);

#[derive(Component)]
struct RotatableCube;

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct AxisMarker {
    local_pos: Vec3,
    base_rotation: Quat,
}

#[derive(Component)]
struct CompassAxis {
    local_pos: Vec3,
    base_rotation: Quat,
}

#[derive(Component)]
struct CompassLabel {
    local_pos: Vec3,
}

// Helper: Möller–Trumbore ray-triangle intersection
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

// Helper: Closest point on a line segment to a target point
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
    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };

    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
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

    image.resize(size);
    let image_handle = images.add(image);
    commands.insert_resource(ViewportImage(image_handle.clone()));

    // spawn the central test cube
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            material: materials.add(Color::srgb(0.8, 0.3, 0.3)),
            transform: Transform::from_xyz(0.0, 0.0, 0.0).with_rotation(Quat::from_euler(
                EulerRot::XYZ,
                0.0_f32.to_radians(),
                0.0_f32.to_radians(),
                0.0_f32.to_radians(),
            )),
            ..default()
        },
        RotatableCube,
    ));

    // spawn light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // spawn main camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: RenderTarget::Image(image_handle),
                ..default()
            },
            projection: Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical(5.0),
                ..default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        MainCamera,
    ));

    // TODO: i may need a 2nd scene for this
    let compass_center = Vec3::new(0.0, 100.0, 0.0);
    let compass_size = Extent3d {
        width: 128,
        height: 128,
        ..default()
    };
    let mut compass_img = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: compass_size,
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
    compass_img.resize(compass_size);
    let compass_handle = images.add(compass_img);
    commands.insert_resource(CompassImage(compass_handle.clone()));

    // spawn compass camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: RenderTarget::Image(compass_handle.clone()),
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
    ));

    // 2d camera the render  axis label of compass camera
    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                target: RenderTarget::Image(compass_handle.clone()),
                order: 2,
                clear_color: ClearColorConfig::None,
                ..default()
            },
            ..default()
        },
        RenderLayers::layer(1),
    ));

    // compass axis colour
    let cx_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.0, 0.0),
        unlit: true,
        ..default()
    });
    let cy_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.8, 0.0),
        unlit: true,
        ..default()
    });
    let cz_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.0, 1.0),
        unlit: true,
        ..default()
    });

    // axis space indicator TODO: select shown indicator based on zoom value
    let compass_mesh = meshes.add(Cylinder::new(0.015, 0.8));

    // spawn compass axis
    // x axis
    commands.spawn((
        PbrBundle {
            mesh: compass_mesh.clone(),
            material: cx_mat,
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
    // y axis
    commands.spawn((
        PbrBundle {
            mesh: compass_mesh.clone(),
            material: cy_mat,
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
    // z axis
    commands.spawn((
        PbrBundle {
            mesh: compass_mesh.clone(),
            material: cz_mat,
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

    // axis label
    let text_style = TextStyle {
        font_size: 18.0,
        color: Color::WHITE,
        ..default()
    };
    for (axis_offset, label) in [(Vec3::X, "X"), (Vec3::Y, "Y"), (Vec3::Z, "Z")] {
        commands.spawn((
            Text2dBundle {
                text: Text::from_section(label, text_style.clone()),
                ..default()
            },
            CompassLabel {
                local_pos: axis_offset,
            },
            RenderLayers::layer(1),
        ));
    }

    // main system axis space indicator colour
    let mat_x = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let mat_y = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let mat_z = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let cyl_mesh = meshes.add(Cylinder::new(0.1, 0.01));

    // TODO: also make this loop bounds dependent an zoom value
    for i in -20..=20 {
        if i == 0 {
            continue;
        }
        let val = i as f32;

        commands.spawn((
            PbrBundle {
                mesh: cyl_mesh.clone(),
                material: mat_x.clone(),
                ..default()
            },
            AxisMarker {
                local_pos: Vec3::X * val,
                base_rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            },
            NotShadowCaster,
        ));
        commands.spawn((
            PbrBundle {
                mesh: cyl_mesh.clone(),
                material: mat_y.clone(),
                ..default()
            },
            AxisMarker {
                local_pos: Vec3::Y * val,
                base_rotation: Quat::IDENTITY,
            },
            NotShadowCaster,
        ));
        commands.spawn((
            PbrBundle {
                mesh: cyl_mesh.clone(),
                material: mat_z.clone(),
                ..default()
            },
            AxisMarker {
                local_pos: Vec3::Z * val,
                base_rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            },
            NotShadowCaster,
        ));
    }
}

fn ui_system(
    mut contexts: EguiContexts,
    (viewport, compass_res, time, keys, mouse_buttons): (
        Res<ViewportImage>,
        Res<CompassImage>,
        Res<Time>,
        Res<ButtonInput<KeyCode>>,
        Res<ButtonInput<MouseButton>>,
    ),
    (mut images, mut materials, meshes): (
        ResMut<Assets<Image>>,
        ResMut<Assets<StandardMaterial>>,
        Res<Assets<Mesh>>,
    ),
    mut state: Local<AppState>,
    mut mouse_wheel_events: EventReader<bevy::input::mouse::MouseWheel>,
    mut mouse_motion_events: EventReader<bevy::input::mouse::MouseMotion>,
    mut gizmos: Gizmos,
    mut cube_query: Query<
        (&mut Transform, &Handle<Mesh>),
        (
            With<RotatableCube>,
            Without<AxisMarker>,
            Without<CompassAxis>,
            Without<CompassLabel>,
        ),
    >,
    mut camera_query: Query<
        (&Camera, &mut Transform, &GlobalTransform, &mut Projection),
        (
            With<MainCamera>,
            Without<RotatableCube>,
            Without<AxisMarker>,
            Without<CompassAxis>,
            Without<CompassLabel>,
        ),
    >,
    mut marker_query: Query<
        (&mut Transform, &AxisMarker, &Handle<StandardMaterial>),
        (
            Without<RotatableCube>,
            Without<Camera>,
            Without<CompassAxis>,
            Without<CompassLabel>,
        ),
    >,
    mut compass_query: Query<
        (&mut Transform, &CompassAxis),
        (
            Without<RotatableCube>,
            Without<Camera>,
            Without<AxisMarker>,
            Without<CompassLabel>,
        ),
    >,
    mut compass_label_query: Query<
        (&mut Transform, &CompassLabel),
        (
            Without<RotatableCube>,
            Without<Camera>,
            Without<AxisMarker>,
            Without<CompassAxis>,
        ),
    >,
) {
    let texture_id = contexts.add_image(viewport.0.clone());
    let compass_texture_id = contexts.add_image(compass_res.0.clone());
    let ctx = contexts.ctx_mut();

    let (camera, mut cam_transform, cam_global, mut projection) = camera_query.single_mut();

    // Compute current rotation at the start of the frame so panning calculates correctly
    let current_rotation = Quat::from_euler(
        EulerRot::XYZ,
        -state.rotation_x.to_radians(),
        state.rotation_y.to_radians(),
        -state.rotation_z.to_radians(),
    );

    let rot_speed = 90.0 * time.delta_seconds();
    let pan_speed = state.zoom * time.delta_seconds();

    let mut delta_rot = Quat::IDENTITY;
    let mut view_pan_delta = Vec3::ZERO; // Represents desired screen-space mouse movement
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
            // Accumulate desired view-space panning
            view_pan_delta.x += mouse_delta.x * mouse_pan_speed;
            view_pan_delta.y -= mouse_delta.y * mouse_pan_speed;
        } else {
            // Rotate around the screen's fixed axes (Y is up, X is right)
            delta_rot =
                Quat::from_axis_angle(Vec3::Y, (mouse_delta.x * mouse_rot_speed).to_radians())
                    * delta_rot;
            delta_rot =
                Quat::from_axis_angle(Vec3::X, (mouse_delta.y * mouse_rot_speed).to_radians())
                    * delta_rot;
        }
    }

    // define movement and rotation via keyboard
    if ctrl_pressed {
        if keys.pressed(KeyCode::Numpad1) {
            view_pan_delta.x -= pan_speed;
        }
        if keys.pressed(KeyCode::Numpad3) {
            view_pan_delta.x += pan_speed;
        }
        if keys.pressed(KeyCode::Numpad4) {
            view_pan_delta.y -= pan_speed;
        }
        if keys.pressed(KeyCode::Numpad6) {
            view_pan_delta.y += pan_speed;
        }
        if keys.pressed(KeyCode::Numpad7) {
            view_pan_delta.z += pan_speed;
        }
        if keys.pressed(KeyCode::Numpad9) {
            view_pan_delta.z -= pan_speed;
        }
    } else {
        if keys.pressed(KeyCode::Numpad1) {
            delta_rot = Quat::from_axis_angle(Vec3::Y, -rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad3) {
            delta_rot = Quat::from_axis_angle(Vec3::Y, rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad4) {
            delta_rot = Quat::from_axis_angle(Vec3::X, rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad6) {
            delta_rot = Quat::from_axis_angle(Vec3::X, -rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad7) {
            delta_rot = Quat::from_axis_angle(Vec3::Z, rot_speed.to_radians()) * delta_rot;
        }
        if keys.pressed(KeyCode::Numpad9) {
            delta_rot = Quat::from_axis_angle(Vec3::Z, -rot_speed.to_radians()) * delta_rot;
        }
    }

    // Transform the 2D view-space pan into a 3D focal offset using the inverse rotation
    if view_pan_delta != Vec3::ZERO {
        let dp = current_rotation.inverse() * -view_pan_delta;
        state.pos_x += dp.x;
        state.pos_y += dp.y;
        state.pos_z += dp.z;
    }

    if let Projection::Orthographic(ortho) = &mut *projection {
        ortho.scaling_mode = ScalingMode::FixedVertical(state.zoom);
    }

    // TODO: understand why this is here
    cam_transform.translation = Vec3::new(state.pos_x, state.pos_y, state.pos_z + state.zoom);

    if delta_rot != Quat::IDENTITY {
        let new_rot = delta_rot * current_rotation;
        let (ex, ey, ez) = new_rot.to_euler(EulerRot::XYZ);

        state.rotation_x = -ex.to_degrees();
        state.rotation_y = ey.to_degrees();
        state.rotation_z = -ez.to_degrees();
    }

    // Recompute target_rotation in case mouse dragging changed it this frame
    let target_rotation = Quat::from_euler(
        EulerRot::XYZ,
        -state.rotation_x.to_radians(),
        state.rotation_y.to_radians(),
        -state.rotation_z.to_radians(),
    );

    // display top panel buttons
    // TODO: implement when i reach that point in development
    egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            if ui.button("file").clicked() {
                println!("file clicked!");
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

    // display bottom panel buttons and status information
    egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
        ui.add_space(3.0);

        ui.horizontal(|ui| {
            if ui.button("l").clicked() {
                state.rotation_x = 90.0;
                state.rotation_y = 0.0;
                state.rotation_z = 270.0;
            }
            if ui.button("t").clicked() {
                state.rotation_x = 0.0;
                state.rotation_y = 0.0;
                state.rotation_z = 0.0;
            }
            if ui.button("b").clicked() {
                state.rotation_x = 180.0;
                state.rotation_y = 0.0;
                state.rotation_z = 0.0;
            }
            if ui.button("r").clicked() {
                state.rotation_x = 90.0;
                state.rotation_y = 0.0;
                state.rotation_z = 90.0;
            }
            if ui.button("f").clicked() {
                state.rotation_x = 90.0;
                state.rotation_y = 0.0;
                state.rotation_z = 0.0;
            }
            if ui.button("b").clicked() {
                state.rotation_x = 90.0;
                state.rotation_y = 0.0;
                state.rotation_z = 180.0;
            }
            if ui.button("p").clicked() {
                state.view_mode = ViewMode::Perspective;
                *projection = Projection::Perspective(PerspectiveProjection::default());
            }
            if ui.button("o").clicked() {
                state.view_mode = ViewMode::Orthogonal;
                *projection = Projection::Orthographic(OrthographicProjection {
                    scaling_mode: ScalingMode::FixedVertical(state.zoom),
                    ..default()
                });
            }
        });

        // The Raycasting Selection Mode Tools
        ui.horizontal(|ui| {
            ui.label("Selection Mode:");
            ui.radio_value(&mut state.selection_mode, SelectionMode::None, "None");
            ui.radio_value(&mut state.selection_mode, SelectionMode::Point, "Point");
            ui.radio_value(&mut state.selection_mode, SelectionMode::Edge, "Edge");
            ui.radio_value(&mut state.selection_mode, SelectionMode::Area, "Area");
        });

        // The Output Label for the Highlighted Data
        ui.horizontal(|ui| {
            ui.label(format!(
                "pos: [{:.1},{:.1},{:.1}]",
                state.pos_x, state.pos_y, state.pos_z
            ));
            ui.label(format!(
                "rot: [{:.0},{:.0},{:.0}]",
                state.rotation_x.rem_euclid(360.0),
                state.rotation_y.rem_euclid(360.0),
                state.rotation_z.rem_euclid(360.0)
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

        // Generate Ray from UI Hover Position
        if state.selection_mode != SelectionMode::None {
            if let Some(hover_pos) = image_response.hover_pos() {
                // Determine the pixel coordinate strictly over the logical viewport image
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

    // Calculate the mathematical pivot center for the camera and the world
    let focal_point = Vec3::new(state.pos_x, state.pos_y, state.pos_z);

    // The cube and axes naturally rest at (0,0,0). Orbit that origin point around the focal point.
    let world_origin = focal_point + target_rotation * (-focal_point);

    // --- RAYCASTING MESH INTERSECTION ---
    state.hovered_point = None;
    state.hovered_edge = None;
    state.hovered_area = None;

    if let (Some((ray_origin, ray_dir)), Ok((_, mesh_handle))) = (view_ray, cube_query.get_single())
    {
        if let Some(mesh) = meshes.get(mesh_handle) {
            let pos_attr = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
            let positions: &[[f32; 3]] = match pos_attr {
                VertexAttributeValues::Float32x3(p) => p,
                _ => &[],
            };

            let indices: Vec<usize> = match mesh.indices().unwrap() {
                bevy::render::mesh::Indices::U32(i) => i.iter().map(|idx| *idx as usize).collect(),
                bevy::render::mesh::Indices::U16(i) => i.iter().map(|idx| *idx as usize).collect(),
            };

            // Create the matrix using the current frame's position to prevent 1-frame raycast lag
            let cube_matrix = Mat4::from_rotation_translation(target_rotation, world_origin);
            let inverse_matrix = cube_matrix.inverse();

            let ray_origin_local = inverse_matrix.transform_point3(ray_origin);
            let ray_dir_local = inverse_matrix.transform_vector3(ray_dir).normalize();

            let mut closest_hit: Option<(f32, Vec3, Vec3, Vec3)> = None;

            for chunk in indices.chunks(3) {
                let v0 = Vec3::from(positions[chunk[0]]);
                let v1 = Vec3::from(positions[chunk[1]]);
                let v2 = Vec3::from(positions[chunk[2]]);

                if let Some(t) = ray_triangle_intersect(ray_origin_local, ray_dir_local, v0, v1, v2)
                {
                    if closest_hit.is_none() || t < closest_hit.unwrap().0 {
                        let wv0 = cube_matrix.transform_point3(v0);
                        let wv1 = cube_matrix.transform_point3(v1);
                        let wv2 = cube_matrix.transform_point3(v2);
                        closest_hit = Some((t, wv0, wv1, wv2));
                    }
                }
            }

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
    }

    // Main scene Gizmos for the Axes Grid
    let local_x = target_rotation * Vec3::X;
    let local_y = target_rotation * Vec3::Y;
    let local_z = target_rotation * Vec3::Z;
    let length = 20.0;

    // Draw the main white origin axes pivoting around the world origin
    gizmos.line(
        world_origin + local_x * -length,
        world_origin + local_x * length,
        Color::srgb(1.0, 1.0, 1.0),
    );
    gizmos.line(
        world_origin + local_y * -length,
        world_origin + local_y * length,
        Color::srgb(1.0, 1.0, 1.0),
    );
    gizmos.line(
        world_origin + local_z * -length,
        world_origin + local_z * length,
        Color::srgb(1.0, 1.0, 1.0),
    );

    // Camera Focal Center Crosshair along the cube's space diagonals (corners)
    let crosshair_length = 50.0;
    let diag1 = (target_rotation * Vec3::new(1.0, 1.0, 1.0)).normalize();
    let diag2 = (target_rotation * Vec3::new(1.0, -1.0, 1.0)).normalize();
    let diag3 = (target_rotation * Vec3::new(1.0, 1.0, -1.0)).normalize();
    let diag4 = (target_rotation * Vec3::new(-1.0, 1.0, 1.0)).normalize();
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
    gizmos.line(
        focal_point - diag3 * crosshair_length,
        focal_point + diag3 * crosshair_length,
        cross_color,
    );
    gizmos.line(
        focal_point - diag4 * crosshair_length,
        focal_point + diag4 * crosshair_length,
        cross_color,
    );

    // Draw the active raycast highlights on top of the geometry
    if let Some(p) = state.hovered_point {
        gizmos.sphere(p, Quat::IDENTITY, 0.05, Color::srgb(1.0, 1.0, 0.0));
    }
    if let Some((a, b)) = state.hovered_edge {
        gizmos.line(a, b, Color::srgb(1.0, 1.0, 0.0));
    }
    if let Some([a, b, c]) = state.hovered_area {
        gizmos.line(a, b, Color::srgb(1.0, 1.0, 0.0));
        gizmos.line(b, c, Color::srgb(1.0, 1.0, 0.0));
        gizmos.line(c, a, Color::srgb(1.0, 1.0, 0.0));
    }

    let cam_forward = cam_transform.rotation * Vec3::NEG_Z;

    for (mut marker_transform, marker, mat_handle) in marker_query.iter_mut() {
        // Offset the markers from the orbiting world_origin
        marker_transform.translation = world_origin + target_rotation * marker.local_pos;
        marker_transform.rotation = target_rotation * marker.base_rotation;

        let cylinder_axis = marker_transform.rotation * Vec3::Y;
        let alignment = cylinder_axis.dot(cam_forward).abs();
        let alpha = (1.0 - alignment.powi(4)).clamp(0.0, 1.0);

        if let Some(mat) = materials.get_mut(mat_handle) {
            mat.base_color.set_alpha(alpha);
        }
    }

    let compass_center = Vec3::new(0.0, 100.0, 0.0);
    for (mut compass_transform, compass_axis) in compass_query.iter_mut() {
        compass_transform.translation = compass_center + target_rotation * compass_axis.local_pos;
        compass_transform.rotation = target_rotation * compass_axis.base_rotation;
    }

    for (mut label_transform, compass_label) in compass_label_query.iter_mut() {
        let rotated_pos = target_rotation * compass_label.local_pos;

        // View from +Z means X dictates screen width and Y dictates screen height.
        // Both are positive mappings.
        let pixel_x = rotated_pos.x * 50.0;
        let pixel_y = rotated_pos.y * 50.0; // CHANGED: Removed the minus sign here

        label_transform.translation = Vec3::new(pixel_x, pixel_y, 0.0);
    }

    if let Ok((mut transform, _)) = cube_query.get_single_mut() {
        transform.rotation = target_rotation;
        // Orbit the cube's position perfectly around the focal point
        transform.translation = world_origin;
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
