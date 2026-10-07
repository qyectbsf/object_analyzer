use bevy::prelude::*;
use std::path::Path;

pub mod mesh_3mf;
pub mod mesh_scad;
pub mod mesh_stl;

pub fn load_mesh(path: &Path) -> Result<Mesh, String> {
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    let (mut positions, uvs, colors) = if ext == "stl" {
        mesh_stl::load(path)?
    } else if ext == "3mf" {
        mesh_3mf::load(path)?
    } else if ext == "scad" {
        mesh_scad::load(path)?
    } else {
        return Err(format!("Unsupported file extension: {}", ext));
    };

    if positions.is_empty() {
        return Err(format!(
            "{} loaded but zero geometry was found.",
            ext.to_uppercase()
        ));
    }

    // Calculate bounding box to snap the object to the (0,0,0) view center
    let mut min_bound = Vec3::splat(f32::MAX);
    let mut max_bound = Vec3::splat(f32::MIN);
    for p in &positions {
        let v = Vec3::new(p[0], p[1], p[2]);
        min_bound = min_bound.min(v);
        max_bound = max_bound.max(v);
    }

    let center_offset = (min_bound + max_bound) / 2.0;

    // Recenter all points
    for p in &mut positions {
        p[0] -= center_offset.x;
        p[1] -= center_offset.y;
        p[2] -= center_offset.z;
    }

    let mut new_mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );

    new_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    new_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    new_mesh.compute_flat_normals();

    Ok(new_mesh)
}
