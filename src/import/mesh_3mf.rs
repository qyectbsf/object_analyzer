use bevy::prelude::*;
use std::path::Path;

pub fn load(path: &Path) -> Result<(Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 4]>), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Failed to open file: {:?}", e))?;

    let model = lib3mf::Model::from_reader(file)
        .map_err(|e| format!("Failed to parse 3MF data: {:?}", e))?;

    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut colors = Vec::new();

    for obj in &model.resources.objects {
        if let Some(mesh_data) = &obj.mesh {
            for tri in &mesh_data.triangles {
                let p0 = &mesh_data.vertices[tri.v1 as usize];
                let p1 = &mesh_data.vertices[tri.v2 as usize];
                let p2 = &mesh_data.vertices[tri.v3 as usize];

                let v0 = Vec3::new(p0.x as f32, p0.y as f32, p0.z as f32);
                let v1 = Vec3::new(p1.x as f32, p1.y as f32, p1.z as f32);
                let v2 = Vec3::new(p2.x as f32, p2.y as f32, p2.z as f32);

                let normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();
                let r = 0.5 + (normal.x * 0.5);
                let g = 0.5 + (normal.y * 0.5);
                let b = 0.5 + (normal.z * 0.5);

                for p in [p0, p1, p2] {
                    positions.push([p.x as f32, p.y as f32, p.z as f32]);
                    uvs.push([0.0, 0.0]);
                    colors.push([r, g, b, 1.0]);
                }
            }
        }
    }

    Ok((positions, uvs, colors))
}
