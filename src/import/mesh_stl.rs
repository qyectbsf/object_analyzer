use bevy::prelude::*;
use std::path::Path;

pub fn load(path: &Path) -> Result<(Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 4]>), String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Failed to open file: {:?}", e))?;

    let stl =
        stl_io::read_stl(&mut file).map_err(|e| format!("Failed to parse STL data: {:?}", e))?;

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

    Ok((positions, uvs, colors))
}
