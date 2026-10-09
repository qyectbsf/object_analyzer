use std::env::temp_dir;
use std::path::Path;
use std::process::Command;

pub fn load(path: &Path) -> Result<(Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 4]>), String> {
    // Define a temporary output file in the OS temp directory
    let out_path = temp_dir().join("rhasta_preview.3mf");

    // Execute openscad strictly for export, using the fast-csg experimental features
    let status = Command::new("openscad_nightly")
        .arg("-o")
        .arg(&out_path)
        .arg("--enable=lazy-union")
        .arg(path)
        .status()
        .map_err(|e| format!("Failed to execute openscad: {}", e))?;

    if !status.success() {
        return Err(
            "OpenSCAD compilation failed. Check your .scad file for syntax errors.".to_string(),
        );
    }

    // Hand the generated temporary file directly to your existing 3MF loader
    crate::import::mesh_3mf::load(&out_path)
}
