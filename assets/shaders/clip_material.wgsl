#import bevy_pbr::forward_io::VertexOutput

struct ClipMaterialUniform {
    color: vec4<f32>,
    clip_plane: vec4<f32>,
    enabled: u32,
};

@group(2) @binding(0)
var<uniform> material: ClipMaterialUniform;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if material.enabled != 0u {
        // Calculate distance from the world point to the plane
        let dist = dot(in.world_position.xyz, material.clip_plane.xyz) - material.clip_plane.w;
        
        // If distance is positive, the point is in front of the plane, so discard the pixel
        if dist > 0.0 {
            discard;
        }
    }
    
    // If we are looking at the inside of the mesh, render it as a solid dark color 
    // to simulate a solid cut surface instead of a hollow shell.
    if !is_front {
        return vec4<f32>(0.25, 0.25, 0.25, 1.0);
    }
    
    var output_color = material.color;
    
    #ifdef VERTEX_COLORS
    output_color = output_color * in.color;
    #endif
    
    return output_color;
}