use crate::get_engine;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug,bytemuck::Pod,bytemuck::Zeroable)]
pub struct ElemDims {
    pub length : u32,
    pub rank : u32,
    pub is_contigous : u32,
    pub _pad1 : u32,

    pub shape: [u32 ; 8],
    pub strides_a : [u32;8],
    pub strides_b : [u32;8],
    pub strides_c : [u32;8],
}
pub const ELEM_SHADER : &str = r#";
struct Dims{
    length : u32,
    rank : u32,
    is_contigous : u32,
    _p1 : u32,
    shape : array<u32,8>,
    strides_a : array<u32,8>,
    strides_b : array<u32,8>,
    strides_c : array<u32,8>,
}
@group(0) @binding(0) var<storage,read> arrayA : array<f32>;
@group(0) @binding(1) var<storage,read> arrayB : array<f32>;
@group(0) @binding(2) var<storage,read_write> arrayC : array<f32>;
@group(0) @binding(3) var<uniform> dims : Dims;

fn get_indices(logical_idx : u32) -> vec3<u32> { 
    var remaining = logical_idx;
    var offset_a = 0u;
    var offset_b = 0u;
    var offset_c = 0u;
    for(var d_idx = 0u ; d_idx < dims.rank; d_idx = d_idx + 1u){
        let d = dims.rank - 1u - d_idx;
        let dim_size = dims.shape[d];

        if(dim_size == 0u) { continue; }

        let coord = remaining % dim_size;
        remaining = remaining / dim_size;

        offset_a = offset_a + coord * dims.strides_a[d];
        offset_b = offset_b + coord * dims.strides_b[d];    
        offset_c = offset_c + coord*dims.strides_c[d];
    }
    return vec3<u32>(offset_a,offset_b,offset_c);
}

@compute @workgroup_size(256)
fn add_main(@builtin(global_invocation_id) gid : vec3<u32>){
    if(dims.is_contigous == 1u){
        let base = gid.x * 4u;
        if(base + 3u < dims.length){
            let val_a = vec4<f32>(arrayA[base],arrayA[base+1],arrayA[base+2],arrayA[base+3]);
            let val_b = vec4<f32>(arrayB[base],arrayB[base+1],arrayB[base+2],arrayB[base+3]);
            let res = val_a + val_b;
            arrayC[base] = res.x;
            arrayC[base+1] = res.y;
            arrayC[base+2] = res.z;
            arrayC[base+3] = res.w;
        }
        else {
         for(var i = base ; i <dims.length ;i=i + 1u){
                arrayC[i] = arrayA[i] + arrayB[i];
         }
    }
    else { 
        let i = gid.x;
        if(i < dims.length){
            let idxs = get_indices(i);
            arrayC[idxs.z] = arrayA[idxs.x] + arrayB[idxs.y];
        }
    }
}
@compute @workgroup_size(256)
fn sub_main(global_invocation_id gid : vec3<u32>){
       if(dims.is_contigous == 1u){
       let base = gid.x * 4u;
       if(base + 3u < dims.length){
         let val_a = vec4<f32>(arrayA[base],arrayA[base+1],arrayA[base+2],arrayA[base+3]);
         let val_b = vec4<f32>(arrayB[base],arrayB[base+1],arrayB[base+2],arrayB[base+3]);
         let res = val_a - val_b;
         arrayC[base] = res.x;
         arrayC[base+1] = res.y;
         arrayC[base+2]=res.z;
         arrayC[base+3] = res.w;
       }
         else{
           for(var i = base ; i<dims.length ;i = i + 1u){
           arrayC[i] = arrayA[i]-arrayB[i]; // tailing loop
           }
         }
        else{
        let i = gid.x;
        if(i < dims.length){
          let idxs = get_indices(i);
          arrayC[idxs.z] = arrayA[idxs.x] - arrayB[idsx.y];
          }
       }
    }
@compute @workgroup_size(256)
fn mul_main(@builtin(gloabal_invocation_id) gid: vec3<u32>){
if(dims.is_contigous == 1u){
   let base = gid.x * 4u;
   if(base + 3u < dims.length){
   let val_a = vec4<f32>(arrayA[base],arrayA[base+1],arrayA[base+2],arrayA[base+3]);
   let val_b = vec4<f32>(arrayB[base],arrayB[base+1],arrayB[base+2],arrayB[base+3]);
   let res = val_a * val_b;
   arrayC[base] = res.x;
   arrayC[base+1] = res.y;
   arrayC[base+2] = res.z;
   arrayC[base+3] = res.w;
   }
   else{
       for(var i = base ; i< dims.length ;i = i +1u){
        arrayC[i] = arrayA[i] * arrayB[i];
       }
   }
   else{
   let i = gid.x;
   if(i < dims.length){
   let idxs = get_indices(i);
   arrayC[idxs.z] = arrayA[idxs.x]*arrayB[idxs.y];
}
}
}
@compute @workgroup_size(256)
fn div_main(@builtin(gloabal_invocation_id) gid: vec3<u32>){
if(dims.is_contigous == 1u){
   let base = gid.x * 4u;
   if(base + 3u < dims.length){
   let val_a = vec4<f32>(arrayA[base],arrayA[base+1],arrayA[base+2],arrayA[base+3]);
   let val_b = vec4<f32>(arrayB[base],arrayB[base+1],arrayB[base+2],arrayB[base+3]);
   let res = val_a / val_b;
   arrayC[base] = res.x;
   arrayC[base+1] = res.y;
   arrayC[base+2] = res.z;
   arrayC[base+3] = res.w;
   }
   else{
       for(var i = base ; i< dims.length ;i = i +1u){
        arrayC[i] = arrayA[i] / arrayB[i];
       }
   }
   else{
   let i = gid.x;
   if(i < dims.length){
   let idxs = get_indices(i);
   arrayC[idxs.z] = arrayA[idxs.x]/arrayB[idxs.y];
}
}
}
"#;














