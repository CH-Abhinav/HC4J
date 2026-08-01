use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};


type Registry = Mutex<HashMap<u64,wgpu::Buffer>>;

static REGISTRY:OnceLock<Registry> =OnceLock::new();
static NEXT_ID:AtomicU64 = AtomicU64::new(1);

pub fn get_registry()->&'static Registry{
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub const HC4J_SUCCESS:i32 =0;
pub const HC4J_ERR_NOT_FOUND:i32 =-1;
pub const HC4J_ERR_INVALID_PARAM:i32 =-2;
pub const HC4J_ERR_GPU_READBACK:i32 =-3;

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_alloc(length:usize) -> u64{
    if length==0 {return 0;}

    let eng=crate::get_engine();
    let size_bytes=match (length as u64).checked_mul(std::mem::size_of::<f32>() as u64){
        Some(bytes)=> bytes,
        None=> return 0,
    };

    let buffer= eng.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HC4J persistent Tensor Buffer"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let id= NEXT_ID.fetch_add(1,Ordering::SeqCst);

    let mut guard = match get_registry().lock(){
        Ok(g)=> g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.insert(id, buffer);
    id
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_write(id: u64, ptr_in: *const f32, length:usize)-> i32{
    if ptr_in.is_null() || length == 0 { return HC4J_ERR_INVALID_PARAM; }

    let eng = crate::get_engine();

    let buffer ={
        let guard= match get_registry().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        match guard.get(&id){
            Some(buf) => buf.clone(),
            None => return HC4J_ERR_NOT_FOUND,
        }
    };

    let slice = unsafe{std::slice::from_raw_parts(ptr_in, length)};
    eng.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(slice));

    HC4J_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_free(id: u64) -> i32{
    let buffer_opt ={
        let mut guard = match get_registry().lock(){
            Ok(g)=>g,
            Err(poisoned)=> poisoned.into_inner(),
        };
        guard.remove(&id)
    };

    if let Some(buffer)= buffer_opt{
        buffer.destroy();
        HC4J_SUCCESS
    } else {
        HC4J_ERR_NOT_FOUND
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_download(id: u64, ptr_out: *mut f32, length:usize)-> i32{
    if ptr_out.is_null() || length == 0 { return HC4J_ERR_INVALID_PARAM; }

    let eng = crate::get_engine();

    let size_bytes = match (length as u64).checked_mul(std::mem::size_of::<f32>() as u64){
        Some(bytes)=> bytes,
        None => return HC4J_ERR_INVALID_PARAM,
    };

    let source_buffer = {
        let guard = match get_registry().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        match guard.get(&id) {
            Some(buf) => buf.clone(),
            None => return HC4J_ERR_NOT_FOUND,
        }
    };

    let staging_buffer = eng.device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("HC4J Download starging buffer"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = eng.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None});
        encoder.copy_buffer_to_buffer(&source_buffer, 0, &staging_buffer, 0, size_bytes);
        eng.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _=sender.send(v);
        });

        eng.device.poll(wgpu::Maintain::Wait);

        match receiver.recv(){
            Ok(Ok(()))=>{
                {
                let data=slice.get_mapped_range();
                unsafe {
                    std::slice::from_raw_parts_mut(ptr_out, length)
                        .copy_from_slice(bytemuck::cast_slice(&data[..]));
                }
            }
            
            staging_buffer.unmap();
            HC4J_SUCCESS
        }
        _=>HC4J_ERR_GPU_READBACK,
    }
}