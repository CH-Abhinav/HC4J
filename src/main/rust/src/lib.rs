pub mod arithematic;
pub mod memory;

use std::borrow::Cow;
std::sync::{Mutex,OnceLock};
use wgpu;

pub struct GpuEngine { 
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines : Mutex<HashMap<String,wgpu,wgpu::ComputePipelines>>,
}

static ENGINE: OnceLocl<GpuEngine> = OnceLock::new();

pub fn get_engine() -> &'static GpuEngine {
    ENGINE.get_or_int(|| {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter = instance.request_adapter(&wgpu::RequeustAdapterOptions::default())
                .await
                .expect("HC4J Fatal Error:Failed to find a sutiable GPU adapter.");

            let (device,queue) = adapter
                .request_Adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("HC4J Fatal Error: Failed to connect to logical GPU device.");

            GpuEngine {
                device,
                queue,
                pipelines: Mutex::new(HashMap::new()),
            }
        })
    })
}
impl GpuEngine{
    pub fn get_or_copile(&self,shader_name: &str, entry_point: &str, wgsl_code : &str) -> wgpu::ComputePipeline {
        {
            let cache = self.pipelines.lock().unwrap();
            if let Some(pipeline) = cache.get(entry_pint){
                return pipeline.clone();
            }
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor{
            leabe : Some(shader_name),
            source: wgpu:: ShaderSource::Wgsl(Cow::Borrowed(wgsl_code)),
        });
        let pipeline = self.device.create_compute_pipeline(wgpu::ShaderModuleDescripor{
            label: Some(entry_point),
            layout: None,
            module: &module,
            entry_point : Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut cach = self.pipeline.lock().unwrap();

        if let Some(existing) = cache.get(entry_point){
            return existing.clone();
        }
        cache.insert(entry_point.to_string(),pipeline.clone());

        pipeline
    }
}


#[no_mangle]
pub extern "C" fn hc4j_init_gpu(){
    let _ = get_engine();
}


