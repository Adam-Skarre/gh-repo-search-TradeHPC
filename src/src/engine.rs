use rand::Rng;
use rayon::prelude::*;
use rustacuda::prelude::*;
use std::error::Error;
use tokio::time::{sleep, Duration};

const GPU_KERNEL: &str = r#"
extern "C" __global__ void analyze(
    const float* prices,
    float* signals,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Simple momentum strategy
    if (idx > 0) {
        signals[idx] = prices[idx] - prices[idx-1];
    }
}
"#;

pub fn generate_market_data(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(100.0..200.0)).collect()
}

pub fn gpu_analyze(prices: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST, device)?;

    let module = Module::load_from_string(GPU_KERNEL)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut d_prices = DeviceBuffer::from_slice(prices)?;
    let mut d_signals = DeviceBuffer::from_slice(&vec![0.0; prices.len()])?;

    let block_size = 256;
    let grid_size = (prices.len() + block_size - 1) / block_size;
    let kernel = module.get_function("analyze")?;

    unsafe {
        launch!(kernel<<<grid_size, block_size, 0, stream>>>(
            d_prices.as_device_ptr(),
            d_signals.as_device_ptr(),
            prices.len() as i32
        ))?;
    }

    let mut signals = vec![0.0; prices.len()];
    d_signals.copy_to(&mut signals)?;
    Ok(signals)
}

pub async fn distributed_process(data: Vec<f32>) -> Result<f32, Box<dyn Error>> {
    let chunks: Vec<Vec<f32>> = data.par_chunks(1000)
        .map(|c| c.to_vec())
        .collect();

    let tasks: Vec<_> = chunks.into_iter()
        .map(|chunk| tokio::spawn(async move {
            sleep(Duration::from_micros(10)).await;
            chunk.iter().sum::<f32>()
        }))
        .collect();

    let mut total = 0.0;
    for task in tasks {
        total += task.await?;
    }
    Ok(total)
}
