use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

#[cfg(all(feature = "model-small", not(feature = "model-medium"), not(feature = "model-large")))]
const EMBEDDED_MODEL: &[u8] = include_bytes!("../assets/small.onnx");

#[cfg(all(feature = "model-medium", not(feature = "model-large")))]
const EMBEDDED_MODEL: &[u8] = include_bytes!("../assets/medium.onnx");

#[cfg(feature = "model-large")]
const EMBEDDED_MODEL: &[u8] = include_bytes!("../assets/large.onnx");

pub(crate) fn onnx_session(onnx_model_file: &str) -> ort::Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_execution_providers([
            ort::ep::CUDA::default().build(),
            ort::ep::CoreML::default().build(),
            ort::ep::CPU::default().build(),
        ])?
        .commit_from_file(onnx_model_file)?;
    Ok(session)
}

#[cfg(any(feature = "model-small", feature = "model-medium", feature = "model-large"))]
pub(crate) fn onnx_session_embedded() -> ort::Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_execution_providers([
            ort::ep::CUDA::default().build(),
            ort::ep::CoreML::default().build(),
            ort::ep::CPU::default().build(),
        ])?
        .commit_from_memory(EMBEDDED_MODEL)?;
    Ok(session)
}
