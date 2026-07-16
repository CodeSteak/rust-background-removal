use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

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
