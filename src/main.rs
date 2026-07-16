use anyhow::{Context, Result};
use clap::Parser;

use image::io::Reader as ImageReader;
use image::{imageops, GenericImage, ImageBuffer, Luma, RgbaImage};
use image::{DynamicImage, GenericImageView, ImageFormat};
use infer::image::is_jpeg;

use ndarray::Array;
use once_cell::sync::OnceCell;
use ort::value::Tensor;
use tokio::sync::Mutex;

use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod http;
mod onnx;

static SESSION: OnceCell<Mutex<ort::session::Session>> = OnceCell::new();
static THRESHOLD_BG: OnceCell<f32> = OnceCell::new();
pub(crate) static USE_TILE: OnceCell<bool> = OnceCell::new();

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct App {
    /// Input image files or directories
    #[arg(value_name = "PATH")]
    inputs: Vec<String>,

    /// Single output file path (requires exactly 1 input)
    #[clap(short, long, value_name = "FILE", requires = "inputs")]
    output_file: Option<String>,

    /// Output naming pattern. {stem} = filename without ext, {ext} = original ext
    #[clap(long, default_value = "{stem}.nbg.png")]
    pattern: String,

    /// Output directory for batch processing
    #[clap(short = 'O', long)]
    output_dir: Option<String>,

    #[clap(short, long)]
    verbose: bool,

    #[clap(short, long, default_value = "false")]
    crop: bool,

    #[clap(short = 'S', long, conflicts_with = "inputs")]
    stdin: bool,

    #[clap(short = 's', long, conflicts_with = "output_file")]
    stdout: bool,

    #[clap(short = 'H', long)]
    http: bool,

    #[clap(short, long, default_value = "0.0.0.0")]
    address: String,

    #[clap(short, long, default_value = "9876")]
    port: u16,

    #[clap(long, default_value = "0.5", help = "Alpha floor (0-1)")]
    threshold_bg: f32,

    #[clap(short, long, default_value = "assets/medium.onnx")]
    model: String,

    #[clap(short, long, default_value = "false")]
    tile: bool,
}

#[tokio::main(worker_threads = 10)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = App::parse();
    THRESHOLD_BG.set(args.threshold_bg).ok();

    if args.http {
        http::start_http_server(&args).await?;
    }
    #[cfg(any(feature="model-small", feature="model-medium", feature="model-large"))]
    let mut session = if args.model == "assets/medium.onnx" {
        onnx::onnx_session_embedded()?
    } else {
        onnx::onnx_session(&args.model)?
    };
    #[cfg(not(any(feature="model-small", feature="model-medium", feature="model-large")))]
    let mut session = onnx::onnx_session(&args.model)?;

    let img: Option<DynamicImage> =     if args.stdin {
        let mut buffer = Vec::new();
        io::stdin().read_to_end(&mut buffer)?;
        if is_jpeg(&buffer) {
            Some(
                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Jpeg)
                    .decode()
                    .context("Failed to decode image from stdin")?,
            )
        } else {
            Some(
                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Png)
                    .decode()
                    .context("Failed to decode image from stdin")?,
            )
        }
    } else {
        None
    };

    if img.is_some() {
        let processed_dynamic_img = process_dynamic_image(&mut session, img.unwrap(), args.tile)?;
        if args.crop {
            let mut output_img = processed_dynamic_img.to_rgba8();
            let alpha_bounds = find_alpha_bounds(&output_img);
            if let Some((min_x, min_y, max_x, max_y)) = alpha_bounds {
                let cropped_img =
                    imageops::crop(&mut output_img, min_x, min_y, max_x - min_x, max_y - min_y)
                        .to_image();
                let mut full_cropped_img = ImageBuffer::new(max_x - min_x, max_y - min_y);
                full_cropped_img.copy_from(&cropped_img, 0, 0).ok();

                let mut buffer = Cursor::new(Vec::new());
                full_cropped_img.write_to(&mut buffer, ImageFormat::Png)?;
                let buffer_content = buffer.into_inner();
                io::stdout().write_all(&buffer_content)?;
            }
        } else {
            let mut buffer = Cursor::new(Vec::new());
            processed_dynamic_img.write_to(&mut buffer, ImageFormat::Png)?;
            let buffer_content = buffer.into_inner();
            io::stdout().write_all(&buffer_content)?;
        }
        std::process::exit(0);
    }

    let image_files = collect_inputs(&args.inputs);

    let model_dims = get_model_dims(&session);

    for input_path in &image_files {
        let start_time = Instant::now();
        let input_img = image::open(input_path).unwrap().into_rgba8();

        let output_img = if args.tile {
            process_tiled(&mut session, &input_img, model_dims)?
        } else {
            process_single(&mut session, &input_img, model_dims)?
        };

        let output_path = if let Some(ref out) = args.output_file {
            PathBuf::from(out)
        } else {
            let stem = input_path.file_stem().unwrap().to_str().unwrap();
            let fname = args.pattern.replace("{stem}", stem);
            let parent = if let Some(ref dir) = args.output_dir {
                PathBuf::from(dir)
            } else if args.inputs.is_empty() || args.inputs.iter().any(|p| Path::new(p).is_dir()) {
                PathBuf::from("output_images")
            } else {
                PathBuf::from(".")
            };
            parent.join(fname)
        };

        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        output_img.save(&output_path)?;

        let elapsed_time = start_time.elapsed();
        println!(
            "Processed {} in {}.{:03} seconds",
            input_path.display(),
            elapsed_time.as_secs(),
            elapsed_time.subsec_millis()
        );

        if args.crop {
            let mut output_img = output_img;
            let alpha_bounds = find_alpha_bounds(&output_img);

            if let Some((min_x, min_y, max_x, max_y)) = alpha_bounds {
                let cropped_img =
                    imageops::crop(&mut output_img, min_x, min_y, max_x - min_x, max_y - min_y)
                        .to_image();
                let mut full_cropped_img = ImageBuffer::new(max_x - min_x, max_y - min_y);
                full_cropped_img.copy_from(&cropped_img, 0, 0).ok();

                let mut cropped_path = output_path.clone();
                if let Some(stem) = cropped_path.file_stem() {
                    let new_stem = format!("{}_cropped", stem.to_str().unwrap());
                    cropped_path.set_file_name(new_stem);
                    cropped_path.set_extension("png");
                }
                full_cropped_img.save(cropped_path)?;
            }
        }
    }
    Ok(())
}

fn collect_inputs(paths: &[String]) -> Vec<PathBuf> {
    if paths.is_empty() {
        if let Ok(entries) = fs::read_dir("images") {
            return entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "jpg" || ext == "png")
                        .unwrap_or(false)
                })
                .map(|e| e.path())
                .collect();
        }
        return vec![];
    }

    let mut files = Vec::new();
    for path in paths {
        let p = Path::new(path);
        if p.is_dir() {
            if let Ok(entries) = fs::read_dir(p) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let ep = entry.path();
                    if ep.extension().map(|ext| ext == "jpg" || ext == "png").unwrap_or(false) {
                        files.push(ep);
                    }
                }
            }
        } else if p.is_file() {
            files.push(p.to_path_buf());
        }
    }
    files.sort();
    files
}

fn get_model_dims(session: &ort::session::Session) -> (u32, u32) {
    let shape = session.inputs()[0].dtype().tensor_shape().unwrap();
    (shape[3] as u32, shape[2] as u32)
}

fn run_inference(
    session: &mut ort::session::Session,
    region: &RgbaImage,
) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
    let input_shape: Vec<usize> = session
        .inputs()[0]
        .dtype()
        .tensor_shape()
        .unwrap()
        .iter()
        .map(|&dim| dim as usize)
        .collect();

    let model_w = input_shape[3] as u32;
    let model_h = input_shape[2] as u32;

    let resized = imageops::resize(region, model_w, model_h, imageops::FilterType::Triangle);

    let input_tensor = Array::from_shape_fn(input_shape, |indices| {
        let mean = 128.;
        let std = 256.;
        (resized[(indices[3] as u32, indices[2] as u32)][indices[1]] as f32 - mean) / std
    });

    let inputs = ort::inputs![Tensor::from_array(input_tensor)?];
    let outputs = session.run(inputs)?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let output_dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let array_view = ndarray::ArrayViewD::from_shape(output_dims, data)
        .map_err(|e| anyhow::anyhow!("shape error: {}", e))?;

    let mut alpha_mask = ImageBuffer::new(model_w, model_h);
    for (indices, alpha) in array_view.indexed_iter() {
        alpha_mask.put_pixel(
            indices[3] as u32,
            indices[2] as u32,
            Luma([(*alpha * 255.) as u8]),
        );
    }
    Ok(alpha_mask)
}

fn process_single(
    session: &mut ort::session::Session,
    input_img: &RgbaImage,
    _model_dims: (u32, u32),
) -> Result<RgbaImage> {
    let alpha = run_inference(session, input_img)?;

    let mut upscaled_alpha = imageops::resize(
        &alpha,
        input_img.width(),
        input_img.height(),
        imageops::FilterType::Lanczos3,
    );

    enhance_alpha(&mut upscaled_alpha);

    let mut output = input_img.clone();
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        pixel[3] = upscaled_alpha.get_pixel(x, y)[0];
    }
    Ok(output)
}

fn process_tiled(
    session: &mut ort::session::Session,
    input_img: &RgbaImage,
    model_dims: (u32, u32),
) -> Result<RgbaImage> {
    let (model_w, model_h) = model_dims;
    let img_w = input_img.width();
    let img_h = input_img.height();

    if img_w < model_w || img_h < model_h {
        return process_single(session, input_img, model_dims);
    }

    let alpha_global_lowres = run_inference(session, input_img)?;
    let alpha_global = imageops::resize(&alpha_global_lowres, img_w, img_h, imageops::FilterType::Lanczos3);

    let overlap = model_w / 4;
    let step = model_w - overlap;

    let mut alpha_accum = vec![0.0f32; (img_w * img_h) as usize];
    let mut weight_accum = vec![0.0f32; (img_w * img_h) as usize];

    let mut tile_positions_x: Vec<u32> = Vec::new();
    {
        let mut tx: u32 = 0;
        while tx < img_w {
            let tile_x = tx.min(img_w.saturating_sub(model_w));
            if tile_positions_x.is_empty() || tile_x != *tile_positions_x.last().unwrap() {
                tile_positions_x.push(tile_x);
            }
            tx += step;
        }
    }
    let mut tile_positions_y: Vec<u32> = Vec::new();
    {
        let mut ty: u32 = 0;
        while ty < img_h {
            let tile_y = ty.min(img_h.saturating_sub(model_h));
            if tile_positions_y.is_empty() || tile_y != *tile_positions_y.last().unwrap() {
                tile_positions_y.push(tile_y);
            }
            ty += step;
        }
    }

    for &tile_y in &tile_positions_y {
        for &tile_x in &tile_positions_x {
            let tile_img = input_img
                .view(tile_x, tile_y, model_w, model_h)
                .to_image();

            let alpha = run_inference(session, &tile_img)?;

            let global_tile = alpha_global.view(tile_x, tile_y, model_w, model_h).to_image();

            let calibrated = calibrate_to_global(&alpha, &global_tile, 8);

            let mut enhanced = calibrated;
            enhance_alpha(&mut enhanced);

            for py in 0..model_h {
                for px in 0..model_w {
                    let abs_x = tile_x + px;
                    let abs_y = tile_y + py;
                    if abs_x >= img_w || abs_y >= img_h {
                        continue;
                    }

                    let alpha_val = enhanced.get_pixel(px, py)[0] as f32 / 255.0;

                    let wx = overlap_dist(px, model_w, overlap);
                    let wy = overlap_dist(py, model_h, overlap);
                    let weight = wx.min(wy);

                    let idx = (abs_y * img_w + abs_x) as usize;
                    alpha_accum[idx] += alpha_val * weight;
                    weight_accum[idx] += weight;
                }
            }
        }
    }

    let mut output = input_img.clone();
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let idx = (y * img_w + x) as usize;
        let alpha = if weight_accum[idx] > 0.0 {
            ((alpha_accum[idx] / weight_accum[idx]) * 255.0) as u8
        } else {
            0u8
        };
        pixel[3] = alpha;
    }
    Ok(output)
}

fn calibrate_to_global(
    tile: &ImageBuffer<Luma<u8>, Vec<u8>>,
    global: &ImageBuffer<Luma<u8>, Vec<u8>>,
    block_size: u32,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let w = tile.width();
    let h = tile.height();
    let mut result = ImageBuffer::new(w, h);

    let cols = ((w + block_size - 1) / block_size) as usize + 1;
    let rows = ((h + block_size - 1) / block_size) as usize + 1;
    let mut offsets = vec![0.0f32; cols * rows];

    for r in 0..rows {
        let cy = ((r as u32) * block_size).min(h - 1);
        for c in 0..cols {
            let cx = ((c as u32) * block_size).min(w - 1);
            offsets[r * cols + c] =
                global.get_pixel(cx, cy)[0] as f32 - tile.get_pixel(cx, cy)[0] as f32;
        }
    }

    for py in 0..h {
        let by = py / block_size;
        let fy = (py % block_size) as f32 / (block_size as f32);
        for px in 0..w {
            let bx = px / block_size;
            let fx = (px % block_size) as f32 / (block_size as f32);

            let o00 = offsets[by as usize * cols + bx as usize];
            let o10 = offsets[by as usize * cols + (bx + 1).min(cols as u32 - 1) as usize];
            let o01 = offsets[(by + 1).min(rows as u32 - 1) as usize * cols + bx as usize];
            let o11 = offsets[(by + 1).min(rows as u32 - 1) as usize * cols
                + (bx + 1).min(cols as u32 - 1) as usize];

            let offset = bilinear(o00, o10, o01, o11, fx, fy);
            let calib = (tile.get_pixel(px, py)[0] as f32 + offset).clamp(0.0, 255.0);

            result.put_pixel(px, py, Luma([calib as u8]));
        }
    }

    result
}

fn bilinear(a00: f32, a10: f32, a01: f32, a11: f32, fx: f32, fy: f32) -> f32 {
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    a00 * w00 + a10 * w10 + a01 * w01 + a11 * w11
}

fn enhance_alpha(alpha: &mut ImageBuffer<Luma<u8>, Vec<u8>>) {
    let cutoff = THRESHOLD_BG.get().unwrap_or(&0.2).clamp(0.0, 1.0);
    let scale = 1.0 / (1.0 - cutoff);
    for pixel in alpha.pixels_mut() {
        let a = pixel[0] as f32 / 255.0;
        let enhanced = ((a - cutoff) * scale).clamp(0.0, 1.0);
        pixel[0] = (enhanced * 255.0) as u8;
    }
}

fn overlap_dist(px: u32, dim: u32, overlap: u32) -> f32 {
    let left = px.min(overlap).max(1);
    let right = (dim - px - 1).min(overlap).max(1);
    (left.min(right) as f32 / overlap as f32).clamp(0.0, 1.0)
}

fn find_alpha_bounds(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0;
    let mut min_y = u32::MAX;
    let mut max_y = 0;
    let thres_b = 10u8;

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] > thres_b {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    if min_x <= max_x && min_y <= max_y {
        Some((min_x, min_y, max_x, max_y))
    } else {
        println!("found NONE {:?}", (min_x, min_y, max_x, max_y));
        None
    }
}

fn process_dynamic_image(
    session: &mut ort::session::Session,
    dynamic_img: DynamicImage,
    tile: bool,
) -> Result<DynamicImage, anyhow::Error> {
    let input_img = dynamic_img.into_rgba8();
    let model_dims = get_model_dims(session);

    let output = if tile {
        process_tiled(session, &input_img, model_dims)?
    } else {
        process_single(session, &input_img, model_dims)?
    };

    Ok(DynamicImage::ImageRgba8(output))
}
