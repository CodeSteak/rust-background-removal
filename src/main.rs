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
static THRESHOLD_BG: OnceCell<u8> = OnceCell::new();
pub(crate) static USE_TILE: OnceCell<bool> = OnceCell::new();

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct App {
    #[clap(short, long, value_name = "INPUT_FILE", default_value = "")]
    input_file: String,
    #[clap(
        short = 'I',
        long,
        value_name = "INPUT_FOLDER",
        default_value = "images"
    )]
    input_images_folder: String,
    #[clap(short, long, value_name = "OUTPUT_FILE", default_value = "")]
    output_file: String,
    #[clap(short, long)]
    verbose: bool,
    #[clap(short, long, default_value = "false")]
    crop: bool,
    #[clap(short = 'S', long, conflicts_with("input_file"))]
    stdin: bool,
    #[clap(short = 's', long, conflicts_with("output_file"))]
    stdout: bool,
    #[clap(short = 'H', long)]
    http: bool,
    #[clap(short, long, default_value = "0.0.0.0")]
    address: String,
    #[clap(short, long, default_value = "9876")]
    port: u16,
    #[clap(short, long, default_value = "10")]
    threshold_bg: u8,
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
    let mut session = onnx::onnx_session(&args.model)?;

    let img: Option<DynamicImage> = if args.stdin {
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

    let input_images_folder = Path::new(&args.input_images_folder);
    let output_images_folder = Path::new("output_images");
    fs::create_dir_all(&output_images_folder)?;

    let image_files: Vec<PathBuf>;
    if !args.input_file.is_empty() {
        image_files = vec![PathBuf::from(&args.input_file)];
    } else {
        image_files = fs::read_dir(&input_images_folder)?
            .filter_map(|entry| {
                if let Ok(entry) = entry {
                    if let Some(extension) = entry.path().extension() {
                        if extension == "jpg" || extension == "png" {
                            return Some(entry.path());
                        }
                    }
                }
                None
            })
            .collect();
    }

    let model_dims = get_model_dims(&session);

    for input_img_file in &image_files {
        let output_img_file = output_images_folder.join(
            input_img_file
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
                + "_nbg.png",
        );

        let start_time = Instant::now();
        let input_img = image::open(input_img_file).unwrap().into_rgba8();

        let output_img = if args.tile {
            process_tiled(&mut session, &input_img, model_dims)?
        } else {
            process_single(&mut session, &input_img, model_dims)?
        };

        output_img.save(&output_img_file)?;

        let elapsed_time = start_time.elapsed();
        println!(
            "Processed {} in {}.{:03} seconds",
            input_img_file.display(),
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

                let mut output_img_file_cropped = output_img_file.clone();
                if let Some(_extension) = output_img_file_cropped.extension() {
                    let file_stem = output_img_file_cropped.file_stem().unwrap();
                    let new_file_stem = format!("{}_cropped", file_stem.to_str().unwrap());
                    output_img_file_cropped.set_file_name(new_file_stem);
                }

                if let Some(extension) = output_img_file.extension() {
                    output_img_file_cropped.set_extension(extension);
                }

                full_cropped_img.save(output_img_file_cropped)?;
            }
        }
    }
    Ok(())
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

    let upscaled_alpha = imageops::resize(
        &alpha,
        input_img.width(),
        input_img.height(),
        imageops::FilterType::Lanczos3,
    );

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

    let alpha_global = {
        let alpha = run_inference(session, input_img)?;
        imageops::resize(&alpha, img_w, img_h, imageops::FilterType::Lanczos3)
    };

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

            for py in 0..model_h {
                for px in 0..model_w {
                    let abs_x = tile_x + px;
                    let abs_y = tile_y + py;
                    if abs_x >= img_w || abs_y >= img_h {
                        continue;
                    }

                    let tile_a = alpha.get_pixel(px, py)[0] as f32 / 255.0;
                    let global_a = alpha_global.get_pixel(abs_x, abs_y)[0] as f32 / 255.0;

                    let blended = tile_a * 0.3 + global_a * 0.7;

                    let wx = overlap_dist(px, model_w, overlap);
                    let wy = overlap_dist(py, model_h, overlap);
                    let weight = wx.min(wy);

                    let idx = (abs_y * img_w + abs_x) as usize;
                    alpha_accum[idx] += blended * weight;
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
    let thres_b = THRESHOLD_BG.get().unwrap();

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] > *thres_b {
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
