//! Generate one MiniMax-H3-Image picture from a prompt and write it as a binary PPM (`P6`), timing each request.
//!
//! ```text
//! cargo run --release -p apxinf-model --features cuda --example minimax_h3_image_generate -- \
//!     --package /models/h3-image-int8 --prompt "A red fox in the snow." --size 1024 --seed 42 --repeat 2 --output fox.ppm
//! ```
//! The package directory holds `apxinf_h3_image.json` (see `cuda_minimax_h3_image::package`).
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::cuda_minimax_h3_image::PDD_STEPS;
use apxinf_model::{AutoModel, ImageRequest, LoadOptions};

struct Arguments {
    package: PathBuf,
    prompt: String,
    width: usize,
    height: usize,
    seed: u64,
    repeat: usize,
    device: usize,
    autotune: bool,
    output: Option<PathBuf>,
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut args = Arguments {
        package: PathBuf::new(),
        prompt: "A red fox trotting through a snowy pine forest.".into(),
        width: 1024,
        height: 1024,
        seed: 42,
        repeat: 1,
        device: 0,
        autotune: true,
        output: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        let number = |name: &str, text: String| {
            text.parse::<usize>()
                .map_err(|_| format!("{name}: `{text}` is not a number"))
        };
        match flag.as_str() {
            "--package" => args.package = PathBuf::from(value("--package")?),
            "--prompt" => args.prompt = value("--prompt")?,
            "--size" => {
                let size = number("--size", value("--size")?)?;
                (args.width, args.height) = (size, size);
            }
            "--width" => args.width = number("--width", value("--width")?)?,
            "--height" => args.height = number("--height", value("--height")?)?,
            "--seed" => {
                args.seed = value("--seed")?
                    .parse()
                    .map_err(|_| "--seed must be an integer".to_string())?
            }
            "--repeat" => args.repeat = number("--repeat", value("--repeat")?)?.max(1),
            "--device" => args.device = number("--device", value("--device")?)?,
            "--no-autotune" => args.autotune = false,
            "--output" => args.output = Some(PathBuf::from(value("--output")?)),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if args.package.as_os_str().is_empty() {
        return Err("--package <dir> is required".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments().map_err(|e| format!("{e}\nusage: minimax_h3_image_generate --package DIR [--prompt TEXT] [--size N | --width W --height H] [--seed N] [--repeat N] [--output FILE.ppm]"))?;
    let options = LoadOptions {
        autotune: args.autotune,
        ..LoadOptions::default()
    };
    let started = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(args.device), &args.package, &options)?;
    println!("load {:.1} s", started.elapsed().as_secs_f64());
    let LoadedImage(runtime) = LoadedImage::from(&mut model)?;
    let request = ImageRequest {
        prompt: &args.prompt,
        width: args.width,
        height: args.height,
        seed: args.seed,
        steps: PDD_STEPS,
    };
    let mut last = None;
    for repeat in 0..args.repeat {
        let started = Instant::now();
        let image = runtime.generate(&request)?;
        println!(
            "request {repeat}: {}x{} in {:.2} s",
            image.width,
            image.height,
            started.elapsed().as_secs_f64()
        );
        last = Some(image);
    }
    if let (Some(path), Some(image)) = (&args.output, last) {
        let mut file = std::fs::File::create(path)?;
        write!(file, "P6\n{} {}\n255\n", image.width, image.height)?;
        file.write_all(&image.rgb)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

struct LoadedImage<'a>(&'a mut dyn apxinf_model::ImageRuntime);

impl<'a> LoadedImage<'a> {
    fn from(model: &'a mut apxinf_model::LoadedModel) -> Result<Self, Box<dyn std::error::Error>> {
        match model {
            apxinf_model::LoadedModel::Image(runtime) => Ok(Self(&mut **runtime)),
            _ => Err("the package did not load as an image model".into()),
        }
    }
}
