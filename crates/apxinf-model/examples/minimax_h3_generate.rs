//! Generate one MiniMax-H3 clip (video and audio) from a prompt and write it as raw files, timing each request.
//!
//! ```text
//! cargo run --release -p apxinf-model --features cuda --example minimax_h3_generate -- \
//!     --package /models/h3-pkg --prompt "A red fox trotting through a snowy pine forest, birds singing." \
//!     --width 768 --height 448 --frames 22 --steps 4 --seed 42 --repeat 2 --output fox
//! ```
//! `--output fox` writes `fox.rgb` (RGB8 frames, `frames x height x width x 3`), `fox.pcm` (little-endian `f32`, interleaved stereo) and
//! `fox.json` with the geometry; the package directory holds `apxinf_h3.json` (see `cuda_minimax_h3::package`). To make a playable file:
//! `ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r 24 -i fox.rgb -f f32le -ar 32000 -ac 2 -i fox.pcm -c:v libx264 -pix_fmt yuv420p -c:a aac fox.mp4`.
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{AutoModel, LoadOptions, VideoRequest};

struct Arguments {
    package: PathBuf,
    /// One or more prompts; request `i` uses prompt `i % len`, so several prompts measure a cold text stage after the first round.
    prompts: Vec<String>,
    width: usize,
    height: usize,
    frames: usize,
    steps: usize,
    seed: u64,
    repeat: usize,
    device: usize,
    autotune: bool,
    output: Option<PathBuf>,
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut args = Arguments {
        package: PathBuf::new(),
        prompts: Vec::new(),
        width: 768,
        height: 448,
        frames: 22,
        steps: 4,
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
            "--prompt" => args.prompts.push(value("--prompt")?),
            "--width" => args.width = number("--width", value("--width")?)?,
            "--height" => args.height = number("--height", value("--height")?)?,
            "--frames" => args.frames = number("--frames", value("--frames")?)?,
            "--steps" => args.steps = number("--steps", value("--steps")?)?,
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
    if args.prompts.is_empty() {
        args.prompts
            .push("A red fox trotting through a snowy pine forest, birds singing.".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments().map_err(|e| {
        format!("{e}\nusage: minimax_h3_generate --package DIR [--prompt TEXT (repeatable)] [--width W --height H] [--frames N] [--steps N] [--seed N] [--repeat N] [--output PREFIX]")
    })?;
    let options = LoadOptions {
        autotune: args.autotune,
        ..LoadOptions::default()
    };
    let started = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(args.device), &args.package, &options)?;
    println!("load {:.1} s", started.elapsed().as_secs_f64());
    let mut last = None;
    for repeat in 0..args.repeat {
        let prompt = &args.prompts[repeat % args.prompts.len()];
        let request = VideoRequest {
            prompt,
            width: args.width,
            height: args.height,
            frames: args.frames,
            seed: args.seed,
            steps: args.steps,
        };
        let started = Instant::now();
        let clip = model.generate_video(&request)?;
        println!(
            "request {repeat}: {}x{} x {} frames + {:.2} s of {}-channel audio in {:.2} s",
            clip.width,
            clip.height,
            clip.frames,
            clip.audio.len() as f64 / clip.audio_channels as f64 / clip.audio_sample_rate as f64,
            clip.audio_channels,
            started.elapsed().as_secs_f64()
        );
        last = Some(clip);
    }
    if let (Some(prefix), Some(clip)) = (&args.output, last) {
        let with = |ext: &str| {
            let mut name = prefix.clone().into_os_string();
            name.push(ext);
            PathBuf::from(name)
        };
        std::fs::File::create(with(".rgb"))?.write_all(&clip.rgb)?;
        let pcm: Vec<u8> = clip.audio.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::File::create(with(".pcm"))?.write_all(&pcm)?;
        std::fs::write(
            with(".json"),
            format!(
                "{{\"width\":{},\"height\":{},\"frames\":{},\"frame_rate\":{},\"audio_sample_rate\":{},\"audio_channels\":{}}}\n",
                clip.width, clip.height, clip.frames, clip.frame_rate, clip.audio_sample_rate, clip.audio_channels
            ),
        )?;
        println!("wrote {0}.rgb, {0}.pcm and {0}.json", prefix.display());
    }
    Ok(())
}
