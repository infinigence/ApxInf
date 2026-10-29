use apxinf_serving::{
    diagnostics, gateway,
    host_pressure::HostPressurePolicy,
    supervisor::{self, Config, WorkerRuntime},
};
use clap::Parser;
use futures_util::FutureExt;
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    process::ExitCode,
    time::Duration,
};

const HTTP_DRAIN_TIMEOUT: Duration = Duration::from_secs(20);

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;

#[derive(Parser)]
#[command(about = "Serve one local model through bounded HTTP APIs.")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "apxinf-local")]
    model_id: String,
    #[arg(long, default_value = "python-mlx-lm", value_parser = ["python-mlx-lm", "mlx-vlm-minicpmv47", "python-clef", "native-minicpm5", "native-minicpm5-prepared", "native-minicpm5-sampled", "native-minicpm5-thinking", "native-minicpm5-thinking-bounded"])]
    provider: String,
    #[arg(long)]
    native_worker: Option<PathBuf>,
    #[arg(long)]
    metal_library: Option<PathBuf>,
    #[arg(long)]
    python: Option<PathBuf>,
    #[arg(long)]
    clef_loader: Option<PathBuf>,
    #[arg(long)]
    worker: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value_t = 16384)]
    max_context: usize,
    #[arg(long, default_value_t = 2048)]
    max_output_tokens: usize,
    #[arg(long, default_value_t = 256)]
    prefill_step_size: usize,
    #[arg(long, default_value_t = 1)]
    output_batch_tokens: usize,
    #[arg(long, default_value_t = 16)]
    queue_capacity: usize,
    #[arg(long, default_value_t = 300)]
    timeout_seconds: u64,
    #[arg(long, default_value_t = 10737418240)]
    memory_budget_bytes: u64,
    #[arg(long, default_value_t = 1073741824)]
    sequence_reservation_bytes: u64,
    #[arg(long)]
    allocator_cache_limit_bytes: Option<u64>,
    #[arg(long, value_enum, default_value = "macos")]
    host_pressure_policy: HostPressurePolicy,
}

#[tokio::main]
async fn main() -> ExitCode {
    let exit = match Args::try_parse() {
        Ok(args) => match run(args).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                diagnostics::emit(format_args!("ApxInf failed: {error}"));
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            if error.use_stderr() {
                diagnostics::emit(format_args!("{error}"));
                ExitCode::from(error.exit_code() as u8)
            } else if error.print().is_err() {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
    };
    diagnostics::finish().await;
    exit
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    diagnostics::initialize()?;
    let mut shutdown = Box::pin(termination_signal()?);
    if !args.host.is_loopback() {
        return Err("This local profile requires a loopback address.".into());
    }
    if args.queue_capacity == 0
        || args.queue_capacity > 256
        || args.max_context == 0
        || args.max_context > 131072
        || args.max_output_tokens > 65536
        || args.output_batch_tokens == 0
        || args.output_batch_tokens > 256
        || args.prefill_step_size == 0
        || args.timeout_seconds == 0
        || args.timeout_seconds > 3600
    {
        return Err("A service limit is outside the supported range.".into());
    }
    if args.provider != "python-clef" && args.clef_loader.is_some() {
        return Err("The --clef-loader option requires --provider python-clef.".into());
    }
    if args.allocator_cache_limit_bytes.is_some() && args.provider != "native-minicpm5-prepared" {
        return Err("Allocator cache limit requires native-minicpm5-prepared.".into());
    }
    let runtime = if matches!(
        args.provider.as_str(),
        "native-minicpm5"
            | "native-minicpm5-prepared"
            | "native-minicpm5-sampled"
            | "native-minicpm5-thinking"
            | "native-minicpm5-thinking-bounded"
    ) {
        let executable = std::fs::canonicalize(
            args.native_worker
                .ok_or("Native provider requires --native-worker.")?,
        )?;
        let metal_library = std::fs::canonicalize(
            args.metal_library
                .ok_or("Native provider requires --metal-library.")?,
        )?;
        if args.provider == "native-minicpm5-prepared" {
            WorkerRuntime::NativeMiniCpm5Prepared { executable, metal_library,
                cache_limit_bytes: args.allocator_cache_limit_bytes.unwrap_or(2147483648) }
        } else if args.provider == "native-minicpm5-thinking-bounded" {
            WorkerRuntime::NativeMiniCpm5BoundedThinking {
                executable,
                metal_library,
            }
        } else if args.provider == "native-minicpm5-thinking" {
            WorkerRuntime::NativeMiniCpm5Thinking {
                executable,
                metal_library,
            }
        } else if args.provider == "native-minicpm5-sampled" {
            WorkerRuntime::NativeMiniCpm5Sampled {
                executable,
                metal_library,
            }
        } else {
            WorkerRuntime::NativeMiniCpm5 {
                executable,
                metal_library,
            }
        }
    } else {
        if args.native_worker.is_some() || args.metal_library.is_some() {
            return Err("Native runtime paths require a native MiniCPM provider.".into());
        }
        if args.provider == "python-clef" {
            if !(1..=4096).contains(&args.max_context) || args.max_output_tokens != 0 {
                return Err(
                    "Clef requires --max-context at most 4096 and --max-output-tokens 0.".into(),
                );
            }
            WorkerRuntime::PythonClef {
                loader: std::fs::canonicalize(
                    args.clef_loader
                        .ok_or("The python-clef provider requires --clef-loader.")?,
                )?,
            }
        } else if args.provider == "mlx-vlm-minicpmv47" {
            WorkerRuntime::PythonMiniCpmVision
        } else {
            WorkerRuntime::Python
        }
    };
    let (python, worker) = worker_paths(&runtime, args.python, args.worker)?;
    let service = supervisor::start_with_shutdown(
        Config {
            runtime,
            python,
            worker,
            model: std::fs::canonicalize(args.model)?,
            model_id: args.model_id,
            max_context: args.max_context,
            max_output: args.max_output_tokens,
            prefill_step_size: args.prefill_step_size,
            output_batch_tokens: args.output_batch_tokens,
            queue_capacity: args.queue_capacity,
            timeout: Duration::from_secs(args.timeout_seconds),
            memory_budget: args.memory_budget_bytes,
            sequence_reservation: args.sequence_reservation_bytes,
            host_pressure_policy: args.host_pressure_policy,
        },
        shutdown.as_mut(),
    )
    .await?;
    let Some(service) = service else {
        return Ok(());
    };
    let shutdown = shutdown.shared();
    let http_shutdown = shutdown.clone();
    let serving = async {
        let listener = tokio::net::TcpListener::bind(SocketAddr::new(args.host, args.port)).await?;
        diagnostics::emit(format_args!(
            "ApxInf listener bound at http://{}",
            listener.local_addr()?
        ));
        let shutdown_service = service.clone();
        axum::serve(listener, gateway::router(service.clone()))
            .with_graceful_shutdown(async move {
                http_shutdown.await;
                shutdown_service.shutdown();
            })
            .await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    };
    tokio::pin!(serving);
    let stopping = async {
        shutdown.await;
        service.shutdown();
        service.wait_stopped().await
    };
    tokio::select! {
        result = &mut serving => {
            service.shutdown();
            let stopped = service.wait_stopped().await;
            result?;
            stopped?;
        }
        stopped = stopping => {
            stopped?;
            match tokio::time::timeout(HTTP_DRAIN_TIMEOUT, &mut serving).await {
                Ok(result) => result?,
                Err(_) => diagnostics::emit(format_args!(
                    "HTTP connections did not drain within 20 seconds after worker reaping."
                )),
            }
        }
    }
    Ok(())
}

fn worker_paths(
    runtime: &WorkerRuntime,
    python: Option<PathBuf>,
    worker: Option<PathBuf>,
) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let vision = matches!(runtime, WorkerRuntime::PythonMiniCpmVision);
    let clef = matches!(runtime, WorkerRuntime::PythonClef { .. });
    if (vision || clef) && python.is_none() {
        return Err(if vision {
            "The mlx-vlm-minicpmv47 provider requires an explicit --python path."
        } else {
            "The python-clef provider requires an explicit --python path."
        }
        .into());
    }
    let mut python =
        python.unwrap_or_else(|| ".apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python".into());
    let mut worker = worker.unwrap_or_else(|| {
        if clef {
            "python/apxinf/apxinf/serving/clef_worker.py".into()
        } else if vision {
            "python/apxinf/apxinf/serving/vision_worker.py".into()
        } else {
            "python/apxinf/apxinf/serving/text_worker.py".into()
        }
    });
    if vision || clef {
        let absolute = std::path::absolute(&python)?;
        let parent = absolute
            .parent()
            .ok_or("The Python path needs a parent directory.")?;
        python = std::fs::canonicalize(parent)?.join(
            absolute
                .file_name()
                .ok_or("The Python path needs a filename.")?,
        );
        if !python.is_file() {
            return Err("The selected Python executable does not exist.".into());
        }
        worker = std::fs::canonicalize(worker)?;
    } else if matches!(runtime, WorkerRuntime::Python) {
        python = std::fs::canonicalize(python)?;
        worker = std::fs::canonicalize(worker)?;
    }
    Ok((python, worker))
}

#[cfg(unix)]
fn termination_signal() -> std::io::Result<impl std::future::Future<Output = ()>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    Ok(async move {
        tokio::select! {
            _ = interrupt.recv() => {},
            _ = terminate.recv() => {},
        }
    })
}

#[cfg(not(unix))]
fn termination_signal() -> std::io::Result<impl std::future::Future<Output = ()>> {
    Ok(async {
        let _ = tokio::signal::ctrl_c().await;
    })
}
