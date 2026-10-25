use apxinf_audio_serving::{
    config::Config,
    gateway::{self, Gateway},
    media::{MediaRoot, MetalLock},
    service::AudioService,
};
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, sync::Semaphore};

struct Options {
    manifest: PathBuf,
    python: PathBuf,
    worker: PathBuf,
    listen: String,
    media_parent: PathBuf,
    fixture: bool,
    pressure: bool,
    fixture_delay: u64,
    metal_lock: Option<PathBuf>,
}
fn options() -> Result<Options, String> {
    let mut values = std::collections::HashMap::new();
    let mut flags = std::collections::HashSet::new();
    let mut args = std::env::args().skip(1);
    while let Some(name) = args.next() {
        if ["--allow-cpu-fixture", "--disable-pressure-check"].contains(&name.as_str()) {
            if !flags.insert(name) {
                return Err("duplicate option".into());
            }
        } else if [
            "--manifest",
            "--python",
            "--worker",
            "--listen",
            "--media-parent",
            "--fixture-delay-ms",
            "--metal-lock",
        ]
        .contains(&name.as_str())
        {
            if values
                .insert(name, args.next().ok_or("missing option value")?)
                .is_some()
            {
                return Err("duplicate option".into());
            }
        } else {
            return Err("Use --manifest FILE --python PYTHON. Optional: --listen HOST:PORT --metal-lock FILE. CPU tests require --allow-cpu-fixture --disable-pressure-check.".into());
        }
    }
    let fixture = flags.contains("--allow-cpu-fixture");
    let fixture_delay = values
        .remove("--fixture-delay-ms")
        .unwrap_or_else(|| "0".into())
        .parse::<u64>()
        .map_err(|_| "invalid fixture delay")?;
    if fixture_delay > 130000 || (fixture_delay != 0 && !fixture) {
        return Err("fixture delay requires the CPU fixture".into());
    }
    Ok(Options {
        manifest: values
            .remove("--manifest")
            .ok_or("--manifest is required")?
            .into(),
        python: values
            .remove("--python")
            .ok_or("--python is required")?
            .into(),
        worker: values
            .remove("--worker")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../python/apxinf/apxinf/serving/audio_worker.py")
            }),
        listen: values
            .remove("--listen")
            .unwrap_or_else(|| "127.0.0.1:8010".into()),
        media_parent: values
            .remove("--media-parent")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir),
        fixture,
        pressure: !flags.contains("--disable-pressure-check"),
        fixture_delay,
        metal_lock: values.remove("--metal-lock").map(PathBuf::from),
    })
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("audio service: {error}");
        std::process::exit(2);
    }
}
async fn run() -> Result<(), String> {
    let options = options()?;
    // Signal streams exist before manifest checks or model loading.
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| e.to_string())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .map_err(|e| e.to_string())?;
    let _metal = if options.pressure {
        Some(MetalLock::acquire(options.metal_lock.as_deref().ok_or(
            "production audio requires --metal-lock with the shared Metal lock path",
        )?)?)
    } else {
        None
    };
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let startup_cancelled = cancelled.clone();
    let manifest = options.manifest;
    let python = options.python;
    let worker = options.worker;
    let mut startup = tokio::task::spawn_blocking(move || {
        Config::load(
            &manifest,
            python,
            worker,
            options.fixture,
            options.pressure,
            startup_cancelled,
        )
    });
    let config = tokio::select! {
        result = &mut startup => result.map_err(|e| e.to_string())??,
        _ = terminate.recv() => { cancelled.store(true,std::sync::atomic::Ordering::SeqCst); let _ = startup.await; return Ok(()); },
        _ = interrupt.recv() => { cancelled.store(true,std::sync::atomic::Ordering::SeqCst); let _ = startup.await; return Ok(()); },
    };
    let media = MediaRoot::create(options.media_parent)?;
    let listener = TcpListener::bind(&options.listen)
        .await
        .map_err(|e| e.to_string())?;
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    let (service, actor) = AudioService::start(config, media, options.fixture_delay).await?;
    let gateway = Gateway {
        service: service.clone(),
        uploads: Arc::new(Semaphore::new(2)),
    };
    let connections = Arc::new(Semaphore::new(16));
    let mut tasks = tokio::task::JoinSet::new();
    println!("apxinf-audio-serve listening {address}");
    let mut accept_error = None;
    loop {
        tokio::select! {
            _ = terminate.recv() => break,
            _ = interrupt.recv() => break,
            accepted = listener.accept() => {
                let (socket,_) = match accepted { Ok(value) => value, Err(error) => { accept_error = Some(error.to_string()); break; } };
                let Ok(permit) = connections.clone().try_acquire_owned() else { drop(socket); continue; };
                let gateway = gateway.clone();
                let received = Instant::now();
                tasks.spawn(async move {
                    let _permit = permit;
                    let response_available = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let response_flag = response_available.clone();
                    let service = hyper::service::service_fn(move |request| {
                        let gateway = gateway.clone();
                        let response_flag = response_flag.clone();
                        async move {
                            let response = gateway::handle(request,gateway,received).await;
                            response_flag.store(true,std::sync::atomic::Ordering::SeqCst);
                            response
                        }
                    });
                    let mut builder = http1::Builder::new();
                    builder.keep_alive(false).max_buf_size(16384).timer(TokioTimer::new()).header_read_timeout(Duration::from_secs(15));
                    let connection = builder.serve_connection(TokioIo::new(apxinf_audio_serving::connection::SendDeadlineIo::new(socket,response_available)),service);
                    let _ = tokio::time::timeout(Duration::from_secs(135),connection).await;
                });
            }
            Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
        }
    }
    drop(listener);
    service.shutdown().await;
    actor.await.map_err(|e| e.to_string())?;
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    if let Some(error) = accept_error {
        return Err(error);
    }
    Ok(())
}
