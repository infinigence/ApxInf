#[path = "../server/mod.rs"]
mod server;

use std::net::{SocketAddr, TcpListener};

use apxinf_model::{CancellationToken, RuntimeCapabilities, RuntimeError, RuntimeRequest};
use clap::Parser;

use server::http::serve;
use server::service::{ProtocolRuntime, ProtocolService, TokenStream};

#[derive(Debug, Parser)]
#[command(name = "apxinf_protocol_stub")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8001")]
    bind: SocketAddr,

    #[arg(long, default_value_t = 32_768)]
    max_model_len: usize,
}

struct StubRuntime {
    capabilities: RuntimeCapabilities,
}

struct StubTokenStream {
    remaining: usize,
    cancel: CancellationToken,
}

impl TokenStream for StubTokenStream {
    fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
        if self.cancel.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(7))
    }

    fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl ProtocolRuntime for StubRuntime {
    fn capabilities(&self) -> RuntimeCapabilities {
        self.capabilities
    }

    fn start(&self, request: RuntimeRequest) -> Result<Box<dyn TokenStream>, RuntimeError> {
        Ok(Box::new(StubTokenStream {
            remaining: request.max_new_tokens,
            cancel: request.cancel,
        }))
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if args.max_model_len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--max-model-len must be positive",
        ));
    }
    let listener = TcpListener::bind(args.bind)?;
    let runtime = StubRuntime {
        capabilities: RuntimeCapabilities::frozen_qwen35(args.max_model_len, 0),
    };
    let service = std::sync::Arc::new(ProtocolService::new(runtime, true));
    serve(listener, service)
}
