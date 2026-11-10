//! Runtime-neutral checks for production `ProtocolRuntime` adapters.

use apxinf_model::{RuntimeCapabilities, RuntimeError, RuntimeRequest};

use super::service::ProtocolRuntime;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    pub capabilities: RuntimeCapabilities,
    pub cancellation_observed: bool,
}

/// Probe the trait contract without depending on a particular model backend.
///
/// The caller supplies a request that is valid for the runtime's advertised
/// capacity. Cancellation is considered observable when the stream terminates
/// or reports `RuntimeError::Cancelled` after `TokenStream::cancel()`.
pub fn probe_protocol_runtime<R: ProtocolRuntime>(
    runtime: &R,
    request: RuntimeRequest,
) -> Result<ConformanceReport, RuntimeError> {
    let capabilities = runtime.capabilities();
    if capabilities.vocab_size == 0
        || capabilities.max_model_len == 0
        || capabilities.parallel_requests == 0
    {
        return Err(RuntimeError::Execution(
            "runtime capabilities must report positive vocab, context, and concurrency".into(),
        ));
    }
    if runtime.capabilities() != capabilities {
        return Err(RuntimeError::Execution(
            "runtime capabilities snapshot changed during probe".into(),
        ));
    }

    let mut stream = runtime.start(request)?;
    let _ = stream.next_token()?;
    stream.cancel();
    let cancellation_observed = match stream.next_token() {
        Ok(None) | Err(RuntimeError::Cancelled) => true,
        Ok(Some(_)) => false,
        Err(error) => return Err(error),
    };
    if !cancellation_observed {
        return Err(RuntimeError::Execution(
            "token stream continued after cancellation".into(),
        ));
    }
    Ok(ConformanceReport {
        capabilities,
        cancellation_observed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::service::TokenStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    struct FakeRuntime {
        cancelled: Arc<AtomicBool>,
    }

    struct FakeStream {
        cancelled: Arc<AtomicBool>,
        yielded: bool,
    }

    impl TokenStream for FakeStream {
        fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(RuntimeError::Cancelled);
            }
            if self.yielded {
                Ok(None)
            } else {
                self.yielded = true;
                Ok(Some(7))
            }
        }

        fn cancel(&self) {
            self.cancelled.store(true, Ordering::Release);
        }
    }

    impl ProtocolRuntime for FakeRuntime {
        fn capabilities(&self) -> RuntimeCapabilities {
            RuntimeCapabilities::frozen_qwen35(32, 0)
        }

        fn start(&self, _request: RuntimeRequest) -> Result<Box<dyn TokenStream>, RuntimeError> {
            Ok(Box::new(FakeStream {
                cancelled: Arc::clone(&self.cancelled),
                yielded: false,
            }))
        }
    }

    #[test]
    fn generic_probe_checks_snapshot_and_observable_cancel() {
        let runtime = FakeRuntime {
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        let report = probe_protocol_runtime(&runtime, RuntimeRequest::new(vec![1; 8], 2)).unwrap();
        assert_eq!(report.capabilities.vocab_size, 248_320);
        assert!(report.cancellation_observed);
    }
}
