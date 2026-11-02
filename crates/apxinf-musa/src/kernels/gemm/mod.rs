//! GEMM operator contracts.
//!
//! The public names mirror `apxinf-cuda`'s GEMM module exactly, including the
//! `pub use` renames: `gemm::bf16` and `gemm::bf16_geglu_fused` are the
//! *functions* `gemm_bf16` and `gemm_bf16_geglu_fused`, not submodules. Model
//! code calls them by those names, so the rename is part of the contract.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{Error, Result, Tensor};

use crate::context::MusaContext;
use crate::tuning::{TuningDb, TuningMode, TuningPaths};

#[cfg_attr(not(apxinf_musa_available), allow(dead_code))]
mod qint8_tiles;
mod bf16;
pub(crate) mod assembly;
mod w8a8;

pub use assembly::{prepare_bf16_nn_weight, prepare_bf16_nt_weight};

pub use bf16::{gemm_bf16 as bf16, gemm_bf16_geglu_fused as bf16_geglu_fused, gemm_bf16_into};
pub use qint8_tiles::serves as w8a8_tile_serves;
pub use w8a8::{
    adaptive_layer_norm_quantize_w8a8_activation, decode_scope as w8a8_decode_scope,
    decode_tiles_supported as w8a8_decode_supported, gemm_quantized_w8a8,
    gemm_quantized_w8a8_into, gemm_w8a8 as w8a8, quantize_w8a8_activation,
    quantize_w8a8_geglu_activation, quantize_w8a8_silu_mul_activation, W8A8Activation,
    W8A8DecodeScope, W8A8Layout, W8A8ScaleMode, W8A8WeightView, SCALE_PAD as W8A8_SCALE_PAD,
};

thread_local! {
    static DECODE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One thread's nesting depth for a scope it owns.
///
/// Two dispatch paths need the same enter/leave bookkeeping over separate
/// counters — the W8A8 scope must not see the BF16 one's depth — so the
/// mechanism lives here and each owner keeps its own `thread_local!`.
pub(super) struct ThreadScope {
    key: &'static std::thread::LocalKey<std::cell::Cell<usize>>,
    _not_send: std::marker::PhantomData<Rc<()>>,
}

impl ThreadScope {
    pub(super) fn enter(key: &'static std::thread::LocalKey<std::cell::Cell<usize>>) -> Self {
        key.with(|depth| depth.set(depth.get() + 1));
        Self {
            key,
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for ThreadScope {
    fn drop(&mut self) {
        self.key.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Marks a synchronous decode traversal for BF16 GEMM dispatch on this thread.
/// The scope is nestable, restores on errors/unwind, and cannot cross threads.
/// Warmup and graph capture must traverse the same scope as eager execution.
#[must_use]
pub struct DecodeScope {
    _scope: ThreadScope,
}

pub fn decode_scope() -> DecodeScope {
    DecodeScope {
        _scope: ThreadScope::enter(&DECODE_DEPTH),
    }
}

pub(super) fn in_decode_scope() -> bool {
    DECODE_DEPTH.with(|depth| depth.get() != 0)
}

#[cfg(test)]
mod decode_scope_tests {
    use super::*;
    #[test]
    fn decode_scope_is_nested_thread_local_and_restored_on_unwind() {
        assert!(!in_decode_scope());
        {
            let _outer = decode_scope();
            assert!(in_decode_scope());
            {
                let _inner = decode_scope();
                assert!(in_decode_scope());
            }
            assert!(in_decode_scope());
            std::thread::spawn(|| assert!(!in_decode_scope()))
                .join()
                .unwrap();
        }
        assert!(!in_decode_scope());
        let _ = std::panic::catch_unwind(|| {
            let _scope = decode_scope();
            panic!("unwind probe");
        });
        assert!(!in_decode_scope());
    }
}

/// Observes BF16 GEMM operands for calibration.
///
/// Thread-local and single-installed, exactly as on CUDA: ordinary inference
/// pays one empty-cell check per GEMM, and concurrent model threads cannot see
/// each other's observer.
pub trait Bf16ActivationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()>;
}

thread_local! {
    static BF16_OBSERVER: RefCell<Option<Rc<dyn Bf16ActivationObserver>>> =
        const { RefCell::new(None) };
}

/// Uninstalls the observer on drop.
pub struct Bf16ObserverGuard;

impl Drop for Bf16ObserverGuard {
    fn drop(&mut self) {
        BF16_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Install a BF16 operand observer until the returned guard is dropped.
pub fn install_bf16_observer(
    observer: Rc<dyn Bf16ActivationObserver>,
) -> Result<Bf16ObserverGuard> {
    BF16_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::Other(
                "a BF16 activation observer is already installed".into(),
            ));
        }
        *slot = Some(observer);
        Ok(Bf16ObserverGuard)
    })
}

/// Notify the installed observer, if any.
///
/// The `Rc` is cloned out of the cell before the call so a re-entrant observer
/// cannot panic on an already-borrowed cell.
pub(super) fn observe_bf16(activation: &Tensor, weight: &Tensor) -> Result<()> {
    let observer = BF16_OBSERVER.with(|slot| slot.borrow().clone());
    match observer {
        Some(observer) => observer.observe(activation, weight),
        None => Ok(()),
    }
}

/// Configure GEMM tuning for a model about to be prepared.
///
/// Provider-native plans are created lazily for the keys a real workload
/// reaches; with no native GEMM provider yet, this records the mode and the
/// database so the seam's callers behave identically, and plan resolution
/// stays a no-op until a native provider needs it.
pub fn configure_tuning(
    ctx: &MusaContext,
    mode: TuningMode,
    databases: &[TuningDb],
    paths: Option<TuningPaths>,
) -> Result<()> {
    crate::tuning::configure(ctx, mode, databases, paths)
}
