use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::GemmArgs;

/// Optional calibration hook for floating-point GEMM operands.
///
/// The hook is thread-local, so normal inference pays one empty-cell check and
/// concurrent model threads cannot observe each other's activations.
pub trait GemmActivationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()>;
}

thread_local! {
    static OBSERVER: RefCell<Option<Rc<dyn GemmActivationObserver>>> =
        const { RefCell::new(None) };
}

/// Uninstalls the calibration observer when dropped.
pub struct GemmObserverGuard;

impl Drop for GemmObserverGuard {
    fn drop(&mut self) {
        OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

pub fn install_gemm_observer(
    observer: Rc<dyn GemmActivationObserver>,
) -> Result<GemmObserverGuard> {
    OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::Other(
                "a GEMM activation observer is already installed".into(),
            ));
        }
        *slot = Some(observer);
        Ok(GemmObserverGuard)
    })
}

pub(super) fn observe(args: &GemmArgs<'_>) -> Result<()> {
    if args.a.dtype() != DType::BF16 || args.b.dtype() != DType::BF16 {
        return Ok(());
    }
    OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe(args.a, args.b)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use apxinf_core::{Result, Tensor};

    use super::{install_gemm_observer, observe, GemmActivationObserver};
    use crate::ops::gemm::GemmArgs;

    struct Counter(Rc<Cell<usize>>);

    impl GemmActivationObserver for Counter {
        fn observe(&self, _activation: &Tensor, _weight: &Tensor) -> Result<()> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn installed_observer_sees_formal_bf16_gemm_operands() {
        let a = Tensor::from_bf16(vec![1, 1], &[half::bf16::ONE]).unwrap();
        let b = Tensor::from_bf16(vec![1, 1], &[half::bf16::ONE]).unwrap();
        let mut out = Tensor::from_bf16(vec![1, 1], &[half::bf16::ZERO]).unwrap();
        let count = Rc::new(Cell::new(0));
        let _guard = install_gemm_observer(Rc::new(Counter(Rc::clone(&count)))).unwrap();

        observe(&GemmArgs::new(&a, &b, &mut out)).unwrap();

        assert_eq!(count.get(), 1);
    }
}
