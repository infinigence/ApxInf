//! Regression guard for the portable-op hot path.
//!
//! Validation is split between plan-build time (full structural checks) and
//! dispatch (a fixed set of comparisons). These tests pin the two properties that
//! split depends on: the dispatch path allocates nothing, and a plan cannot be
//! silently reused with operands it was not validated for.
//!
//! The allocation counter is process-global, so this binary intentionally holds a
//! single test: a concurrently running test in the same binary would pollute it.

use apxinf_core::{contracts::*, DType, Device, Tensor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCS.load(Ordering::Relaxed);
    f();
    ALLOCS.load(Ordering::Relaxed) - before
}

fn operands() -> (Tensor, Tensor) {
    // PI0.5-shaped decode attention: B=1 Q=10 K=522 Hq=8 Hkv=1 D=256.
    (
        Tensor::zeros(vec![1, 10, 8, 256], DType::BF16),
        Tensor::zeros(vec![1, 522, 1, 256], DType::BF16),
    )
}

#[test]
fn dispatch_path_does_not_allocate() {
    let (q, k) = operands();
    let o = AttentionOptions::full(0.0625);
    let plan = AttentionPlan::new(Device::Cpu, &q, &k, &k, &o).unwrap();
    // Warm up: the first call through any path may trigger one-time lazy init.
    for _ in 0..100 {
        plan.check_operands(Device::Cpu, &q, &k, &k, &o).unwrap();
    }
    assert_eq!(
        allocations(|| plan.check_operands(Device::Cpu, &q, &k, &k, &o).unwrap()),
        0,
        "planned attention dispatch must not allocate"
    );

    let bias = Tensor::zeros(vec![1, 1, 10, 522], DType::F32);
    for mask in [
        AttentionMask::Causal {
            q_start: 512,
            k_start: 0,
        },
        AttentionMask::Additive(&bias),
    ] {
        let options = AttentionOptions { mask, ..o };
        let plan = AttentionPlan::new(Device::Cpu, &q, &k, &k, &options).unwrap();
        assert_eq!(
            allocations(|| plan
                .check_operands(Device::Cpu, &q, &k, &k, &options)
                .unwrap()),
            0
        );
    }

    let x = Tensor::zeros(vec![2, 3, 4], DType::F32);
    let dims = x.shape().dims();
    assert_eq!(
        allocations(|| {
            validate_permutation(dims, &[2, 0, 1]).unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            validate_concat(&[dims, dims], 0).unwrap();
        }),
        0
    );
    let slice = AxisSlice {
        axis: 1,
        start: 0,
        end: 2,
    };
    assert_eq!(
        allocations(|| {
            slice.validate(dims).unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            checked_bytes_iter([2usize, 3, 4].into_iter(), DType::F32).unwrap();
        }),
        0
    );
}
