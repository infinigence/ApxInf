//! Lifecycle tests for lazy backend storage. No native runtime is needed.
#[cfg(all(feature = "opaque-storage", target_os = "macos", target_arch = "aarch64"))]
mod opaque {
    use apxinf_core::{DType, Device, Shape, Storage, Tensor};
    use std::{cell::Cell, rc::Rc};

    struct Owner(Rc<Cell<usize>>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn reshape_and_clone_retain_owner_without_exposing_a_pointer() {
        let drops = Rc::new(Cell::new(0));
        let owner = Rc::new(Owner(drops.clone()));
        let t = Tensor::from_opaque_parts(
            Shape::new(vec![2, 3]),
            DType::F32,
            Device::Metal(0),
            24,
            owner.clone(),
        )
        .unwrap();
        drop(owner);
        let reshaped = t.reshape(vec![3, 2]).unwrap();
        let clone = reshaped.clone();
        assert!(clone.storage().as_gpu().is_none());
        assert!(clone.storage().as_cpu().is_none());
        assert!(clone.as_f32().is_err());
        assert!(
            matches!(clone.storage(), Storage::Opaque { device: Device::Metal(0), handle }
            if handle.len() == 24 && handle.owner().downcast_ref::<Owner>().is_some())
        );
        drop(t);
        drop(reshaped);
        assert_eq!(drops.get(), 0);
        drop(clone);
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn invalid_extents_and_non_metal_opaque_owners_are_rejected() {
        for (shape, device, bytes) in [
            (vec![2, 3], Device::Metal(0), 23),
            (vec![usize::MAX, 2], Device::Metal(0), usize::MAX),
            (vec![usize::MAX], Device::Metal(0), usize::MAX),
            (vec![0, 2], Device::Metal(0), 0),
            (vec![1], Device::Cpu, 4),
            (vec![1], Device::Cuda(0), 4),
        ] {
            assert!(Tensor::from_opaque_parts(
                Shape::new(shape),
                DType::F32,
                device,
                bytes,
                Rc::new(())
            )
            .is_err());
        }
        let t = Tensor::from_opaque_parts(
            Shape::new(vec![2, 3]),
            DType::F32,
            Device::Metal(0),
            24,
            Rc::new(()),
        )
        .unwrap();
        assert!(t.reshape(vec![usize::MAX, 2]).is_err());
        assert!(t.reshape(vec![5]).is_err());
    }
}

#[cfg(not(all(feature = "opaque-storage", target_os = "macos", target_arch = "aarch64")))]
#[test]
fn default_storage_keeps_cpu_cuda_thread_traits() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<apxinf_core::Tensor>();
}
