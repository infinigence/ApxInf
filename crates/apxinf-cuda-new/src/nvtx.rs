//! NVTX (NVIDIA Tools Extension) range markers for nsys profiling.
//!
//! Wrap a code region in `let _g = nvtx::range("name");` and it appears as a
//! labeled range in the Nsight Systems timeline. The guard pops on drop.

use std::ffi::CString;

#[cfg(all(feature = "nvtx", nvtx_header))]
mod imp {
    use std::os::raw::c_char;
    extern "C" {
        #[link_name = "apxinf_nvtx_range_push"]
        pub fn nvtxRangePushA(name: *const c_char) -> i32;
        #[link_name = "apxinf_nvtx_range_pop"]
        pub fn nvtxRangePop() -> i32;
    }
}

#[cfg(all(feature = "nvtx", nvtx_v3, not(nvtx_header)))]
mod imp {
    use std::os::raw::c_char;
    #[link(name = "nvtx3interop")]
    extern "C" {
        pub fn nvtxRangePushA(name: *const c_char) -> i32;
        pub fn nvtxRangePop() -> i32;
    }
}

#[cfg(all(feature = "nvtx", nvtx_v2, not(nvtx_v3), not(nvtx_header)))]
mod imp {
    use std::os::raw::c_char;
    #[link(name = "nvToolsExt")]
    extern "C" {
        pub fn nvtxRangePushA(name: *const c_char) -> i32;
        pub fn nvtxRangePop() -> i32;
    }
}

#[cfg(any(
    not(feature = "nvtx"),
    all(not(nvtx_header), not(nvtx_v3), not(nvtx_v2))
))]
#[allow(non_snake_case)]
mod imp {
    use std::os::raw::c_char;
    pub unsafe fn nvtxRangePushA(_name: *const c_char) -> i32 {
        0
    }
    pub unsafe fn nvtxRangePop() -> i32 {
        0
    }
}

pub struct Range {
    _name: CString,
}

impl Range {
    pub fn new(name: &str) -> Self {
        let name = CString::new(name).unwrap_or_else(|_| CString::new("nvtx").unwrap());
        unsafe {
            imp::nvtxRangePushA(name.as_ptr());
        }
        Self { _name: name }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe {
            imp::nvtxRangePop();
        }
    }
}

pub fn range(name: &str) -> Range {
    Range::new(name)
}
