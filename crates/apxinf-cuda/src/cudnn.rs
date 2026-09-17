//! Lazy cuDNN loading: models that do not use convolution need no cuDNN install.
use crate::ffi::cudnn::*;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::OnceLock;

#[cfg_attr(target_os = "linux", link(name = "dl"))]
extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

pub(crate) fn api() -> Result<&'static Api, String> {
    API.get_or_init(load).as_ref().map_err(Clone::clone)
}

fn load() -> Result<Api, String> {
    // Use the OS loader search path. Deployment owns LD_LIBRARY_PATH; no
    // model-specific library directories or Python imports are consulted.
    let library = CString::new("libcudnn.so.9").unwrap();
    let handle = unsafe { dlopen(library.as_ptr(), 2) }; // RTLD_NOW, local scope.
    if handle.is_null() {
        let error = unsafe { dlerror() };
        let detail = if error.is_null() {
            "unknown loader error".into()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        };
        return Err(format!(
            "cuDNN v9 is required for convolution; install it on the loader path: {detail}"
        ));
    }
    let result = (|| {
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let name = CString::new($name).unwrap();
                let address = unsafe { dlsym(handle, name.as_ptr()) };
                if address.is_null() {
                    return Err(format!("cuDNN symbol missing: {}", $name));
                }
                unsafe { std::mem::transmute::<*mut c_void, $ty>(address) }
            }};
        }
        Ok(Api {
            create: symbol!("cudnnCreate", Create),
            destroy: symbol!("cudnnDestroy", Destroy),
            set_stream: symbol!("cudnnSetStream", SetStream),
            create_tensor: symbol!("cudnnCreateTensorDescriptor", Create),
            destroy_tensor: symbol!("cudnnDestroyTensorDescriptor", Destroy),
            set_tensor: symbol!("cudnnSetTensor4dDescriptor", Set4d),
            set_tensor_nd: symbol!("cudnnSetTensorNdDescriptor", SetTensorNd),
            create_filter: symbol!("cudnnCreateFilterDescriptor", Create),
            destroy_filter: symbol!("cudnnDestroyFilterDescriptor", Destroy),
            set_filter: symbol!("cudnnSetFilter4dDescriptor", Set4d),
            set_filter_nd: symbol!("cudnnSetFilterNdDescriptor", SetFilterNd),
            create_conv: symbol!("cudnnCreateConvolutionDescriptor", Create),
            destroy_conv: symbol!("cudnnDestroyConvolutionDescriptor", Destroy),
            set_conv: symbol!("cudnnSetConvolution2dDescriptor", SetConv),
            set_conv_nd: symbol!("cudnnSetConvolutionNdDescriptor", SetConvNd),
            set_math: symbol!("cudnnSetConvolutionMathType", SetInt),
            set_groups: symbol!("cudnnSetConvolutionGroupCount", SetInt),
            forward_workspace: symbol!("cudnnGetConvolutionForwardWorkspaceSize", Workspace),
            backward_workspace: symbol!("cudnnGetConvolutionBackwardDataWorkspaceSize", Workspace),
            forward: symbol!("cudnnConvolutionForward", Convolve),
            backward: symbol!("cudnnConvolutionBackwardData", Convolve),
            add: symbol!("cudnnAddTensor", Add),
            transform: symbol!("cudnnTransformTensor", Add),
            error_string: symbol!("cudnnGetErrorString", ErrorString),
        })
    })();
    if result.is_err() {
        unsafe {
            dlclose(handle);
        }
    }
    // Successful function pointers live for the process lifetime with API.
    result
}

pub(crate) fn check(api: &Api, status: c_int) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    let text = unsafe { (api.error_string)(status) };
    let detail = if text.is_null() {
        format!("status {status}")
    } else {
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    };
    Err(format!("cuDNN: {detail}"))
}

pub(crate) struct Descriptor {
    pub raw: Desc,
    destroy: Destroy,
}
impl Descriptor {
    pub fn new(api: &Api, create: Create, destroy: Destroy) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        check(api, unsafe { create(&mut raw) })?;
        Ok(Self { raw, destroy })
    }
}
impl Drop for Descriptor {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.raw);
        }
    }
}
