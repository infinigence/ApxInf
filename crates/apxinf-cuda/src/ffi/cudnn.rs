//! cuDNN v9 C ABI used by the optional convolution provider.
use std::ffi::{c_char, c_int, c_void};

pub type Desc = *mut c_void;
pub type Create = unsafe extern "C" fn(*mut Desc) -> c_int;
pub type Destroy = unsafe extern "C" fn(Desc) -> c_int;
pub type SetStream = unsafe extern "C" fn(Desc, Desc) -> c_int;
pub type Set4d = unsafe extern "C" fn(Desc, c_int, c_int, c_int, c_int, c_int, c_int) -> c_int;
pub type SetTensorNd =
    unsafe extern "C" fn(Desc, c_int, c_int, *const c_int, *const c_int) -> c_int;
pub type SetFilterNd = unsafe extern "C" fn(Desc, c_int, c_int, c_int, *const c_int) -> c_int;
pub type SetConvNd = unsafe extern "C" fn(
    Desc,
    c_int,
    *const c_int,
    *const c_int,
    *const c_int,
    c_int,
    c_int,
) -> c_int;
pub type SetConv =
    unsafe extern "C" fn(Desc, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int) -> c_int;
pub type SetInt = unsafe extern "C" fn(Desc, c_int) -> c_int;
pub type Workspace = unsafe extern "C" fn(Desc, Desc, Desc, Desc, Desc, c_int, *mut usize) -> c_int;
pub type Convolve = unsafe extern "C" fn(
    Desc,
    *const c_void,
    Desc,
    *const c_void,
    Desc,
    *const c_void,
    Desc,
    c_int,
    *mut c_void,
    usize,
    *const c_void,
    Desc,
    *mut c_void,
) -> c_int;
pub type Add = unsafe extern "C" fn(
    Desc,
    *const c_void,
    Desc,
    *const c_void,
    *const c_void,
    Desc,
    *mut c_void,
) -> c_int;
pub type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

pub struct Api {
    pub create: Create,
    pub destroy: Destroy,
    pub set_stream: SetStream,
    pub create_tensor: Create,
    pub destroy_tensor: Destroy,
    pub set_tensor: Set4d,
    pub set_tensor_nd: SetTensorNd,
    pub create_filter: Create,
    pub destroy_filter: Destroy,
    pub set_filter: Set4d,
    pub set_filter_nd: SetFilterNd,
    pub create_conv: Create,
    pub destroy_conv: Destroy,
    pub set_conv: SetConv,
    pub set_conv_nd: SetConvNd,
    pub set_math: SetInt,
    pub set_groups: SetInt,
    pub forward_workspace: Workspace,
    pub backward_workspace: Workspace,
    pub forward: Convolve,
    pub backward: Convolve,
    pub add: Add,
    pub transform: Add,
    pub error_string: ErrorString,
}
