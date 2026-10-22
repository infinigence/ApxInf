use std::ffi::{c_char, c_void};
pub type Handle = *mut c_void;
pub type Callback =
    unsafe extern "C" fn(Handle, *const *const c_void, usize, *mut Handle, usize) -> i32;
extern "C" {
    pub fn apx_mlx_error() -> *const c_char;
    pub fn apx_mlx_stream_new(gpu: i32, index: i32, out: *mut Handle) -> i32;
    pub fn apx_mlx_stream_free(stream: Handle);
    pub fn apx_mlx_sync(stream: Handle) -> i32;
    pub fn apx_mlx_memory_stats(active: *mut usize, cache: *mut usize, peak: *mut usize) -> i32;
    pub fn apx_mlx_reset_peak_memory() -> i32;
    pub fn apx_mlx_clear_cache() -> i32;
    pub fn apx_mlx_array_new(
        stream: Handle,
        shape: *const i32,
        rank: usize,
        dtype: i32,
        bytes: *const u8,
        len: usize,
        out: *mut Handle,
    ) -> i32;
    pub fn apx_mlx_array_clone(array: *const c_void, out: *mut Handle) -> i32;
    pub fn apx_mlx_array_free(array: Handle);
    pub fn apx_mlx_array_info(
        array: *const c_void,
        shape: *mut i32,
        capacity: usize,
        rank: *mut usize,
        dtype: *mut i32,
    ) -> i32;
    pub fn apx_mlx_array_read(
        stream: Handle,
        array: *const c_void,
        bytes: *mut u8,
        len: usize,
    ) -> i32;
    pub fn apx_mlx_eval(stream: Handle, arrays: *const *const c_void, count: usize) -> i32;
    pub fn apx_mlx_op(
        stream: Handle,
        op: i32,
        inputs: *const *const c_void,
        count: usize,
        ints: *const i32,
        nints: usize,
        floats: *const f32,
        nfloats: usize,
        out: *mut Handle,
    ) -> i32;
    pub fn apx_mlx_compile_new(
        callback: Callback,
        context: Handle,
        outputs: usize,
        shapeless: i32,
        out: *mut Handle,
    ) -> i32;
    pub fn apx_mlx_compile_free(compiled: Handle);
    pub fn apx_mlx_compile_call(
        compiled: Handle,
        inputs: *const *const c_void,
        count: usize,
        outputs: *mut Handle,
        output_count: usize,
    ) -> i32;
    pub fn apx_mlx_quantize(
        stream: Handle,
        input: *const c_void,
        group: i32,
        bits: i32,
        outputs: *mut Handle,
    ) -> i32;
    pub fn apx_mlx_metal_new(
        name: *const c_char,
        source: *const c_char,
        header: *const c_char,
        inputs: *const *const c_char,
        input_count: usize,
        outputs: *const *const c_char,
        output_count: usize,
        out: *mut Handle,
    ) -> i32;
    pub fn apx_mlx_metal_free(kernel: Handle);
    pub fn apx_mlx_metal_call(
        stream: Handle,
        kernel: Handle,
        inputs: *const *const c_void,
        input_count: usize,
        shapes: *const i32,
        ranks: *const usize,
        dtypes: *const i32,
        output_count: usize,
        grid: *const i32,
        group: *const i32,
        template_dtype: i32,
        outputs: *mut Handle,
    ) -> i32;
}
