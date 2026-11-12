use crate::runtime::ComputeUnits;
use apxinf_core::{Error, Result};
use std::collections::BTreeSet;
use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeDeviceKind {
    Unknown,
    Cpu,
    Gpu,
    NeuralEngine,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeDevice {
    pub kind: ComputeDeviceKind,
    pub class_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceUsage {
    pub supported: Vec<ComputeDevice>,
    pub preferred: Option<ComputeDevice>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputBinding {
    pub name: Option<String>,
    pub literal_present: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementInput {
    pub name: String,
    pub bindings: Vec<InputBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationPlacement {
    pub path: Vec<usize>,
    pub operator: String,
    pub inputs: Vec<PlacementInput>,
    pub outputs: Vec<String>,
    pub child_operation_counts: Vec<usize>,
    pub usage: Option<DeviceUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionPlacement {
    pub name: String,
    pub root_operation_count: usize,
    pub operations: Vec<OperationPlacement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedPlacement {
    pub compiled_path: PathBuf,
    pub compute_units: ComputeUnits,
    pub functions: Vec<FunctionPlacement>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeString {
    data: *const u8,
    len: usize,
}

#[repr(C)]
struct NativeDevice {
    kind: i32,
    class_name: NativeString,
}

#[repr(C)]
struct NativeBinding {
    name_present: i32,
    name: NativeString,
    literal_present: i32,
}

#[repr(C)]
struct NativeInput {
    name: NativeString,
    bindings: *const NativeBinding,
    count: usize,
}

#[repr(C)]
struct NativeFunction {
    name: NativeString,
    root_operations: usize,
}

#[repr(C)]
struct NativeRow {
    function_index: usize,
    path: *const usize,
    path_len: usize,
    operator_name: NativeString,
    inputs: *const NativeInput,
    input_count: usize,
    outputs: *const NativeString,
    output_count: usize,
    child_operations: *const usize,
    child_count: usize,
    usage_present: i32,
    supported: *const NativeDevice,
    supported_count: usize,
    preferred_present: i32,
    preferred: NativeDevice,
}

extern "C" {
    fn apx_coreml_inspect_plan(
        compiled: *const std::ffi::c_char,
        units: i32,
        timeout_ms: u64,
        visitor: unsafe extern "C" fn(*mut c_void, *const NativeFunction, *const NativeRow) -> i32,
        context: *mut c_void,
    ) -> i32;
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

unsafe fn native_slice<'a, Value>(data: *const Value, count: usize) -> Result<&'a [Value]> {
    if count == 0 {
        return Ok(&[]);
    }
    if data.is_null() || count > 100000 {
        return Err(invalid("invalid Core ML plan array view"));
    }
    Ok(unsafe { std::slice::from_raw_parts(data, count) })
}

unsafe fn native_string(value: NativeString) -> Result<String> {
    let bytes = unsafe { native_slice(value.data, value.len) }?;
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| invalid("Core ML plan string is not UTF-8"))
}

unsafe fn native_device(value: &NativeDevice) -> Result<ComputeDevice> {
    let kind = match value.kind {
        0 => ComputeDeviceKind::Unknown,
        1 => ComputeDeviceKind::Cpu,
        2 => ComputeDeviceKind::Gpu,
        3 => ComputeDeviceKind::NeuralEngine,
        _ => return Err(invalid("invalid Core ML plan device tag")),
    };
    Ok(ComputeDevice {
        kind,
        class_name: unsafe { native_string(value.class_name) }?,
    })
}

fn flag(value: i32) -> Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid("invalid Core ML plan presence flag")),
    }
}

#[derive(Default)]
struct Collector {
    functions: Vec<FunctionPlacement>,
    pending: BTreeSet<(usize, Vec<usize>)>,
    error: Option<String>,
    rows: usize,
}

impl Collector {
    unsafe fn function(&mut self, function: &NativeFunction) -> Result<()> {
        if !self.pending.is_empty()
            || self.functions.len() >= 1000
            || function.root_operations > 100000
            || function.root_operations.saturating_add(self.rows) > 100000
        {
            return Err(invalid("incomplete or oversized Core ML plan function"));
        }
        let name = unsafe { native_string(function.name) }?;
        if self.functions.iter().any(|previous| previous.name == name) {
            return Err(invalid("duplicate Core ML plan function"));
        }
        let index = self.functions.len();
        for operation in 0..function.root_operations {
            self.pending.insert((index, vec![operation]));
        }
        self.functions.push(FunctionPlacement {
            name,
            root_operation_count: function.root_operations,
            operations: Vec::new(),
        });
        Ok(())
    }

    unsafe fn row(&mut self, row: &NativeRow) -> Result<()> {
        let path = unsafe { native_slice(row.path, row.path_len) }?.to_vec();
        if path.is_empty()
            || path.len() > 129
            || path.len() % 2 == 0
            || !self.pending.remove(&(row.function_index, path.clone()))
            || self.rows >= 100000
        {
            return Err(invalid(
                "invalid, duplicate or unexpected Core ML plan operation path",
            ));
        }
        let children = unsafe { native_slice(row.child_operations, row.child_count) }?.to_vec();
        let mut child_total = 0usize;
        for &count in &children {
            child_total = child_total
                .checked_add(count)
                .ok_or_else(|| invalid("Core ML plan size overflow"))?;
            if child_total > 100000 {
                return Err(invalid("Core ML plan child limit exceeded"));
            }
        }
        if child_total
            .saturating_add(self.pending.len())
            .saturating_add(self.rows + 1)
            > 100000
        {
            return Err(invalid("Core ML plan pending operation limit exceeded"));
        }
        for (block, &count) in children.iter().enumerate() {
            for operation in 0..count {
                let mut child = path.clone();
                child.extend([block, operation]);
                self.pending.insert((row.function_index, child));
            }
        }
        let inputs = unsafe { native_slice(row.inputs, row.input_count) }?
            .iter()
            .map(|input| unsafe {
                let bindings = native_slice(input.bindings, input.count)?
                    .iter()
                    .map(|binding| {
                        Ok(InputBinding {
                            name: if flag(binding.name_present)? {
                                Some(native_string(binding.name)?)
                            } else {
                                None
                            },
                            literal_present: flag(binding.literal_present)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(PlacementInput {
                    name: native_string(input.name)?,
                    bindings,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = unsafe { native_slice(row.outputs, row.output_count) }?
            .iter()
            .map(|value| unsafe { native_string(*value) })
            .collect::<Result<Vec<_>>>()?;
        let usage = if flag(row.usage_present)? {
            Some(DeviceUsage {
                supported: unsafe { native_slice(row.supported, row.supported_count) }?
                    .iter()
                    .map(|device| unsafe { native_device(device) })
                    .collect::<Result<Vec<_>>>()?,
                preferred: if flag(row.preferred_present)? {
                    Some(unsafe { native_device(&row.preferred) }?)
                } else {
                    None
                },
            })
        } else {
            if row.supported_count != 0 || flag(row.preferred_present)? {
                return Err(invalid(
                    "Core ML plan has devices without usage information",
                ));
            }
            None
        };
        self.functions[row.function_index]
            .operations
            .push(OperationPlacement {
                path,
                operator: unsafe { native_string(row.operator_name) }?,
                inputs,
                outputs,
                child_operation_counts: children,
                usage,
            });
        self.rows += 1;
        Ok(())
    }

    fn finish(self) -> Result<Vec<FunctionPlacement>> {
        if let Some(error) = self.error {
            return Err(invalid(error));
        }
        if !self.pending.is_empty() {
            return Err(invalid("Core ML plan traversal was incomplete"));
        }
        Ok(self.functions)
    }
}

unsafe extern "C" fn visit(
    context: *mut c_void,
    function: *const NativeFunction,
    row: *const NativeRow,
) -> i32 {
    if context.is_null() {
        return -1;
    }
    let collector = unsafe { &mut *context.cast::<Collector>() };
    if collector.error.is_some() {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        match (function.as_ref(), row.as_ref()) {
            (Some(function), None) => collector.function(function),
            (None, Some(row)) => collector.row(row),
            _ => Err(invalid("invalid Core ML plan visitor event")),
        }
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            collector.error = Some(error.to_string());
            -1
        }
        Err(_) => {
            collector.error = Some("Core ML plan visitor panicked".into());
            -1
        }
    }
}

pub(crate) fn timeout_millis(timeout: Duration) -> Result<u64> {
    let millis =
        u64::try_from(timeout.as_millis()).map_err(|_| invalid("Core ML plan timeout overflow"))?;
    if millis == 0 || millis > i64::MAX as u64 / 1_000_000 {
        return Err(invalid(
            "Core ML plan timeout must be positive and fit a native deadline",
        ));
    }
    Ok(millis)
}

pub(crate) fn inspect(
    compiled: &Path,
    units: ComputeUnits,
    timeout_ms: u64,
) -> Result<ExpectedPlacement> {
    let path = compiled
        .to_str()
        .ok_or_else(|| invalid("Core ML plan path must be UTF-8"))?;
    let path = CString::new(path).map_err(|_| invalid("Core ML plan path contains NUL"))?;
    let mut collector = Collector::default();
    let status = unsafe {
        apx_coreml_inspect_plan(
            path.as_ptr(),
            units as i32,
            timeout_ms,
            visit,
            (&mut collector as *mut Collector).cast(),
        )
    };
    if collector.error.is_none() {
        crate::runtime::check(status)?;
    }
    Ok(ExpectedPlacement {
        compiled_path: compiled.to_path_buf(),
        compute_units: units,
        functions: collector.finish()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: &str) -> NativeString {
        NativeString {
            data: value.as_ptr(),
            len: value.len(),
        }
    }

    fn row(path: &[usize]) -> NativeRow {
        NativeRow {
            function_index: 0,
            path: path.as_ptr(),
            path_len: path.len(),
            operator_name: text("test"),
            inputs: std::ptr::null(),
            input_count: 0,
            outputs: std::ptr::null(),
            output_count: 0,
            child_operations: std::ptr::null(),
            child_count: 0,
            usage_present: 0,
            supported: std::ptr::null(),
            supported_count: 0,
            preferred_present: 0,
            preferred: NativeDevice {
                kind: 0,
                class_name: text(""),
            },
        }
    }

    #[test]
    fn owned_facts_preserve_nested_empty_blocks_and_unknown_devices() -> Result<()> {
        let mut collector = Collector::default();
        let native_class = String::from("FutureComputeDevice");
        let supported = [NativeDevice {
            kind: 0,
            class_name: text(&native_class),
        }];
        let bindings = [
            NativeBinding {
                name_present: 1,
                name: text("x"),
                literal_present: 0,
            },
            NativeBinding {
                name_present: 0,
                name: text(""),
                literal_present: 1,
            },
        ];
        let inputs = [NativeInput {
            name: text("value"),
            bindings: bindings.as_ptr(),
            count: bindings.len(),
        }];
        let children = [0, 1];
        let mut parent = row(&[0]);
        parent.inputs = inputs.as_ptr();
        parent.input_count = inputs.len();
        parent.child_operations = children.as_ptr();
        parent.child_count = children.len();
        parent.usage_present = 1;
        parent.supported = supported.as_ptr();
        parent.supported_count = 1;
        unsafe {
            collector.function(&NativeFunction {
                name: text("main"),
                root_operations: 1,
            })?;
            collector.row(&parent)?;
            collector.row(&row(&[0, 1, 0]))?;
            collector.function(&NativeFunction {
                name: text("empty"),
                root_operations: 0,
            })?;
        }
        let facts = collector.finish()?;
        drop(native_class);
        assert_eq!(facts.len(), 2);
        assert!(facts[1].operations.is_empty());
        assert_eq!(facts[0].operations[1].path, [0, 1, 0]);
        let parent = &facts[0].operations[0];
        assert_eq!(parent.child_operation_counts, [0, 1]);
        assert_eq!(parent.inputs[0].bindings[0].name.as_deref(), Some("x"));
        assert!(parent.inputs[0].bindings[1].literal_present);
        let usage = parent.usage.as_ref().unwrap();
        assert_eq!(usage.supported[0].kind, ComputeDeviceKind::Unknown);
        assert_eq!(usage.supported[0].class_name, "FutureComputeDevice");
        assert!(usage.preferred.is_none());
        assert!(facts[0].operations[1].usage.is_none());
        Ok(())
    }

    #[test]
    fn incomplete_duplicate_or_malformed_rows_never_finish_successfully() -> Result<()> {
        let mut missing = Collector::default();
        unsafe {
            missing.function(&NativeFunction {
                name: text("main"),
                root_operations: 1,
            })?;
        }
        assert!(missing.finish().is_err());
        let mut duplicate = Collector::default();
        unsafe {
            duplicate.function(&NativeFunction {
                name: text("main"),
                root_operations: 1,
            })?;
            duplicate.row(&row(&[0]))?;
            assert!(duplicate.row(&row(&[0])).is_err());
        }
        let mut invalid_event = Collector::default();
        let context = (&mut invalid_event as *mut Collector).cast();
        assert_eq!(
            unsafe { visit(context, std::ptr::null(), std::ptr::null()) },
            -1
        );
        assert!(invalid_event.finish().is_err());
        assert!(unsafe {
            native_string(NativeString {
                data: [255].as_ptr(),
                len: 1,
            })
        }
        .is_err());
        assert!(unsafe { native_slice::<usize>(std::ptr::null(), 1) }.is_err());
        let mut oversized = Collector::default();
        assert!(unsafe {
            oversized.function(&NativeFunction {
                name: text("main"),
                root_operations: 100001,
            })
        }
        .is_err());
        Ok(())
    }

    #[test]
    fn missing_usage_is_distinct_from_known_empty_support() -> Result<()> {
        let mut collector = Collector::default();
        unsafe {
            collector.function(&NativeFunction {
                name: text("main"),
                root_operations: 2,
            })?;
        }
        let mut known = row(&[0]);
        known.usage_present = 1;
        unsafe {
            collector.row(&known)?;
            collector.row(&row(&[1]))?;
        }
        let facts = collector.finish()?;
        assert_eq!(
            facts[0].operations[0].usage,
            Some(DeviceUsage {
                supported: vec![],
                preferred: None
            })
        );
        assert_eq!(facts[0].operations[1].usage, None);
        Ok(())
    }

    #[test]
    fn timeout_must_fit_a_positive_native_deadline() {
        assert!(timeout_millis(Duration::ZERO).is_err());
        assert!(timeout_millis(Duration::from_nanos(1)).is_err());
        assert!(timeout_millis(Duration::MAX).is_err());
        assert_eq!(timeout_millis(Duration::from_secs(60)).unwrap(), 60000);
    }

    #[test]
    fn budget_counts_each_pending_child_once() -> Result<()> {
        for children in [[40000usize, 40000], [40000, 59999]] {
            let mut collector = Collector::default();
            let context = (&mut collector as *mut Collector).cast();
            let function = NativeFunction {
                name: text("main"),
                root_operations: 1,
            };
            assert_eq!(unsafe { visit(context, &function, std::ptr::null()) }, 0);
            let mut parent = row(&[0]);
            parent.child_operations = children.as_ptr();
            parent.child_count = children.len();
            assert_eq!(
                unsafe { visit(context, std::ptr::null(), &parent) },
                0,
                "{:?}",
                collector.error
            );
            for (block, count) in children.into_iter().enumerate() {
                for index in 0..count {
                    assert_eq!(
                        unsafe { visit(context, std::ptr::null(), &row(&[0, block, index])) },
                        0
                    );
                }
            }
            assert_eq!(
                collector.finish()?[0].operations.len(),
                1 + children.iter().sum::<usize>()
            );
        }
        let mut overflow = Collector::default();
        let context = (&mut overflow as *mut Collector).cast();
        let function = NativeFunction {
            name: text("main"),
            root_operations: 1,
        };
        assert_eq!(unsafe { visit(context, &function, std::ptr::null()) }, 0);
        let children = [40000usize, 60000];
        let mut parent = row(&[0]);
        parent.child_operations = children.as_ptr();
        parent.child_count = children.len();
        assert_eq!(unsafe { visit(context, std::ptr::null(), &parent) }, -1);
        assert!(overflow.finish().is_err());
        Ok(())
    }
}
