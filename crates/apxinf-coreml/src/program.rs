//! A typed builder for single-function FP16 ML Programs (specification 9,
//! opset `CoreML8`) over a fixed operation list. It names operations and
//! their constants as coremltools 9.0 `mb.program` does, so a program built
//! with the same calls decodes to the same message.
use crate::blob::{BlobDataType, BlobWriter};
use crate::proto::{self, binding, data_type, tensor_value, value};
use apxinf_core::{Error, Result};
use half::f16;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

const OPSET: &str = "CoreML8";
const WEIGHTS: &str = "@model_path/weights/weight.bin";
static NEXT_BUILDER_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BuilderId(usize);

impl BuilderId {
    fn new() -> Self {
        Self(NEXT_BUILDER_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A named program value: its MIL data type and static shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Var {
    builder: BuilderId,
    name: String,
    dtype: i32,
    shape: Vec<usize>,
}

impl Var {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// The second operand of a binary operation.
pub enum Operand<'a> {
    Var(&'a Var),
    /// An FP16 scalar constant.
    Scalar(f16),
}

/// Builds one MIL function from builder-owned values with unique names.
pub struct ProgramBuilder {
    id: BuilderId,
    inputs: Vec<Var>,
    operations: Vec<proto::Operation>,
    blobs: BlobWriter,
    counts: BTreeMap<&'static str, usize>,
    symbols: BTreeSet<String>,
}

fn tensor_type(dtype: i32, shape: &[usize]) -> proto::ValueType {
    proto::ValueType {
        tensor_type: Some(proto::TensorType {
            data_type: dtype,
            rank: i64::try_from(shape.len()).expect("validated shape rank"),
            dimensions: shape
                .iter()
                .map(|&size| proto::Dimension {
                    constant: Some(proto::ConstantDimension {
                        size: u64::try_from(size).expect("validated shape dimension"),
                    }),
                })
                .collect(),
        }),
    }
}

fn immediate(dtype: i32, shape: &[usize], tensor: tensor_value::Value) -> proto::Value {
    proto::Value {
        doc_string: String::new(),
        r#type: Some(tensor_type(dtype, shape)),
        value: Some(value::Value::ImmediateValue(proto::ImmediateValue {
            tensor: Some(proto::TensorValue {
                value: Some(tensor),
            }),
        })),
    }
}

fn string_value(s: &str) -> proto::Value {
    immediate(
        data_type::STRING,
        &[],
        tensor_value::Value::Strings(proto::RepeatedStrings {
            values: vec![s.to_owned()],
        }),
    )
}

fn by_name(var: &Var) -> proto::Argument {
    proto::Argument {
        arguments: vec![proto::Binding {
            binding: Some(binding::Binding::Name(var.name.clone())),
        }],
    }
}

fn contract(message: &'static str) -> Error {
    Error::Contract(message)
}

fn shape_elements(shape: &[usize]) -> Result<usize> {
    i64::try_from(shape.len()).map_err(|_| contract("ML Program rank exceeds int64"))?;
    shape.iter().try_fold(1usize, |count, &dimension| {
        if dimension == 0 {
            return Err(contract("ML Program dimensions must be positive"));
        }
        i64::try_from(dimension).map_err(|_| contract("ML Program dimension exceeds int64"))?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| contract("ML Program element count overflows usize"))
    })
}

/// Numpy-style broadcast of two static shapes of equal rank.
fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    if b.is_empty() {
        return Ok(a.to_vec());
    }
    if a.len() != b.len() {
        return Err(contract(
            "ML Program binary operands need equal rank or a scalar",
        ));
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| match (x, y) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(contract(
                "ML Program binary operand shapes do not broadcast",
            )),
        })
        .collect()
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            id: BuilderId::new(),
            inputs: Vec::new(),
            operations: Vec::new(),
            blobs: BlobWriter::new(),
            counts: BTreeMap::new(),
            symbols: BTreeSet::new(),
        }
    }

    /// An FP16 function input.
    pub fn input(&mut self, name: &str, shape: &[usize]) -> Result<Var> {
        shape_elements(shape)?;
        self.reserve_symbols(&[name.to_owned()])?;
        let var = Var {
            builder: self.id,
            name: name.to_owned(),
            dtype: data_type::FLOAT16,
            shape: shape.to_vec(),
        };
        self.inputs.push(var.clone());
        Ok(var)
    }

    fn op_name(&self, kind: &'static str, name: Option<&str>) -> String {
        match name {
            Some(name) => name.to_owned(),
            None => self.op_name_after(kind, self.counts.get(kind).copied().unwrap_or(0)),
        }
    }

    fn op_name_after(&self, kind: &'static str, mut count: usize) -> String {
        loop {
            let name = format!("{kind}_{count}");
            if !self.symbols.contains(&name) {
                return name;
            }
            count += 1;
        }
    }

    fn commit_op_name(&mut self, kind: &'static str, requested: Option<&str>, name: &str) {
        if requested.is_none() {
            let prefix = format!("{kind}_");
            let index = name
                .strip_prefix(&prefix)
                .and_then(|index| index.parse::<usize>().ok())
                .expect("auto-generated operation name");
            self.counts.insert(kind, index + 1);
        }
    }

    fn reserve_symbols(&mut self, names: &[String]) -> Result<()> {
        let mut pending = BTreeSet::new();
        for name in names {
            if name.is_empty() {
                return Err(contract("ML Program value names must be nonempty"));
            }
            if self.symbols.contains(name) || !pending.insert(name.clone()) {
                return Err(contract("ML Program value names must be unique"));
            }
        }
        self.symbols.extend(names.iter().cloned());
        Ok(())
    }

    fn require_value(&self, var: &Var) -> Result<()> {
        if var.builder != self.id {
            return Err(contract("ML Program value belongs to a different builder"));
        }
        if !self.symbols.contains(&var.name) {
            return Err(contract("ML Program value is not defined by this builder"));
        }
        Ok(())
    }

    fn push(
        &mut self,
        kind: &str,
        name: &str,
        inputs: BTreeMap<String, proto::Argument>,
        outputs: &[Var],
    ) {
        self.operations.push(proto::Operation {
            r#type: kind.to_owned(),
            inputs,
            outputs: outputs
                .iter()
                .map(|v| proto::NamedValueType {
                    name: v.name.clone(),
                    r#type: Some(tensor_type(v.dtype, &v.shape)),
                })
                .collect(),
            attributes: [("name".to_owned(), string_value(name))].into(),
        });
    }

    /// A `const` operation named `{op}_{param}_0`.
    fn constant(
        &mut self,
        op: &str,
        param: &str,
        value: proto::Value,
        dtype: i32,
        shape: &[usize],
    ) -> Var {
        let name = format!("{op}_{param}_0");
        let var = Var {
            builder: self.id,
            name: name.clone(),
            dtype,
            shape: shape.to_vec(),
        };
        self.operations.push(proto::Operation {
            r#type: "const".to_owned(),
            inputs: BTreeMap::new(),
            outputs: vec![proto::NamedValueType {
                name: name.clone(),
                r#type: Some(tensor_type(dtype, shape)),
            }],
            attributes: [
                ("name".to_owned(), string_value(&name)),
                ("val".to_owned(), value),
            ]
            .into(),
        });
        var
    }

    fn ints(&mut self, op: &str, param: &str, values: &[i32], scalar: bool) -> Var {
        let shape: &[usize] = if scalar { &[] } else { &[values.len()] };
        let value = immediate(
            data_type::INT32,
            shape,
            tensor_value::Value::Ints(proto::RepeatedInts {
                values: values.to_vec(),
            }),
        );
        self.constant(op, param, value, data_type::INT32, shape)
    }

    fn blob_value(
        &mut self,
        dtype: i32,
        shape: &[usize],
        kind: BlobDataType,
        bytes: &[u8],
    ) -> proto::Value {
        let offset = self.blobs.append(kind, bytes);
        proto::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(dtype, shape)),
            value: Some(value::Value::BlobFileValue(proto::BlobFileValue {
                file_name: WEIGHTS.to_owned(),
                offset,
            })),
        }
    }

    /// `constexpr_blockwise_shift_scale(data, scale)`: INT8 `data` times FP16
    /// `scale` broadcast over its unit dimensions, materialized at load.
    pub fn constexpr_blockwise_shift_scale(
        &mut self,
        data: &[i8],
        shape: &[usize],
        scale: &[f16],
        scale_shape: &[usize],
    ) -> Result<Var> {
        if data.len() != shape_elements(shape)?
            || scale.len() != shape_elements(scale_shape)?
            || broadcast(shape, scale_shape)? != shape
        {
            return Err(contract(
                "blockwise shift/scale data and scale must match their shapes and broadcast",
            ));
        }
        let name = self.op_name("constexpr_blockwise_shift_scale", None);
        self.reserve_symbols(std::slice::from_ref(&name))?;
        self.commit_op_name("constexpr_blockwise_shift_scale", None, &name);
        let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
        let data = self.blob_value(data_type::INT8, shape, BlobDataType::Int8, &bytes);
        let bytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale = self.blob_value(
            data_type::FLOAT16,
            scale_shape,
            BlobDataType::Float16,
            &bytes,
        );
        let arg = |v: proto::Value| proto::Argument {
            arguments: vec![proto::Binding {
                binding: Some(binding::Binding::Value(v)),
            }],
        };
        let out = Var {
            builder: self.id,
            name: name.clone(),
            dtype: data_type::FLOAT16,
            shape: shape.to_vec(),
        };
        let inputs = [
            ("data".to_owned(), arg(data)),
            ("scale".to_owned(), arg(scale)),
        ]
        .into();
        self.push(
            "constexpr_blockwise_shift_scale",
            &name,
            inputs,
            std::slice::from_ref(&out),
        );
        Ok(out)
    }

    /// A `const` FP16 tensor stored in the weight file, named `{op}_{param}_0`.
    fn fp16_blob(&mut self, op: &str, param: &str, values: &[f16], shape: &[usize]) -> Var {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let value = self.blob_value(data_type::FLOAT16, shape, BlobDataType::Float16, &bytes);
        self.constant(op, param, value, data_type::FLOAT16, shape)
    }

    /// The output shape of a conv of `x` with a `weight_shape` kernel.
    fn conv_shape(x: &Var, weight_shape: &[usize]) -> Result<Vec<usize>> {
        let (&[n, cin, h, w], &[cout, wcin, kh, kw]) = (&x.shape[..], weight_shape) else {
            return Err(contract("conv needs rank-4 input and weight"));
        };
        if cin != wcin || kh > h || kw > w {
            return Err(contract("conv weight does not fit the input"));
        }
        let shape = vec![n, cout, h - kh + 1, w - kw + 1];
        shape_elements(&shape)?;
        Ok(shape)
    }

    /// The stride, padding, dilation and group constants of conv `conv`.
    fn conv_params(&mut self, conv: &str) -> [(&'static str, Var); 5] {
        let strides = self.ints(conv, "strides", &[1, 1], false);
        let pad_type = immediate(
            data_type::STRING,
            &[],
            tensor_value::Value::Strings(proto::RepeatedStrings {
                values: vec!["valid".to_owned()],
            }),
        );
        let pad_type = self.constant(conv, "pad_type", pad_type, data_type::STRING, &[]);
        let pad = self.ints(conv, "pad", &[0, 0, 0, 0], false);
        let dilations = self.ints(conv, "dilations", &[1, 1], false);
        let groups = self.ints(conv, "groups", &[1], true);
        [
            ("strides", strides),
            ("pad_type", pad_type),
            ("pad", pad),
            ("dilations", dilations),
            ("groups", groups),
        ]
    }

    fn conv_op(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        tensors: &[(&str, &Var)],
        params: &[(&'static str, Var)],
    ) -> Var {
        let out = Var {
            builder: self.id,
            name: name.to_owned(),
            dtype: data_type::FLOAT16,
            shape,
        };
        let inputs = tensors
            .iter()
            .map(|&(k, v)| (k, v))
            .chain(params.iter().map(|(k, v)| (*k, v)))
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("conv", name, inputs, std::slice::from_ref(&out));
        out
    }

    /// 2-D convolution, stride 1, `valid` padding, dilation 1, one group.
    pub fn conv(&mut self, x: &Var, weight: &Var) -> Result<Var> {
        self.require_value(x)?;
        self.require_value(weight)?;
        let shape = Self::conv_shape(x, &weight.shape)?;
        let name = self.op_name("conv", None);
        let names = std::iter::once(name.clone())
            .chain(
                ["strides", "pad_type", "pad", "dilations", "groups"]
                    .map(|param| format!("{name}_{param}_0")),
            )
            .collect::<Vec<_>>();
        self.reserve_symbols(&names)?;
        self.commit_op_name("conv", None, &name);
        let params = self.conv_params(&name);
        Ok(self.conv_op(&name, shape, &[("x", x), ("weight", weight)], &params))
    }

    /// [`conv`](Self::conv) with FP16 weights in the weight file, as
    /// coremltools stores a NumPy FP16 weight: `{conv}_weight_0` comes first.
    pub fn conv_fp16(&mut self, x: &Var, weight: &[f16], weight_shape: &[usize]) -> Result<Var> {
        self.require_value(x)?;
        if weight.len() != shape_elements(weight_shape)? {
            return Err(contract("FP16 constant values must match their shape"));
        }
        let shape = Self::conv_shape(x, weight_shape)?;
        let name = self.op_name("conv", None);
        let names = std::iter::once(name.clone())
            .chain(std::iter::once(format!("{name}_weight_0")))
            .chain(
                ["strides", "pad_type", "pad", "dilations", "groups"]
                    .map(|param| format!("{name}_{param}_0")),
            )
            .collect::<Vec<_>>();
        self.reserve_symbols(&names)?;
        self.commit_op_name("conv", None, &name);
        let w = self.fp16_blob(&name, "weight", weight, weight_shape);
        let params = self.conv_params(&name);
        Ok(self.conv_op(&name, shape, &[("x", x), ("weight", &w)], &params))
    }

    /// `mul(conv_fp16(x, weight), scale)` after coremltools 9.0's
    /// `fuse_conv_scale` pass: one conv named `name` (the absorbed `mul`'s
    /// name) with weight `weight * scale` and a zero bias, both FP16. Reserving
    /// that symbol makes the conv counter skip it; parameter constants keep
    /// the counter-derived conv name.
    pub fn conv_fp16_scaled(
        &mut self,
        x: &Var,
        weight: &[f16],
        weight_shape: &[usize],
        scale: f16,
        name: &str,
    ) -> Result<Var> {
        self.require_value(x)?;
        if weight.len() != shape_elements(weight_shape)? {
            return Err(contract("FP16 constant values must match their shape"));
        }
        let shape = Self::conv_shape(x, weight_shape)?;
        let conv = self.op_name_after("conv", self.counts.get("conv").copied().unwrap_or(0));
        let conv = if conv == name {
            self.op_name_after(
                "conv",
                conv.strip_prefix("conv_")
                    .and_then(|index| index.parse::<usize>().ok())
                    .unwrap_or(0)
                    + 1,
            )
        } else {
            conv
        };
        let names = [
            name.to_owned(),
            format!("{name}_weight_0"),
            format!("{name}_bias_0"),
        ]
        .into_iter()
        .chain(std::iter::once(conv.clone()))
        .chain(
            ["strides", "pad_type", "pad", "dilations", "groups"]
                .map(|param| format!("{conv}_{param}_0")),
        )
        .collect::<Vec<_>>();
        self.reserve_symbols(&names)?;
        self.commit_op_name("conv", None, &conv);
        let params = self.conv_params(&conv);
        let scaled: Vec<f16> = weight
            .iter()
            .map(|w| f16::from_f32(w.to_f32() * scale.to_f32()))
            .collect();
        let w = self.fp16_blob(name, "weight", &scaled, weight_shape);
        let b = self.fp16_blob(
            name,
            "bias",
            &vec![f16::ZERO; weight_shape[0]],
            &weight_shape[..1],
        );
        Ok(self.conv_op(
            name,
            shape,
            &[("x", x), ("weight", &w), ("bias", &b)],
            &params,
        ))
    }

    fn binary(
        &mut self,
        kind: &'static str,
        x: &Var,
        y: Operand<'_>,
        name: Option<&str>,
    ) -> Result<Var> {
        self.require_value(x)?;
        if let Operand::Var(var) = &y {
            self.require_value(var)?;
        }
        let shape = broadcast(
            &x.shape,
            match &y {
                Operand::Var(v) => &v.shape,
                Operand::Scalar(_) => &[],
            },
        )?;
        shape_elements(&shape)?;
        let requested_name = name;
        let name = self.op_name(kind, requested_name);
        let names = match &y {
            Operand::Var(_) => vec![name.clone()],
            Operand::Scalar(_) => vec![name.clone(), format!("{name}_y_0")],
        };
        self.reserve_symbols(&names)?;
        self.commit_op_name(kind, requested_name, &name);
        let y = match y {
            Operand::Var(v) => v.clone(),
            Operand::Scalar(s) => {
                let value = immediate(
                    data_type::FLOAT16,
                    &[],
                    tensor_value::Value::Bytes(proto::RepeatedBytes {
                        values: s.to_le_bytes().to_vec(),
                    }),
                );
                self.constant(&name, "y", value, data_type::FLOAT16, &[])
            }
        };
        let out = Var {
            builder: self.id,
            name: name.clone(),
            dtype: data_type::FLOAT16,
            shape,
        };
        let inputs = [("x".to_owned(), by_name(x)), ("y".to_owned(), by_name(&y))].into();
        self.push(kind, &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    pub fn mul(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("mul", x, y, name)
    }
    pub fn add(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("add", x, y, name)
    }
    pub fn real_div(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("real_div", x, y, name)
    }
    pub fn maximum(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("maximum", x, y, name)
    }

    fn unary(&mut self, kind: &'static str, x: &Var, name: Option<&str>) -> Result<Var> {
        self.require_value(x)?;
        let requested_name = name;
        let name = self.op_name(kind, requested_name);
        self.reserve_symbols(std::slice::from_ref(&name))?;
        self.commit_op_name(kind, requested_name, &name);
        let out = Var {
            builder: self.id,
            name: name.clone(),
            ..x.clone()
        };
        self.push(
            kind,
            &name,
            [("x".to_owned(), by_name(x))].into(),
            std::slice::from_ref(&out),
        );
        Ok(out)
    }

    pub fn tanh(&mut self, x: &Var) -> Result<Var> {
        self.unary("tanh", x, None)
    }
    pub fn abs(&mut self, x: &Var) -> Result<Var> {
        self.unary("abs", x, None)
    }
    pub fn identity(&mut self, x: &Var, name: Option<&str>) -> Result<Var> {
        self.unary("identity", x, name)
    }

    /// `reduce_max` over `axes` (non-negative).
    pub fn reduce_max(&mut self, x: &Var, axes: &[i32], keep_dims: bool) -> Result<Var> {
        self.require_value(x)?;
        if axes
            .iter()
            .any(|&a| usize::try_from(a).map_or(true, |a| a >= x.shape.len()))
        {
            return Err(contract("reduce_max axes must index the input"));
        }
        let name = self.op_name("reduce_max", None);
        self.reserve_symbols(&[
            name.clone(),
            format!("{name}_axes_0"),
            format!("{name}_keep_dims_0"),
        ])?;
        self.commit_op_name("reduce_max", None, &name);
        let axes_var = self.ints(&name, "axes", axes, false);
        let keep = immediate(
            data_type::BOOL,
            &[],
            tensor_value::Value::Bools(proto::RepeatedBools {
                values: vec![keep_dims],
            }),
        );
        let keep = self.constant(&name, "keep_dims", keep, data_type::BOOL, &[]);
        let shape = x
            .shape
            .iter()
            .enumerate()
            .filter_map(
                |(i, &d)| match axes.iter().any(|&a| usize::try_from(a) == Ok(i)) {
                    true if keep_dims => Some(1),
                    true => None,
                    false => Some(d),
                },
            )
            .collect();
        let out = Var {
            builder: self.id,
            name: name.clone(),
            dtype: x.dtype,
            shape,
        };
        let inputs = [("x", x), ("axes", &axes_var), ("keep_dims", &keep)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("reduce_max", &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    /// `split` along a non-negative `axis` into parts of `sizes`.
    pub fn split(&mut self, x: &Var, sizes: &[usize], axis: usize) -> Result<Vec<Var>> {
        self.require_value(x)?;
        let total = sizes.iter().try_fold(0usize, |sum, &size| {
            sum.checked_add(size)
                .ok_or_else(|| contract("ML Program split size sum overflows usize"))
        })?;
        if axis >= x.shape.len() || total != x.shape[axis] || sizes.is_empty() || sizes.contains(&0)
        {
            return Err(contract("split sizes must add up to the axis"));
        }
        let sizes_i32: Vec<i32> = sizes
            .iter()
            .map(|&s| i32::try_from(s).map_err(|_| contract("ML Program split size exceeds int32")))
            .collect::<Result<_>>()?;
        let axis_i32 =
            i32::try_from(axis).map_err(|_| contract("ML Program split axis exceeds int32"))?;
        let name = self.op_name("split", None);
        let names = std::iter::once(name.clone())
            .chain([format!("{name}_split_sizes_0"), format!("{name}_axis_0")])
            .chain((0..sizes.len()).map(|index| format!("{name}_{index}")))
            .collect::<Vec<_>>();
        self.reserve_symbols(&names)?;
        self.commit_op_name("split", None, &name);
        let sizes_var = self.ints(&name, "split_sizes", &sizes_i32, false);
        let axis_var = self.ints(&name, "axis", &[axis_i32], true);
        let outs: Vec<Var> = sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let mut shape = x.shape.clone();
                shape[axis] = s;
                Var {
                    builder: self.id,
                    name: format!("{name}_{i}"),
                    dtype: x.dtype,
                    shape,
                }
            })
            .collect();
        let inputs = [("x", x), ("split_sizes", &sizes_var), ("axis", &axis_var)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("split", &name, inputs, &outs);
        Ok(outs)
    }

    /// The package: one `main` function returning `outputs`, all FP16.
    pub fn finish(self, outputs: &[&Var]) -> Result<Package> {
        if outputs.is_empty() {
            return Err(contract("ML Program needs at least one output"));
        }
        let mut names = BTreeSet::new();
        for output in outputs {
            self.require_value(output)?;
            if output.dtype != data_type::FLOAT16 {
                return Err(contract("ML Program outputs must be FP16 values"));
            }
            if !names.insert(output.name.as_str()) {
                return Err(contract("ML Program output names must be unique"));
            }
        }
        let feature = |v: &Var| proto::FeatureDescription {
            name: v.name.clone(),
            short_description: String::new(),
            r#type: Some(proto::FeatureType {
                multi_array_type: Some(proto::ArrayFeatureType {
                    shape: v
                        .shape
                        .iter()
                        .map(|&d| i64::try_from(d).expect("validated shape dimension"))
                        .collect(),
                    data_type: proto::ARRAY_FLOAT16,
                }),
                is_optional: false,
            }),
        };
        let block = proto::Block {
            inputs: Vec::new(),
            outputs: outputs.iter().map(|v| v.name.clone()).collect(),
            operations: self.operations,
            attributes: BTreeMap::new(),
        };
        let function = proto::Function {
            inputs: self
                .inputs
                .iter()
                .map(|v| proto::NamedValueType {
                    name: v.name.clone(),
                    r#type: Some(tensor_type(v.dtype, &v.shape)),
                })
                .collect(),
            opset: OPSET.to_owned(),
            block_specializations: [(OPSET.to_owned(), block)].into(),
            attributes: BTreeMap::new(),
        };
        let model = proto::Model {
            specification_version: 9,
            description: Some(proto::ModelDescription {
                input: self.inputs.iter().map(feature).collect(),
                output: outputs.iter().map(|v| feature(v)).collect(),
                metadata: Some(proto::Metadata {
                    short_description: String::new(),
                    user_defined: [(
                        "com.github.apxinf.writer".to_owned(),
                        concat!("apxinf-coreml ", env!("CARGO_PKG_VERSION")).to_owned(),
                    )]
                    .into(),
                }),
            }),
            is_updatable: false,
            ml_program: Some(proto::Program {
                version: 1,
                functions: [("main".to_owned(), function)].into(),
                doc_string: String::new(),
                attributes: BTreeMap::new(),
            }),
        };
        Ok(Package {
            model,
            weights: self.blobs.finish(),
        })
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// An ML Program package: the specification and its weight file.
pub struct Package {
    pub model: proto::Model,
    pub weights: Vec<u8>,
}

const MANIFEST: &str = r#"{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {
        "6C1A3E50-63A5-4E0F-9B53-0D5B4E1E7A01": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel"
        },
        "6C1A3E50-63A5-4E0F-9B53-0D5B4E1E7A02": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Weights",
            "name": "weights",
            "path": "com.apple.CoreML/weights"
        }
    },
    "rootModelIdentifier": "6C1A3E50-63A5-4E0F-9B53-0D5B4E1E7A01"
}
"#;

impl Package {
    pub fn model_bytes(&self) -> Vec<u8> {
        prost::Message::encode_to_vec(&self.model)
    }

    /// Writes `dir` as an `.mlpackage` directory. `dir` must not exist.
    pub fn write(&self, dir: &Path) -> Result<()> {
        let io = |e: std::io::Error| Error::Other(format!("cannot write ML Program package: {e}"));
        let data = dir.join("Data/com.apple.CoreML");
        std::fs::create_dir_all(data.join("weights")).map_err(io)?;
        std::fs::write(dir.join("Manifest.json"), MANIFEST).map_err(io)?;
        std::fs::write(data.join("model.mlmodel"), self.model_bytes()).map_err(io)?;
        std::fs::write(data.join("weights/weight.bin"), &self.weights).map_err(io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_coremltools_counters_and_named_ops() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 4, 1, 8]).unwrap();
        let w = b
            .constexpr_blockwise_shift_scale(&[1; 8], &[2, 4, 1, 1], &[f16::ONE; 2], &[2, 1, 1, 1])
            .unwrap();
        let c = b.conv(&x, &w).unwrap();
        assert_eq!((c.name(), c.shape()), ("conv_0", &[1, 2, 1, 8][..]));
        let m = b
            .mul(&c, Operand::Scalar(f16::from_f32(0.5)), None)
            .unwrap();
        let s = b
            .maximum(&m, Operand::Scalar(f16::ONE), Some("gate_scale"))
            .unwrap();
        let m2 = b.mul(&m, Operand::Var(&s), None).unwrap();
        let r = b.reduce_max(&m2, &[1], true).unwrap();
        let parts = b.split(&m2, &[1, 1], 1).unwrap();
        let y = b.identity(&parts[0], Some("y")).unwrap();
        assert_eq!(
            [
                m.name(),
                s.name(),
                m2.name(),
                r.name(),
                parts[1].name(),
                y.name()
            ],
            [
                "mul_0",
                "gate_scale",
                "mul_1",
                "reduce_max_0",
                "split_0_1",
                "y"
            ]
        );
        assert_eq!(r.shape(), &[1, 1, 1, 8]);
        let names: Vec<String> = b
            .operations
            .iter()
            .map(|op| op.outputs[0].name.clone())
            .collect();
        assert_eq!(
            &names[..8],
            [
                "constexpr_blockwise_shift_scale_0",
                "conv_0_strides_0",
                "conv_0_pad_type_0",
                "conv_0_pad_0",
                "conv_0_dilations_0",
                "conv_0_groups_0",
                "conv_0",
                "mul_0_y_0"
            ]
        );
        let package = b.finish(&[&y]).unwrap();
        let decoded: proto::Model = prost::Message::decode(&package.model_bytes()[..]).unwrap();
        assert_eq!(decoded, package.model);
    }

    /// The order and names of coremltools 9.0's FP16 conv programs: a NumPy FP16
    /// conv stores its weight first; a fused `conv * 0.5` keeps the conv's
    /// parameter names, takes the mul's name and adds a zero bias.
    #[test]
    fn fp16_convs_follow_coremltools_storage_and_fusion() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 4, 1, 8]).unwrap();
        let w = [f16::from_f32(3.0); 8];
        let fused = b
            .conv_fp16_scaled(&x, &w, &[2, 4, 1, 1], f16::from_f32(0.5), "mul_0")
            .unwrap();
        let plain = b.conv_fp16(&x, &w, &[2, 4, 1, 1]).unwrap();
        let m = b.mul(&fused, Operand::Var(&plain), None).unwrap();
        assert_eq!(
            (fused.name(), plain.name(), m.name()),
            ("mul_0", "conv_1", "mul_1")
        );
        let names: Vec<&str> = b
            .operations
            .iter()
            .map(|op| op.outputs[0].name.as_str())
            .collect();
        assert_eq!(
            &names[..14],
            [
                "conv_0_strides_0",
                "conv_0_pad_type_0",
                "conv_0_pad_0",
                "conv_0_dilations_0",
                "conv_0_groups_0",
                "mul_0_weight_0",
                "mul_0_bias_0",
                "mul_0",
                "conv_1_weight_0",
                "conv_1_strides_0",
                "conv_1_pad_type_0",
                "conv_1_pad_0",
                "conv_1_dilations_0",
                "conv_1_groups_0",
            ]
        );
        assert!(b.operations[7].inputs.contains_key("bias"));
        let offsets: Vec<u64> = [5, 6, 8]
            .iter()
            .map(|&i| match &b.operations[i].attributes["val"].value {
                Some(value::Value::BlobFileValue(blob)) => blob.offset,
                _ => panic!("{} is not stored in the weight file", names[i]),
            })
            .collect();
        let package = b.finish(&[&m]).unwrap();
        // In operation order: the scaled weight (1.5), the zero bias, the plain weight (3.0).
        let halves = |offset: u64| -> Vec<f32> {
            let (dtype, bytes) = crate::blob::read(&package.weights, offset).unwrap();
            assert_eq!(dtype, BlobDataType::Float16 as u32);
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        assert!(offsets[0] < offsets[1] && offsets[1] < offsets[2]);
        assert_eq!(halves(offsets[0]), [1.5; 8]);
        assert_eq!(halves(offsets[1]), [0.0; 2]);
        assert_eq!(halves(offsets[2]), [3.0; 8]);
    }

    #[test]
    fn invalid_programs_are_rejected() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 4, 1, 8]).unwrap();
        assert!(b
            .constexpr_blockwise_shift_scale(&[1; 7], &[2, 4, 1, 1], &[f16::ONE; 2], &[2, 1, 1, 1])
            .is_err());
        let w = b
            .constexpr_blockwise_shift_scale(&[1; 6], &[2, 3, 1, 1], &[f16::ONE; 2], &[2, 1, 1, 1])
            .unwrap();
        assert!(b.conv(&x, &w).is_err());
        assert!(b.split(&x, &[3], 1).is_err());
        assert!(b.reduce_max(&x, &[4], true).is_err());
        let wide = b.input("z", &[1, 3, 1, 8]).unwrap();
        assert!(b.mul(&x, Operand::Var(&wide), None).is_err());
    }

    #[test]
    fn values_cannot_cross_builder_boundaries() {
        let mut first = ProgramBuilder::new();
        let foreign = first.input("x", &[1]).unwrap();

        let mut second = ProgramBuilder::new();
        let before = (
            second.inputs.clone(),
            second.operations.clone(),
            second.counts.clone(),
            second.symbols.clone(),
        );
        assert!(second
            .identity(&foreign, None)
            .unwrap_err()
            .to_string()
            .contains("ML Program value belongs to a different builder"));
        assert_eq!(second.inputs, before.0);
        assert_eq!(second.operations, before.1);
        assert_eq!(second.counts, before.2);
        assert_eq!(second.symbols, before.3);

        let local = second.input("local", &[1]).unwrap();
        let before = (
            second.inputs.clone(),
            second.operations.clone(),
            second.counts.clone(),
            second.symbols.clone(),
        );
        assert!(second
            .add(&local, Operand::Var(&foreign), None)
            .unwrap_err()
            .to_string()
            .contains("ML Program value belongs to a different builder"));
        assert!(second
            .conv(&local, &foreign)
            .unwrap_err()
            .to_string()
            .contains("ML Program value belongs to a different builder"));
        assert_eq!(second.inputs, before.0);
        assert_eq!(second.operations, before.1);
        assert_eq!(second.counts, before.2);
        assert_eq!(second.symbols, before.3);

        assert!(second
            .finish(&[&foreign])
            .err()
            .unwrap()
            .to_string()
            .contains("ML Program value belongs to a different builder"));

        let mut third = ProgramBuilder::new();
        let local = third.input("x", &[1]).unwrap();
        assert!(third
            .finish(&[&local, &local])
            .err()
            .unwrap()
            .to_string()
            .contains("ML Program output names must be unique"));

        let empty = ProgramBuilder::new();
        assert!(empty
            .finish(&[])
            .err()
            .unwrap()
            .to_string()
            .contains("ML Program needs at least one output"));
    }

    #[test]
    fn duplicate_and_empty_symbols_are_rejected_without_mutation() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1]).unwrap();
        let before = (
            b.inputs.clone(),
            b.operations.clone(),
            b.counts.clone(),
            b.symbols.clone(),
        );
        assert!(b
            .input("x", &[1])
            .unwrap_err()
            .to_string()
            .contains("ML Program value names must be unique"));
        assert!(b
            .identity(&x, Some("x"))
            .unwrap_err()
            .to_string()
            .contains("ML Program value names must be unique"));
        assert!(b
            .input("", &[1])
            .unwrap_err()
            .to_string()
            .contains("ML Program value names must be nonempty"));
        assert!(b
            .identity(&x, Some(""))
            .unwrap_err()
            .to_string()
            .contains("ML Program value names must be nonempty"));
        assert_eq!(b.inputs, before.0);
        assert_eq!(b.operations, before.1);
        assert_eq!(b.counts, before.2);
        assert_eq!(b.symbols, before.3);

        b.input("mul_0", &[1]).unwrap();
        let before = (b.operations.clone(), b.counts.clone(), b.symbols.clone());
        let y = b.mul(&x, Operand::Scalar(f16::ONE), None).unwrap();
        assert_ne!(y.name(), "mul_0");
        assert_eq!(b.operations.len(), before.0.len() + 2);
        assert_eq!(b.counts["mul"], 2);
        assert!(b.symbols.contains("mul_0"));
        assert!(b.symbols.contains("mul_1"));
    }

    #[test]
    fn invalid_input_shapes_leave_the_builder_unchanged() {
        let mut b = ProgramBuilder::new();
        for shape in [&[0][..], &[2, 0], &[usize::MAX, 2], &[i64::MAX as usize, 3]] {
            assert!(b.input("invalid", shape).is_err());
            assert!(b.inputs.is_empty());
            assert!(b.operations.is_empty());
            assert!(b.counts.is_empty());
        }
        let scalar = b.input("scalar", &[]).unwrap();
        let package = b.finish(&[&scalar]).unwrap();
        assert!(package.model.description.unwrap().input[0]
            .r#type
            .as_ref()
            .unwrap()
            .multi_array_type
            .as_ref()
            .unwrap()
            .shape
            .is_empty());
    }

    #[test]
    fn invalid_weight_shapes_do_not_append_blobs_or_operations() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1]).unwrap();
        for shape in [&[0][..], &[usize::MAX, 2], &[i64::MAX as usize, 3]] {
            assert!(b
                .constexpr_blockwise_shift_scale(&[], shape, &[f16::ONE], &[])
                .is_err());
        }
        assert!(b
            .constexpr_blockwise_shift_scale(&[1], &[1], &[], &[usize::MAX, 2])
            .is_err());
        assert!(b.operations.is_empty());
        assert!(b.counts.is_empty());
        let actual = b.finish(&[&x]).unwrap();
        let mut clean = ProgramBuilder::new();
        let x = clean.input("x", &[1]).unwrap();
        let expected = clean.finish(&[&x]).unwrap();
        assert_eq!(actual.model_bytes(), expected.model_bytes());
        assert_eq!(actual.weights, expected.weights);
    }

    #[test]
    fn split_rejects_overflow_truncation_and_zero_without_mutation() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[2]).unwrap();
        let large = b.input("large", &[i32::MAX as usize + 1]).unwrap();
        for sizes in [&[usize::MAX, 3][..], &[0, 2], &[]] {
            assert!(b.split(&x, sizes, 0).is_err());
        }
        assert!(b.split(&large, &[i32::MAX as usize + 1], 0).is_err());
        assert!(b.operations.is_empty());
        assert!(b.counts.is_empty());
        let parts = b.split(&x, &[1, 1], 0).unwrap();
        assert_eq!(parts[0].name(), "split_0_0");
        let package = b.finish(&[&parts[0]]).unwrap();
        assert_eq!(&package.weights[0..4], &0u32.to_le_bytes());
    }

    #[test]
    fn derived_shapes_are_checked_before_emitting_constants() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[i64::MAX as usize, 1]).unwrap();
        let y = b.input("y", &[1, 3]).unwrap();
        assert!(b.mul(&x, Operand::Var(&y), None).is_err());
        assert!(b.counts.is_empty());
        assert!(b.operations.is_empty());
        let x = b.input("image", &[i64::MAX as usize, 1, 1, 1]).unwrap();
        let w = b
            .constexpr_blockwise_shift_scale(&[1; 3], &[3, 1, 1, 1], &[f16::ONE], &[])
            .unwrap();
        let before = b.operations.clone();
        let counts = b.counts.clone();
        assert!(b.conv(&x, &w).is_err());
        assert_eq!(b.operations, before);
        assert_eq!(b.counts, counts);
    }

    #[test]
    fn scalar_and_int32_boundary_shapes_remain_supported() {
        let mut b = ProgramBuilder::new();
        let scalar = b
            .constexpr_blockwise_shift_scale(&[2], &[], &[f16::ONE], &[])
            .unwrap();
        let scalar = b.mul(&scalar, Operand::Scalar(f16::ONE), None).unwrap();
        assert!(scalar.shape().is_empty());
        let large = b.input("large", &[i32::MAX as usize]).unwrap();
        let parts = b.split(&large, &[i32::MAX as usize], 0).unwrap();
        assert_eq!(parts[0].shape(), &[i32::MAX as usize]);
        let package = b.finish(&[&scalar, &parts[0]]).unwrap();
        let description = package.model.description.unwrap();
        assert_eq!(
            description.output[1]
                .r#type
                .as_ref()
                .unwrap()
                .multi_array_type
                .as_ref()
                .unwrap()
                .shape,
            &[i64::from(i32::MAX)]
        );
    }

    #[test]
    fn invalid_fp16_weights_leave_the_builder_unchanged() {
        for scaled in [false, true] {
            let mut b = ProgramBuilder::new();
            let x = b.input("x", &[1, 4, 1, 8]).unwrap();
            let symbols = b.symbols.clone();
            for (weight, shape) in [
                (&[][..], &[2, 4, 1, 1][..]),
                (&[][..], &[0, 4, 1, 1][..]),
                (&[][..], &[usize::MAX, 4, 1, 1][..]),
                (&[f16::ONE; 16][..], &[2, 4, 2, 1][..]),
            ] {
                let result = if scaled {
                    b.conv_fp16_scaled(&x, weight, shape, f16::ONE, "mul_0")
                } else {
                    b.conv_fp16(&x, weight, shape)
                };
                assert!(result.is_err());
                assert!(b.operations.is_empty());
                assert!(b.counts.is_empty());
                assert_eq!(b.symbols, symbols);
            }
            let package = b.finish(&[&x]).unwrap();
            assert_eq!(package.weights.len(), 64);
            assert_eq!(&package.weights[0..4], &0u32.to_le_bytes());
        }
    }

    #[test]
    fn fp16_output_shape_overflow_precedes_names_and_constants() {
        for scaled in [false, true] {
            let mut b = ProgramBuilder::new();
            let x = b.input("x", &[i64::MAX as usize, 1, 1, 1]).unwrap();
            let symbols = b.symbols.clone();
            let result = if scaled {
                b.conv_fp16_scaled(&x, &[f16::ONE; 3], &[3, 1, 1, 1], f16::ONE, "mul_0")
            } else {
                b.conv_fp16(&x, &[f16::ONE; 3], &[3, 1, 1, 1])
            };
            assert!(result.is_err());
            assert!(b.operations.is_empty());
            assert!(b.counts.is_empty());
            assert_eq!(b.symbols, symbols);
            let package = b.finish(&[&x]).unwrap();
            assert_eq!(package.weights.len(), 64);
        }
    }
}
