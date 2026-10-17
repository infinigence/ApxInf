//! The subset of the Core ML specification (coremltools 9.0 `Model.proto`,
//! `FeatureTypes.proto` and `MIL.proto`, BSD-3-Clause; see NOTICE) that an
//! ML Program package needs. Field numbers are the specification's. Fields
//! outside the subset are skipped when decoding.
use std::collections::BTreeMap;

#[derive(Clone, PartialEq, prost::Message)]
pub struct Model {
    #[prost(int32, tag = "1")]
    pub specification_version: i32,
    #[prost(message, optional, tag = "2")]
    pub description: Option<ModelDescription>,
    #[prost(bool, tag = "10")]
    pub is_updatable: bool,
    #[prost(message, optional, tag = "502")]
    pub ml_program: Option<Program>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ModelDescription {
    #[prost(message, repeated, tag = "1")]
    pub input: Vec<FeatureDescription>,
    #[prost(message, repeated, tag = "10")]
    pub output: Vec<FeatureDescription>,
    #[prost(message, optional, tag = "100")]
    pub metadata: Option<Metadata>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct FeatureDescription {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "2")]
    pub short_description: String,
    #[prost(message, optional, tag = "3")]
    pub r#type: Option<FeatureType>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Metadata {
    #[prost(string, tag = "1")]
    pub short_description: String,
    #[prost(btree_map = "string, string", tag = "100")]
    pub user_defined: BTreeMap<String, String>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct FeatureType {
    /// `oneof Type { ArrayFeatureType multiArrayType = 5; }`; other kinds are
    /// outside the subset.
    #[prost(message, optional, tag = "5")]
    pub multi_array_type: Option<ArrayFeatureType>,
    #[prost(bool, tag = "1000")]
    pub is_optional: bool,
}

/// `ArrayFeatureType.ArrayDataType.FLOAT16`.
pub const ARRAY_FLOAT16: i32 = 0x10000 | 16;

#[derive(Clone, PartialEq, prost::Message)]
pub struct ArrayFeatureType {
    #[prost(int64, repeated, tag = "1")]
    pub shape: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Program {
    #[prost(int64, tag = "1")]
    pub version: i64,
    #[prost(btree_map = "string, message", tag = "2")]
    pub functions: BTreeMap<String, Function>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
    #[prost(btree_map = "string, message", tag = "4")]
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Function {
    #[prost(message, repeated, tag = "1")]
    pub inputs: Vec<NamedValueType>,
    #[prost(string, tag = "2")]
    pub opset: String,
    #[prost(btree_map = "string, message", tag = "3")]
    pub block_specializations: BTreeMap<String, Block>,
    #[prost(btree_map = "string, message", tag = "4")]
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Block {
    #[prost(message, repeated, tag = "1")]
    pub inputs: Vec<NamedValueType>,
    #[prost(string, repeated, tag = "2")]
    pub outputs: Vec<String>,
    #[prost(message, repeated, tag = "3")]
    pub operations: Vec<Operation>,
    #[prost(btree_map = "string, message", tag = "4")]
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Argument {
    #[prost(message, repeated, tag = "1")]
    pub arguments: Vec<Binding>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Binding {
    #[prost(oneof = "binding::Binding", tags = "1, 2")]
    pub binding: Option<binding::Binding>,
}

pub mod binding {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Binding {
        #[prost(string, tag = "1")]
        Name(String),
        #[prost(message, tag = "2")]
        Value(super::Value),
    }
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Operation {
    #[prost(string, tag = "1")]
    pub r#type: String,
    #[prost(btree_map = "string, message", tag = "2")]
    pub inputs: BTreeMap<String, Argument>,
    #[prost(message, repeated, tag = "3")]
    pub outputs: Vec<NamedValueType>,
    #[prost(btree_map = "string, message", tag = "5")]
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct NamedValueType {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<ValueType>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ValueType {
    /// `oneof type { TensorType tensorType = 1; }`; other kinds are outside
    /// the subset.
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorType>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorType {
    #[prost(int32, tag = "1")]
    pub data_type: i32,
    #[prost(int64, tag = "2")]
    pub rank: i64,
    #[prost(message, repeated, tag = "3")]
    pub dimensions: Vec<Dimension>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Dimension {
    /// `oneof dimension { ConstantDimension constant = 1; }`.
    #[prost(message, optional, tag = "1")]
    pub constant: Option<ConstantDimension>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ConstantDimension {
    #[prost(uint64, tag = "1")]
    pub size: u64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Value {
    #[prost(string, tag = "1")]
    pub doc_string: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<ValueType>,
    #[prost(oneof = "value::Value", tags = "3, 5")]
    pub value: Option<value::Value>,
}

pub mod value {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Value {
        #[prost(message, tag = "3")]
        ImmediateValue(super::ImmediateValue),
        #[prost(message, tag = "5")]
        BlobFileValue(super::BlobFileValue),
    }
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ImmediateValue {
    /// `oneof value { TensorValue tensor = 1; }`.
    #[prost(message, optional, tag = "1")]
    pub tensor: Option<TensorValue>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct BlobFileValue {
    #[prost(string, tag = "1")]
    pub file_name: String,
    #[prost(uint64, tag = "2")]
    pub offset: u64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorValue {
    #[prost(oneof = "tensor_value::Value", tags = "2, 3, 4, 7")]
    pub value: Option<tensor_value::Value>,
}

pub mod tensor_value {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Value {
        #[prost(message, tag = "2")]
        Ints(super::RepeatedInts),
        #[prost(message, tag = "3")]
        Bools(super::RepeatedBools),
        #[prost(message, tag = "4")]
        Strings(super::RepeatedStrings),
        #[prost(message, tag = "7")]
        Bytes(super::RepeatedBytes),
    }
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct RepeatedInts {
    #[prost(int32, repeated, tag = "1")]
    pub values: Vec<i32>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct RepeatedBools {
    #[prost(bool, repeated, tag = "1")]
    pub values: Vec<bool>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct RepeatedStrings {
    #[prost(string, repeated, tag = "1")]
    pub values: Vec<String>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct RepeatedBytes {
    #[prost(bytes = "vec", tag = "1")]
    pub values: Vec<u8>,
}

/// `MIL.proto` `DataType` values the writer uses.
pub mod data_type {
    pub const BOOL: i32 = 1;
    pub const STRING: i32 = 2;
    pub const FLOAT16: i32 = 10;
    pub const INT8: i32 = 21;
    pub const INT32: i32 = 23;
}
