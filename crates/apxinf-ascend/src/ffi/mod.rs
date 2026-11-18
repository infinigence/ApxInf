//! Raw FFI bindings to the CANN ACL Runtime and aclnn operator libraries.
//!
//! Only the functions needed by `apxinf-ascend` are declared. Signatures
//! follow the ACL C headers in `/usr/local/Ascend/cann-9.1.0/x86_64-linux/include/acl/`.

pub mod acl;
pub mod aclnn;
