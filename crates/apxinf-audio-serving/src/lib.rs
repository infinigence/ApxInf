// Unix-only: the media and config layers use O_NOFOLLOW, flock and Unix file
// metadata. On other targets this crate is intentionally empty so a
// whole-workspace build still compiles.
#![cfg(unix)]

pub mod config;
pub mod connection;
pub mod contracts;
pub mod gateway;
pub mod media;
pub mod service;
pub mod wav;
