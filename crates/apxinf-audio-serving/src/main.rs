// The audio service relies on Unix-only APIs (O_NOFOLLOW, flock, POSIX signals),
// so non-Unix targets build a stub that reports the platform as unsupported.
#[cfg(unix)]
mod serve;

#[cfg(unix)]
fn main() {
    serve::main();
}

#[cfg(not(unix))]
fn main() {
    eprintln!("audio service: apxinf-audio-serve requires a Unix host");
    std::process::exit(2);
}
