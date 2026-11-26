use crate::{
    config::sha256,
    contracts::{MediaGrant, MAX_OUTPUT, MEDIA_QUOTA},
    wav,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub struct MediaRoot {
    pub path: PathBuf,
    pub reserved: AtomicUsize,
}
impl MediaRoot {
    pub fn create(parent: PathBuf) -> Result<Arc<Self>, String> {
        let path = parent.join(format!("apxinf-audio-{}", uuid::Uuid::new_v4()));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            path,
            reserved: AtomicUsize::new(0),
        }))
    }
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Arc<MediaLease>, String> {
        self.reserved
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                n.checked_add(bytes).filter(|v| *v <= MEDIA_QUOTA)
            })
            .map_err(|_| "media_quota")?;
        Ok(Arc::new(MediaLease {
            root: self.clone(),
            bytes,
            handles: std::sync::Mutex::new(Vec::new()),
        }))
    }
}
use std::os::unix::fs::DirBuilderExt;
impl Drop for MediaRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.path);
    }
}

pub struct MediaLease {
    pub root: Arc<MediaRoot>,
    bytes: usize,
    handles: std::sync::Mutex<Vec<String>>,
}
impl MediaLease {
    pub fn add_handle(&self, handle: &str) {
        self.handles.lock().unwrap().push(handle.into());
    }
    pub fn publish(&self, handle: &str, bytes: &[u8]) -> Result<(), String> {
        self.add_handle(handle);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(self.root.path.join(handle))
            .map_err(|e| e.to_string())?;
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.set_permissions(fs::Permissions::from_mode(0o400))
            .map_err(|e| e.to_string())
    }
    pub fn read_output(
        &self,
        grant: &MediaGrant,
        expected_handle: &str,
    ) -> Result<Vec<u8>, String> {
        if grant.handle != expected_handle || grant.bytes > MAX_OUTPUT as u64 {
            return Err("invalid output grant".into());
        }
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(self.root.path.join(&grant.handle))
            .map_err(|e| e.to_string())?;
        let before = file.metadata().map_err(|e| e.to_string())?;
        if !before.is_file() || before.len() != grant.bytes || before.nlink() != 1 {
            return Err("invalid output file".into());
        }
        let mut bytes = Vec::with_capacity(grant.bytes as usize);
        (&mut file)
            .take(grant.bytes + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        let after = file.metadata().map_err(|e| e.to_string())?;
        if before.ino() != after.ino()
            || before.len() != after.len()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || bytes.len() as u64 != grant.bytes
            || sha256(&bytes) != grant.sha256
        {
            return Err("output file changed".into());
        }
        let info = wav::inspect(&bytes, false)?;
        if info.frames != grant.frames || info.sample_rate as u64 != grant.sample_rate {
            return Err("output shape mismatch".into());
        }
        Ok(bytes)
    }
}
impl Drop for MediaLease {
    fn drop(&mut self) {
        for handle in self.handles.get_mut().unwrap().iter() {
            let _ = fs::remove_file(self.root.path.join(handle));
        }
        self.root.reserved.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

pub struct ResponseMedia {
    pub bytes: Vec<u8>,
    pub lease: Arc<MediaLease>,
}
impl AsRef<[u8]> for ResponseMedia {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

pub struct MetalLock {
    _file: File,
}
impl MetalLock {
    pub fn acquire(path: &std::path::Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .map_err(|e| e.to_string())?;
        let meta = file.metadata().map_err(|e| e.to_string())?;
        if !meta.is_file()
            || meta.uid() != unsafe { libc::geteuid() }
            || unsafe {
                libc::flock(
                    std::os::fd::AsRawFd::as_raw_fd(&file),
                    libc::LOCK_EX | libc::LOCK_NB,
                )
            } != 0
        {
            return Err("Metal device lock is unavailable".into());
        }
        Ok(Self { _file: file })
    }
}

pub fn normal_pressure() -> bool {
    #[cfg(target_os = "macos")]
    {
        let mut level: libc::c_int = 0;
        let mut size = std::mem::size_of_val(&level);
        let result = unsafe {
            libc::sysctlbyname(
                c"kern.memorystatus_vm_pressure_level".as_ptr(),
                (&mut level as *mut libc::c_int).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        result == 0 && size == std::mem::size_of_val(&level) && level == 1
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
