//! 各框架 crate 共用的文件系统安全原语。

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

/// 为一个文件承载的权威（authority）持有的、与其生命周期绑定的独占租约。
#[derive(Debug)]
pub struct ExclusiveFileLease {
    /// 持有独占锁的底层文件句柄。
    file: std::fs::File,
    /// 该租约对应的锁文件路径。
    path: PathBuf,
}

impl ExclusiveFileLease {
    /// 返回该租约对应的锁文件路径。
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ExclusiveFileLease {
    /// 释放持有的文件锁。
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/** 创建目录树，并持久化发布每一个新建的祖先目录。

创建顺序从最近的已存在祖先目录开始，逐级向 `path` 推进。每创建一个目录，
都会先同步该目录及其父目录后才创建下一个子目录。重试时会先同步最近的已存在
祖先目录，用于修复上一次失败——失败发生在 `mkdir` 之后、目录项屏障（barrier）
完成之前的情形。
*/
pub fn create_dir_all_durable(path: &Path) -> std::io::Result<()> {
    create_dir_all_durable_with(path, sync_parent_directory)
}

/// `create_dir_all_durable` 的可注入实现，允许测试替换目录同步回调。
fn create_dir_all_durable_with(
    path: &Path,
    mut sync_directory: impl FnMut(&Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "durable directory path must not be empty",
        ));
    }
    let mut missing = Vec::new();
    let mut nearest = path.to_path_buf();
    loop {
        match std::fs::metadata(&nearest) {
            Ok(metadata) if metadata.is_dir() => break,
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("directory path is not a directory: {}", nearest.display()),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                missing.push(nearest.clone());
                nearest = nearest
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));
            }
            Err(error) => return Err(error),
        }
    }

    sync_directory(&nearest)?;
    if let Some(parent) = nearest.parent() {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        sync_directory(parent)?;
    }
    for directory in missing.into_iter().rev() {
        match std::fs::create_dir(&directory) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if !std::fs::metadata(&directory)?.is_dir() {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
        sync_directory(&directory)?;
        if let Some(parent) = directory.parent() {
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            sync_directory(parent)?;
        }
    }
    Ok(())
}

/** 为 `authority_path` 获取一个非阻塞的独占旁路（sidecar）租约。

同一时刻仅有一个进程能在该权威（authority）的生命周期内持有该租约；同一进程
内的其它句柄必须共享同一个权威，而不是再获取一个新租约。竞争的进程会直接
失败并返回错误，而不是静默地产生写入竞争。
*/
pub fn try_exclusive_file_lease(authority_path: &Path) -> std::io::Result<ExclusiveFileLease> {
    use fs2::FileExt;

    let parent = authority_path.parent().ok_or_else(|| {
        std::io::Error::other(format!(
            "authority path has no parent: {}",
            authority_path.display()
        ))
    })?;
    std::fs::create_dir_all(parent)?;
    let name = authority_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("authority");
    let path = parent.join(format!(".{name}.lease"));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)?;
    file.try_lock_exclusive().map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!(
                "file-backed authority {} is already open in another process: {error}",
                authority_path.display()
            ),
        )
    })?;
    Ok(ExclusiveFileLease { file, path })
}

/// 在将外部标识符用作单个文件系统路径段之前对其进行校验。
pub fn validate_path_segment(value: &str) -> std::io::Result<&str> {
    if value.is_empty()
        || Path::new(value).is_absolute()
        || Path::new(value)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || value.contains('/')
        || value.contains('\\')
        || value.chars().any(char::is_control)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("value is not a safe path segment: {value:?}"),
        ));
    }
    Ok(value)
}

/** 将精确的 UTF-8 字节编码为一个与文件系统排序规则无关的身份标识。

输出仅使用小写十六进制 ASCII 字符，因此文件系统执行的大小写折叠和 Unicode
归一化都不会让不同的输入字符串产生别名冲突。该函数对任意输入都是全函数
（total function），不会校验 `value` 是否是一个可接受的外部标识符。
*/
pub fn encode_utf8_path_identity(value: &str) -> String {
    let capacity = value.len().saturating_mul(2).saturating_add(3);
    let mut encoded = String::with_capacity(capacity);
    encoded.push_str("id-");
    for byte in value.as_bytes() {
        encoded.push(char::from_digit(u32::from(*byte >> 4), 16).unwrap_or('0'));
        encoded.push(char::from_digit(u32::from(*byte & 0x0f), 16).unwrap_or('0'));
    }
    encoded
}

/// 校验外部标识符并将其编码为精确的文件系统身份标识。
pub fn encode_path_segment_identity(value: &str) -> std::io::Result<String> {
    validate_path_segment(value)?;
    Ok(encode_utf8_path_identity(value))
}

/// 将经过校验的外部标识符拼接到 `root` 之下。
pub fn join_path_segment(root: &Path, value: &str) -> std::io::Result<PathBuf> {
    let joined = root.join(validate_path_segment(value)?);
    match std::fs::symlink_metadata(&joined) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "path segment resolves through a symlink: {}",
                joined.display()
            ),
        )),
        Ok(_) => Ok(joined),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(joined),
        Err(error) => Err(error),
    }
}

/// 借助一个唯一的同级临时文件和持久化重命名，原子性地替换一个文件。
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other(format!("path has no parent: {}", path.display())))?;
    std::fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data");
    let tmp = parent.join(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4()));
    let existing_permissions = std::fs::metadata(path)
        .ok()
        .map(|metadata| metadata.permissions());
    let write_result = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        if let Some(permissions) = existing_permissions {
            file.set_permissions(permissions)?;
        }
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = replace_temp_file(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    sync_parent_directory(parent)
}

/** 删除一个文件，并持久化发布该目录项删除操作。

`NotFound` 视为幂等成功，但仍会同步父目录，以便重试时修复此前发生在
`unlink` 之后、目录屏障之前的崩溃。
*/
pub fn remove_file_durable(path: &Path) -> std::io::Result<bool> {
    remove_file_durable_with(path, sync_parent_directory)
}

/// `remove_file_durable` 的可注入实现，允许测试替换父目录同步回调。
fn remove_file_durable_with(
    path: &Path,
    sync_parent: impl Fn(&Path) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other(format!("path has no parent: {}", path.display())))?;
    let removed = match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error),
    };
    sync_parent(parent)?;
    Ok(removed)
}

/// 将临时文件重命名替换为目标文件（非 Windows 平台）。
#[cfg(not(windows))]
fn replace_temp_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

/// 将临时文件重命名替换为目标文件（Windows 平台，使用 `MoveFileExW`）。
#[cfg(windows)]
fn replace_temp_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let mut source_wide = source.as_os_str().encode_wide().collect::<Vec<_>>();
    source_wide.push(0);
    let mut destination_wide = destination.as_os_str().encode_wide().collect::<Vec<_>>();
    destination_wide.push(0);
    /*
    SAFETY: 两个缓冲区都以 NUL 结尾，在调用期间保持存活，并且 MoveFileExW
    返回后不会保留对它们指针的引用。
    */
    let replaced = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/** 仅当文件当前（已加锁的）字节内容满足调用方提供的版本判定条件时，
才原子性地替换该文件。

协作的写入方通过一个同级旁路锁进行串行化。判定条件的检查和文件替换都在
持有该锁期间完成，从而关闭了带版本号的编辑器/存储惯用的“读取/检查/写入”
之间的竞争窗口。
*/
pub fn atomic_compare_and_swap<F>(
    path: &Path,
    bytes: &[u8],
    matches_expected: F,
) -> std::io::Result<bool>
where
    F: FnOnce(&[u8]) -> bool,
{
    use fs2::FileExt;

    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other(format!("path has no parent: {}", path.display())))?;
    std::fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data");
    let lock_path = parent.join(format!(".{file_name}.lock"));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    lock.lock_exclusive()?;
    let result = (|| {
        let current = std::fs::read(path)?;
        if !matches_expected(&current) {
            return Ok(false);
        }
        atomic_write(path, bytes)?;
        Ok(true)
    })();
    let unlock_result = lock.unlock();
    match (result, unlock_result) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(replaced), Ok(())) => Ok(replaced),
    }
}

/// 修改一个已存在文件后所请求的持久化级别。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileDurability {
    /// 仅刷新语言层缓冲区，不请求磁盘屏障。
    Flush,
    /// 要求已修改的文件数据落盘到持久存储。
    SyncData,
}

/// 打开一个已存在文件时所使用的访问模式。
#[derive(Clone, Copy)]
enum ExistingFileAccess {
    /// 只读打开。
    Read,
    /// 以追加模式打开。
    Append,
    /// 以写入模式打开。
    Write,
}

/**
用于判定文件在磁盘上身份是否发生变化的稳定标识（设备号+inode 或
Windows 上的卷序列号+文件索引）。
*/
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExistingFileIdentity {
    /// Unix 上该文件所在的设备号。
    #[cfg(unix)]
    device: u64,
    /// Unix 上该文件的 inode 号。
    #[cfg(unix)]
    inode: u64,
    /// Windows 上该文件所在卷的序列号。
    #[cfg(windows)]
    volume_serial_number: u32,
    /// Windows 上该文件的文件索引。
    #[cfg(windows)]
    file_index: u64,
}

/** 保持一个已存在的普通文件身份存活的打开描述符。

文件承载的权威（authority）会在其完整生命周期内持有该 guard。之后以
no-follow 方式打开同一路径时，即可区分出原始文件与“删除后重建”的替换文件，
即使替换文件长度相同、文件系统原本会复用其身份标识。
*/
#[derive(Debug)]
pub struct ExistingRegularFileGuard {
    /// 持有身份标识的底层文件句柄。
    file: std::fs::File,
    /// 打开时记录下的文件身份标识，用于后续比对。
    identity: ExistingFileIdentity,
}

/// 保持一个已存在目录身份存活的打开描述符。
#[derive(Debug)]
pub struct ExistingDirectoryGuard {
    /// 持有目录身份的底层目录句柄（不直接使用，仅用于保持句柄存活）。
    _directory: std::fs::File,
    /// 打开时记录下的目录身份标识，用于后续比对。
    identity: ExistingFileIdentity,
}

impl ExistingRegularFileGuard {
    /// 返回被守护文件描述符的当前长度。
    pub fn len(&self) -> std::io::Result<u64> {
        self.file.metadata().map(|metadata| metadata.len())
    }

    /// 判断被守护文件当前是否没有数据。
    pub fn is_empty(&self) -> std::io::Result<bool> {
        self.len().map(|len| len == 0)
    }
}

/// 读取文件句柄的身份标识（Unix 上为设备号+inode，Windows 上为卷序列号+文件索引）。
fn existing_file_identity(file: &std::fs::File) -> std::io::Result<ExistingFileIdentity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata()?;
        Ok(ExistingFileIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };

        let mut information = BY_HANDLE_FILE_INFORMATION::default();
        /*
        SAFETY: std 的 File 在整个调用期间持有一个有效句柄，且
        `information` 可写、对齐正确，GetFileInformationByHandle 返回后
        不会保留对它的引用。
        */
        let succeeded =
            unsafe { GetFileInformationByHandle(file.as_raw_handle() as HANDLE, &mut information) };
        if succeeded == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let file_index =
            (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
        Ok(ExistingFileIdentity {
            volume_serial_number: information.dwVolumeSerialNumber,
            file_index,
        })
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = file;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "existing regular-file identity is unavailable on this platform",
        ))
    }
}

/// 以 no-follow 方式打开一个已存在的普通文件（拒绝跟随符号链接、拒绝目录）。
fn open_existing_regular(
    path: &Path,
    access: ExistingFileAccess,
) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    if matches!(access, ExistingFileAccess::Read) {
        options.read(true);
    } else {
        options.write(true);
    }
    if matches!(access, ExistingFileAccess::Append) {
        options.append(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(not(any(unix, windows)))]
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "atomic no-follow existing-file mutation is unavailable on this platform",
        ));
    }

    let file = options.open(path)?;
    let metadata = file.metadata()?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("refusing to mutate a reparse point: {}", path.display()),
            ));
        }
    }
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("mutation target is not a regular file: {}", path.display()),
        ));
    }
    Ok(file)
}

/// 以 no-follow 方式打开一个已存在的目录（拒绝跟随符号链接）。
fn open_existing_directory(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(
            windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS
                | windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT,
        );
    }
    #[cfg(not(any(unix, windows)))]
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "atomic no-follow existing-directory access is unavailable on this platform",
        ));
    }
    let directory = options.open(path)?;
    let metadata = directory.metadata()?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to open a directory reparse point: {}",
                    path.display()
                ),
            ));
        }
    }
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("directory authority is not a directory: {}", path.display()),
        ));
    }
    Ok(directory)
}

/// 校验刚打开的文件句柄仍与 `guard` 记录的身份（及可选的期望长度）一致。
fn verify_open_file_matches(
    path: &Path,
    file: &std::fs::File,
    guard: &ExistingRegularFileGuard,
    expected_len: Option<u64>,
) -> std::io::Result<u64> {
    let metadata = file.metadata()?;
    let identity = existing_file_identity(file)?;
    if identity != guard.identity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "existing regular file identity changed; verified reopen required: {}",
                path.display()
            ),
        ));
    }
    let len = metadata.len();
    if let Some(expected_len) = expected_len
        && len != expected_len
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "existing regular file length changed from {expected_len} to {len}; verified reopen required: {}",
                path.display()
            ),
        ));
    }
    Ok(len)
}

/// 以 no-follow 方式打开一个已存在的普通文件，并保留其稳定身份标识。
pub fn open_existing_regular_guard(path: &Path) -> std::io::Result<ExistingRegularFileGuard> {
    let file = open_existing_regular(path, ExistingFileAccess::Read)?;
    let identity = existing_file_identity(&file)?;
    Ok(ExistingRegularFileGuard { file, identity })
}

/// 以 no-follow 方式打开一个已存在的目录，并保留其稳定身份标识。
pub fn open_existing_directory_guard(path: &Path) -> std::io::Result<ExistingDirectoryGuard> {
    let directory = open_existing_directory(path)?;
    let identity = existing_file_identity(&directory)?;
    Ok(ExistingDirectoryGuard {
        _directory: directory,
        identity,
    })
}

/// 校验 `path` 是否仍指向被 guard 守护的那个目录。
pub fn verify_existing_directory(
    path: &Path,
    guard: &ExistingDirectoryGuard,
) -> std::io::Result<()> {
    let directory = open_existing_directory(path)?;
    let identity = existing_file_identity(&directory)?;
    if identity != guard.identity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "existing directory identity changed; verified reopen required: {}",
                path.display()
            ),
        ));
    }
    Ok(())
}

/// 通过用于身份校验的同一个描述符，同步被 guard 守护的目录。
pub fn sync_existing_directory_matching(
    path: &Path,
    guard: &ExistingDirectoryGuard,
) -> std::io::Result<()> {
    let directory = open_existing_directory(path)?;
    let identity = existing_file_identity(&directory)?;
    if identity != guard.identity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "existing directory identity changed; verified reopen required: {}",
                path.display()
            ),
        ));
    }
    sync_verified_directory_handle(&directory)
}

/// 同步已校验的目录句柄（Unix 平台）。
#[cfg(unix)]
fn sync_verified_directory_handle(directory: &std::fs::File) -> std::io::Result<()> {
    directory.sync_all()
}

/// 同步已校验的目录句柄（非 Unix 平台的占位实现）。
#[cfg(not(unix))]
fn sync_verified_directory_handle(_directory: &std::fs::File) -> std::io::Result<()> {
    /*
    Windows 上为稳定身份而打开的目录句柄并不可靠地支持
    FlushFileBuffers。身份标识仍会在上面完成校验；在该平台上，
    目录项屏障只能是尽力而为的空操作。
    */
    Ok(())
}

/// 仅当 `path` 仍指向 `guard` 所守护的文件时，才返回其当前长度。
pub fn matching_existing_regular_len(
    path: &Path,
    guard: &ExistingRegularFileGuard,
) -> std::io::Result<u64> {
    let file = open_existing_regular(path, ExistingFileAccess::Read)?;
    verify_open_file_matches(path, &file, guard, None)
}

/** 校验身份和大小之后，通过同一个 no-follow 描述符执行追加写入。

检查与修改操作使用同一个描述符，因此路径替换无法在两者之间插入竞争。
*/
pub fn append_existing_matching(
    path: &Path,
    guard: &ExistingRegularFileGuard,
    expected_len: u64,
    bytes: &[u8],
    durability: FileDurability,
) -> std::io::Result<()> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Append)?;
    verify_open_file_matches(path, &file, guard, Some(expected_len))?;
    file.write_all(bytes)?;
    finish_existing_mutation(&mut file, durability)
}

/// 仅当该文件仍是 `guard` 所守护的文件时，以 no-follow 方式读取其完整内容。
pub fn read_existing_matching(
    path: &Path,
    guard: &ExistingRegularFileGuard,
) -> std::io::Result<Vec<u8>> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    verify_open_file_matches(path, &file, guard, None)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// 仅当该路径仍指向 `guard` 所守护的文件时，才读取从指定偏移开始的后缀数据。
pub fn read_existing_from_matching(
    path: &Path,
    guard: &ExistingRegularFileGuard,
    offset: u64,
) -> std::io::Result<Vec<u8>> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    let len = verify_open_file_matches(path, &file, guard, None)?;
    if offset > len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "read offset {offset} exceeds file length {len}: {}",
                path.display()
            ),
        ));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// 仅当身份标识和总长度都匹配时，读取数量受限的若干条完整记录。
pub fn read_existing_lines_from_matching(
    path: &Path,
    guard: &ExistingRegularFileGuard,
    expected_len: u64,
    offset: u64,
    limit: usize,
) -> std::io::Result<Vec<u8>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    verify_open_file_matches(path, &file, guard, Some(expected_len))?;
    if offset > expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "read offset {offset} exceeds file length {expected_len}: {}",
                path.display()
            ),
        ));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    for _ in 0..limit {
        let read = reader.read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        if bytes.last() != Some(&b'\n') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("incomplete newline-terminated record in {}", path.display()),
            ));
        }
    }
    Ok(bytes)
}

/**
仅当 no-follow 普通文件的总长度与调用方持有的不可变元数据完全一致时，
才从中读取数量受限的若干条完整记录。
*/
pub fn read_existing_lines_from_exact_len(
    path: &Path,
    expected_len: u64,
    offset: u64,
    limit: usize,
) -> std::io::Result<Vec<u8>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    let len = file.metadata()?.len();
    if len != expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "existing regular file length changed from {expected_len} to {len}; verified reopen required: {}",
                path.display()
            ),
        ));
    }
    if offset > expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "read offset {offset} exceeds file length {expected_len}: {}",
                path.display()
            ),
        ));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    for _ in 0..limit {
        let read = reader.read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        if bytes.last() != Some(&b'\n') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("incomplete newline-terminated record in {}", path.display()),
            ));
        }
    }
    Ok(bytes)
}

/// 校验身份和大小之后，通过同一个 no-follow 描述符执行截断操作。
pub fn truncate_existing_matching(
    path: &Path,
    guard: &ExistingRegularFileGuard,
    expected_len: u64,
    len: u64,
    durability: FileDurability,
) -> std::io::Result<()> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Write)?;
    verify_open_file_matches(path, &file, guard, Some(expected_len))?;
    file.set_len(len)?;
    finish_existing_mutation(&mut file, durability)
}

/// 依据请求的持久化级别，完成一次修改后的收尾（flush 或 sync_data）。
fn finish_existing_mutation(
    file: &mut std::fs::File,
    durability: FileDurability,
) -> std::io::Result<()> {
    match durability {
        FileDurability::Flush => file.flush(),
        FileDurability::SyncData => file.sync_data(),
    }
}

/** 在不跟随末端符号链接的前提下，向一个已存在的普通文件追加写入。

持久化的 JSONL 存储使用此函数而非“检查后打开”的两步序列，从而使路径替换
无法把追加写入重定向到存储根目录之外。
*/
pub fn append_existing(
    path: &Path,
    bytes: &[u8],
    durability: FileDurability,
) -> std::io::Result<()> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Append)?;
    file.write_all(bytes)?;
    finish_existing_mutation(&mut file, durability)
}

/** 在不跟随末端符号链接的前提下，读取一个已存在的普通文件。

文件先被打开，然后再校验其句柄元数据，从而避免在最后一个路径分量上出现
“检查后打开”的竞争。
*/
pub fn read_existing(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/** 在不跟随末端符号链接的前提下，从一个精确的字节偏移开始读取已存在的
普通文件。

带检查点的日志（journal）使用此函数，只解码经过校验的序列号之后的事件
后缀部分，而不是每次回放都重新读取完整日志。
*/
pub fn read_existing_from(path: &Path, offset: u64) -> std::io::Result<Vec<u8>> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    let len = file.metadata()?.len();
    if offset > len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "read offset {offset} exceeds file length {len}: {}",
                path.display()
            ),
        ));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/** 从 `offset` 开始，最多读取 `limit` 条以换行符结尾的完整记录。

与 [`read_existing_from`] 不同，本函数按请求的记录数量限制实际占用的内存，
并在看到最后一条所需的换行符后立即停止。
*/
pub fn read_existing_lines_from(
    path: &Path,
    offset: u64,
    limit: usize,
) -> std::io::Result<Vec<u8>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut file = open_existing_regular(path, ExistingFileAccess::Read)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    for _ in 0..limit {
        let read = reader.read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        if bytes.last() != Some(&b'\n') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("incomplete newline-terminated record in {}", path.display()),
            ));
        }
    }
    Ok(bytes)
}

/** 在不跟随末端符号链接的前提下，截断一个已存在的普通文件。

恢复（recovery）代码在校验完最后一条完整记录后使用此函数；本函数从不会
创建一个不存在的目标文件。
*/
pub fn truncate_existing(path: &Path, len: u64, durability: FileDurability) -> std::io::Result<()> {
    let mut file = open_existing_regular(path, ExistingFileAccess::Write)?;
    file.set_len(len)?;
    finish_existing_mutation(&mut file, durability)
}

/// 同步父目录，以持久化发布目录项的创建/删除（Unix 平台）。
#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> std::io::Result<()> {
    std::fs::File::open(parent)?.sync_all()
}

/// 同步父目录的占位实现（非 Unix 平台上为空操作）。
#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_nested_creation_stops_at_failed_barrier_and_retries() -> std::io::Result<()> {
        let root =
            std::env::temp_dir().join(format!("echo-core-durable-dir-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let target = root.join("missing-a").join("missing-b").join("journal");
        let failed_directory = root.join("missing-a").join("missing-b");
        let mut visited = Vec::new();
        let error = create_dir_all_durable_with(&target, |directory| {
            visited.push(directory.to_path_buf());
            if directory == failed_directory {
                Err(std::io::Error::other(
                    "injected nested directory barrier failure",
                ))
            } else {
                Ok(())
            }
        })
        .expect_err("nested barrier failure must surface");
        assert!(
            error
                .to_string()
                .contains("nested directory barrier failure")
        );
        assert!(failed_directory.is_dir());
        assert!(!target.is_dir());
        assert!(visited.contains(&root.join("missing-a")));
        assert!(visited.contains(&failed_directory));

        create_dir_all_durable(&target)?;
        assert!(target.is_dir());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn durable_relative_creation_syncs_dot_for_top_level_entry() -> std::io::Result<()> {
        let top = PathBuf::from(format!(
            "echo-core-relative-durable-{}",
            uuid::Uuid::new_v4()
        ));
        let nested = top.join("a").join("b");
        let mut visited = Vec::new();
        create_dir_all_durable_with(&nested, |directory| {
            visited.push(directory.to_path_buf());
            Ok(())
        })?;
        let top_barrier = visited
            .windows(2)
            .any(|pair| pair.first() == Some(&top) && pair.get(1) == Some(&PathBuf::from(".")));
        let nested_barrier = visited
            .windows(2)
            .any(|pair| pair.first() == Some(&top.join("a")) && pair.get(1) == Some(&top));
        assert!(top_barrier);
        assert!(nested_barrier);
        std::fs::remove_dir_all(top)?;
        Ok(())
    }

    #[test]
    fn path_segment_rejects_escape_and_separators() {
        for invalid in ["", ".", "..", "../x", "a/b", "a\\b", "/tmp"] {
            assert!(
                validate_path_segment(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        assert!(validate_path_segment("case-01_中文").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn joined_path_segment_rejects_existing_symlink() -> std::io::Result<()> {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("echo-core-join-{}", uuid::Uuid::new_v4()));
        let outside =
            std::env::temp_dir().join(format!("echo-core-join-outside-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(&outside)?;
        symlink(&outside, root.join("linked"))?;

        assert!(join_path_segment(&root, "linked").is_err());

        std::fs::remove_file(root.join("linked"))?;
        std::fs::remove_dir_all(root)?;
        std::fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[test]
    fn atomic_write_replaces_content() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-fs-{}", uuid::Uuid::new_v4()));
        let path = root.join("value.json");
        atomic_write(&path, b"one")?;
        atomic_write(&path, b"two")?;
        assert_eq!(std::fs::read(&path)?, b"two");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn durable_remove_is_idempotent() -> std::io::Result<()> {
        let root =
            std::env::temp_dir().join(format!("echo-core-durable-remove-{}", uuid::Uuid::new_v4()));
        create_dir_all_durable(&root)?;
        let path = root.join("record.json");
        atomic_write(&path, b"record")?;
        assert!(remove_file_durable(&path)?);
        assert!(!path.exists());
        assert!(!remove_file_durable(&path)?);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn durable_remove_retry_reconciles_unlink_before_parent_barrier() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "echo-core-durable-remove-retry-{}",
            uuid::Uuid::new_v4()
        ));
        create_dir_all_durable(&root)?;
        let path = root.join("record.json");
        atomic_write(&path, b"record")?;
        let sync_attempts = std::cell::Cell::new(0_u8);
        let first = remove_file_durable_with(&path, |_parent| {
            let next = sync_attempts.get().saturating_add(1);
            sync_attempts.set(next);
            if next == 1 {
                Err(std::io::Error::other("injected parent barrier failure"))
            } else {
                Ok(())
            }
        });
        assert!(first.is_err());
        assert!(!path.exists());
        assert!(!remove_file_durable_with(&path, |_parent| Ok(()))?);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn exclusive_file_lease_rejects_a_second_authority() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-lease-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let authority = root.join("store.json");
        let first = try_exclusive_file_lease(&authority)?;
        assert!(try_exclusive_file_lease(&authority).is_err());
        drop(first);
        let reacquired = try_exclusive_file_lease(&authority)?;
        assert!(reacquired.path().exists());
        drop(reacquired);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_existing_from_returns_exact_suffix() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-read-{}", uuid::Uuid::new_v4()));
        let path = root.join("events.jsonl");
        std::fs::create_dir_all(&root)?;
        std::fs::write(&path, b"prefix-suffix")?;

        assert_eq!(read_existing_from(&path, 7)?, b"suffix");
        assert!(read_existing_from(&path, 14).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_existing_lines_from_materializes_only_the_requested_prefix() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-lines-{}", uuid::Uuid::new_v4()));
        let path = root.join("events.jsonl");
        std::fs::create_dir_all(&root)?;
        let mut contents = String::new();
        for index in 0..10_000usize {
            use std::fmt::Write as _;
            writeln!(&mut contents, "record-{index:05}")
                .map_err(|error| std::io::Error::other(error.to_string()))?;
        }
        std::fs::write(&path, contents.as_bytes())?;

        let bounded = read_existing_lines_from(&path, 0, 2)?;
        assert_eq!(bounded, b"record-00000\nrecord-00001\n");
        assert!(bounded.len() < contents.len());
        assert_eq!(
            read_existing_lines_from_exact_len(
                &path,
                u64::try_from(contents.len()).unwrap_or(u64::MAX),
                0,
                2,
            )?,
            bounded
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn guarded_existing_operations_reject_same_length_replacement() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-guarded-{}", uuid::Uuid::new_v4()));
        let path = root.join("events.jsonl");
        std::fs::create_dir_all(&root)?;
        std::fs::write(&path, b"one\n")?;
        let guard = open_existing_regular_guard(&path)?;

        assert_eq!(matching_existing_regular_len(&path, &guard)?, 4);
        assert_eq!(
            read_existing_lines_from_matching(&path, &guard, 4, 0, 1)?,
            b"one\n"
        );
        append_existing_matching(&path, &guard, 4, b"x", FileDurability::Flush)?;
        assert_eq!(read_existing_from_matching(&path, &guard, 4)?, b"x");
        truncate_existing_matching(&path, &guard, 5, 4, FileDurability::Flush)?;

        std::fs::remove_file(&path)?;
        std::fs::write(&path, b"two\n")?;
        assert!(matching_existing_regular_len(&path, &guard).is_err());
        assert!(read_existing_lines_from_matching(&path, &guard, 4, 0, 1).is_err());
        assert!(append_existing_matching(&path, &guard, 4, b"x", FileDurability::Flush).is_err());
        assert!(truncate_existing_matching(&path, &guard, 4, 0, FileDurability::Flush).is_err());

        drop(guard);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn guarded_directory_rejects_missing_and_replacement_path() -> std::io::Result<()> {
        let root =
            std::env::temp_dir().join(format!("echo-core-directory-{}", uuid::Uuid::new_v4()));
        let directory = root.join("authority");
        let displaced = root.join("displaced");
        std::fs::create_dir_all(&directory)?;
        let guard = open_existing_directory_guard(&directory)?;
        verify_existing_directory(&directory, &guard)?;
        sync_existing_directory_matching(&directory, &guard)?;

        std::fs::rename(&directory, &displaced)?;
        assert!(verify_existing_directory(&directory, &guard).is_err());
        std::fs::create_dir(&directory)?;
        assert!(verify_existing_directory(&directory, &guard).is_err());
        assert!(sync_existing_directory_matching(&directory, &guard).is_err());

        drop(guard);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_preserves_existing_permissions() -> std::io::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!("echo-core-mode-{}", uuid::Uuid::new_v4()));
        let path = root.join("secret.yaml");
        atomic_write(&path, b"one")?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        atomic_write(&path, b"two")?;
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o600
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn atomic_compare_and_swap_rejects_stale_bytes() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-cas-{}", uuid::Uuid::new_v4()));
        let path = root.join("document.txt");
        atomic_write(&path, b"current")?;

        assert!(!atomic_compare_and_swap(&path, b"new", |bytes| bytes == b"stale")?);
        assert_eq!(std::fs::read(&path)?, b"current");
        assert!(atomic_compare_and_swap(&path, b"new", |bytes| bytes == b"current")?);
        assert_eq!(std::fs::read(&path)?, b"new");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn append_existing_supports_flush_and_sync_data() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-append-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("run.jsonl");
        std::fs::write(&path, b"start\n")?;

        append_existing(&path, b"flush\n", FileDurability::Flush)?;
        append_existing(&path, b"sync\n", FileDurability::SyncData)?;

        assert_eq!(std::fs::read(&path)?, b"start\nflush\nsync\n");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn append_existing_rejects_symlink_target_for_both_modes() -> std::io::Result<()> {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("echo-core-append-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let outside = root.join("outside.jsonl");
        let link = root.join("run.jsonl");
        std::fs::write(&outside, b"outside\n")?;
        symlink(&outside, &link)?;

        for durability in [FileDurability::Flush, FileDurability::SyncData] {
            assert!(append_existing(&link, b"event\n", durability).is_err());
        }
        assert_eq!(std::fs::read(&outside)?, b"outside\n");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn truncate_existing_supports_flush_and_sync_data() -> std::io::Result<()> {
        let root =
            std::env::temp_dir().join(format!("echo-core-truncate-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("journal.jsonl");
        std::fs::write(&path, b"abcdef")?;

        truncate_existing(&path, 4, FileDurability::Flush)?;
        assert_eq!(std::fs::read(&path)?, b"abcd");
        truncate_existing(&path, 2, FileDurability::SyncData)?;
        assert_eq!(std::fs::read(&path)?, b"ab");

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn truncate_existing_rejects_symlink_target_for_both_modes() -> std::io::Result<()> {
        use std::os::unix::fs::symlink;

        let root =
            std::env::temp_dir().join(format!("echo-core-truncate-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let outside = root.join("outside.jsonl");
        let link = root.join("journal.jsonl");
        std::fs::write(&outside, b"outside\n")?;
        symlink(&outside, &link)?;

        for durability in [FileDurability::Flush, FileDurability::SyncData] {
            assert!(truncate_existing(&link, 0, durability).is_err());
        }
        assert_eq!(std::fs::read(&outside)?, b"outside\n");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn read_existing_reads_regular_file() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("echo-core-read-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("events.jsonl");
        std::fs::write(&path, "first\n第二\n")?;

        assert_eq!(read_existing(&path)?, "first\n第二\n".as_bytes());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn read_existing_rejects_symlink_target() -> std::io::Result<()> {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("echo-core-read-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root)?;
        let target = root.join("target.json");
        let link = root.join("link.json");
        std::fs::write(&target, b"outside")?;
        symlink(&target, &link)?;

        assert!(read_existing(&link).is_err());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
