use std::fmt;

/// Errors raised while compressing or decompressing.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The caller passed an invalid configuration or input.
    InvalidArgument(String),
    /// The compressed stream is truncated, damaged, or inconsistent.
    Corrupt(String),
    /// The output does not fit in the space allowed for it.
    BufferTooSmall,
    /// The stream uses a feature this port does not support yet.
    Unsupported(String),
    /// The lossless backend failed.
    Lossless(String),
    /// The data holds values the algorithm cannot code; the dispatcher stores it losslessly.
    Fallback(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::Corrupt(m) => write!(f, "corrupt data: {m}"),
            Error::BufferTooSmall => write!(f, "The buffer for compressed data is not large enough."),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Lossless(m) => write!(f, "lossless backend: {m}"),
            Error::Fallback(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn corrupt<T>(msg: &str) -> Result<T> {
    Err(Error::Corrupt(msg.to_string()))
}
