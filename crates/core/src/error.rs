use thiserror::Error;

/// Error types for the parquet batch writer library
#[derive(Error, Debug)]
pub enum ParquetBatchWriterError {
    /// Error from the Arrow library
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    /// Error from the GeoArrow library
    #[cfg(feature = "geo")]
    #[error("GeoArrow error: {0}")]
    GeoArrow(#[from] geoarrow_schema::error::GeoArrowError),

    /// Error from the Parquet library
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Custom error for batch writing operations
    #[error("Batch writer error: {0}")]
    BatchWriter(String),
}

/// Result type for the parquet batch writer library
pub type Result<T> = std::result::Result<T, ParquetBatchWriterError>;
