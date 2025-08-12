use std::{fs::File, io::BufWriter, path::Path, sync::Arc};

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
use geoparquet::writer::{GeoParquetRecordBatchEncoder, GeoParquetWriterOptionsBuilder};
use parquet::arrow::ArrowWriter;

pub use error::{GeoParquetBatchWriterError, Result};
pub use geoparquet_batch_writer_derive::{GeoParquetRowData, GeoParquetRowStruct};

mod error;

/// Trait for types that can be represented as Arrow data types and arrays.
///
/// This trait allows custom types to define how they should be converted to Arrow
/// schemas and arrays, making the system extensible for new data types.
///
/// Similar to how serde works, this trait can be implemented by downstream crates
/// to support custom types in GeoParquet files.
pub trait ArrowDataType: Send + Sync + 'static {
    /// The Arrow array type that represents this data type
    type Array: Array + 'static;

    /// Get the Arrow DataType for this type
    fn data_type() -> DataType;

    /// Create an Arrow array from an iterator of values (non-nullable)
    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
        Self: Sized;

    /// Create an Arrow array from an iterator of optional values (nullable)
    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
        Self: Sized;
}

// Implementations for primitive types
use arrow_array::*;

impl ArrowDataType for u64 {
    type Array = UInt64Array;

    fn data_type() -> DataType {
        DataType::UInt64
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(UInt64Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(UInt64Array::from_iter(iter))
    }
}

impl ArrowDataType for i64 {
    type Array = Int64Array;

    fn data_type() -> DataType {
        DataType::Int64
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(Int64Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(Int64Array::from_iter(iter))
    }
}

impl ArrowDataType for u32 {
    type Array = UInt32Array;

    fn data_type() -> DataType {
        DataType::UInt32
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(UInt32Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(UInt32Array::from_iter(iter))
    }
}

impl ArrowDataType for i32 {
    type Array = Int32Array;

    fn data_type() -> DataType {
        DataType::Int32
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(Int32Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(Int32Array::from_iter(iter))
    }
}

impl ArrowDataType for f64 {
    type Array = Float64Array;

    fn data_type() -> DataType {
        DataType::Float64
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(Float64Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(Float64Array::from_iter(iter))
    }
}

impl ArrowDataType for f32 {
    type Array = Float32Array;

    fn data_type() -> DataType {
        DataType::Float32
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(Float32Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(Float32Array::from_iter(iter))
    }
}

impl ArrowDataType for bool {
    type Array = BooleanArray;

    fn data_type() -> DataType {
        DataType::Boolean
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(BooleanArray::from(iter.into_iter().collect::<Vec<_>>()))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(BooleanArray::from_iter(iter))
    }
}

impl ArrowDataType for String {
    type Array = StringArray;

    fn data_type() -> DataType {
        DataType::Utf8
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(StringArray::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(StringArray::from_iter(iter))
    }
}

/// Internal re-exports used by the proc-macro expansion.
/// This lets downstream users only depend on `geoparquet-batch-writer`.
#[doc(hidden)]
pub mod __dep {
    pub use crate::ArrowDataType;
    pub use crate::error::{GeoParquetBatchWriterError, Result};
    pub use arrow_array;
    pub use arrow_buffer;
    pub use arrow_schema;
    pub use geoarrow_array;
    pub use geoarrow_schema;
}

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of rows per batch. When reached, the batch is written.
    pub max_rows_per_batch: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_rows_per_batch: 10_000, // Default to 10k rows per batch
        }
    }
}

/// Trait for data that can be written to GeoParquet files
pub trait GeoParquetRowData: Send + Sync + Sized {
    /// Get the Arrow schema for this row type
    fn schema() -> Arc<Schema>;

    /// Convert a batch of rows into Arrow arrays
    /// The arrays must be in the same order as the schema fields
    fn to_arrays(rows: &[Self]) -> Result<Vec<Arc<dyn Array>>>;
}

/// A batch writer for GeoParquet files that handle batching automatically.
///
/// Use the Derive trait to create row records.
pub struct GeoParquetBatchWriter<T: GeoParquetRowData> {
    encoder: GeoParquetRecordBatchEncoder,
    writer: ArrowWriter<BufWriter<File>>,
    schema: Arc<Schema>,
    config: BatchConfig,
    current_batch: Vec<T>,
    batch_num: usize,
}

impl<T: GeoParquetRowData> GeoParquetBatchWriter<T> {
    /// Create a new GeoParquetBatchWriter
    pub fn new<P: AsRef<Path>>(output_path: P, config: BatchConfig) -> Result<Self> {
        let schema = T::schema();
        let options = GeoParquetWriterOptionsBuilder::default()
            .set_generate_covering(true)
            .build();
        let encoder = GeoParquetRecordBatchEncoder::try_new(&schema, &options)?;
        let out_f = File::create(output_path.as_ref())?;
        let out_buf = BufWriter::new(out_f);
        let writer = ArrowWriter::try_new(out_buf, encoder.target_schema(), None)?;

        Ok(Self {
            encoder,
            writer,
            schema,
            config,
            current_batch: Vec::new(),
            batch_num: 0,
        })
    }

    /// Add a single row to the batch writer
    pub fn add_row(&mut self, row: T) -> Result<()> {
        self.current_batch.push(row);

        // If we've reached the max batch size, write the batch
        if self.current_batch.len() >= self.config.max_rows_per_batch {
            self.write_current_batch()?;
        }

        Ok(())
    }

    /// Add multiple rows to the batch writer
    pub fn add_rows<I>(&mut self, rows: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        for row in rows {
            self.add_row(row)?;
        }
        Ok(())
    }

    /// Write the current batch
    fn write_current_batch(&mut self) -> Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Build arrays for current rows
        let arrays = T::to_arrays(&self.current_batch)?;
        self.batch_num += 1;
        // println!(
        //     "Processing batch {} ({} rows)",
        //     self.batch_num,
        //     self.current_batch.len()
        // );

        let batch = RecordBatch::try_new(self.schema.clone(), arrays)?;
        let encoded_batch = self.encoder.encode_record_batch(&batch)?;
        self.writer.write(&encoded_batch)?;

        // Update state for next batch
        self.current_batch.clear();

        Ok(())
    }

    /// Flush any remaining rows in the current batch
    pub fn flush(&mut self) -> Result<()> {
        if !self.current_batch.is_empty() {
            self.write_current_batch()?;
        }
        Ok(())
    }

    /// Finish writing and close the file
    pub fn finish(mut self) -> Result<()> {
        // Flush any remaining data
        self.flush()?;

        // Finalize the parquet file
        let kv_metadata = self.encoder.into_keyvalue()?;
        self.writer.append_key_value_metadata(kv_metadata);
        self.writer.finish()?;

        Ok(())
    }

    /// Get the total number of batches written so far
    pub fn batch_count(&self) -> usize {
        self.batch_num
    }

    /// Get the number of points in the current (unflushed) batch
    pub fn current_batch_size(&self) -> usize {
        self.current_batch.len()
    }
}
