//! Batch writer for Parquet files with optional GeoParquet support.
//!
//! The `geo` feature is enabled by default. Disable default features to build a
//! plain parquet-only configuration:
//!
//! ```toml
//! [dependencies]
//! parquet-batch-writer = { version = "0.1.4", default-features = false }
//! ```
//!
//! Re-enable geometry support explicitly with `features = ["geo"]`.

use std::{fs::File, io::BufWriter, path::Path, sync::Arc};

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
#[cfg(feature = "geo")]
use arrow_schema::{Field, extension::EXTENSION_TYPE_NAME_KEY};
#[cfg(feature = "geo")]
use geoparquet::writer::{GeoParquetRecordBatchEncoder, GeoParquetWriterOptionsBuilder};
use parquet::arrow::ArrowWriter;

pub use error::{ParquetBatchWriterError, Result};
pub use parquet_batch_writer_derive::{ParquetRowData, ParquetRowStruct};

mod error;

/// Trait for types that can be represented as Arrow data types and arrays.
///
/// This trait allows custom types to define how they should be converted to Arrow
/// schemas and arrays, making the system extensible for new data types.
///
/// Similar to how serde works, this trait can be implemented by downstream crates
/// to support custom types in Parquet files.
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

impl ArrowDataType for u16 {
    type Array = UInt16Array;

    fn data_type() -> DataType {
        DataType::UInt16
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(UInt16Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(UInt16Array::from_iter(iter))
    }
}

impl ArrowDataType for i16 {
    type Array = Int16Array;

    fn data_type() -> DataType {
        DataType::Int16
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(Int16Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(Int16Array::from_iter(iter))
    }
}

impl ArrowDataType for u8 {
    type Array = UInt8Array;

    fn data_type() -> DataType {
        DataType::UInt8
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(UInt8Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(UInt8Array::from_iter(iter))
    }
}

impl ArrowDataType for i8 {
    type Array = Int8Array;

    fn data_type() -> DataType {
        DataType::Int8
    }

    fn from_iter_values<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Self>,
    {
        Arc::new(Int8Array::from_iter_values(iter))
    }

    fn from_iter<I>(iter: I) -> Arc<Self::Array>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        Arc::new(Int8Array::from_iter(iter))
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
/// This lets downstream users only depend on `parquet-batch-writer`.
#[doc(hidden)]
pub mod __dep {
    pub use crate::ArrowDataType;
    pub use crate::error::{ParquetBatchWriterError, Result};
    pub use arrow_array;
    pub use arrow_buffer;
    pub use arrow_schema;
    #[cfg(feature = "geo")]
    pub use geoarrow_array;
    #[cfg(feature = "geo")]
    pub use geoarrow_schema;
}

#[doc(hidden)]
#[cfg(feature = "geo")]
#[macro_export]
macro_rules! __parquet_batch_writer_require_geo {
    () => {};
}

#[doc(hidden)]
#[cfg(not(feature = "geo"))]
#[macro_export]
macro_rules! __parquet_batch_writer_require_geo {
    () => {
        compile_error!(
            "geometry support requires the `geo` feature on `parquet-batch-writer`; enable default features or set features = [\"geo\"]"
        );
    };
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
            max_rows_per_batch: 1_000_000, // Default to 1m rows per batch
        }
    }
}

/// Trait for data that can be written to Parquet files.
///
/// Schemas containing GeoArrow extension types are written as GeoParquet automatically.
pub trait ParquetRowData: Send + Sync + Sized {
    /// Get the Arrow schema for this row type
    fn schema() -> Arc<Schema>;

    /// Convert a batch of rows into Arrow arrays
    /// The arrays must be in the same order as the schema fields
    fn to_arrays(rows: &[Self]) -> Result<Vec<Arc<dyn Array>>>;
}

enum OutputEncoding {
    Plain,
    #[cfg(feature = "geo")]
    Geo(GeoParquetRecordBatchEncoder),
}

impl OutputEncoding {
    #[cfg(feature = "geo")]
    fn try_new(schema: &Schema) -> Result<Self> {
        if !schema_has_geometry(schema) {
            return Ok(Self::Plain);
        }

        let options = GeoParquetWriterOptionsBuilder::default()
            .set_generate_covering(true)
            .build();
        let encoder = GeoParquetRecordBatchEncoder::try_new(schema, &options)?;
        Ok(Self::Geo(encoder))
    }

    #[cfg(not(feature = "geo"))]
    fn try_new(_schema: &Schema) -> Result<Self> {
        Ok(Self::Plain)
    }

    fn target_schema(&self, input_schema: &Arc<Schema>) -> Arc<Schema> {
        match self {
            Self::Plain => input_schema.clone(),
            #[cfg(feature = "geo")]
            Self::Geo(encoder) => encoder.target_schema(),
        }
    }

    fn encode_record_batch(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        match self {
            Self::Plain => Ok(batch),
            #[cfg(feature = "geo")]
            Self::Geo(encoder) => Ok(encoder.encode_record_batch(&batch)?),
        }
    }

    fn append_metadata(self, writer: &mut ArrowWriter<BufWriter<File>>) -> Result<()> {
        #[cfg(not(feature = "geo"))]
        let _ = writer;

        #[cfg(feature = "geo")]
        if let Self::Geo(encoder) = self {
            writer.append_key_value_metadata(encoder.into_keyvalue()?);
        }
        Ok(())
    }
}

#[cfg(feature = "geo")]
fn schema_has_geometry(schema: &Schema) -> bool {
    schema
        .fields()
        .iter()
        .any(|field| field_has_geometry(field))
}

#[cfg(feature = "geo")]
fn field_has_geometry(field: &Field) -> bool {
    if field
        .metadata()
        .get(EXTENSION_TYPE_NAME_KEY)
        .is_some_and(|ext_name| ext_name.starts_with("geoarrow"))
    {
        return true;
    }

    match field.data_type() {
        DataType::List(child) | DataType::LargeList(child) => field_has_geometry(child.as_ref()),
        DataType::FixedSizeList(child, _) => field_has_geometry(child.as_ref()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|child| field_has_geometry(child.as_ref())),
        _ => false,
    }
}

/// A batch writer for Parquet files that handles batching automatically.
///
/// Use the Derive trait to create row records.
pub struct ParquetBatchWriter<T: ParquetRowData> {
    encoding: OutputEncoding,
    writer: ArrowWriter<BufWriter<File>>,
    schema: Arc<Schema>,
    config: BatchConfig,
    current_batch: Vec<T>,
    batch_num: usize,
}

impl<T: ParquetRowData> ParquetBatchWriter<T> {
    /// Create a new ParquetBatchWriter
    pub fn new<P: AsRef<Path>>(output_path: P, config: BatchConfig) -> Result<Self> {
        let schema = T::schema();
        let encoding = OutputEncoding::try_new(&schema)?;
        let out_f = File::create(output_path.as_ref())?;
        let out_buf = BufWriter::new(out_f);
        let writer = ArrowWriter::try_new(out_buf, encoding.target_schema(&schema), None)?;

        Ok(Self {
            encoding,
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
        let encoded_batch = self.encoding.encode_record_batch(batch)?;
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
        self.encoding.append_metadata(&mut self.writer)?;
        self.writer.finish()?;

        Ok(())
    }

    /// Get the total number of batches written so far
    pub fn batch_count(&self) -> usize {
        self.batch_num
    }

    /// Get the number of rows in the current (unflushed) batch
    pub fn current_batch_size(&self) -> usize {
        self.current_batch.len()
    }
}
