use arrow_array::{Array, RecordBatch};
use arrow_schema::Schema;
use anyhow::Result;
use geoparquet::writer::GeoParquetRecordBatchEncoder;
use parquet::arrow::ArrowWriter;
use std::{fs::File, sync::Arc};

pub use geoparquet_batch_writer_derive::GeoParquetRowData;

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of rows to accumulate before checking memory usage
    pub check_interval: usize,
    /// Memory threshold in bytes - batch will be written when exceeded
    pub memory_threshold: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            check_interval: 1000,         // Check memory every 1000 items
            memory_threshold: 100 * 1024, // 100KB threshold
        }
    }
}

/// Trait for data that can be written to GeoParquet files
pub trait GeoParquetRowData: Clone + Send + Sync {
    /// Get the Arrow schema for this row type
    fn schema() -> Arc<Schema>;

    /// Convert a batch of rows into Arrow arrays
    /// The arrays must be in the same order as the schema fields
    fn to_arrays(rows: &[Self]) -> Result<Vec<Arc<dyn Array>>>;

    /// Estimated memory size per row (used optionally for batching decisions)
    fn estimated_row_memory_size() -> usize { 0 }
}

/// A batch writer for GeoParquet files that handles memory-based batching automatically
pub struct GeoParquetBatchWriter<T: GeoParquetRowData> {
    encoder: GeoParquetRecordBatchEncoder,
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    config: BatchConfig,
    current_batch: Vec<T>,
    batch_num: usize,
}

impl<T: GeoParquetRowData> GeoParquetBatchWriter<T> {
    /// Create a new GeoParquetBatchWriter
    pub fn new(output_path: &str, config: BatchConfig) -> Result<Self> {
        let schema = T::schema();
        let encoder = GeoParquetRecordBatchEncoder::try_new(&schema, &Default::default())?;
        let writer =
            ArrowWriter::try_new(File::create(output_path)?, encoder.target_schema(), None)?;

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

        // Check if we should evaluate memory usage
        if self.current_batch.len() % self.config.check_interval == 0 {
            self.check_and_write_if_needed()?;
        }

        Ok(())
    }

    /// Add multiple rows to the batch writer
    pub fn add_rows(&mut self, rows: &[T]) -> Result<()> {
        for row in rows {
            self.add_row(row.clone())?;
        }
        Ok(())
    }

    /// Check current memory usage and write batch if threshold is exceeded
    fn check_and_write_if_needed(&mut self) -> Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Build arrays to check memory size
        let arrays = T::to_arrays(&self.current_batch)?;

        // Calculate total memory size
        let memory_size: usize = arrays
            .iter()
            .map(|array| array.get_array_memory_size())
            .sum();

        // Write batch if memory threshold exceeded
        if memory_size > self.config.memory_threshold {
            self.write_current_batch_with_arrays(arrays, memory_size)?;
        }

        Ok(())
    }

    /// Write the current batch using pre-built arrays
    fn write_current_batch_with_arrays(
        &mut self,
        arrays: Vec<Arc<dyn Array>>,
        memory_size: usize,
    ) -> Result<()> {
        self.batch_num += 1;
        println!(
            "Processing batch {} ({} rows, {} bytes)",
            self.batch_num,
            self.current_batch.len(),
            memory_size
        );

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
            // Build arrays for remaining rows
            let arrays = T::to_arrays(&self.current_batch)?;
            let memory_size: usize = arrays
                .iter()
                .map(|array| array.get_array_memory_size())
                .sum();

            self.write_current_batch_with_arrays(arrays, memory_size)?;
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
