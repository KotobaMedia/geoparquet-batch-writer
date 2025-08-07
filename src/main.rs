/*!
 * GeoParquet Batch Writer with Trait-Based Architecture
 *
 * This module provides a flexible, trait-based system for writing GeoParquet files with automatic batching.
 *
 * Key Features:
 * - `GeoParquetRowData` trait allows any struct to be written to GeoParquet
 * - Automatic memory-based batching to handle large datasets efficiently
 * - Type-safe schema generation and array conversion
 * - Examples include simple `Row` and enriched `EnrichedRow` structures
 *
 * Usage:
 * 1. Implement `GeoParquetRowData` for your struct
 * 2. Create a `GeoParquetBatchWriter<YourStruct>`
 * 3. Add rows using `add_row()` or `add_rows()`
 * 4. Call `finish()` to flush and close the file
 *
 * The writer automatically handles batching based on memory usage, ensuring
 * consistent performance even with large datasets.
 */

use anyhow::{Result, anyhow};
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use geo::{Point, point};
use geoarrow_array::GeoArrowArray;
use geoarrow_array::builder::PointBuilder;
use geoarrow_schema::{Dimension, PointType};
use geoparquet::writer::GeoParquetRecordBatchEncoder;
use parquet::arrow::arrow_writer::ArrowWriter;
use rand::{Rng, thread_rng};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Trait for data that can be written to GeoParquet files
pub trait GeoParquetRowData: Clone + Send + Sync {
    /// Get the Arrow schema for this row type
    fn schema() -> Arc<Schema>;

    /// Convert a batch of rows into Arrow arrays
    /// The arrays must be in the same order as the schema fields
    fn to_arrays(rows: &[Self]) -> Result<Vec<Arc<dyn Array>>>;

    /// Estimated memory size per row (used for batching decisions)
    fn estimated_row_memory_size() -> usize;
}

/// Example row structure with ID and geometry
#[derive(Debug, Clone)]
pub struct Row {
    pub id: u64,
    pub geometry: Point<f64>,
}

impl GeoParquetRowData for Row {
    fn schema() -> Arc<Schema> {
        let point_type = PointType::new(Dimension::XY, Default::default());
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            point_type.to_field("geometry", false),
        ]))
    }

    fn to_arrays(rows: &[Self]) -> Result<Vec<Arc<dyn Array>>> {
        // Build ID array
        let id_array = Arc::new(UInt64Array::from_iter(rows.iter().map(|row| row.id)));

        // Build geometry array
        let point_type = PointType::new(Dimension::XY, Default::default());
        let mut geom_builder = PointBuilder::new(point_type);
        for row in rows {
            geom_builder.push_point(Some(&point!(x: row.geometry.x(), y: row.geometry.y())));
        }
        let geom_array = Arc::new(geom_builder.finish().into_array_ref());

        Ok(vec![id_array, geom_array])
    }

    fn estimated_row_memory_size() -> usize {
        // Rough estimate: 8 bytes for u64 + ~32 bytes for Point coordinates and overhead
        40
    }
}

/// Example of a more complex row structure with additional attributes
#[derive(Debug, Clone)]
pub struct EnrichedRow {
    pub id: u64,
    pub geometry: Point<f64>,
    pub name: String,
    pub value: f64,
}

impl GeoParquetRowData for EnrichedRow {
    fn schema() -> Arc<Schema> {
        let point_type = PointType::new(Dimension::XY, Default::default());
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            point_type.to_field("geometry", false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]))
    }

    fn to_arrays(rows: &[Self]) -> Result<Vec<Arc<dyn Array>>> {
        use arrow_array::{Float64Array, StringArray};

        // Build arrays for each field
        let id_array = Arc::new(UInt64Array::from_iter(rows.iter().map(|row| row.id)));

        let point_type = PointType::new(Dimension::XY, Default::default());
        let mut geom_builder = PointBuilder::new(point_type);
        for row in rows {
            geom_builder.push_point(Some(&point!(x: row.geometry.x(), y: row.geometry.y())));
        }
        let geom_array = Arc::new(geom_builder.finish().into_array_ref());

        let name_array = Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| &row.name),
        ));

        let value_array = Arc::new(Float64Array::from_iter(rows.iter().map(|row| row.value)));

        Ok(vec![id_array, geom_array, name_array, value_array])
    }

    fn estimated_row_memory_size() -> usize {
        // Rough estimate: 8 bytes for u64 + ~32 bytes for Point + ~50 bytes for String + 8 bytes for f64
        98
    }
}

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of points to accumulate before checking memory usage
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

#[derive(Parser)]
#[command(name = "geoparquet-generator")]
#[command(about = "A CLI tool to generate random geospatial data and save it as GeoParquet")]
struct Cli {
    /// Output file path for the GeoParquet file
    #[arg(short, long, default_value = "output.parquet")]
    output: String,

    /// Number of random points to generate
    #[arg(short, long, default_value = "10000")]
    count: usize,

    /// Bounding box for random points: min_lon,min_lat,max_lon,max_lat
    #[arg(short, long, default_value = "-180,-90,180,90")]
    bbox: String,

    /// Add random attributes to the points
    #[arg(short, long)]
    attributes: bool,

    /// Use enriched row format with additional fields
    #[arg(short, long)]
    enriched: bool,
}

fn parse_bbox(bbox_str: &str) -> Result<(f64, f64, f64, f64)> {
    let coords: Vec<f64> = bbox_str
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;

    if coords.len() != 4 {
        return Err(anyhow!("Bounding box must have exactly 4 coordinates"));
    }

    Ok((coords[0], coords[1], coords[2], coords[3]))
}

fn generate_random_points(count: usize, bbox: (f64, f64, f64, f64)) -> Vec<Point<f64>> {
    let mut rng = thread_rng();
    let (min_lon, min_lat, max_lon, max_lat) = bbox;

    (0..count)
        .map(|_| {
            let lon = rng.gen_range(min_lon..=max_lon);
            let lat = rng.gen_range(min_lat..=max_lat);
            Point::new(lon, lat)
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Parse bounding box
    let bbox = parse_bbox(&cli.bbox)?;
    println!(
        "Generating {} random points within bbox: {:?}",
        cli.count, bbox
    );

    // Generate random points
    let points = generate_random_points(cli.count, bbox);

    // Create output directory if it doesn't exist
    if let Some(parent) = Path::new(&cli.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Create batch writer with custom configuration
    let batch_config = BatchConfig {
        check_interval: 1000,         // Check memory every 1000 items
        memory_threshold: 100 * 1024, // 100KB threshold
    };

    if cli.enriched {
        // Use enriched row format
        let mut writer: GeoParquetBatchWriter<EnrichedRow> =
            GeoParquetBatchWriter::new(&cli.output, batch_config)?;

        let names = vec!["Restaurant", "Park", "School", "Hospital", "Store"];
        let mut rng = thread_rng();

        for (i, point) in points.iter().enumerate() {
            let row = EnrichedRow {
                id: i as u64,
                geometry: *point,
                name: format!("{} #{}", names[i % names.len()], i),
                value: rng.gen_range(0.0..100.0),
            };
            writer.add_row(row)?;
        }

        writer.finish()?;
        println!(
            "Successfully wrote {} enriched rows to {}",
            cli.count, cli.output
        );
    } else {
        // Use simple row format
        let mut writer: GeoParquetBatchWriter<Row> =
            GeoParquetBatchWriter::new(&cli.output, batch_config)?;

        for (i, point) in points.iter().enumerate() {
            let row = Row {
                id: i as u64,
                geometry: *point,
            };
            writer.add_row(row)?;
        }

        writer.finish()?;
        println!(
            "Successfully wrote {} simple rows to {}",
            cli.count, cli.output
        );
    }

    Ok(())
}
