use anyhow::{Result, anyhow};
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use geo::{Point, point};
use geoarrow_array::GeoArrowArray;
use geoarrow_array::builder::PointBuilder;
use geoarrow_schema::{Dimension, PointType};
use geoparquet::writer::GeoParquetRecordBatchEncoder;
use parquet::arrow::arrow_writer::ArrowWriter;
use rand::{Rng, thread_rng};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

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
pub struct GeoParquetBatchWriter {
    encoder: GeoParquetRecordBatchEncoder,
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    point_type: PointType,
    config: BatchConfig,
    current_batch: Vec<Point<f64>>,
    current_batch_start_id: i64,
    batch_num: usize,
}

impl GeoParquetBatchWriter {
    /// Create a new GeoParquetBatchWriter
    pub fn new(output_path: &str, config: BatchConfig) -> Result<Self> {
        let point_type = PointType::new(Dimension::XY, Default::default());
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            point_type.to_field("geometry", false),
        ]));

        let encoder = GeoParquetRecordBatchEncoder::try_new(&schema, &Default::default())?;
        let writer =
            ArrowWriter::try_new(File::create(output_path)?, encoder.target_schema(), None)?;

        Ok(Self {
            encoder,
            writer,
            schema,
            point_type,
            config,
            current_batch: Vec::new(),
            current_batch_start_id: 0,
            batch_num: 0,
        })
    }

    /// Add a single point to the batch writer
    pub fn add_point(&mut self, point: Point<f64>) -> Result<()> {
        self.current_batch.push(point);

        // Check if we should evaluate memory usage
        if self.current_batch.len() % self.config.check_interval == 0 {
            self.check_and_write_if_needed()?;
        }

        Ok(())
    }

    /// Add multiple points to the batch writer
    pub fn add_points(&mut self, points: &[Point<f64>]) -> Result<()> {
        for point in points {
            self.add_point(*point)?;
        }
        Ok(())
    }

    /// Check current memory usage and write batch if threshold is exceeded
    fn check_and_write_if_needed(&mut self) -> Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Build arrays to check memory size
        let id_array = Arc::new(Int64Array::from_iter(
            self.current_batch
                .iter()
                .enumerate()
                .map(|(i, _)| (self.current_batch_start_id + i as i64)),
        ));

        let mut geom_builder = PointBuilder::new(self.point_type.clone());
        for point in &self.current_batch {
            geom_builder.push_point(Some(&point!(x: point.x(), y: point.y())));
        }
        let geom_array = Arc::new(geom_builder.finish().into_array_ref());

        // Calculate total memory size
        let memory_size = id_array.get_array_memory_size() + geom_array.get_array_memory_size();

        // Write batch if memory threshold exceeded
        if memory_size > self.config.memory_threshold {
            self.write_current_batch_with_arrays(id_array, geom_array, memory_size)?;
        }

        Ok(())
    }

    /// Write the current batch using pre-built arrays
    fn write_current_batch_with_arrays(
        &mut self,
        id_array: Arc<Int64Array>,
        geom_array: Arc<dyn Array>,
        memory_size: usize,
    ) -> Result<()> {
        self.batch_num += 1;
        println!(
            "Processing batch {} ({} points, {} bytes)",
            self.batch_num,
            self.current_batch.len(),
            memory_size
        );

        let batch = RecordBatch::try_new(self.schema.clone(), vec![id_array, geom_array])?;
        let encoded_batch = self.encoder.encode_record_batch(&batch)?;
        self.writer.write(&encoded_batch)?;

        // Update state for next batch
        self.current_batch_start_id += self.current_batch.len() as i64;
        self.current_batch.clear();

        Ok(())
    }

    /// Flush any remaining points in the current batch
    pub fn flush(&mut self) -> Result<()> {
        if !self.current_batch.is_empty() {
            // Build arrays for remaining points
            let id_array = Arc::new(Int64Array::from_iter(
                self.current_batch
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (self.current_batch_start_id + i as i64)),
            ));

            let mut geom_builder = PointBuilder::new(self.point_type.clone());
            for point in &self.current_batch {
                geom_builder.push_point(Some(&point!(x: point.x(), y: point.y())));
            }
            let geom_array = Arc::new(geom_builder.finish().into_array_ref());

            let memory_size = id_array.get_array_memory_size() + geom_array.get_array_memory_size();
            self.write_current_batch_with_arrays(id_array, geom_array, memory_size)?;
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

fn generate_random_attributes(count: usize) -> Vec<HashMap<String, String>> {
    let mut rng = thread_rng();
    let cities = vec![
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ];
    let categories = vec!["Restaurant", "Park", "School", "Hospital", "Store"];

    (0..count)
        .map(|i| {
            let mut attrs = HashMap::new();
            attrs.insert("id".to_string(), i.to_string());
            attrs.insert("name".to_string(), format!("Point_{}", i));
            attrs.insert(
                "city".to_string(),
                cities[rng.gen_range(0..cities.len())].to_string(),
            );
            attrs.insert(
                "category".to_string(),
                categories[rng.gen_range(0..categories.len())].to_string(),
            );
            attrs.insert("value".to_string(), rng.gen_range(1..=100).to_string());
            attrs
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

    // Generate attributes if requested
    let attributes = if cli.attributes {
        Some(generate_random_attributes(cli.count))
    } else {
        None
    };

    // Create output directory if it doesn't exist
    if let Some(parent) = Path::new(&cli.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // For now, let's create a simple CSV-like structure that we can convert
    // Since geoparquet crate usage might be complex, let's first show the data generation
    println!("Generated {} points", points.len());

    if let Some(ref attrs) = attributes {
        println!("Sample point with attributes:");
        if let (Some(point), Some(attr)) = (points.first(), attrs.first()) {
            println!("  Point: ({}, {})", point.x(), point.y());
            println!("  Attributes: {:?}", attr);
        }
    } else {
        println!("Sample points (first 5):");
        for (i, point) in points.iter().take(5).enumerate() {
            println!("  Point {}: ({}, {})", i, point.x(), point.y());
        }
    }

    // Create batch writer with custom configuration
    let batch_config = BatchConfig {
        check_interval: 1000,         // Check memory every 1000 items
        memory_threshold: 100 * 1024, // 100KB threshold
    };

    let mut writer = GeoParquetBatchWriter::new(&cli.output, batch_config)?;

    // Add all points to the writer (batching is handled internally)
    writer.add_points(&points)?;

    // Finish writing (this flushes any remaining data)
    writer.finish()?;

    println!("Successfully wrote {} points to {}", cli.count, cli.output);

    Ok(())
}
