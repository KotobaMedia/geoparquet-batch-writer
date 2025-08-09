use anyhow::{Ok, Result, anyhow};
use clap::Parser;
use geo::Point;
use geoparquet_batch_writer::{GeoParquetBatchWriter, GeoParquetRowData};
use rand::prelude::*;
use std::path::Path;

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

#[derive(Clone, GeoParquetRowData)]
struct DataRow {
    id: u64,
    geometry: Point<f64>,
    name: String,
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

    // Write points to GeoParquet file
    let mut writer = GeoParquetBatchWriter::new(&cli.output, Default::default())?;
    for (i, point) in points.into_iter().enumerate() {
        let row = DataRow {
            id: i as u64,
            geometry: point,
            name: format!("Point {}", i),
        };
        writer.add_row(row)?;
    }
    writer.finish()?;

    Ok(())
}
