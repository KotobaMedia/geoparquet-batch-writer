# GeoParquet Batch Writer

[![Crates.io Version](https://img.shields.io/crates/v/geoparquet-batch-writer)](https://crates.io/crates/geoparquet-batch-writer)
[![docs.rs](https://img.shields.io/docsrs/geoparquet-batch-writer)](https://docs.rs/geoparquet-batch-writer/latest/geoparquet-batch-writer/)

Rust library (plus derive macro) for writing GeoParquet files efficiently in batches using GeoArrow/Arrow. Define a simple row struct with a geometry field, derive `GeoParquetRowData`, and stream rows to an on-disk GeoParquet file.

## Features
- Derive macro to turn your struct into Arrow arrays + schema
- Automatic batching with a configurable `max_rows_per_batch`
- Supports geo-types: Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, Geometry, GeometryCollection (all `f64`)
- Optional fields via `Option<T>` (including optional geometry)
- Column rename and geometry dimension hints (XY/XYZ/XYM)

## Workspace layout
- `crates/core`: library crate `geoparquet-batch-writer`
- `crates/derive`: proc-macro crate exporting `#[derive(GeoParquetRowData)]`
- `crates/example-cli`: example CLI demonstrating how to generate random data and write GeoParquet using the library (not published)

## Build
- Prereqs: Rust (stable) with Cargo

```sh
# build the workspace (library + derive + example CLI)
cargo build -q

# run tests (core crate has unit tests)
cargo test -q
```

## Library usage

`cargo add geoparquet-batch-writer`

Add a row type and derive `GeoParquetRowData`. Mark exactly one geometry field with `#[geo(geometry)]`. Optionally rename columns or set geometry dimension.

```rust
use anyhow::Result;
use geo_types::Point;
use geoparquet_batch_writer::{BatchConfig, GeoParquetBatchWriter, GeoParquetRowData};

#[derive(Clone, GeoParquetRowData)]
struct Row {
    id: u64,
    #[geo(name = "geom", geometry, dim = "XY")] // XY | XYZ | XYM
    point: Point<f64>,
    note: Option<String>,
}

fn main() -> Result<()> {
    let mut w: GeoParquetBatchWriter<Row> = GeoParquetBatchWriter::new(
        "output.parquet",
        BatchConfig { max_rows_per_batch: 10_000 },
    )?;

    for i in 0..25_000u64 {
        w.add_row(Row {
            id: i,
            point: Point::new(-120.0 + (i as f64 * 0.0001), 35.0),
            note: (i % 2 == 0).then(|| format!("row {i}")),
        })?;
    }

    w.finish()?; // flush remaining, write metadata, close file
    Ok(())
}
```

Notes
- Only one geometry field is supported per row at the moment
- Geometry can be optional (`Option<Point<f64>>`) and will produce nulls
- Non-geometry columns support typical Arrow scalar types (e.g., integers, floats, strings)

## Example CLI
An example CLI lives in `crates/example-cli` to illustrate how to consume the library. It is for demonstration only and is not published to crates.io.

Run it like this:

```sh
# from repo root
cargo run -q -p geoparquet-batch-writer-example-cli -- \
    --output output.parquet \
    --count 10000 \
    --bbox "-180,-90,180,90"
```

Flags
- `--output` (path): where to write the GeoParquet file (default `output.parquet`)
- `--count` (usize): number of random points (default `10000`)
- `--bbox` (min_lon,min_lat,max_lon,max_lat): bounding box for random points (default `-180,-90,180,90`)
