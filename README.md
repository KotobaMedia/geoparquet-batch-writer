# Parquet Batch Writer

[![Crates.io Version](https://img.shields.io/crates/v/parquet-batch-writer)](https://crates.io/crates/parquet-batch-writer)
[![docs.rs](https://img.shields.io/docsrs/parquet-batch-writer)](https://docs.rs/parquet-batch-writer/latest/parquet-batch-writer/)

Rust library (plus derive macro) for writing Parquet files efficiently in batches using Arrow. Define a row struct, derive `ParquetRowData`, and stream rows to an on-disk Parquet file. If the generated schema contains GeoArrow geometry fields, the writer automatically switches to GeoParquet encoding and metadata.

## Features
- Derive macro to turn your struct into Arrow arrays + schema
- Automatic batching with a configurable `max_rows_per_batch`
- Plain parquet writing works without any geo dependencies
- Automatic GeoParquet output when the schema contains geometry fields and the `geo` feature is enabled
- Supports geo-types: Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, Geometry, GeometryCollection (all `f64`)
- Optional fields via `Option<T>` (including optional geometry)
- Column rename and geometry dimension hints (XY/XYZ/XYM)

## Workspace layout
- `crates/core`: library crate `parquet-batch-writer`
- `crates/derive`: proc-macro crate exporting `#[derive(ParquetRowData)]`
- `crates/example-cli`: example CLI demonstrating how geometry rows automatically produce GeoParquet output (not published)

## Build
- Prereqs: Rust (stable) with Cargo

```sh
# build the workspace (library + derive + example CLI)
cargo build -q

# run tests (core crate has unit tests)
cargo test -q

# verify the plain parquet-only library build
cargo test -q -p parquet-batch-writer --no-default-features
```

## Library usage

`cargo add parquet-batch-writer`

Geo support is enabled by default. If you only need plain parquet output, disable default features:

```toml
[dependencies]
parquet-batch-writer = { version = "0.1.4", default-features = false }
```

Use `features = ["geo"]` or default features when you want geometry columns and GeoParquet metadata.

Add a row type and derive `ParquetRowData`. Geometry fields are detected automatically from supported `geo-types`, or can be marked explicitly with `#[parquet(geometry)]`. Optionally rename columns or set geometry dimension.

```rust
use anyhow::Result;
use geo_types::Point;
use parquet_batch_writer::{BatchConfig, ParquetBatchWriter, ParquetRowData};

#[derive(ParquetRowData)]
struct Row {
    id: u64,
    #[parquet(name = "geom", geometry, dim = "XY")] // XY | XYZ | XYM
    point: Point<f64>,
    note: Option<String>,
}

fn main() -> Result<()> {
    let mut w: ParquetBatchWriter<Row> = ParquetBatchWriter::new(
        "output.parquet",
        Default::default(),
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
- Geometry fields trigger GeoParquet automatically when the `geo` feature is enabled; rows without geometry write plain Parquet
- Multiple geometry fields are supported
- Geometry can be optional (`Option<Point<f64>>`) and will produce nulls
- Non-geometry columns support typical Arrow scalar types (e.g., integers, floats, strings)

## Example CLI
An example CLI lives in `crates/example-cli` to illustrate how to consume the library. It is for demonstration only and is not published to crates.io.

Run it like this:

```sh
# from repo root
cargo run -q -p parquet-batch-writer-example-cli -- \
    --output output.parquet \
    --count 10000 \
    --bbox "-180,-90,180,90"
```

Flags
- `--output` (path): where to write the Parquet file (or GeoParquet, when geometry is present) (default `output.parquet`)
- `--count` (usize): number of random points (default `10000`)
- `--bbox` (min_lon,min_lat,max_lon,max_lat): bounding box for random points (default `-180,-90,180,90`)
