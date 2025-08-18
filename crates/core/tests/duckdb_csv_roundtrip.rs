use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use duckdb::Connection;
use geo_types::Point;
use geoparquet_batch_writer::{BatchConfig, GeoParquetBatchWriter, GeoParquetRowData};

#[derive(GeoParquetRowData)]
struct RowPoint {
    id: u64,
    name: String,
    #[geo(geometry)]
    geom: Point<f64>,
}

struct TmpPath {
    _dir: tempfile::TempDir,
    parquet: PathBuf,
    csv: PathBuf,
}

fn tmp_files(prefix: &str) -> TmpPath {
    let dir = tempfile::tempdir().expect("tmpdir");
    let mut parquet = dir.path().to_path_buf();
    parquet.push(format!("{prefix}.parquet"));
    let mut csv = dir.path().to_path_buf();
    csv.push(format!("{prefix}.csv"));
    TmpPath { _dir: dir, parquet, csv }
}


#[test]
fn write_parquet_export_csv_with_duckdb_and_compare() -> Result<()> {
    // Prepare small dataset
    let rows = vec![
        RowPoint { id: 1, name: "a".into(), geom: Point::new(1.0, 2.0) },
        RowPoint { id: 2, name: "b".into(), geom: Point::new(-3.5, 4.25) },
        RowPoint { id: 3, name: "c".into(), geom: Point::new(0.0, 0.0) },
    ];

    // Write parquet
    let tmp = tmp_files("duckdb_roundtrip");
    let mut writer: GeoParquetBatchWriter<RowPoint> = GeoParquetBatchWriter::new(
        &tmp.parquet,
        BatchConfig { max_rows_per_batch: 2 },
    )?;
    writer.add_rows(rows.into_iter())?;
    writer.finish()?;

    // Use DuckDB to export to CSV, with geometry as lowercase hex to avoid case-mismatch
    let conn = Connection::open_in_memory()?;
    let parquet_path = tmp.parquet.to_str().unwrap().replace('"', "\"");
    let csv_path = tmp.csv.to_str().unwrap().replace('"', "\"");
    let sql = format!(
        "COPY (SELECT id, name, lower(hex(geom)) AS geom FROM read_parquet('{}') ORDER BY id) TO '{}' WITH (HEADER, DELIMITER ',');",
        parquet_path, csv_path
    );
    conn.execute_batch(&sql)?;

    // Read exported CSV as string for easier diffs
    let out_str = fs::read_to_string(&tmp.csv)?;

    // Expected CSV string comes from a fixture for strict comparison
    let expected_str: &'static str = include_str!("fixtures/points_expected.csv");

    assert_eq!(out_str, expected_str);
    Ok(())
}
