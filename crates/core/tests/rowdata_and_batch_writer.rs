use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use geo::{Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
use geoparquet_batch_writer::{BatchConfig, GeoParquetBatchWriter, GeoParquetRowData};

#[derive(Clone, GeoParquetRowData)]
struct RowPoint {
    id: u64,
    name: String,
    #[geo(geometry)]
    geom: Point<f64>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowOptional {
    id: i32,
    flag: Option<bool>,
    note: Option<String>,
    #[geo(geometry)]
    geom: Option<Point<f64>>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowLineString {
    id: u32,
    #[geo(geometry)]
    line: LineString<f64>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowPolygon {
    #[geo(geometry)]
    poly: Polygon<f64>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowMultiPoint {
    label: String,
    #[geo(geometry)]
    mpt: MultiPoint<f64>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowMultiLineString {
    id: u64,
    #[geo(geometry)]
    mls: MultiLineString<f64>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowMultiPolygon {
    #[geo(geometry)]
    mpoly: MultiPolygon<f64>,
}

#[derive(Clone, GeoParquetRowData)]
struct RowGeometryEnum {
    id: i64,
    #[geo(geometry)]
    geom: Geometry<f64>,
}

struct TmpPath {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

fn tmp_file(name: &str) -> TmpPath {
    let dir = tempfile::tempdir().expect("tmpdir");
    let mut p = dir.path().to_path_buf();
    p.push(format!("{name}.parquet"));
    TmpPath { _dir: dir, path: p }
}

#[test]
fn schema_and_arrays_point() -> Result<()> {
    // Schema fields and array conversion for points with scalars
    let schema = RowPoint::schema();
    assert_eq!(schema.fields().len(), 3);
    let rows = vec![
        RowPoint {
            id: 1,
            name: "a".into(),
            geom: Point::new(1.0, 2.0),
        },
        RowPoint {
            id: 2,
            name: "b".into(),
            geom: Point::new(3.0, 4.0),
        },
    ];
    let arrays = RowPoint::to_arrays(&rows)?;
    assert_eq!(arrays.len(), 3);
    // basic memory size sanity
    let total_mem: usize = arrays.iter().map(|a| a.get_array_memory_size()).sum();
    assert!(total_mem > 0);
    Ok(())
}

#[test]
fn optional_fields_and_nulls() -> Result<()> {
    let rows = vec![
        RowOptional {
            id: 1,
            flag: None,
            note: None,
            geom: None,
        },
        RowOptional {
            id: 2,
            flag: Some(true),
            note: Some("hi".into()),
            geom: Some(Point::new(0.0, 0.0)),
        },
    ];
    let arrays = RowOptional::to_arrays(&rows)?;
    // id, flag, note, geom
    assert_eq!(arrays.len(), 4);
    Ok(())
}

#[test]
fn different_geometries_basic_arrays() -> Result<()> {
    let _ = RowLineString::to_arrays(&[RowLineString {
        id: 1,
        line: LineString::from(vec![(0.0, 0.0), (1.0, 1.0)]),
    }])?;
    let _ = RowPolygon::to_arrays(&[RowPolygon {
        poly: Polygon::new(
            LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
            vec![],
        ),
    }])?;
    let _ = RowMultiPoint::to_arrays(&[RowMultiPoint {
        label: "x".into(),
        mpt: MultiPoint::from(vec![Point::new(1.0, 2.0), Point::new(2.0, 3.0)]),
    }])?;
    let _ = RowMultiLineString::to_arrays(&[RowMultiLineString {
        id: 7,
        mls: MultiLineString(vec![LineString::from(vec![(0.0, 0.0), (2.0, 2.0)])]),
    }])?;
    let _ = RowMultiPolygon::to_arrays(&[RowMultiPolygon {
        mpoly: MultiPolygon(vec![Polygon::new(
            LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
            vec![],
        )]),
    }])?;
    let _ = RowGeometryEnum::to_arrays(&[RowGeometryEnum {
        id: 3,
        geom: Geometry::from(Point::new(9.0, 9.0)),
    }])?;
    Ok(())
}

#[test]
fn batch_writer_writes_batches_by_row_count() -> Result<()> {
    let out = tmp_file("points_batch");
    let mut writer: GeoParquetBatchWriter<RowPoint> = GeoParquetBatchWriter::new(
        out.path.to_str().unwrap(),
        BatchConfig {
            max_rows_per_batch: 3,
        }, // trigger frequent writes
    )?;

    for i in 0..10u64 {
        writer.add_row(RowPoint {
            id: i,
            name: format!("n{i}"),
            geom: Point::new(i as f64, i as f64),
        })?;
    }
    // Force flush remaining
    writer.finish()?;

    // Parquet file created and non-empty
    let meta = fs::metadata(out.path)?;
    assert!(meta.len() > 0);
    Ok(())
}

#[test]
fn batch_writer_handles_varied_structs() -> Result<()> {
    // lines
    let out1 = tmp_file("lines");
    let mut w1: GeoParquetBatchWriter<RowLineString> = GeoParquetBatchWriter::new(
        out1.path.to_str().unwrap(),
        BatchConfig {
            max_rows_per_batch: 2,
        },
    )?;
    w1.add_row(RowLineString {
        id: 1,
        line: LineString::from(vec![(0.0, 0.0), (1.0, 1.0)]),
    })?;
    w1.add_row(RowLineString {
        id: 2,
        line: LineString::from(vec![(1.0, 1.0), (2.0, 2.0)]),
    })?;
    w1.finish()?;

    // polygons
    let out2 = tmp_file("polys");
    let mut w2: GeoParquetBatchWriter<RowPolygon> =
        GeoParquetBatchWriter::new(out2.path.to_str().unwrap(), BatchConfig::default())?;
    w2.add_row(RowPolygon {
        poly: Polygon::new(
            LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
            vec![],
        ),
    })?;
    w2.finish()?;

    // geometry enum
    let out3 = tmp_file("geometry_enum");
    let mut w3: GeoParquetBatchWriter<RowGeometryEnum> = GeoParquetBatchWriter::new(
        out3.path.to_str().unwrap(),
        BatchConfig {
            max_rows_per_batch: 1,
        },
    )?;
    w3.add_row(RowGeometryEnum {
        id: 7,
        geom: Geometry::from(Point::new(0.0, 0.0)),
    })?;
    w3.add_row(RowGeometryEnum {
        id: 8,
        geom: Geometry::from(LineString::from(vec![(0.0, 0.0), (2.0, 0.0)])),
    })?;
    w3.finish()?;

    for p in [out1.path, out2.path, out3.path] {
        let meta = fs::metadata(p)?;
        assert!(meta.len() > 0);
    }
    Ok(())
}
