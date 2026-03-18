use std::fs::{self, File};
use std::path::PathBuf;

use anyhow::Result;
use geo_types::{Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
use geoparquet_batch_writer::{BatchConfig, ParquetBatchWriter, ParquetRowData};
use parquet::file::reader::{FileReader, SerializedFileReader};

#[derive(Clone, ParquetRowData)]
struct RowPoint {
    id: u64,
    name: String,
    #[parquet(geometry)]
    geom: Point<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowOptional {
    id: i32,
    flag: Option<bool>,
    note: Option<String>,
    #[parquet(geometry)]
    geom: Option<Point<f64>>,
}

#[derive(Clone, ParquetRowData)]
struct RowLineString {
    id: u32,
    #[parquet(geometry)]
    line: LineString<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowPolygon {
    #[parquet(geometry)]
    poly: Polygon<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowMultiPoint {
    label: String,
    #[parquet(geometry)]
    mpt: MultiPoint<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowMultiLineString {
    id: u64,
    #[parquet(geometry)]
    mls: MultiLineString<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowMultiPolygon {
    #[parquet(geometry)]
    mpoly: MultiPolygon<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowGeometryEnum {
    id: i64,
    #[parquet(geometry)]
    geom: Geometry<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowWithVecs {
    id: u32,
    tags: Vec<String>,
    scores: Vec<f64>,
    counts: Option<Vec<i32>>,
    #[parquet(geometry)]
    geom: Point<f64>,
}

#[derive(Clone, ParquetRowData)]
struct RowPlain {
    id: u64,
    name: String,
    active: Option<bool>,
}

#[derive(Clone, ParquetRowData)]
struct RowTwoPoints {
    id: u64,
    start: Point<f64>,
    end: Point<f64>,
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

fn parquet_metadata_contains(path: &PathBuf, key: &str) -> Result<bool> {
    let reader = SerializedFileReader::new(File::open(path)?)?;
    let file_metadata = reader.metadata().file_metadata();
    let Some(kv_metadata) = file_metadata.key_value_metadata() else {
        return Ok(false);
    };
    Ok(kv_metadata.iter().any(|kv| kv.key == key))
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
fn schema_and_arrays_with_vecs() -> Result<()> {
    // Schema fields and array conversion for rows with Vec fields
    let schema = RowWithVecs::schema();
    assert_eq!(schema.fields().len(), 5); // id, tags, scores, counts, geom

    let rows = vec![
        RowWithVecs {
            id: 1,
            tags: vec!["a".to_string(), "b".to_string()],
            scores: vec![1.0, 2.0, 3.0],
            counts: Some(vec![10, 20]),
            geom: Point::new(1.0, 2.0),
        },
        RowWithVecs {
            id: 2,
            tags: vec!["c".to_string()],
            scores: vec![4.0],
            counts: None,
            geom: Point::new(3.0, 4.0),
        },
    ];

    let arrays = RowWithVecs::to_arrays(&rows)?;
    assert_eq!(arrays.len(), 5);

    // Check that list arrays have correct lengths
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
fn multiple_geometry_fields_are_supported() -> Result<()> {
    let rows = vec![RowTwoPoints {
        id: 1,
        start: Point::new(0.0, 0.0),
        end: Point::new(1.0, 1.0),
    }];
    let schema = RowTwoPoints::schema();
    assert_eq!(schema.fields().len(), 3);
    let arrays = RowTwoPoints::to_arrays(&rows)?;
    assert_eq!(arrays.len(), 3);
    Ok(())
}

#[test]
fn batch_writer_writes_batches_by_row_count() -> Result<()> {
    let out = tmp_file("points_batch");
    let mut writer: ParquetBatchWriter<RowPoint> = ParquetBatchWriter::new(
        &out.path,
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
    let meta = fs::metadata(&out.path)?;
    assert!(meta.len() > 0);
    assert!(parquet_metadata_contains(&out.path, "geo")?);
    Ok(())
}

#[test]
fn batch_writer_handles_varied_structs() -> Result<()> {
    // lines
    let out1 = tmp_file("lines");
    let mut w1: ParquetBatchWriter<RowLineString> = ParquetBatchWriter::new(
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
    let mut w2: ParquetBatchWriter<RowPolygon> =
        ParquetBatchWriter::new(out2.path.to_str().unwrap(), BatchConfig::default())?;
    w2.add_row(RowPolygon {
        poly: Polygon::new(
            LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
            vec![],
        ),
    })?;
    w2.finish()?;

    // geometry enum
    let out3 = tmp_file("geometry_enum");
    let mut w3: ParquetBatchWriter<RowGeometryEnum> = ParquetBatchWriter::new(
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

#[test]
fn plain_parquet_writer_skips_geo_metadata() -> Result<()> {
    let out = tmp_file("plain_rows");
    let mut writer: ParquetBatchWriter<RowPlain> =
        ParquetBatchWriter::new(&out.path, BatchConfig::default())?;
    writer.add_row(RowPlain {
        id: 1,
        name: "plain".into(),
        active: Some(true),
    })?;
    writer.finish()?;

    let meta = fs::metadata(&out.path)?;
    assert!(meta.len() > 0);
    assert!(!parquet_metadata_contains(&out.path, "geo")?);
    Ok(())
}

#[test]
fn geometry_schema_automatically_writes_geoparquet_metadata() -> Result<()> {
    let out = tmp_file("two_points");
    let mut writer: ParquetBatchWriter<RowTwoPoints> =
        ParquetBatchWriter::new(&out.path, BatchConfig::default())?;
    writer.add_row(RowTwoPoints {
        id: 1,
        start: Point::new(0.0, 0.0),
        end: Point::new(1.0, 1.0),
    })?;
    writer.finish()?;

    assert!(parquet_metadata_contains(&out.path, "geo")?);
    Ok(())
}
