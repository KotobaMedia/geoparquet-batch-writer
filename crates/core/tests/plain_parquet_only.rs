use std::fs::{self, File};
use std::path::PathBuf;

use anyhow::Result;
use geoparquet_batch_writer::{BatchConfig, ParquetBatchWriter, ParquetRowData};
use parquet::file::reader::{FileReader, SerializedFileReader};

#[derive(Clone, ParquetRowData)]
struct RowPlain {
    id: u64,
    name: String,
    active: Option<bool>,
}

struct TmpPath {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

fn tmp_file(name: &str) -> TmpPath {
    let dir = tempfile::tempdir().expect("tmpdir");
    let mut path = dir.path().to_path_buf();
    path.push(format!("{name}.parquet"));
    TmpPath { _dir: dir, path }
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
fn schema_and_arrays_plain() -> Result<()> {
    let schema = RowPlain::schema();
    assert_eq!(schema.fields().len(), 3);

    let rows = vec![
        RowPlain {
            id: 1,
            name: "a".into(),
            active: Some(true),
        },
        RowPlain {
            id: 2,
            name: "b".into(),
            active: None,
        },
    ];

    let arrays = RowPlain::to_arrays(&rows)?;
    assert_eq!(arrays.len(), 3);
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
