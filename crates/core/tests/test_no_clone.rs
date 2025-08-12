use geo_types::Point;
use geoparquet_batch_writer::{GeoParquetBatchWriter, GeoParquetRowData};

// This struct intentionally does NOT derive Clone
// to demonstrate that Clone is no longer required
#[derive(GeoParquetRowData)]
struct NonCloneableRow {
    id: u64,
    #[geo(geometry)]
    point: Point<f64>,
}

#[test]
fn test_no_clone_requirement() {
    let mut writer = GeoParquetBatchWriter::new("/tmp/test_no_clone.parquet", Default::default()).unwrap();

    // Create a row - this works without Clone
    let row = NonCloneableRow {
        id: 1,
        point: Point::new(0.0, 0.0),
    };

    // This now works without requiring Clone!
    writer.add_row(row).unwrap();

    // We can also use add_rows with an iterator of owned values
    let rows = vec![
        NonCloneableRow {
            id: 2,
            point: Point::new(1.0, 1.0),
        },
        NonCloneableRow {
            id: 3,
            point: Point::new(2.0, 2.0),
        },
    ];

    // This works because we're passing ownership of the Vec elements
    writer.add_rows(rows).unwrap();

    writer.finish().unwrap();
}
