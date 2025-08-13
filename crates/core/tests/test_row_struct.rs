use geoparquet_batch_writer::{ArrowDataType, GeoParquetRowStruct};
use arrow_schema::DataType;
use arrow_array::Array;

#[derive(GeoParquetRowStruct, Clone, Default)]
struct SimpleStruct {
    id: u64,
    name: String,
    value: f64,
}

#[derive(GeoParquetRowStruct, Clone, Default)]
struct OptionalFieldsStruct {
    id: u64,
    name: Option<String>,
    value: Option<f64>,
}

#[derive(GeoParquetRowStruct, Clone, Default)]
struct CustomNameStruct {
    id: u64,
    #[geo(name = "display_name")]
    name: String,
}

#[test]
fn test_simple_struct_datatype() {
    let dt = SimpleStruct::data_type();
    match dt {
        DataType::Struct(fields) => {
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].name(), "id");
            assert_eq!(fields[0].data_type(), &DataType::UInt64);
            assert!(!fields[0].is_nullable());

            assert_eq!(fields[1].name(), "name");
            assert_eq!(fields[1].data_type(), &DataType::Utf8);
            assert!(!fields[1].is_nullable());

            assert_eq!(fields[2].name(), "value");
            assert_eq!(fields[2].data_type(), &DataType::Float64);
            assert!(!fields[2].is_nullable());
        },
        _ => panic!("Expected Struct data type"),
    }
}

#[test]
fn test_optional_fields_struct_datatype() {
    let dt = OptionalFieldsStruct::data_type();
    match dt {
        DataType::Struct(fields) => {
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].name(), "id");
            assert!(!fields[0].is_nullable());

            assert_eq!(fields[1].name(), "name");
            assert!(fields[1].is_nullable());

            assert_eq!(fields[2].name(), "value");
            assert!(fields[2].is_nullable());
        },
        _ => panic!("Expected Struct data type"),
    }
}

#[test]
fn test_custom_name_struct_datatype() {
    let dt = CustomNameStruct::data_type();
    match dt {
        DataType::Struct(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name(), "id");
            assert_eq!(fields[1].name(), "display_name");
        },
        _ => panic!("Expected Struct data type"),
    }
}

#[test]
fn test_simple_struct_from_iter_values() {
    let data = vec![
        SimpleStruct { id: 1, name: "Alice".to_string(), value: 3.14 },
        SimpleStruct { id: 2, name: "Bob".to_string(), value: 2.71 },
    ];

    let array = SimpleStruct::from_iter_values(data.clone());
    assert_eq!(array.len(), 2);
    assert_eq!(array.num_columns(), 3);

    // Check that the array is not null (since we used from_iter_values)
    assert!(array.nulls().is_none());
}

#[test]
fn test_optional_struct_from_iter() {
    let data = vec![
        Some(OptionalFieldsStruct { id: 1, name: Some("Alice".to_string()), value: Some(3.14) }),
        None,
        Some(OptionalFieldsStruct { id: 3, name: None, value: Some(2.71) }),
    ];

    let array = OptionalFieldsStruct::from_iter(data);
    assert_eq!(array.len(), 3);
    assert_eq!(array.num_columns(), 3);

    // Check nullability - the second element should be null
    let nulls = array.nulls().unwrap();
    assert!(!nulls.is_null(0)); // first element is not null
    assert!(nulls.is_null(1));  // second element is null
    assert!(!nulls.is_null(2)); // third element is not null
}

// Test that darling migration is working properly with custom names
#[derive(GeoParquetRowStruct, Clone, Default)]
struct DarlingTestStruct {
    id: u64,
    #[geo(name = "special_name")]
    field_with_custom_name: String,
    optional_field: Option<f64>,
}

#[test]
fn test_darling_custom_attribute_parsing() {
    let dt = DarlingTestStruct::data_type();
    match dt {
        DataType::Struct(fields) => {
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].name(), "id");
            assert_eq!(fields[1].name(), "special_name"); // Should use darling-parsed custom name
            assert_eq!(fields[2].name(), "optional_field");

            // Check nullability
            assert!(!fields[0].is_nullable());
            assert!(!fields[1].is_nullable());
            assert!(fields[2].is_nullable());
        },
        _ => panic!("Expected Struct data type"),
    }
}
