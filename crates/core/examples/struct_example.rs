use geoparquet_batch_writer::{ArrowDataType, GeoParquetRowStruct};
use arrow_schema::DataType;
use arrow_array::Array;

/// Example showing the GeoParquetRowStruct derive macro in action
#[derive(GeoParquetRowStruct, Clone, Default, Debug)]
struct Person {
    id: u64,
    name: String,
    age: Option<u32>,
    #[geo(name = "email_address")]
    email: Option<String>,
}

fn main() {
    // Create some sample data
    let people = vec![
        Person {
            id: 1,
            name: "Alice".to_string(),
            age: Some(30),
            email: Some("alice@example.com".to_string()),
        },
        Person {
            id: 2,
            name: "Bob".to_string(),
            age: None,
            email: Some("bob@example.com".to_string()),
        },
        Person {
            id: 3,
            name: "Charlie".to_string(),
            age: Some(25),
            email: None,
        },
    ];

    // Show the generated Arrow schema
    let data_type = Person::data_type();
    println!("Generated DataType: {:?}", data_type);

    // Verify it's a struct type
    match &data_type {
        DataType::Struct(fields) => {
            println!("\nStruct Fields:");
            for field in fields.iter() {
                println!("  - {}: {:?} (nullable: {})", 
                    field.name(), 
                    field.data_type(), 
                    field.is_nullable()
                );
            }
        }
        _ => panic!("Expected Struct data type"),
    }

    // Create Arrow arrays from the data
    let struct_array = Person::from_iter_values(people.clone());
    println!("\nCreated StructArray with {} rows", struct_array.len());

    // Test with nullable structs (some None values)
    let nullable_people = vec![
        Some(people[0].clone()),
        None,
        Some(people[2].clone()),
    ];

    let nullable_struct_array = Person::from_iter(nullable_people);
    println!("Created nullable StructArray with {} rows", nullable_struct_array.len());
    println!("Has nulls: {:?}", nullable_struct_array.nulls().is_some());
}
