use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, Ident, Lit, PathArguments, Type, parse_macro_input, spanned::Spanned,
};

#[derive(Clone, Copy, Debug)]
enum GeometryKind {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    Geometry,
    GeometryCollection,
}

#[proc_macro_derive(GeoParquetRowData, attributes(geo))]
pub fn derive_geo_parquet_row_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_geoparquet_row_data(&input) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error().into(),
    }
}

#[proc_macro_derive(GeoParquetRowStruct, attributes(geo))]
pub fn derive_geo_parquet_row_struct(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_geoparquet_row_struct(&input) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_geoparquet_row_data(input: &DeriveInput) -> syn::Result<TokenStream> {
    let struct_ident = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                // DataStruct doesn't implement Spanned in syn 2; use the fields span.
                return Err(syn::Error::new(
                    s.fields.span(),
                    "GeoParquetRowData requires named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                input.span(),
                "GeoParquetRowData supports only structs",
            ));
        }
    };

    struct FieldInfo {
        ident: Ident,
        col_name: String,
        ty: syn::Type,
        is_option: bool,
        is_geometry: bool,
        geom_kind: Option<GeometryKind>,
        dim: Option<String>, // XY, XYZ, XYM
    }

    let mut finfos: Vec<FieldInfo> = Vec::new();
    let mut geom_count = 0usize;

    for f in fields {
        let ident = f.ident.clone().unwrap();
        let mut col_name = ident.to_string();
        let mut is_geometry = false;
        let mut dim: Option<String> = None;

        // Parse #[geo(...)] attrs
        for attr in &f.attrs {
            if !attr.path().is_ident("geo") {
                continue;
            }
            // syn 2.0: use parse_nested_meta to handle attribute arguments
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("geometry") {
                    is_geometry = true;
                    return Ok(());
                }
                if meta.path.is_ident("name") {
                    let lit: Lit = meta.value()?.parse()?;
                    if let Lit::Str(s) = lit {
                        col_name = s.value();
                    }
                    return Ok(());
                }
                if meta.path.is_ident("dim") {
                    let lit: Lit = meta.value()?.parse()?;
                    if let Lit::Str(s) = lit {
                        dim = Some(s.value());
                    }
                    return Ok(());
                }
                Ok(())
            })?;
        }

        // Detect Option<T>
        let (is_option, inner_ty) = match option_inner(&f.ty) {
            Some(t) => (true, t),
            None => (false, &f.ty),
        };

        // Heuristic geometry detection if not annotated
        let detected_kind = geometry_kind(inner_ty);
        if !is_geometry && detected_kind.is_some() {
            is_geometry = true;
        }
        if is_geometry {
            geom_count += 1;
        }

        finfos.push(FieldInfo {
            ident,
            col_name,
            ty: inner_ty.clone(),
            is_option,
            is_geometry,
            geom_kind: detected_kind,
            dim,
        });
    }

    if geom_count == 0 {
        return Err(syn::Error::new(
            input.span(),
            "No geometry field found. Mark one with #[geo(geometry)] or use a type like geo_types::Point<f64>, LineString<f64>, Polygon<f64>, MultiPoint<f64>, MultiLineString<f64>, MultiPolygon<f64>, Geometry<f64>, or GeometryCollection<f64>.",
        ));
    }
    if geom_count > 1 {
        return Err(syn::Error::new(
            input.span(),
            "Multiple geometry fields found; currently only one geometry field is supported.",
        ));
    }

    // Ensure geometry field type is recognized if explicitly marked
    for fi in &finfos {
        if fi.is_geometry && fi.geom_kind.is_none() {
            return Err(syn::Error::new(
                fi.ty.span(),
                "Unsupported geometry type. Expected geo_types::Point<f64>, LineString<f64>, Polygon<f64>, MultiPoint<f64>, MultiLineString<f64>, MultiPolygon<f64>, Geometry<f64>, or GeometryCollection<f64>.",
            ));
        }
    }

    // Generate schema fields & array builders
    let mut schema_field_tokens = Vec::new();
    let mut array_expr_tokens = Vec::new();

    // geometry shared setup (Type depends on geometry kind)
    let geom_type_ident = format_ident!("__gp_geom_type");
    let geom_setup_tokens = |kind: GeometryKind,
                             dim_string: Option<&str>|
     -> proc_macro2::TokenStream {
        let dim_expr = match dim_string.unwrap_or("XY") {
            "XYZ" => quote!(::geoparquet_batch_writer::__dep::geoarrow_schema::Dimension::XYZ),
            "XYM" => quote!(::geoparquet_batch_writer::__dep::geoarrow_schema::Dimension::XYM),
            _ => quote!(::geoparquet_batch_writer::__dep::geoarrow_schema::Dimension::XY),
        };
        let ty_ctor_two = match kind {
            GeometryKind::Point => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::PointType::new
            )),
            GeometryKind::LineString => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::LineStringType::new
            )),
            GeometryKind::Polygon => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::PolygonType::new
            )),
            GeometryKind::MultiPoint => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::MultiPointType::new
            )),
            GeometryKind::MultiLineString => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::MultiLineStringType::new
            )),
            GeometryKind::MultiPolygon => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::MultiPolygonType::new
            )),
            GeometryKind::Geometry => None,
            GeometryKind::GeometryCollection => None,
        };
        let one_arg_ctor = match kind {
            GeometryKind::Geometry => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::GeometryType::new
            )),
            GeometryKind::GeometryCollection => Some(quote!(
                ::geoparquet_batch_writer::__dep::geoarrow_schema::GeometryCollectionType::new
            )),
            _ => None,
        };

        if let Some(ctor) = ty_ctor_two {
            quote! {
                let #geom_type_ident = {
                    let dim = #dim_expr;
                    #ctor(dim, ::std::sync::Arc::new(::std::default::Default::default()))
                };
            }
        } else if let Some(ctor1) = one_arg_ctor {
            quote! {
                let #geom_type_ident = {
                    #ctor1(::std::sync::Arc::new(::std::default::Default::default()))
                };
            }
        } else {
            // Fallback, should not happen
            quote! {
                let #geom_type_ident = {
                    let dim = #dim_expr;
                    ::geoparquet_batch_writer::__dep::geoarrow_schema::GeometryType::new(::std::sync::Arc::new(::std::default::Default::default()))
                };
            }
        }
    };

    // Flag whether we need geometry init
    let mut geometry_init_tokens: Option<proc_macro2::TokenStream> = None;

    for (idx, fi) in finfos.iter().enumerate() {
        if fi.is_geometry {
            let col_name_lit = syn::LitStr::new(&fi.col_name, fi.ident.span());
            let is_option = fi.is_option;
            // Determine geometry kind (from detection or default to Geometry)
            let kind = fi.geom_kind.unwrap_or(GeometryKind::Geometry);

            // Schema field (nullable if Option). to_field may return Result in some versions.
            schema_field_tokens.push(quote! {
                #geom_type_ident.to_field(#col_name_lit, #is_option)
            });

            // Arrays: choose correct builder and push method
            let b_ident = format_ident!("__gp_geom_builder");
            let arr_ident = format_ident!("__gp_arr_{}", idx);
            let ident = &fi.ident;

            let (builder_path, push_method_ident) = match kind {
                GeometryKind::Point => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::PointBuilder),
                    format_ident!("push_point"),
                ),
                GeometryKind::LineString => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::LineStringBuilder),
                    format_ident!("push_line_string"),
                ),
                GeometryKind::Polygon => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::PolygonBuilder),
                    format_ident!("push_polygon"),
                ),
                GeometryKind::MultiPoint => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::MultiPointBuilder),
                    format_ident!("push_multi_point"),
                ),
                GeometryKind::MultiLineString => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::MultiLineStringBuilder),
                    format_ident!("push_multi_line_string"),
                ),
                GeometryKind::MultiPolygon => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::MultiPolygonBuilder),
                    format_ident!("push_multi_polygon"),
                ),
                GeometryKind::Geometry => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::GeometryBuilder),
                    format_ident!("push_geometry"),
                ),
                GeometryKind::GeometryCollection => (
                    quote!(::geoparquet_batch_writer::__dep::geoarrow_array::builder::GeometryCollectionBuilder),
                    format_ident!("push_geometry_collection"),
                ),
            };

            // Make builder pushes respect nullability and silence any unused Result from push_null
            let push_tokens = if fi.is_option {
                quote! {
                    for row in rows {
                        if let Some(g) = row.#ident.as_ref() {
                            let _ = #b_ident.#push_method_ident(Some(g));
                        } else {
                            // If push_null returns a Result in this geoarrow version, ignore it explicitly
                            let _ = #b_ident.push_null();
                        }
                    }
                }
            } else {
                quote! {
                    for row in rows {
                        let g = &row.#ident;
                        let _ = #b_ident.#push_method_ident(Some(g));
                    }
                }
            };

            array_expr_tokens.push(quote! {{
                use ::geoparquet_batch_writer::__dep::geoarrow_array::GeoArrowArray as _;
                let mut #b_ident = #builder_path::new(#geom_type_ident.clone());
                #push_tokens
                let #arr_ident = ::std::sync::Arc::new(#b_ident.finish().into_array_ref());
                #arr_ident
            }});

            // Initialize the appropriate GeoArrow schema type with requested dim
            geometry_init_tokens = Some(geom_setup_tokens(kind, fi.dim.as_deref()));
        } else {
            // Scalar or String
            let dt = arrow_datatype(&fi.ty)?;
            let (array_ty, from_tokens) = array_ctor(&fi.ty, fi.is_option)?;
            let col_name_lit = syn::LitStr::new(&fi.col_name, fi.ident.span());
            let is_option = fi.is_option;
            schema_field_tokens.push(quote! {
                ::geoparquet_batch_writer::__dep::arrow_schema::Field::new(#col_name_lit, #dt, #is_option)
            });
            let arr_ident = format_ident!("__gp_arr_{}", idx);

            // rows.iter().map(|r| r.field) vs map(|r| r.field.as_ref()…) for Option & String
            let map_expr = value_mapper(&fi.ty, &fi.ident, fi.is_option);

            array_expr_tokens.push(quote! {{
                let it = rows.iter().map(#map_expr);
                let arr: ::std::sync::Arc<#array_ty> = #from_tokens;
                let #arr_ident: ::std::sync::Arc<dyn ::geoparquet_batch_writer::__dep::arrow_array::Array> = arr;
                #arr_ident
            }});
        }
    }

    let schema_vec_tokens = quote! {
        {
            ::std::sync::Arc::new(::geoparquet_batch_writer::__dep::arrow_schema::Schema::new(vec![
                #(#schema_field_tokens),*
            ]))
        }
    };

    let arrays_vec_tokens = quote! {
        {
            ::std::result::Result::Ok(vec![
                #(#array_expr_tokens),*
            ])
        }
    };

    let expanded = quote! {
        impl ::geoparquet_batch_writer::GeoParquetRowData for #struct_ident
        where Self: Send + Sync
        {
            fn schema() -> ::std::sync::Arc<::geoparquet_batch_writer::__dep::arrow_schema::Schema> {
                #geometry_init_tokens
                #schema_vec_tokens
            }

            fn to_arrays(rows: &[Self]) -> ::geoparquet_batch_writer::__dep::Result<Vec<::std::sync::Arc<dyn ::geoparquet_batch_writer::__dep::arrow_array::Array>>> {
                #geometry_init_tokens
                #arrays_vec_tokens
            }
        }
    };

    Ok(expanded.into())
}

fn impl_geoparquet_row_struct(input: &DeriveInput) -> syn::Result<TokenStream> {
    let struct_ident = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new(
                    s.fields.span(),
                    "GeoParquetRowStruct requires named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                input.span(),
                "GeoParquetRowStruct supports only structs",
            ));
        }
    };

    struct FieldInfo {
        ident: Ident,
        col_name: String,
        ty: syn::Type,
        is_option: bool,
    }

    let mut finfos: Vec<FieldInfo> = Vec::new();

    for f in fields {
        let ident = f.ident.clone().unwrap();
        let mut col_name = ident.to_string();

        // Parse #[geo(...)] attrs for custom column names
        for attr in &f.attrs {
            if !attr.path().is_ident("geo") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let lit: Lit = meta.value()?.parse()?;
                    if let Lit::Str(s) = lit {
                        col_name = s.value();
                    }
                    return Ok(());
                }
                Ok(())
            })?;
        }

        // Detect Option<T>
        let (is_option, inner_ty) = match option_inner(&f.ty) {
            Some(t) => (true, t),
            None => (false, &f.ty),
        };

        finfos.push(FieldInfo {
            ident,
            col_name,
            ty: inner_ty.clone(),
            is_option,
        });
    }

    // Generate struct fields for DataType::Struct
    let mut struct_field_tokens = Vec::new();

    for fi in &finfos {
        let col_name_lit = syn::LitStr::new(&fi.col_name, fi.ident.span());
        let ty = &fi.ty;
        let is_option = fi.is_option;

        struct_field_tokens.push(quote! {
            ::geoparquet_batch_writer::__dep::arrow_schema::Field::new(
                #col_name_lit,
                <#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::data_type(),
                #is_option
            )
        });
    }

    // Generate array creation for StructArray
    let mut field_array_tokens_iter_values = Vec::new();
    let mut field_array_tokens_iter = Vec::new();

    for (idx, fi) in finfos.iter().enumerate() {
        let ident = &fi.ident;
        let ty = &fi.ty;
        let is_option = fi.is_option;
        let arr_ident = format_ident!("__gp_arr_{}", idx);

        if is_option {
            // Field is Option<T>, extract Option<T> and use T::from_iter
            field_array_tokens_iter_values.push(quote! {
                let field_iter = values.iter().map(|r: &Self| r.#ident.clone());
                let #arr_ident = <#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter(field_iter);
            });
            field_array_tokens_iter.push(quote! {
                let field_iter = values.iter().map(|r: &Self| r.#ident.clone());
                let #arr_ident = <#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter(field_iter);
            });
        } else {
            // Field is T, extract T and use T::from_iter_values
            field_array_tokens_iter_values.push(quote! {
                let field_iter = values.iter().map(|r: &Self| r.#ident.clone());
                let #arr_ident = <#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter_values(field_iter);
            });
            field_array_tokens_iter.push(quote! {
                let field_iter = values.iter().map(|r: &Self| r.#ident.clone());
                let #arr_ident = <#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter_values(field_iter);
            });
        }
    }

    let field_array_refs: Vec<_> = (0..finfos.len())
        .map(|idx| format_ident!("__gp_arr_{}", idx))
        .collect();

    let expanded = quote! {
        impl ::geoparquet_batch_writer::__dep::ArrowDataType for #struct_ident
        where Self: Send + Sync + Clone + 'static
        {
            type Array = ::geoparquet_batch_writer::__dep::arrow_array::StructArray;

            fn data_type() -> ::geoparquet_batch_writer::__dep::arrow_schema::DataType {
                ::geoparquet_batch_writer::__dep::arrow_schema::DataType::Struct(
                    ::geoparquet_batch_writer::__dep::arrow_schema::Fields::from(vec![
                        #(#struct_field_tokens),*
                    ])
                )
            }

            fn from_iter_values<I>(iter: I) -> ::std::sync::Arc<Self::Array>
            where
                I: IntoIterator<Item = Self>,
            {
                let values: Vec<Self> = iter.into_iter().collect();
                #(#field_array_tokens_iter_values)*

                let field_arrays: Vec<::std::sync::Arc<dyn ::geoparquet_batch_writer::__dep::arrow_array::Array>> = vec![
                    #(#field_array_refs as ::std::sync::Arc<dyn ::geoparquet_batch_writer::__dep::arrow_array::Array>),*
                ];

                let fields = match Self::data_type() {
                    ::geoparquet_batch_writer::__dep::arrow_schema::DataType::Struct(fields) => fields,
                    _ => unreachable!(),
                };

                ::std::sync::Arc::new(
                    ::geoparquet_batch_writer::__dep::arrow_array::StructArray::new(
                        fields,
                        field_arrays,
                        None
                    )
                )
            }

            fn from_iter<I>(iter: I) -> ::std::sync::Arc<Self::Array>
            where
                I: IntoIterator<Item = Option<Self>>,
            {
                let collected: Vec<_> = iter.into_iter().collect();
                let len = collected.len();

                // Create validity bitmap for the struct array
                let mut validity = ::geoparquet_batch_writer::__dep::arrow_array::builder::BooleanBufferBuilder::new(len);
                let values: Vec<Self> = collected.into_iter().map(|opt| {
                    match opt {
                        Some(val) => {
                            validity.append(true);
                            val
                        },
                        None => {
                            validity.append(false);
                            // We need a default value for None cases
                            // This is a limitation - structs need default values for missing data
                            ::std::default::Default::default()
                        }
                    }
                }).collect();

                let validity_buffer = validity.finish();
                #(#field_array_tokens_iter)*

                let field_arrays: Vec<::std::sync::Arc<dyn ::geoparquet_batch_writer::__dep::arrow_array::Array>> = vec![
                    #(#field_array_refs as ::std::sync::Arc<dyn ::geoparquet_batch_writer::__dep::arrow_array::Array>),*
                ];

                let fields = match Self::data_type() {
                    ::geoparquet_batch_writer::__dep::arrow_schema::DataType::Struct(fields) => fields,
                    _ => unreachable!(),
                };

                ::std::sync::Arc::new(
                    ::geoparquet_batch_writer::__dep::arrow_array::StructArray::new(
                        fields,
                        field_arrays,
                        Some(validity_buffer.into())
                    )
                )
            }
        }
    };

    Ok(expanded.into())
}

fn option_inner(ty: &Type) -> Option<&Type> {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Option" {
                if let PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}

fn vec_inner(ty: &Type) -> Option<&Type> {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Vec" {
                if let PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}

fn geometry_kind(ty: &Type) -> Option<GeometryKind> {
    // Expect something like geo_types::Point<f64>, geo_types::LineString<f64>, etc.
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if let PathArguments::AngleBracketed(args) = &seg.arguments {
                if let Some(syn::GenericArgument::Type(Type::Path(inner))) = args.args.first() {
                    if let Some(seg2) = inner.path.segments.last() {
                        if seg2.ident == "f64" {
                            return match name.as_str() {
                                "Point" => Some(GeometryKind::Point),
                                "LineString" => Some(GeometryKind::LineString),
                                "Polygon" => Some(GeometryKind::Polygon),
                                "MultiPoint" => Some(GeometryKind::MultiPoint),
                                "MultiLineString" => Some(GeometryKind::MultiLineString),
                                "MultiPolygon" => Some(GeometryKind::MultiPolygon),
                                "Geometry" => Some(GeometryKind::Geometry),
                                "GeometryCollection" => Some(GeometryKind::GeometryCollection),
                                _ => None,
                            };
                        }
                    }
                }
            }
        }
    }
    None
}

fn arrow_datatype(ty: &Type) -> syn::Result<proc_macro2::TokenStream> {
    // Check if this is a Vec<T> type
    if let Some(inner_ty) = vec_inner(ty) {
        let inner_dt = arrow_datatype(inner_ty)?;
        return Ok(quote!(
            ::geoparquet_batch_writer::__dep::arrow_schema::DataType::List(
                ::std::sync::Arc::new(
                    ::geoparquet_batch_writer::__dep::arrow_schema::Field::new(
                        "item",
                        #inner_dt,
                        true
                    )
                )
            )
        ));
    }

    // Use the ArrowDataType trait to get the data type
    Ok(quote!(<#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::data_type()))
}

fn array_ctor(
    ty: &Type,
    is_option: bool,
) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    // Check if this is a Vec<T> type
    if let Some(inner_ty) = vec_inner(ty) {
        let inner_type_name = type_name(inner_ty);

        // First try primitive types with specific builders for better performance
        let primitive_builder = match inner_type_name.as_str() {
            "u64" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::UInt64Builder),
                quote!(values.append_value(*val)),
            )),
            "i64" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::Int64Builder),
                quote!(values.append_value(*val)),
            )),
            "u32" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::UInt32Builder),
                quote!(values.append_value(*val)),
            )),
            "i32" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::Int32Builder),
                quote!(values.append_value(*val)),
            )),
            "f64" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::Float64Builder),
                quote!(values.append_value(*val)),
            )),
            "f32" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::Float32Builder),
                quote!(values.append_value(*val)),
            )),
            "bool" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::BooleanBuilder),
                quote!(values.append_value(*val)),
            )),
            "String" => Some((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::builder::StringBuilder),
                quote!(values.append_value(val)),
            )),
            _ => None,
        };

        if let Some((values_builder_type, values_append)) = primitive_builder {
            let list_construction = if is_option {
                quote! {
                    {
                        let mut builder = ::geoparquet_batch_writer::__dep::arrow_array::builder::ListBuilder::new(#values_builder_type::new());
                        for opt_vec in it {
                            if let Some(vec_val) = opt_vec {
                                let values = builder.values();
                                for val in vec_val {
                                    #values_append;
                                }
                                builder.append(true);
                            } else {
                                builder.append(false);
                            }
                        }
                        builder.finish()
                    }
                }
            } else {
                quote! {
                    {
                        let mut builder = ::geoparquet_batch_writer::__dep::arrow_array::builder::ListBuilder::new(#values_builder_type::new());
                        for vec_val in it {
                            let values = builder.values();
                            for val in vec_val {
                                #values_append;
                            }
                            builder.append(true);
                        }
                        builder.finish()
                    }
                }
            };

            return Ok((
                quote!(::geoparquet_batch_writer::__dep::arrow_array::ListArray),
                quote! { ::std::sync::Arc::new(#list_construction) },
            ));
        }

        // For complex types that implement ArrowDataType, use a builder-based approach
        let list_construction = if is_option {
            quote! {
                {
                    // Collect the iterator once to avoid use-after-move issues
                    let all_list_values: Vec<_> = it.collect();

                    // Now create a simple list array by concatenating all values and tracking offsets
                    let mut all_values = Vec::new();
                    let mut validity = ::geoparquet_batch_writer::__dep::arrow_array::builder::BooleanBufferBuilder::new(all_list_values.len());

                    for list_opt in &all_list_values {
                        match list_opt {
                            Some(vec_val) => {
                                validity.append(true);
                                all_values.extend(vec_val.iter().cloned());
                            }
                            None => {
                                validity.append(false);
                                // No values to add for null case
                            }
                        }
                    }

                    let values_array = <#inner_ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter_values(all_values);
                    let validity_buffer = validity.finish();

                    // Use try_new instead of new_unchecked
                    ::geoparquet_batch_writer::__dep::arrow_array::ListArray::try_new(
                        ::std::sync::Arc::new(::geoparquet_batch_writer::__dep::arrow_schema::Field::new(
                            "item",
                            <#inner_ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::data_type(),
                            true
                        )),
                        ::geoparquet_batch_writer::__dep::arrow_buffer::OffsetBuffer::from_lengths(
                            all_list_values.iter().map(|opt| {
                                opt.as_ref().map(|v| v.len()).unwrap_or(0)
                            })
                        ),
                        values_array,
                        Some(validity_buffer.into())
                    ).unwrap()
                }
            }
        } else {
            quote! {
                {
                    let all_vecs: Vec<_> = it.collect();
                    let mut all_values = Vec::new();

                    for vec_val in &all_vecs {
                        all_values.extend(vec_val.iter().cloned());
                    }

                    let values_array = <#inner_ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter_values(all_values);

                    ::geoparquet_batch_writer::__dep::arrow_array::ListArray::try_new(
                        ::std::sync::Arc::new(::geoparquet_batch_writer::__dep::arrow_schema::Field::new(
                            "item",
                            <#inner_ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::data_type(),
                            true
                        )),
                        ::geoparquet_batch_writer::__dep::arrow_buffer::OffsetBuffer::from_lengths(
                            all_vecs.iter().map(|v| v.len())
                        ),
                        values_array,
                        None
                    ).unwrap()
                }
            }
        };
        return Ok((
            quote!(::geoparquet_batch_writer::__dep::arrow_array::ListArray),
            quote! { ::std::sync::Arc::new(#list_construction) },
        ));
    }

    // Use the ArrowDataType trait for array construction
    let arr_ty = quote!(<#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::Array);
    let from = if is_option {
        quote!(<#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter(it))
    } else {
        quote!(<#ty as ::geoparquet_batch_writer::__dep::ArrowDataType>::from_iter_values(it))
    };
    Ok((arr_ty, from))
}

fn value_mapper(ty: &Type, ident: &Ident, is_option: bool) -> proc_macro2::TokenStream {
    // Check if this is a Vec<T> type
    if vec_inner(ty).is_some() {
        if is_option {
            quote!(|r: &Self| r.#ident.as_ref())
        } else {
            quote!(|r: &Self| &r.#ident)
        }
    } else {
        let t = type_name(ty);
        if t == "String" {
            if is_option {
                quote!(|r: &Self| r.#ident.as_ref().map(|s| s.clone()))
            } else {
                quote!(|r: &Self| r.#ident.clone())
            }
        } else if is_copy_scalar(&t) {
            if is_option {
                quote!(|r: &Self| r.#ident)
            } else {
                quote!(|r: &Self| r.#ident)
            }
        } else {
            // Shouldn't happen given our supported types
            quote!(|r: &Self| r.#ident)
        }
    }
}

fn type_name(ty: &Type) -> String {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            return seg.ident.to_string();
        }
    }
    format!("{}", quote!(#ty))
}

fn is_copy_scalar(name: &str) -> bool {
    matches!(name, "u64" | "i64" | "u32" | "i32" | "f64" | "f32" | "bool")
}
