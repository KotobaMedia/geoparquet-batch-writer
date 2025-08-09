use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, Ident, Lit, PathArguments, Type, parse_macro_input, spanned::Spanned,
};

#[proc_macro_derive(GeoParquetRowData, attributes(geo))]
pub fn derive_geo_parquet_row_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_geoparquet_row_data(&input) {
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
        if !is_geometry && is_point_f64(inner_ty) {
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
            dim,
        });
    }

    if geom_count == 0 {
        return Err(syn::Error::new(
            input.span(),
            "No geometry field found. Mark one with #[geo(geometry)] or use geo::Point<f64>.",
        ));
    }
    if geom_count > 1 {
        return Err(syn::Error::new(
            input.span(),
            "Multiple geometry fields found; currently only one geometry field is supported.",
        ));
    }

    // Generate schema fields & array builders
    let mut schema_field_tokens = Vec::new();
    let mut array_expr_tokens = Vec::new();
    let mut mem_est_addends = Vec::new();

    // geometry shared setup (PointType)
    let point_type_ident = format_ident!("__gp_point_type");
    let geom_setup = quote! {
        let #point_type_ident = {
            let dim = ::geoarrow_schema::Dimension::XY;
            ::geoarrow_schema::PointType::new(dim, ::std::default::Default::default())
        };
    };
    let geom_setup_with_dim = |dim_string: &str| -> proc_macro2::TokenStream {
        let dim_expr = match dim_string {
            "XYZ" => quote!(::geoarrow_schema::Dimension::XYZ),
            "XYM" => quote!(::geoarrow_schema::Dimension::XYM),
            "XY" | _ => quote!(::geoarrow_schema::Dimension::XY),
        };
        quote! {
            let #point_type_ident = {
                let dim = #dim_expr;
                ::geoarrow_schema::PointType::new(dim, ::std::default::Default::default())
            };
        }
    };

    // Flag whether we need geometry init
    let mut geometry_init_tokens: Option<proc_macro2::TokenStream> = None;

    for (idx, fi) in finfos.iter().enumerate() {
        if fi.is_geometry {
            let col_name_lit = syn::LitStr::new(&fi.col_name, fi.ident.span());
            let is_option = fi.is_option;
            // Schema field (nullable if Option)
            schema_field_tokens.push(quote! {
                #point_type_ident.to_field(#col_name_lit, #is_option)
            });

            // Arrays: use PointBuilder
            let b_ident = format_ident!("__gp_geom_builder");
            let arr_ident = format_ident!("__gp_arr_{}", idx);
            let ident = &fi.ident;

            // Make builder pushes respect nullability
            let push_tokens = if fi.is_option {
                quote! {
                    for row in rows {
                        if let Some(g) = row.#ident {
                            let p = ::geo::point!(x: g.x(), y: g.y());
                            #b_ident.push_point(Some(&p));
                        } else {
                            #b_ident.push_null();
                        }
                    }
                }
            } else {
                quote! {
                    for row in rows {
                        let g = &row.#ident;
                        let p = ::geo::point!(x: g.x(), y: g.y());
                        #b_ident.push_point(Some(&p));
                    }
                }
            };

            array_expr_tokens.push(quote! {{
                use ::geoarrow_array::GeoArrowArray as _;
                let mut #b_ident = ::geoarrow_array::builder::PointBuilder::new(#point_type_ident.clone());
                #push_tokens
                let #arr_ident = ::std::sync::Arc::new(#b_ident.finish().into_array_ref());
                #arr_ident
            }});

            // Memory estimate: rough ~32 bytes per point
            mem_est_addends.push(quote!(32usize));

            // Remember to create the PointType with requested dim
            geometry_init_tokens = Some(match &fi.dim {
                Some(s) => geom_setup_with_dim(s),
                None => geom_setup.clone(),
            });
        } else {
            // Scalar or String
            let dt = arrow_datatype(&fi.ty)?;
            let (array_ty, from_tokens, mem_bytes) = array_ctor(&fi.ty, fi.is_option)?;
            let col_name_lit = syn::LitStr::new(&fi.col_name, fi.ident.span());
            let is_option = fi.is_option;
            schema_field_tokens.push(quote! {
                ::arrow_schema::Field::new(#col_name_lit, #dt, #is_option)
            });
            let arr_ident = format_ident!("__gp_arr_{}", idx);

            // rows.iter().map(|r| r.field) vs map(|r| r.field.as_ref()…) for Option & String
            let map_expr = value_mapper(&fi.ty, &fi.ident, fi.is_option);

            array_expr_tokens.push(quote! {{
                let it = rows.iter().map(#map_expr);
                let arr: ::std::sync::Arc<#array_ty> = #from_tokens;
                let #arr_ident: ::std::sync::Arc<dyn ::arrow_array::Array> = arr;
                #arr_ident
            }});

            mem_est_addends.push(quote!(#mem_bytes));
        }
    }

    let schema_vec_tokens = quote! {
        {
            #geometry_init_tokens
            ::std::sync::Arc::new(::arrow_schema::Schema::new(vec![
                #(#schema_field_tokens),*
            ]))
        }
    };

    let arrays_vec_tokens = quote! {
        {
            #geometry_init_tokens
            ::anyhow::Ok(vec![
                #(#array_expr_tokens),*
            ])
        }
    };

    let mem_sum_tokens = quote! {{
        0usize #(+ #mem_est_addends)*
    }};

    let expanded = quote! {
        impl ::geoparquet_batch_writer_core::GeoParquetRowData for #struct_ident
        where Self: Clone + Send + Sync
        {
            fn schema() -> ::std::sync::Arc<::arrow_schema::Schema> {
                #schema_vec_tokens
            }

            fn to_arrays(rows: &[Self]) -> ::anyhow::Result<Vec<::std::sync::Arc<dyn ::arrow_array::Array>>> {
                #arrays_vec_tokens
            }

            fn estimated_row_memory_size() -> usize {
                #mem_sum_tokens
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

fn is_point_f64(ty: &Type) -> bool {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Point" {
                if let PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(Type::Path(inner))) = args.args.first() {
                        if let Some(seg2) = inner.path.segments.last() {
                            return seg2.ident == "f64";
                        }
                    }
                }
            }
        }
    }
    false
}

fn arrow_datatype(ty: &Type) -> syn::Result<proc_macro2::TokenStream> {
    Ok(match type_name(ty).as_str() {
        "u64" => quote!(::arrow_schema::DataType::UInt64),
        "i64" => quote!(::arrow_schema::DataType::Int64),
        "u32" => quote!(::arrow_schema::DataType::UInt32),
        "i32" => quote!(::arrow_schema::DataType::Int32),
        "f64" => quote!(::arrow_schema::DataType::Float64),
        "f32" => quote!(::arrow_schema::DataType::Float32),
        "bool" => quote!(::arrow_schema::DataType::Boolean),
        "String" => quote!(::arrow_schema::DataType::Utf8),
        other => {
            return Err(syn::Error::new(
                ty.span(),
                format!("Unsupported field type `{}`", other),
            ));
        }
    })
}

fn array_ctor(
    ty: &Type,
    is_option: bool,
) -> syn::Result<(
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
)> {
    let t = type_name(ty);
    let (arr_ty, from_vals, from_opts, est): (_, _, _, proc_macro2::TokenStream) = match t.as_str()
    {
        "u64" => (
            quote!(::arrow_array::UInt64Array),
            quote!(::arrow_array::UInt64Array::from_iter_values(it)),
            quote!(::arrow_array::UInt64Array::from_iter(it)),
            quote!(::std::mem::size_of::<u64>()),
        ),
        "i64" => (
            quote!(::arrow_array::Int64Array),
            quote!(::arrow_array::Int64Array::from_iter_values(it)),
            quote!(::arrow_array::Int64Array::from_iter(it)),
            quote!(::std::mem::size_of::<i64>()),
        ),
        "u32" => (
            quote!(::arrow_array::UInt32Array),
            quote!(::arrow_array::UInt32Array::from_iter_values(it)),
            quote!(::arrow_array::UInt32Array::from_iter(it)),
            quote!(::std::mem::size_of::<u32>()),
        ),
        "i32" => (
            quote!(::arrow_array::Int32Array),
            quote!(::arrow_array::Int32Array::from_iter_values(it)),
            quote!(::arrow_array::Int32Array::from_iter(it)),
            quote!(::std::mem::size_of::<i32>()),
        ),
        "f64" => (
            quote!(::arrow_array::Float64Array),
            quote!(::arrow_array::Float64Array::from_iter_values(it)),
            quote!(::arrow_array::Float64Array::from_iter(it)),
            quote!(::std::mem::size_of::<f64>()),
        ),
        "f32" => (
            quote!(::arrow_array::Float32Array),
            quote!(::arrow_array::Float32Array::from_iter_values(it)),
            quote!(::arrow_array::Float32Array::from_iter(it)),
            quote!(::std::mem::size_of::<f32>()),
        ),
        "bool" => (
            quote!(::arrow_array::BooleanArray),
            quote!(::arrow_array::BooleanArray::from_iter_values(it)),
            quote!(::arrow_array::BooleanArray::from_iter(it)),
            quote!(1usize),
        ),
        "String" => (
            quote!(::arrow_array::StringArray),
            quote!(::arrow_array::StringArray::from_iter_values(it)),
            quote!(::arrow_array::StringArray::from_iter(it)),
            quote!(48usize),
        ),
        other => {
            return Err(syn::Error::new(
                ty.span(),
                format!("Unsupported field type `{}`", other),
            ));
        }
    };
    let from = if is_option { from_opts } else { from_vals };
    Ok((arr_ty, quote! { ::std::sync::Arc::new(#from) }, est))
}

fn value_mapper(ty: &Type, ident: &Ident, is_option: bool) -> proc_macro2::TokenStream {
    let t = type_name(ty);
    if t == "String" {
        if is_option {
            quote!(|r: &Self| r.#ident.as_deref())
        } else {
            quote!(|r: &Self| r.#ident.as_str())
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
