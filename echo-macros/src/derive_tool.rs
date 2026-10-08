/*!
`#[derive(Tool)]` —— 根据结构体定义生成 `Tool` trait 实现。

## 用法

```rust,ignore
#[derive(Tool)]
#[tool(name = "read_file", description = "Read a file", risk_level = "ReadOnly")]
struct ReadFileTool {
    #[tool_param(skip)]
    base_dir: PathBuf,

    #[tool_param(description = "File path")]
    path: String,

    #[tool_param(description = "Start line")]
    start_line: Option<usize>,
}

impl ToolRunner<ReadFileToolParams> for ReadFileTool {
    async fn run(&self, params: ReadFileToolParams) -> Result<ToolResult> {
        // ... business logic ...
    }
}
```

该宏会生成：
- 实现了 `Deserialize + JsonSchema` 的 `ReadFileToolParams` 结构体
- 完整的 `impl Tool for ReadFileTool`（name、description、parameters、execute 等）
*/

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Attribute, Data, DeriveInput, Fields, Ident, LitStr, Token};

// ── 解析后的属性 ──────────────────────────────────────────────────────────

/// 解析后的结构体级 `#[tool(...)]` 属性
struct ToolStructAttrs {
    /// 工具名称
    name: String,
    /// 工具描述
    description: String,
    /// 风险等级（`ReadOnly` / `Standard` / `Dangerous`），未指定则使用默认
    risk_level: Option<String>,
    /// 工具所需的权限列表
    permissions: Vec<Ident>,
}

/// 解析后的字段级 `#[tool_param(...)]` 属性
struct ToolParamAttrs {
    /// 是否跳过该字段（内部状态，不暴露为参数）
    skip: bool,
    /// 参数描述
    description: Option<String>,
}

// ── 解析辅助函数 ────────────────────────────────────────────────────────────

impl syn::parse::Parse for ToolStructAttrs {
    /// 解析 `name`、`description`、`risk_level`、`permissions` 键值对。
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name: Option<String> = None;
        let mut description: Option<String> = None;
        let mut risk_level: Option<String> = None;
        let mut permissions: Vec<Ident> = Vec::new();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if ident == "name" {
                let _eq: Token![=] = input.parse()?;
                let value: LitStr = input.parse()?;
                name = Some(value.value());
            } else if ident == "description" {
                let _eq: Token![=] = input.parse()?;
                let value: LitStr = input.parse()?;
                description = Some(value.value());
            } else if ident == "risk_level" {
                let _eq: Token![=] = input.parse()?;
                let value: LitStr = input.parse()?;
                risk_level = Some(value.value());
            } else if ident == "permissions" {
                let _eq: Token![=] = input.parse()?;
                let content;
                syn::bracketed!(content in input);
                while !content.is_empty() {
                    let perm: Ident = content.parse()?;
                    permissions.push(perm);
                    if !content.is_empty() {
                        let _comma: Token![,] = content.parse()?;
                    }
                }
            } else {
                return Err(syn::Error::new_spanned(
                    ident,
                    "unknown attribute, expected `name`, `description`, `risk_level`, or `permissions`",
                ));
            }
            if !input.is_empty() {
                let _comma: Token![,] = input.parse()?;
            }
        }

        Ok(ToolStructAttrs {
            name: name.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[tool] requires `name = \"...\"`",
                )
            })?,
            description: description.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[tool] requires `description = \"...\"`",
                )
            })?,
            risk_level,
            permissions,
        })
    }
}

/// 从字段上解析 `#[tool_param(skip)]` 或 `#[tool_param(description = "...")]`。
fn parse_tool_param_attrs(attrs: &[Attribute]) -> syn::Result<ToolParamAttrs> {
    let mut skip = false;
    let mut description: Option<String> = None;

    for attr in attrs {
        if !attr.path().is_ident("tool_param") {
            continue;
        }
        let parsed = attr.parse_args::<ToolParamRaw>()?;
        skip = skip || parsed.skip;
        if parsed.description.is_some() {
            description = parsed.description;
        }
    }

    // 同时检查字段上的文档注释
    if description.is_none() {
        description = super::extract_doc_comments(attrs);
    }

    Ok(ToolParamAttrs { skip, description })
}

/// tool_param 参数的原始解析结果（同时支持 `skip` 与 `description = "..."`）
struct ToolParamRaw {
    /// 是否出现了 `skip`
    skip: bool,
    /// `description` 的值
    description: Option<String>,
}

impl syn::parse::Parse for ToolParamRaw {
    /// 解析以逗号分隔的 `skip` / `description = "..."` 参数。
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut skip = false;
        let mut description: Option<String> = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if ident == "skip" {
                skip = true;
            } else if ident == "description" {
                let _eq: Token![=] = input.parse()?;
                let value: LitStr = input.parse()?;
                description = Some(value.value());
            } else {
                return Err(syn::Error::new_spanned(
                    ident,
                    "expected `skip` or `description`",
                ));
            }
            if !input.is_empty() {
                let _comma: Token![,] = input.parse()?;
            }
        }
        Ok(ToolParamRaw { skip, description })
    }
}

/// 从结构体上取出第一个 `#[tool(...)]` 属性并解析，返回 ToolStructAttrs。
fn extract_tool_attrs(attrs: &[Attribute]) -> syn::Result<ToolStructAttrs> {
    for attr in attrs {
        if attr.path().is_ident("tool") {
            return attr.parse_args::<ToolStructAttrs>();
        }
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "#[derive(Tool)] requires a #[tool(name = \"...\", description = \"...\")] attribute on the struct",
    ))
}

// ── 结构体字段信息 ──────────────────────────────────────────────────────────

/// 需要暴露给 LLM 的参数字段
struct ParamField {
    /// 字段名
    ident: Ident,
    /// 字段类型
    ty: syn::Type,
    /// 附加到生成字段上的 `#[schemars(...)]` 属性
    schemars_attr: TokenStream,
}

// ── 主入口 ───────────────────────────────────────────────────────────

/// `#[derive(Tool)]` 的实现入口：生成参数结构体与 `Tool` trait 实现。
pub fn derive_tool_impl(input: DeriveInput) -> syn::Result<TokenStream> {
    let echo_crate = crate::resolve_echo_crate_path(crate::MacroCrate::Core)?;
    let serde_crate = crate::macro_support_crate_path(&echo_crate, "serde");
    let schemars_crate = crate::macro_support_crate_path(&echo_crate, "schemars");
    let struct_ident = &input.ident;
    let struct_name_str = struct_ident.to_string();
    let params_ident = format_ident!("{}Params", struct_name_str);

    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "#[derive(Tool)] does not support generic structs; use a concrete wrapper type",
        ));
    }

    // 提取结构体级的 #[tool(...)] 属性
    let tool_attrs = extract_tool_attrs(&input.attrs)?;
    let tool_name = &tool_attrs.name;
    let tool_desc = &tool_attrs.description;

    // 解析结构体字段
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    &input,
                    "#[derive(Tool)] only supports named fields (struct with { })",
                ));
            }
            Fields::Unit => {
                // 单元结构体：参数生成为空结构体
                return generate_unit_tool(
                    &echo_crate,
                    struct_ident,
                    params_ident,
                    tool_name,
                    tool_desc,
                    &tool_attrs,
                );
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input,
                "#[derive(Tool)] only supports structs",
            ));
        }
    };

    // 将参数字段与状态字段分开
    let mut param_fields: Vec<ParamField> = Vec::new();
    let mut field_names: Vec<Ident> = Vec::new();

    for field in fields {
        let field_ident = field.ident.as_ref().ok_or_else(|| {
            syn::Error::new_spanned(field, "#[derive(Tool)] requires named fields")
        })?;

        let param_attrs = parse_tool_param_attrs(&field.attrs)?;
        if param_attrs.skip {
            continue; // 内部状态，不暴露给 LLM
        }

        let schemars_attr = if let Some(desc) = &param_attrs.description {
            quote! { #[schemars(description = #desc)] }
        } else {
            quote! {}
        };

        param_fields.push(ParamField {
            ident: field_ident.clone(),
            ty: field.ty.clone(),
            schemars_attr,
        });
        field_names.push(field_ident.clone());
    }

    // 生成参数结构体
    let param_field_defs: Vec<TokenStream> = param_fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            let ty = &f.ty;
            let schemars_attr = &f.schemars_attr;
            quote! {
                #schemars_attr
                pub #name: #ty
            }
        })
        .collect();

    // ── 风险等级覆盖 ────────────────────────────────────────────────
    let risk_level_override = if let Some(level) = &tool_attrs.risk_level {
        let level_path = match level.as_str() {
            "ReadOnly" => quote! { #echo_crate::tools::ToolRiskLevel::ReadOnly },
            "Standard" => quote! { #echo_crate::tools::ToolRiskLevel::Standard },
            "Dangerous" => quote! { #echo_crate::tools::ToolRiskLevel::Dangerous },
            other => {
                return Err(syn::Error::new_spanned(
                    &input,
                    format!(
                        "risk_level must be `ReadOnly`, `Standard`, or `Dangerous`, got `{}`",
                        other
                    ),
                ));
            }
        };
        quote! {
            fn risk_level(&self) -> #echo_crate::tools::ToolRiskLevel {
                #level_path
            }
        }
    } else {
        quote! {}
    };

    // ── 权限覆盖 ──────────────────────────────────────────────
    let permissions_override = if tool_attrs.permissions.is_empty() {
        quote! {}
    } else {
        let perms = tool_attrs.permissions.iter().map(|p| {
            quote! { #echo_crate::tools::permission::ToolPermission::#p }
        });
        quote! {
            fn permissions(&self) -> Vec<#echo_crate::tools::permission::ToolPermission> {
                vec![#(#perms),*]
            }
        }
    };

    // ── 构建生成代码块 ──────────────────────────────────────────
    let generated = quote! {
        /// 为 [`#struct_ident`] 自动生成的参数结构体。
        #[derive(#echo_crate::__macro_support::serde::Deserialize, #echo_crate::__macro_support::schemars::JsonSchema)]
        #[serde(crate = #serde_crate)]
        #[schemars(crate = #schemars_crate)]
        #[allow(non_camel_case_types)]
        pub struct #params_ident {
            #(#param_field_defs),*
        }

        #[allow(dead_code)]
        impl #echo_crate::tools::Tool for #struct_ident {
            fn name(&self) -> &str { #tool_name }
            fn description(&self) -> &str { #tool_desc }

            fn parameters(&self) -> #echo_crate::__macro_support::serde_json::Value {
                let schema = #echo_crate::__macro_support::schemars::schema_for!(#params_ident);
                #echo_crate::__macro_support::serde_json::to_value(schema).unwrap_or_default()
            }

            #risk_level_override

            #permissions_override

            fn validate_parameters<'a>(
                &'a self,
                params: &'a #echo_crate::tools::ToolParameters,
            ) -> #echo_crate::__macro_support::futures::future::BoxFuture<'a, #echo_crate::error::Result<()>> {
                Box::pin(async move {
                    Self::deserialize_params(params)?;
                    Ok(())
                })
            }

            fn execute<'a>(
                &'a self,
                parameters: #echo_crate::tools::ToolParameters,
            ) -> #echo_crate::__macro_support::futures::future::BoxFuture<'a, #echo_crate::error::Result<#echo_crate::tools::ToolResult>> {
                Box::pin(async move {
                    let params = #struct_ident::deserialize_params(& parameters)?;
                    <Self as #echo_crate::tools::ToolRunner<#params_ident>>::run(self, params).await
                })
            }
        }

        impl #struct_ident {
            #[doc(hidden)]
            fn deserialize_params(params: &#echo_crate::tools::ToolParameters) -> #echo_crate::error::Result<#params_ident> {
                let value = #echo_crate::__macro_support::serde_json::Value::Object(params.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
                #echo_crate::__macro_support::serde_json::from_value(value).map_err(|e| {
                    let msg = e.to_string();
                    let field = msg
                        .strip_prefix("missing field `")
                        .and_then(|s| s.strip_suffix('`'))
                        .or_else(|| {
                            msg.split(" at `").nth(1).and_then(|s| s.strip_suffix('`'))
                        })
                        .unwrap_or("(deserialization)");
                    #echo_crate::error::ToolError::InvalidParameter {
                        name: field.to_string(),
                        message: msg,
                    }.into()
                })
            }
        }
    };

    Ok(generated)
}

// ── 单元结构体特例 ───────────────────────────────────────────────────

/// 为单元结构体生成空参数结构体与 `Tool` trait 实现。
fn generate_unit_tool(
    echo_crate: &syn::Path,
    struct_ident: &Ident,
    params_ident: Ident,
    tool_name: &str,
    tool_desc: &str,
    tool_attrs: &ToolStructAttrs,
) -> syn::Result<TokenStream> {
    let serde_crate = crate::macro_support_crate_path(echo_crate, "serde");
    let schemars_crate = crate::macro_support_crate_path(echo_crate, "schemars");
    let risk_level_override = if let Some(level) = &tool_attrs.risk_level {
        let level_path = match level.as_str() {
            "ReadOnly" => quote! { #echo_crate::tools::ToolRiskLevel::ReadOnly },
            "Standard" => quote! { #echo_crate::tools::ToolRiskLevel::Standard },
            "Dangerous" => quote! { #echo_crate::tools::ToolRiskLevel::Dangerous },
            _ => {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown risk_level: {}", level),
                ));
            }
        };
        quote! {
            fn risk_level(&self) -> #echo_crate::tools::ToolRiskLevel {
                #level_path
            }
        }
    } else {
        quote! {}
    };

    let permissions_override = if tool_attrs.permissions.is_empty() {
        quote! {}
    } else {
        let perms = tool_attrs.permissions.iter().map(|p| {
            quote! { #echo_crate::tools::permission::ToolPermission::#p }
        });
        quote! {
            fn permissions(&self) -> Vec<#echo_crate::tools::permission::ToolPermission> {
                vec![#(#perms),*]
            }
        }
    };

    Ok(quote! {
        /// 为 [`#struct_ident`] 自动生成的空参数结构体。
        #[derive(#echo_crate::__macro_support::serde::Deserialize, #echo_crate::__macro_support::schemars::JsonSchema)]
        #[serde(crate = #serde_crate)]
        #[schemars(crate = #schemars_crate)]
        #[allow(non_camel_case_types)]
        pub struct #params_ident {}

        #[allow(dead_code)]
        impl #echo_crate::tools::Tool for #struct_ident {
            fn name(&self) -> &str { #tool_name }
            fn description(&self) -> &str { #tool_desc }

            fn parameters(&self) -> #echo_crate::__macro_support::serde_json::Value {
                #echo_crate::__macro_support::serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }

            #risk_level_override

            #permissions_override

            fn execute<'a>(
                &'a self,
                _parameters: #echo_crate::tools::ToolParameters,
            ) -> #echo_crate::__macro_support::futures::future::BoxFuture<'a, #echo_crate::error::Result<#echo_crate::tools::ToolResult>> {
                Box::pin(async move {
                    <Self as #echo_crate::tools::ToolRunner<#params_ident>>::run(self, #params_ident {}).await
                })
            }
        }
    })
}
