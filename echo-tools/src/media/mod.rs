/*!
 * 面向图片与文档的媒体处理工具。
 *
 * 提供以下工具：
 * - 从 URL 获取图片并转换为 base64
 */

/// 从 URL 获取图片的工具。
pub mod image_fetch;

pub use image_fetch::ImageFetchTool;
