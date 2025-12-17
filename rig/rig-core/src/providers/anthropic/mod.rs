//! Anthropic API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::anthropic;
//!
//! let client = anthropic::Client::new("YOUR_API_KEY");
//!
//! let sonnet = client.completion_model(anthropic::CLAUDE_3_5_SONNET);
//! ```
//!
//! # OAuth Authentication (Claude Code)
//!
//! For Claude Code OAuth tokens, use `OAuthClient`:
//!
//! ```
//! use rig::providers::anthropic;
//!
//! // Using environment variable CLAUDE_CODE_AUTH_TOKEN
//! let client = anthropic::OAuthClient::from_env();
//!
//! // Or directly with token
//! let client = anthropic::OAuthClient::builder()
//!     .api_key("your-oauth-token")
//!     .build()
//!     .unwrap();
//!
//! let sonnet = client.completion_model(anthropic::CLAUDE_3_5_SONNET);
//! ```

pub mod client;
pub mod completion;
pub mod decoders;
pub mod streaming;

pub use client::{Client, ClientBuilder, OAuthClient, OAuthClientBuilder};
pub use completion::OAuthCompletionModel;
