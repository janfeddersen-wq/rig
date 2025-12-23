//! Context compression module for managing LLM context window limits.
//!
//! This module provides pluggable compression strategies that can reduce
//! chat history to fit within token budgets while preserving conversation
//! coherence and tool call/result integrity.
//!
//! ## Available Strategies
//!
//! - [`TruncationCompressor`]: Simple FIFO removal of oldest messages
//! - [`SlidingWindowCompressor`]: Preserves first/last messages, trims middle
//! - [`SummarizingCompressor`]: Uses an LLM to summarize removed context
//!
//! ## Example
//!
//! ```ignore
//! use rig::compression::{SummarizingCompressor, SlidingWindowCompressor};
//!
//! // Simple sliding window
//! let compressor = SlidingWindowCompressor::new()
//!     .with_preserve_first(1);
//!
//! // LLM-based summarization (can use a different model)
//! let summarizer = SummarizingCompressor::new(small_fast_model)
//!     .with_preserve_first(1)
//!     .with_preserve_recent(3);
//! ```

mod estimator;
mod traits;
mod truncation;
mod sliding_window;
mod summarizing;

pub use estimator::{estimate_tokens, estimate_message_tokens, estimate_messages_tokens};
pub use traits::{ContextCompressor, CompressionError};
pub use truncation::TruncationCompressor;
pub use sliding_window::SlidingWindowCompressor;
pub use summarizing::SummarizingCompressor;
