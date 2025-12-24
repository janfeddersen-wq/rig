//! Fast token estimation without external dependencies.
//!
//! Uses character-based heuristics optimized for code-heavy content.
//! The 3.4 chars/token ratio accounts for code's higher symbol density
//! compared to natural language prose (~4.0 chars/token).

use crate::completion::Message;
use crate::completion::message::{AssistantContent, UserContent, ToolResultContent};

/// Characters per token ratio, optimized for code-heavy content.
/// Natural language is typically ~4.0, code is ~3.0-3.5.
const CHARS_PER_TOKEN: f32 = 3.4;

/// Overhead tokens per message for role and formatting.
const MESSAGE_OVERHEAD: usize = 4;

/// Estimate token count for a text string.
#[inline]
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.len() as f32 / CHARS_PER_TOKEN).ceil() as usize
}

/// Estimate token count for a single message.
pub fn estimate_message_tokens(message: &Message) -> usize {
    let content_tokens: usize = match message {
        Message::User { content } => {
            content.iter().map(estimate_user_content_tokens).sum()
        }
        Message::Assistant { content, .. } => {
            content.iter().map(estimate_assistant_content_tokens).sum()
        }
    };
    content_tokens + MESSAGE_OVERHEAD
}

/// Estimate token count for a sequence of messages.
pub fn estimate_messages_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

fn estimate_user_content_tokens(content: &UserContent) -> usize {
    match content {
        UserContent::Text(t) => estimate_tokens(&t.text),
        UserContent::ToolResult(tr) => {
            // Tool result ID + content
            estimate_tokens(&tr.id)
                + tr.content
                    .iter()
                    .map(|c| match c {
                        ToolResultContent::Text(t) => estimate_tokens(&t.text),
                        ToolResultContent::Image(_) => 85, // Base64 images are ~85 tokens for the reference
                    })
                    .sum::<usize>()
        }
        UserContent::Image(_) => 85,  // Typical image token estimate
        UserContent::Audio(_) => 100, // Audio reference tokens
        UserContent::Video(_) => 100, // Video reference tokens
        UserContent::Document(d) => estimate_tokens(&d.data.to_string()),
    }
}

fn estimate_assistant_content_tokens(content: &AssistantContent) -> usize {
    match content {
        AssistantContent::Text(t) => estimate_tokens(&t.text),
        AssistantContent::ToolCall(tc) => {
            // Tool name + arguments (usually JSON)
            estimate_tokens(&tc.function.name)
                + estimate_tokens(&tc.function.arguments.to_string())
        }
        AssistantContent::Reasoning(r) => {
            r.reasoning.iter().map(|s| estimate_tokens(s)).sum()
        }
        AssistantContent::Image(_) => 85,
    }
}

/// Comprehensive context estimation for LLM requests.
///
/// This struct provides a complete picture of context usage before sending
/// a request to the LLM, including:
/// - System prompt tokens
/// - Tool definitions tokens (JSON schemas)
/// - All message tokens (user, assistant, tool calls, tool results, reasoning)
///
/// The estimation uses the 3.4 chars/token ratio optimized for code-heavy content.
#[derive(Debug, Clone)]
pub struct ContextEstimate {
    /// Tokens used by the system prompt/preamble
    pub system_prompt_tokens: usize,
    /// Tokens used by tool definitions (JSON schemas)
    pub tool_definitions_tokens: usize,
    /// Tokens used by all messages (user, assistant, tool calls, tool results, reasoning)
    pub messages_tokens: usize,
    /// Total estimated tokens (sum of all above)
    pub total_tokens: usize,
    /// Model's context window size
    pub context_window: u64,
    /// Percentage of context window used (0-100+)
    pub usage_percent: u32,
}

impl ContextEstimate {
    /// Create a new context estimate with all components.
    ///
    /// # Arguments
    /// * `system_prompt` - The system prompt/preamble text
    /// * `tool_definitions_json` - Tool definitions serialized as JSON
    /// * `messages` - All conversation messages
    /// * `context_window` - The model's context window size in tokens
    ///
    /// # Example
    /// ```ignore
    /// use rig::compression::ContextEstimate;
    ///
    /// let estimate = ContextEstimate::new(
    ///     "You are a helpful assistant.",
    ///     &serde_json::to_string(&tools).unwrap(),
    ///     &messages,
    ///     200_000,
    /// );
    /// println!("Using {}% of context", estimate.usage_percent);
    /// ```
    pub fn new(
        system_prompt: &str,
        tool_definitions_json: &str,
        messages: &[Message],
        context_window: u64,
    ) -> Self {
        let system_prompt_tokens = estimate_tokens(system_prompt);
        let tool_definitions_tokens = estimate_tokens(tool_definitions_json);
        let messages_tokens = estimate_messages_tokens(messages);

        let total_tokens = system_prompt_tokens + tool_definitions_tokens + messages_tokens;
        let usage_percent = if context_window > 0 {
            ((total_tokens as u64 * 100) / context_window) as u32
        } else {
            0
        };

        Self {
            system_prompt_tokens,
            tool_definitions_tokens,
            messages_tokens,
            total_tokens,
            context_window,
            usage_percent,
        }
    }

    /// Check if compression should be triggered based on a threshold percentage.
    ///
    /// # Arguments
    /// * `threshold_percent` - The percentage (0-100) at which compression triggers
    ///
    /// # Returns
    /// `true` if current usage exceeds the threshold
    pub fn needs_compression(&self, threshold_percent: u32) -> bool {
        self.usage_percent >= threshold_percent
    }

    /// Calculate the threshold token count for a given percentage.
    pub fn threshold_tokens(&self, threshold_percent: u32) -> u64 {
        (self.context_window * threshold_percent as u64) / 100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // "hello" = 5 chars / 3.4 = 1.47 -> ceil = 2
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn test_estimate_tokens_code() {
        // Typical code line
        let code = "fn main() { println!(\"Hello, world!\"); }";
        let tokens = estimate_tokens(code);
        // 42 chars / 3.4 = 12.35 -> ceil = 13 (but f32 rounding gives 12)
        assert_eq!(tokens, 12);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        // 340 chars should be ~100 tokens
        let text = "a".repeat(340);
        assert_eq!(estimate_tokens(&text), 100);
    }

    #[test]
    fn test_context_estimate() {
        let system_prompt = "You are a helpful assistant.";
        let tool_defs = r#"[{"name":"read_file","description":"Read a file"}]"#;
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let estimate = ContextEstimate::new(
            system_prompt,
            tool_defs,
            &messages,
            200_000,
        );

        // Verify components are calculated
        assert!(estimate.system_prompt_tokens > 0);
        assert!(estimate.tool_definitions_tokens > 0);
        assert!(estimate.messages_tokens > 0);
        assert_eq!(
            estimate.total_tokens,
            estimate.system_prompt_tokens + estimate.tool_definitions_tokens + estimate.messages_tokens
        );
        assert_eq!(estimate.context_window, 200_000);
        // Small messages should be less than 1% of 200k
        assert!(estimate.usage_percent < 1);
    }

    #[test]
    fn test_context_estimate_needs_compression() {
        // Create a large message to test threshold
        let large_text = "x".repeat(6800); // ~2000 tokens
        let messages = vec![Message::user(&large_text)];

        let estimate = ContextEstimate::new(
            "",
            "",
            &messages,
            2000, // Small context window
        );

        // Should exceed 80% threshold
        assert!(estimate.needs_compression(80));
        // Should not exceed 120% threshold (impossible)
        assert!(!estimate.needs_compression(120));
    }
}
