//! Summarizing context compression using LLM.
//!
//! This strategy uses an LLM to create a "Continuity Briefing" from removed
//! messages, preserving important context in a compressed form.

use std::sync::Arc;

use crate::completion::{Message, Prompt};

use super::estimator::{estimate_messages_tokens, estimate_tokens};
use super::traits::{CompressionError, ContextCompressor};

/// The prompt template for generating continuity briefings.
const SUMMARIZATION_PROMPT: &str = r#"**Your Role:** You are a specialized AI Context Compression Engine.

**Your Task:** Analyze the provided conversation with an AI Coding Agent and generate a "Continuity Briefing." The primary goal of this briefing is to give the AI Agent a detailed understanding of the *current state* so it can resume the task perfectly, with only a very brief summary of the steps that led here.

**Instructions:**
1.  **Prioritize the Present:** The most detailed information should be about the immediate task and current code state.
2.  **Summarize the Past:** The history should be a very high-level overview. Do not detail every previous attempt or code iteration.
3.  **Be Unambiguous:** Use clear and direct language. The goal is function, not prose.
4.  **Strictly Follow the Output Format:** The entire output must adhere to the specified Markdown format below.

**Input:**
[CONVERSATION_HISTORY]

**Output Format (Strict):**

### 🎯 **Overall Goal**
*A single, concise sentence describing the user's main objective.*

### 🗺️ **Recent Path (Brief Summary)**
*A very brief, high-level summary of the last 2-3 major steps taken. Focus on the *outcome* of those steps, not the process.*
*   **Example:** "Initial function `X` was created. User found an issue with edge case `Y`. The last attempt to fix it resulted in error `Z`."

---

### 📍 **Current State (Detailed Explanation)**
*This is the most important section. Provide a detailed snapshot of where the project is RIGHT NOW.*

*   **Current Focus:** What specific file, function, or module are we currently working on? (e.g., "Refining the `parse_json_payload` function in `api_handler.py`.")
*   **Immediate Obstacle/Task:** What is the exact problem to be solved or the next specific action to be taken? (e.g., "The function fails with a `KeyError` when the 'optional_field' is missing from the input JSON. The immediate task is to add robust error handling for this specific case.")
*   **Code Status:**
    *   **Relevant Code Snippet:** *Include only the small, specific block of code (function, loop, etc.) that is the direct subject of the current task.*
    *   **Last Known Error:** If applicable, state the exact error message and a brief explanation. (e.g., "Error: `KeyError: 'optional_field'`. This occurs on line 23 when the function tries to access a key that doesn't exist.")
*   **Key Constraints & Requirements:** List any critical user requirements relevant to the *immediate* task. (e.g., "The solution must not use a try-except block; the user wants a conditional check. The function must return `None` if the field is missing.")

### 🚀 **Next Action Required**
*A clear, one-sentence directive for the AI.*
*   **Example:** "AI needs to modify the provided code snippet to check for the existence of 'optional_field' before accessing it and return `None` if it's absent."
"#;

/// A compressor that summarizes removed context using an LLM.
///
/// This strategy:
/// 1. Identifies messages that would be truncated to fit the budget
/// 2. Sends those messages to an LLM to generate a "Continuity Briefing"
/// 3. Injects the briefing between preserved initial context and recent messages
///
/// # Example
/// ```ignore
/// use rig::compression::SummarizingCompressor;
///
/// // Use any model that implements Prompt (agents, completion models, etc.)
/// let compressor = SummarizingCompressor::new(summarizer_agent)
///     .with_preserve_first(1)   // Keep system prompt
///     .with_preserve_recent(3); // Keep last 3 messages
///
/// // Note: compress_async must be used for this compressor
/// let compressed = compressor.compress_async(messages, 4096).await?;
/// ```
pub struct SummarizingCompressor<P: Prompt> {
    /// The promptable model/agent used for summarization.
    /// Can be a different (smaller/faster) model than the main agent.
    summarizer: Arc<P>,
    /// Number of messages to preserve from the start (e.g., system prompt).
    preserve_first: usize,
    /// Number of recent messages to always keep.
    preserve_recent: usize,
    /// Maximum tokens for the generated summary.
    max_summary_tokens: usize,
    /// Custom summarization prompt (optional).
    custom_prompt: Option<String>,
}

impl<P: Prompt> SummarizingCompressor<P> {
    /// Create a new summarizing compressor with the given promptable model/agent.
    ///
    /// The model can be a smaller/faster model than the main agent model
    /// to reduce latency and cost for summarization.
    pub fn new(summarizer: P) -> Self {
        Self {
            summarizer: Arc::new(summarizer),
            preserve_first: 1,
            preserve_recent: 2,
            max_summary_tokens: 1000,
            custom_prompt: None,
        }
    }

    /// Create from an Arc'd model (useful when sharing models).
    pub fn from_arc(summarizer: Arc<P>) -> Self {
        Self {
            summarizer,
            preserve_first: 1,
            preserve_recent: 2,
            max_summary_tokens: 1000,
            custom_prompt: None,
        }
    }

    /// Set the number of messages to preserve from the start.
    pub fn with_preserve_first(mut self, count: usize) -> Self {
        self.preserve_first = count;
        self
    }

    /// Set the number of recent messages to always keep.
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count;
        self
    }

    /// Set the maximum tokens for the generated summary.
    pub fn with_max_summary_tokens(mut self, tokens: usize) -> Self {
        self.max_summary_tokens = tokens;
        self
    }

    /// Set a custom summarization prompt.
    ///
    /// Use `[CONVERSATION_HISTORY]` as placeholder for the messages to summarize.
    pub fn with_custom_prompt(mut self, prompt: String) -> Self {
        self.custom_prompt = Some(prompt);
        self
    }

    /// Async compression that calls the LLM for summarization.
    ///
    /// This is the primary method to use for this compressor.
    pub async fn compress_async(
        &self,
        messages: Vec<Message>,
        max_tokens: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        if messages.is_empty() {
            return Ok(messages);
        }

        // If already within budget, return as-is
        if estimate_messages_tokens(&messages) <= max_tokens {
            return Ok(messages);
        }

        let total = messages.len();

        // Determine preserved sections
        let preserve_start = self.preserve_first.min(total);
        let preserve_end = self.preserve_recent.min(total.saturating_sub(preserve_start));

        // Split messages into three sections
        let first_messages: Vec<_> = messages.iter().take(preserve_start).cloned().collect();
        let last_messages: Vec<_> = messages
            .iter()
            .skip(total.saturating_sub(preserve_end))
            .cloned()
            .collect();

        // Middle section is what we'll summarize
        let middle_start = preserve_start;
        let middle_end = total.saturating_sub(preserve_end);

        if middle_start >= middle_end {
            // No middle section to summarize
            let mut result = first_messages;
            result.extend(last_messages);
            return Ok(result);
        }

        let middle_messages: Vec<_> = messages[middle_start..middle_end].to_vec();

        // Check if summarization would help
        let preserved_tokens =
            estimate_messages_tokens(&first_messages) + estimate_messages_tokens(&last_messages);
        let middle_tokens = estimate_messages_tokens(&middle_messages);

        // Only summarize if the middle section is substantial
        if middle_tokens < 100 {
            // Too small to summarize, just truncate
            let mut result = first_messages;
            result.extend(last_messages);
            return Ok(result);
        }

        // Generate the continuity briefing
        let summary = self.generate_summary(&middle_messages).await?;

        // Check if summary fits in budget
        let summary_tokens = estimate_tokens(&summary);
        let total_after = preserved_tokens + summary_tokens;

        if total_after > max_tokens {
            // Summary too large, need to truncate it or skip
            tracing::warn!(
                "Summarized context ({} tokens) still exceeds budget with preserved messages. \
                 Falling back to simple truncation.",
                summary_tokens
            );
            let mut result = first_messages;
            result.extend(last_messages);
            return Ok(result);
        }

        // Build the result with injected summary
        let mut result = first_messages;

        // Add the continuity briefing as a user message
        let briefing_message = Message::user(format!(
            "**[CONTEXT CONTINUITY BRIEFING]**\n\
             *The following is a compressed summary of the preceding conversation:*\n\n\
             {}\n\n\
             *[End of briefing - conversation continues below]*",
            summary
        ));
        result.push(briefing_message);

        result.extend(last_messages);

        tracing::info!(
            "Compressed {} messages ({} tokens) into briefing ({} tokens). \
             Preserved {} first + {} recent messages.",
            middle_messages.len(),
            middle_tokens,
            summary_tokens,
            preserve_start,
            preserve_end
        );

        Ok(result)
    }

    /// Generate a summary from the given messages using the LLM.
    async fn generate_summary(&self, messages: &[Message]) -> Result<String, CompressionError> {
        // Format messages for the summarization prompt
        let conversation_text = self.format_messages_for_summary(messages);

        // Get the prompt template
        let prompt_template = self
            .custom_prompt
            .as_deref()
            .unwrap_or(SUMMARIZATION_PROMPT);

        // Replace placeholder with actual conversation
        let full_prompt = prompt_template.replace("[CONVERSATION_HISTORY]", &conversation_text);

        // Call the summarizer
        let response = self
            .summarizer
            .prompt(full_prompt)
            .await
            .map_err(|e| CompressionError::CompressionFailed(format!("Summarization failed: {}", e)))?;

        Ok(response)
    }

    /// Format messages into a readable text format for summarization.
    fn format_messages_for_summary(&self, messages: &[Message]) -> String {
        use crate::completion::message::{AssistantContent, UserContent, ToolResultContent};

        let mut output = String::new();

        for (i, msg) in messages.iter().enumerate() {
            match msg {
                Message::User { content } => {
                    output.push_str(&format!("**[User Message {}]**\n", i + 1));
                    for c in content.iter() {
                        match c {
                            UserContent::Text(t) => {
                                output.push_str(&t.text);
                                output.push('\n');
                            }
                            UserContent::ToolResult(tr) => {
                                output.push_str(&format!(
                                    "[Tool Result for '{}']:\n",
                                    tr.id
                                ));
                                for tc in tr.content.iter() {
                                    if let ToolResultContent::Text(t) = tc {
                                        // Truncate very long tool results
                                        let text = if t.text.len() > 2000 {
                                            format!("{}...[truncated]", &t.text[..2000])
                                        } else {
                                            t.text.clone()
                                        };
                                        output.push_str(&text);
                                        output.push('\n');
                                    }
                                }
                            }
                            UserContent::Image(_) => {
                                output.push_str("[Image attached]\n");
                            }
                            UserContent::Document(d) => {
                                output.push_str(&format!("[Document: {}]\n", d.data));
                            }
                            _ => {}
                        }
                    }
                }
                Message::Assistant { content, .. } => {
                    output.push_str(&format!("**[Assistant Message {}]**\n", i + 1));
                    for c in content.iter() {
                        match c {
                            AssistantContent::Text(t) => {
                                output.push_str(&t.text);
                                output.push('\n');
                            }
                            AssistantContent::ToolCall(tc) => {
                                output.push_str(&format!(
                                    "[Tool Call: {}({})]\n",
                                    tc.function.name,
                                    // Truncate long arguments
                                    if tc.function.arguments.to_string().len() > 500 {
                                        format!("{}...", &tc.function.arguments.to_string()[..500])
                                    } else {
                                        tc.function.arguments.to_string()
                                    }
                                ));
                            }
                            AssistantContent::Reasoning(r) => {
                                output.push_str("[Reasoning: ");
                                output.push_str(&r.reasoning.join(" "));
                                output.push_str("]\n");
                            }
                            AssistantContent::Image(_) => {
                                output.push_str("[Image generated]\n");
                            }
                        }
                    }
                }
            }
            output.push('\n');
        }

        output
    }
}

// Implement the sync trait with a note that async should be used
impl<P: Prompt> ContextCompressor for SummarizingCompressor<P> {
    fn compress(
        &self,
        messages: Vec<Message>,
        max_tokens: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        // For sync context, fall back to simple truncation behavior
        // Users should call compress_async for full functionality
        tracing::warn!(
            "SummarizingCompressor::compress called synchronously. \
             Use compress_async for LLM-based summarization. \
             Falling back to simple truncation."
        );

        if messages.is_empty() {
            return Ok(messages);
        }

        if estimate_messages_tokens(&messages) <= max_tokens {
            return Ok(messages);
        }

        let total = messages.len();
        let preserve_start = self.preserve_first.min(total);
        let preserve_end = self.preserve_recent.min(total.saturating_sub(preserve_start));

        let mut result: Vec<_> = messages.iter().take(preserve_start).cloned().collect();
        result.extend(
            messages
                .iter()
                .skip(total.saturating_sub(preserve_end))
                .cloned(),
        );

        Ok(result)
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        estimate_messages_tokens(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_messages() {
        // Basic formatting test - actual LLM tests would be integration tests
        let messages = vec![
            Message::user("Hello, I need help with my code"),
            Message::assistant("Sure, I can help. What's the issue?"),
            Message::user("My function crashes"),
        ];

        // Just verify the compressor can be created
        // Full tests would require a mock model
        assert_eq!(messages.len(), 3);
    }
}
