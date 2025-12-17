//! Example of using Anthropic with OAuth authentication (Claude Code tokens).
//!
//! This example demonstrates how to use the OAuthClient which uses Bearer
//! authentication instead of x-api-key, required for Claude Code OAuth tokens.
//!
//! Set the CLAUDE_CODE_AUTH_TOKEN environment variable before running.

use rig::prelude::*;
use rig::{
    completion::Prompt,
    providers::anthropic::{self, completion::CLAUDE_3_5_SONNET},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Anthropic OAuth client from environment variable
    // Uses CLAUDE_CODE_AUTH_TOKEN instead of ANTHROPIC_API_KEY
    let client = anthropic::OAuthClient::from_env();

    // Or create manually with custom configuration:
    // let client = anthropic::OAuthClient::builder()
    //     .api_key("your-oauth-token")
    //     .anthropic_beta("interleaved-thinking-2025-05-14")  // Add more betas
    //     .user_agent("my-app/1.0.0")  // Custom User-Agent
    //     .build()
    //     .unwrap();

    // Create agent with a system prompt
    let agent = client
        .agent(CLAUDE_3_5_SONNET)
        .preamble("You are a helpful coding assistant. Be precise and concise.")
        .temperature(0.5)
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("What is the difference between &str and String in Rust?")
        .await?;

    println!("{response}");

    Ok(())
}
