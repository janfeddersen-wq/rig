//! Anthropic client api implementation
use http::{HeaderName, HeaderValue};

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel, OAuthCompletionModel};
use crate::{
    client::{
        self, ApiKey, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider,
        ProviderBuilder, ProviderClient,
    },
    http_client,
};

// ================================================================
// Main Anthropic Client
// ================================================================
#[derive(Debug, Default, Clone)]
pub struct AnthropicExt;

impl Provider for AnthropicExt {
    type Builder = AnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build<H>(
        _builder: &client::ClientBuilder<Self::Builder, AnthropicKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for AnthropicExt {
    type Completion = Capable<CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

#[derive(Debug, Clone)]
pub struct AnthropicBuilder {
    anthropic_version: String,
    anthropic_betas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AnthropicKey(String);

impl<S> From<S> for AnthropicKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl ApiKey for AnthropicKey {
    fn into_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(&self.0)
                .map(|val| (HeaderName::from_static("x-api-key"), val))
                .map_err(Into::into),
        )
    }
}

pub type Client<H = reqwest::Client> = client::Client<AnthropicExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<AnthropicBuilder, AnthropicKey, H>;

// ================================================================
// OAuth Client (for Claude Code tokens)
// ================================================================

/// OAuth extension for Anthropic, used with Claude Code OAuth tokens.
/// Uses Bearer authentication instead of x-api-key header.
#[derive(Debug, Default, Clone)]
pub struct AnthropicOAuthExt;

impl Provider for AnthropicOAuthExt {
    type Builder = AnthropicOAuthBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build<H>(
        _builder: &client::ClientBuilder<Self::Builder, BearerAuth, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for AnthropicOAuthExt {
    type Completion = Capable<OAuthCompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

/// Builder for OAuth-based Anthropic client.
/// Automatically configures required headers for Claude Code OAuth.
#[derive(Debug, Clone)]
pub struct AnthropicOAuthBuilder {
    anthropic_version: String,
    anthropic_betas: Vec<String>,
    user_agent: String,
    x_app: String,
}

impl Default for AnthropicOAuthBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            // OAuth requires specific beta flags
            anthropic_betas: vec!["oauth-2025-04-20".into()],
            user_agent: "claude-cli/2.0.61 (external, cli)".into(),
            x_app: "cli".into(),
        }
    }
}

impl ProviderBuilder for AnthropicOAuthBuilder {
    type Output = AnthropicOAuthExt;
    type ApiKey = BearerAuth;

    const BASE_URL: &'static str = "https://api.anthropic.com";

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, BearerAuth, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, BearerAuth, H>> {
        builder.headers_mut().insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version)?,
        );

        if !self.anthropic_betas.is_empty() {
            builder.headers_mut().insert(
                "anthropic-beta",
                HeaderValue::from_str(&self.anthropic_betas.join(","))?,
            );
        }

        // Set required headers for OAuth authentication
        builder
            .headers_mut()
            .insert("x-app", HeaderValue::from_str(&self.x_app)?);
        builder
            .headers_mut()
            .insert("user-agent", HeaderValue::from_str(&self.user_agent)?);

        Ok(builder)
    }
}

impl DebugExt for AnthropicOAuthExt {}

/// Client type for OAuth-based authentication (Claude Code tokens).
/// Uses `Authorization: Bearer <token>` instead of `x-api-key` header.
pub type OAuthClient<H = reqwest::Client> = client::Client<AnthropicOAuthExt, H>;

/// Builder for OAuth-based Anthropic client.
pub type OAuthClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<AnthropicOAuthBuilder, BearerAuth, H>;

impl ProviderClient for OAuthClient {
    type Input = String;

    fn from_env() -> Self
    where
        Self: Sized,
    {
        let token =
            std::env::var("CLAUDE_CODE_AUTH_TOKEN").expect("CLAUDE_CODE_AUTH_TOKEN not set");

        Self::builder().api_key(token).build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self
    where
        Self: Sized,
    {
        Self::builder().api_key(input).build().unwrap()
    }
}

/// Builder methods for OAuth client configuration
impl<H> OAuthClientBuilder<H> {
    /// Set the Anthropic API version
    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|ext| AnthropicOAuthBuilder {
            anthropic_version: anthropic_version.into(),
            ..ext
        })
    }

    /// Add multiple beta features
    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));
            ext
        })
    }

    /// Add a single beta feature
    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas.push(anthropic_beta.into());
            ext
        })
    }

    /// Set custom User-Agent header
    pub fn user_agent(self, user_agent: &str) -> Self {
        self.over_ext(|ext| AnthropicOAuthBuilder {
            user_agent: user_agent.into(),
            ..ext
        })
    }

    /// Set custom x-app header
    pub fn x_app(self, x_app: &str) -> Self {
        self.over_ext(|ext| AnthropicOAuthBuilder {
            x_app: x_app.into(),
            ..ext
        })
    }
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
        }
    }
}

impl ProviderBuilder for AnthropicBuilder {
    type Output = AnthropicExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = "https://api.anthropic.com";

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        builder.headers_mut().insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version)?,
        );

        if !self.anthropic_betas.is_empty() {
            builder.headers_mut().insert(
                "anthropic-beta",
                HeaderValue::from_str(&self.anthropic_betas.join(","))?,
            );
        }

        Ok(builder)
    }
}

impl DebugExt for AnthropicExt {}

impl ProviderClient for Client {
    type Input = String;

    fn from_env() -> Self
    where
        Self: Sized,
    {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

        Self::builder().api_key(key).build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self
    where
        Self: Sized,
    {
        Self::builder().api_key(input).build().unwrap()
    }
}

/// Create a new anthropic client using the builder
///
/// # Example
/// ```
/// use rig::providers::anthropic::{ClientBuilder, self};
///
/// // Initialize the Anthropic client
/// let anthropic_client = ClientBuilder::new("your-claude-api-key")
///    .anthropic_version(ANTHROPIC_VERSION_LATEST)
///    .anthropic_beta("prompt-caching-2024-07-31")
///    .build()
/// ```
impl<H> ClientBuilder<H> {
    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|ext| AnthropicBuilder {
            anthropic_version: anthropic_version.into(),
            ..ext
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));

            ext
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas.push(anthropic_beta.into());

            ext
        })
    }
}
