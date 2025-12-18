//! Stream statistics for tracking bytes received during streaming.
//!
//! Provides a wrapper stream that counts bytes as they pass through,
//! allowing applications to monitor streaming progress and calculate
//! throughput metrics like tokens per second.

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;

use crate::http_client::Result;

// Global active counter - set this before starting a stream to automatically count bytes
static ACTIVE_COUNTER: OnceLock<Mutex<Option<StreamBytesCounter>>> = OnceLock::new();

fn get_counter_lock() -> &'static Mutex<Option<StreamBytesCounter>> {
    ACTIVE_COUNTER.get_or_init(|| Mutex::new(None))
}

/// Set the active byte counter. All streaming requests will use this counter
/// until it's cleared with `clear_active_counter()`.
pub fn set_active_counter(counter: StreamBytesCounter) {
    *get_counter_lock().lock().unwrap() = Some(counter);
}

/// Clear the active byte counter.
pub fn clear_active_counter() {
    *get_counter_lock().lock().unwrap() = None;
}

/// Get a clone of the active counter if one is set.
pub fn get_active_counter() -> Option<StreamBytesCounter> {
    get_counter_lock().lock().unwrap().clone()
}

/// Shared counter for tracking bytes received during streaming.
///
/// This can be cloned and shared across threads to monitor streaming progress.
/// Use `bytes_received()` to get the current count, and `reset()` to clear it.
#[derive(Debug, Clone, Default)]
pub struct StreamBytesCounter {
    inner: Arc<AtomicUsize>,
}

impl StreamBytesCounter {
    /// Create a new byte counter initialized to zero.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get the total bytes received since creation or last reset.
    pub fn bytes_received(&self) -> usize {
        self.inner.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero and return the previous value.
    pub fn reset(&self) -> usize {
        self.inner.swap(0, Ordering::Relaxed)
    }

    /// Add bytes to the counter.
    pub(crate) fn add(&self, bytes: usize) {
        self.inner.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get the inner Arc for sharing.
    pub fn inner(&self) -> Arc<AtomicUsize> {
        self.inner.clone()
    }
}

pin_project! {
    /// A stream wrapper that counts bytes as they pass through.
    ///
    /// Wraps any stream of `Result<Bytes>` and increments a shared counter
    /// for each chunk received. This allows monitoring streaming progress
    /// without modifying the underlying stream behavior.
    pub struct ByteCountingStream<S> {
        #[pin]
        inner: S,
        counter: Arc<AtomicUsize>,
    }
}

impl<S> ByteCountingStream<S> {
    /// Wrap a stream with byte counting.
    pub fn new(inner: S, counter: StreamBytesCounter) -> Self {
        Self {
            inner,
            counter: counter.inner(),
        }
    }

    /// Wrap a stream with a raw atomic counter.
    pub fn with_counter(inner: S, counter: Arc<AtomicUsize>) -> Self {
        Self { inner, counter }
    }
}

impl<S> Stream for ByteCountingStream<S>
where
    S: Stream<Item = Result<Bytes>>,
{
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        match this.inner.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                // Count the bytes as they pass through
                let len = bytes.len();
                let total = this.counter.fetch_add(len, Ordering::Relaxed) + len;
                tracing::trace!("ByteCountingStream: received {} bytes, total now {}", len, total);
                Poll::Ready(Some(Ok(bytes)))
            }
            other => other,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_byte_counting() {
        let counter = StreamBytesCounter::new();
        let data = vec![
            Ok(Bytes::from("hello")),
            Ok(Bytes::from(" ")),
            Ok(Bytes::from("world")),
        ];
        let inner_stream = stream::iter(data);
        let mut counting_stream = ByteCountingStream::new(inner_stream, counter.clone());

        // Consume the stream
        while let Some(_) = counting_stream.next().await {}

        // Check the count
        assert_eq!(counter.bytes_received(), 11); // "hello world" = 11 bytes
    }

    #[tokio::test]
    async fn test_reset() {
        let counter = StreamBytesCounter::new();
        counter.add(100);
        assert_eq!(counter.bytes_received(), 100);

        let old = counter.reset();
        assert_eq!(old, 100);
        assert_eq!(counter.bytes_received(), 0);
    }
}
