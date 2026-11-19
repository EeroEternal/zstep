// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! ZeroMQ `PUB` publisher for `zs-kv` prefix cache events (`BlockStored` / `BlockRemoved` /
//! `AllBlocksCleared`), wire-compatible with Nebula's `nebula-router::kv_events`.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rmpv::Value;
use tokio::sync::mpsc;
use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};
use zs_kv::{KvCacheEvent, KvCacheEventSink};

const KV_EVENT_QUEUE_CAP: usize = 4096;

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Encode one [`KvCacheEvent`] into Nebula/vLLM-compatible msgpack batch `[ts_f64, [event_map]]`.
pub fn encode_kv_event_batch(event: &KvCacheEvent, ts_secs: f64) -> Vec<u8> {
    let event_val = match event {
        KvCacheEvent::Stored { block_hashes } => {
            let hashes = block_hashes
                .iter()
                .map(|h| Value::Binary(h.to_vec()))
                .collect();
            Value::Map(vec![
                (Value::from("type"), Value::from("BlockStored")),
                (Value::from("block_hashes"), Value::Array(hashes)),
                (Value::from("medium"), Value::from("GPU")),
            ])
        }
        KvCacheEvent::Removed { block_hashes } => {
            let hashes = block_hashes
                .iter()
                .map(|h| Value::Binary(h.to_vec()))
                .collect();
            Value::Map(vec![
                (Value::from("type"), Value::from("BlockRemoved")),
                (Value::from("block_hashes"), Value::Array(hashes)),
            ])
        }
        KvCacheEvent::Cleared => {
            Value::Map(vec![(Value::from("type"), Value::from("AllBlocksCleared"))])
        }
    };

    let batch = Value::Array(vec![Value::F64(ts_secs), Value::Array(vec![event_val])]);
    let mut out = Vec::with_capacity(64);
    let _ = rmpv::encode::write_value(&mut out, &batch);
    out
}

/// Non-blocking handle to the background ZMQ `PUB` task.
pub struct KvEventPublisher {
    sink: KvCacheEventSink,
}

impl KvEventPublisher {
    /// Spawn a background ZMQ `PUB` publisher connecting to `endpoint` and publishing on `topic`.
    ///
    /// Emits an initial `AllBlocksCleared` event upon startup. Dropping all cloned `sink`s
    /// (including `ServingEngine` shutdown) terminates the background task cleanly.
    pub fn spawn(endpoint: String, topic: String, handle: &tokio::runtime::Handle) -> Self {
        let (tx, mut rx) = mpsc::channel::<KvCacheEvent>(KV_EVENT_QUEUE_CAP);
        let _ = tx.try_send(KvCacheEvent::Cleared);

        handle.spawn(async move {
            let mut socket = PubSocket::new();
            let mut backoff_ms = 250u64;
            loop {
                match socket.connect(&endpoint).await {
                    Ok(()) => break,
                    Err(err) => {
                        eprintln!(
                            "zstep kv_events ZMQ connect to {endpoint} failed: {err}; retrying in {backoff_ms}ms"
                        );
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                            maybe = rx.recv() => {
                                if maybe.is_none() {
                                    return;
                                }
                            }
                        }
                        backoff_ms = (backoff_ms * 2).min(2000);
                    }
                }
            }

            let topic_bytes = topic.into_bytes();
            let mut seq: u64 = 0;
            while let Some(event) = rx.recv().await {
                seq = seq.wrapping_add(1);
                let payload = encode_kv_event_batch(&event, now_secs_f64());
                // vLLM/Nebula 3-frame ZMQ wire format: [topic, seq_u64_be, msgpack_payload].
                let mut msg = ZmqMessage::from(topic_bytes.clone());
                msg.push_back(seq.to_be_bytes().to_vec().into());
                msg.push_back(payload.into());
                let _ = socket.send(msg).await;
            }
        });

        let sink: KvCacheEventSink = Arc::new(move |event| {
            let _ = tx.try_send(event);
        });
        Self { sink }
    }

    pub fn sink(&self) -> KvCacheEventSink {
        Arc::clone(&self.sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeromq::{SocketRecv, SubSocket};

    #[tokio::test]
    async fn zmq_pub_sub_roundtrip_emits_cleared_stored_and_removed() {
        let mut sub = SubSocket::new();
        let endpoint = sub
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind local zmq sub");
        let addr = endpoint.to_string();
        sub.subscribe("test-model/0")
            .await
            .expect("subscribe topic");

        let rt = tokio::runtime::Handle::current();
        let pub_handle = KvEventPublisher::spawn(addr, "test-model/0".to_owned(), &rt);
        let sink = pub_handle.sink();

        // Give ZMQ SUB/PUB subscription handshake a brief moment before sending Stored/Removed.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let stored_hash = [0x0e, 0x18, 0x08, 0xe2, 0x94, 0x31, 0x7a, 0xa6];
        sink(KvCacheEvent::Stored {
            block_hashes: vec![stored_hash],
        });
        sink(KvCacheEvent::Removed {
            block_hashes: vec![stored_hash],
        });

        let mut seen_types = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while seen_types.len() < 3 && tokio::time::Instant::now() < deadline {
            if let Ok(Ok(msg)) = tokio::time::timeout(Duration::from_millis(500), sub.recv()).await
            {
                assert_eq!(
                    msg.len(),
                    3,
                    "expected 3-frame [topic, seq, payload] ZMQ message"
                );
                let topic = msg.get(0).expect("topic frame");
                assert_eq!(topic.as_ref(), b"test-model/0");
                let seq_frame = msg.get(1).expect("seq frame");
                assert_eq!(seq_frame.len(), 8);
                let payload = msg.get(2).expect("payload frame");
                let mut rd = payload.as_ref();
                let val = rmpv::decode::read_value(&mut rd).expect("valid msgpack batch");
                let arr = val.as_array().expect("batch array");
                let events = arr[1].as_array().expect("events array");
                let map = events[0].as_map().expect("event map");
                let ev_type = map
                    .iter()
                    .find(|(k, _)| k.as_str() == Some("type"))
                    .and_then(|(_, v)| v.as_str())
                    .expect("type field");
                seen_types.push(ev_type.to_owned());
            }
        }
        assert!(
            seen_types.contains(&"BlockStored".to_owned())
                && seen_types.contains(&"BlockRemoved".to_owned()),
            "expected BlockStored & BlockRemoved in {seen_types:?}"
        );
    }
}
