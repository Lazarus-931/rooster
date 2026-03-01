use std::collections::HashMap;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info, warn};





#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}



#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]   // matches Python's _infer_dtype strings
pub enum MetricType {
    Bool,
    Int,
    Float,
    Str,
}


// ── Metric definition ─────────────────────────────────────────────────────────
//
// dtype is absent from the registration payload (user doesn't declare it).
// It is inferred from the first log entry that arrives for this metric and
// filled in then, so the session always reflects the true wire type.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDef {
    #[serde(default)]
    pub dtype: Option<MetricType>,
    pub rate: u32,
}

impl From<&MetricValue> for MetricType {
    fn from(v: &MetricValue) -> Self {
        match v {
            MetricValue::Bool(_)  => MetricType::Bool,
            MetricValue::Int(_)   => MetricType::Int,
            MetricValue::Float(_) => MetricType::Float,
            MetricValue::Str(_)   => MetricType::Str,
        }
    }
}


// ── Log — general purpose entry ───────────────────────────────────────────────
//
// kind  : user-defined label ("loss", "gradient", "custom", anything)
// step  : training step index
// data  : any key → any scalar value the user wants recorded
//
// Wire format:
//   { "kind": "loss", "step": 5, "timestamp": "...", "data": { "loss": 0.4, ... } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Log {
    pub kind: String,
    pub step: u64,
    pub timestamp: String,
    pub dtype: String,                      // inferred by Python parser, e.g. "float"
    pub data: HashMap<String, MetricValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Framework {
    Jax,
    Pytorch,
    Tensorflow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub project_id: String,
    pub name: String,
    pub framework: Framework,
}

#[derive(Debug, Deserialize)]
pub struct RegisterPayload {
    pub project: ProjectInfo,
    pub session: SessionInfo,
    pub metrics: HashMap<String, MetricDef>,
}

#[derive(Debug, Deserialize)]
pub struct LogPayload {
    /// Present in every Log wire frame; the active session is tracked via
    /// `SessionState` so this field is deserialized but not accessed in code.
    #[allow(dead_code)]
    pub session_id: String,
    pub entries: Vec<Log>,
}

#[derive(Debug, Deserialize)]
pub struct EndPayload {
    /// Deserialized from the wire frame but not accessed — session identity is
    /// carried by `SessionState`.
    #[allow(dead_code)]
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClientMessage {
    Register(RegisterPayload),
    Log(LogPayload),
    /// Inner payload deserialized from the wire but not accessed in code —
    /// receiving this variant alone is sufficient to mark the session complete.
    #[allow(dead_code)]
    End(EndPayload),
}





pub struct SessionState {
    pub session: SessionInfo,
    pub project: ProjectInfo,
    /// Metric schema: name → MetricDef.
    /// dtype starts as None and is filled in from the first log entry for each metric.
    pub metrics: HashMap<String, MetricDef>,
    pub last_log: Option<Log>,
    pub ended_cleanly: bool,
}

impl SessionState {
    pub fn new(session: SessionInfo, project: ProjectInfo, metrics: HashMap<String, MetricDef>) -> Self {
        Self {
            session,
            project,
            metrics,
            last_log: None,
            ended_cleanly: false,
        }
    }
}

/// Wait for the client's opening `Register` frame and return a live `SessionState`.
///
/// Fails if:
///   - no message arrives within `REGISTER_TIMEOUT`
///   - the socket closes before Register
///   - the first frame is not a valid Register message
pub async fn establish_connection(
    socket: &mut WebSocket,
) -> Result<SessionState, Box<dyn std::error::Error + Send + Sync>> {
    const REGISTER_TIMEOUT: Duration = Duration::from_secs(10);

    let frame = timeout(REGISTER_TIMEOUT, socket.recv())
        .await
        .map_err(|_| {
            warn!("timed out waiting for Register message (10s)");
            "timed out waiting for Register message"
        })?
        .ok_or("socket closed before Register message")?
        .map_err(|e| format!("WebSocket error: {e}"))?;

    let text = match frame {
        Message::Text(t) => t,
        other => {
            warn!(frame = ?other, "expected text frame for Register, got something else");
            return Err(format!("expected text frame, got {:?}", other).into());
        }
    };

    match serde_json::from_str::<ClientMessage>(&text)
        .map_err(|e| format!("invalid Register payload: {e}"))?
    {
        ClientMessage::Register(p) => {
            info!(
                session_id = %p.session.session_id,
                session_name = %p.session.name,
                metrics = ?p.metrics.keys().collect::<Vec<_>>(),
                "Register received"
            );
            Ok(SessionState::new(p.session, p.project, p.metrics))
        }
        other => {
            warn!(msg = ?other, "first message must be Register");
            Err(format!("first message must be Register, got {:?}", other).into())
        }
    }
}


