use std::sync::Arc;

use axum::{
    extract::{ws::{Message, WebSocket, WebSocketUpgrade}, State},
    response::Response,
    routing::get,
    Router,
};
use tracing::{debug, info, warn};

use crate::core::db::Db;
use crate::core::session::{establish_connection, ClientMessage, MetricType, SessionState};
use crate::core::storage::persist_last_record;


pub fn router(db: Arc<Db>) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .with_state(db)
}


async fn ws_handler(State(db): State<Arc<Db>>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, db))
}


async fn handle_socket(mut socket: WebSocket, db: Arc<Db>) {
    info!("websocket connection accepted");

    let mut state = match establish_connection(&mut socket).await {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "session not established, closing connection");
            return;
        }
    };

    info!(
        session_id = %state.session.session_id,
        session_name = %state.session.name,
        "session established"
    );

    // Register the session and create per-metric tables.
    // Non-fatal — server continues even if DB is unavailable.
    if let Err(e) = db.register_session(&state).await {
        warn!(
            session_id = %state.session.session_id,
            error = %e,
            "could not register session in DB"
        );
    }

    loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => {
                if dispatch(&text, &mut state, &db).await {
                    break;
                }
            }
            Some(Ok(Message::Close(_))) => {
                info!(session_id = %state.session.session_id, "socket closed by client");
                break;
            }
            None | Some(Err(_)) => {
                warn!(session_id = %state.session.session_id, "socket dropped without End message");
                break;
            }
            _ => {}
        }
    }

    if !state.ended_cleanly {
        warn!(
            session_id = %state.session.session_id,
            "session did not end cleanly — persisting last record"
        );
        persist_last_record(&state);
    } else {
        info!(session_id = %state.session.session_id, "session complete");
    }
}


/// Parse a raw text frame into a `ClientMessage` and act on it.
/// Returns `true` when the session should end (End message received).
async fn dispatch(text: &str, state: &mut SessionState, db: &Db) -> bool {
    let sid = &state.session.session_id;

    let msg = match serde_json::from_str::<ClientMessage>(text) {
        Ok(m) => m,
        Err(e) => {
            warn!(session_id = %sid, error = %e, "failed to parse message");
            return false;
        }
    };

    match msg {
        ClientMessage::Log(payload) => {
            for entry in &payload.entries {
                debug!(
                    session_id = %sid,
                    kind = %entry.kind,
                    step = entry.step,
                    "log entry received"
                );

                // On first log for this metric, infer dtype from the first data value.
                if let Some(def) = state.metrics.get_mut(&entry.kind) {
                    if def.dtype.is_none() {
                        def.dtype = entry.data.values().next().map(MetricType::from);
                        debug!(
                            session_id = %sid,
                            kind = %entry.kind,
                            dtype = ?def.dtype,
                            "dtype inferred for metric"
                        );
                    }
                }

                // Persist to DB. Non-fatal — warn and continue if it fails.
                if let Err(e) = db.insert_log(sid, entry).await {
                    warn!(
                        session_id = %sid,
                        kind = %entry.kind,
                        step = entry.step,
                        error = %e,
                        "could not write log entry to DB"
                    );
                }
            }

            state.last_log = payload.entries.into_iter().last();
            false
        }
        ClientMessage::End(_) => {
            info!(session_id = %sid, "end message received");
            state.ended_cleanly = true;
            true
        }
        ClientMessage::Register(_) => {
            warn!(session_id = %sid, "unexpected re-registration ignored");
            false
        }
    }
}
