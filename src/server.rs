use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Response,
    routing::get,
    Router,
};

use crate::core::session::{establish_connection, ClientMessage, MetricType, SessionState};
use crate::core::storage::persist_last_record;


pub fn router() -> Router {
    Router::new().route("/ws", get(ws_handler))
}


async fn ws_handler(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_socket)
}


async fn handle_socket(mut socket: WebSocket) {
    let mut state = match establish_connection(&mut socket).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rooster: session not established: {e}");
            return;
        }
    };

    loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => {
                if dispatch(&text, &mut state) {
                    break;
                }
            }
            Some(Ok(Message::Close(_))) => break,
            None | Some(Err(_)) => break,
            _ => {}
        }
    }

    if !state.ended_cleanly {
        persist_last_record(&state);
    }
}


/// Parse a raw text frame into a `ClientMessage` and act on it.
/// Returns `true` when the session should end (End message received).
fn dispatch(text: &str, state: &mut SessionState) -> bool {
    let msg = match serde_json::from_str::<ClientMessage>(text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("rooster: parse error: {e}");
            return false;
        }
    };

    match msg {
        ClientMessage::Log(payload) => {
            for entry in &payload.entries {
                // On first log for this metric, infer dtype from the first data value
                // and record it in the session's metrics map.
                if let Some(def) = state.metrics.get_mut(&entry.kind) {
                    if def.dtype.is_none() {
                        def.dtype = entry.data.values().next().map(MetricType::from);
                    }
                }
            }
            state.last_log = payload.entries.into_iter().last();
            false
        }
        ClientMessage::End(_) => {
            state.ended_cleanly = true;
            true
        }
        ClientMessage::Register(_) => {
            eprintln!("rooster: unexpected re-registration, ignoring");
            false
        }
    }
}
