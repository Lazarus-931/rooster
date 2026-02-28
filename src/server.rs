use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Response,
    routing::get,
    Router,
};

use crate::core::session::{establish_connection, ClientMessage};
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
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Metrics(payload)) => {
                        state.last_record = payload.records.into_iter().last();
                    }
                    Ok(ClientMessage::End(_)) => {
                        state.ended_cleanly = true;
                        break;
                    }
                    Ok(ClientMessage::Register(_)) => {
                        eprintln!("rooster: unexpected re-registration, ignoring");
                    }
                    Err(e) => {
                        eprintln!("rooster: parse error: {e}");
                    }
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
