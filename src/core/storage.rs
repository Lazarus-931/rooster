use std::fs;
use std::path::PathBuf;

use tracing::{error, info, warn};

use crate::core::session::SessionState;


fn session_dir(session_id: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".rooster")
        .join("sessions")
        .join(session_id)
}


// Called when a connection closes without a clean End message.
// Writes the last received Log entry to disk so the CLI can recover it.
pub fn persist_last_record(state: &SessionState) {
    let sid = &state.session.session_id;

    let Some(ref log) = state.last_log else {
        info!(session_id = %sid, "no log entries received — nothing to persist");
        return;
    };

    let dir = session_dir(sid);
    let path = dir.join("last.json");

    warn!(session_id = %sid, path = ?path, kind = %log.kind, step = log.step,
          "persisting last log after abrupt disconnect");

    if let Err(e) = fs::create_dir_all(&dir) {
        error!(session_id = %sid, path = ?dir, error = %e, "could not create session dir");
        return;
    }

    match serde_json::to_string_pretty(log) {
        Ok(data) => {
            if let Err(e) = fs::write(&path, data) {
                error!(session_id = %sid, path = ?path, error = %e, "could not write last log");
            } else {
                info!(session_id = %sid, path = ?path, "last log persisted");
            }
        }
        Err(e) => error!(session_id = %sid, error = %e, "could not serialize last log"),
    }
}
