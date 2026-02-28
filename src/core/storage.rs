use std::fs;
use std::path::PathBuf;

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
    let Some(ref log) = state.last_log else {
        return;
    };

    let dir = session_dir(&state.session.session_id);
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("rooster: could not create session dir {:?}: {e}", dir);
        return;
    }

    let path = dir.join("last.json");
    match serde_json::to_string_pretty(log) {
        Ok(data) => {
            if let Err(e) = fs::write(&path, data) {
                eprintln!("rooster: could not write last log to {:?}: {e}", path);
            }
        }
        Err(e) => eprintln!("rooster: could not serialize last log: {e}"),
    }
}
