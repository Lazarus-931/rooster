use std::path::PathBuf;
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous};
use tracing::{debug, error, info};

use crate::core::session::{Log, SessionState};


const SESSIONS_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    project_id          TEXT NOT NULL,
    name                TEXT NOT NULL,
    framework           TEXT NOT NULL,
    project_name        TEXT NOT NULL,
    project_description TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
";

pub struct Db {
    pool: SqlitePool,
}

impl Db {

    pub async fn open() -> Result<Self, sqlx::Error> {
        Self::open_at(default_db_path()).await
    }

    pub async fn open_at(path: PathBuf) -> Result<Self, sqlx::Error> {
        info!(path = ?path, "opening rooster database");

        std::fs::create_dir_all(path.parent().unwrap()).ok();

        let opts = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            // WAL: server can write while the CLI reads — no "database is locked" errors
            .journal_mode(SqliteJournalMode::Wal)
            // NORMAL: data survives OS crashes; use FULL for power-loss safety
            .synchronous(SqliteSynchronous::Normal)
            // Wait up to 5s on lock contention before returning an error
            .busy_timeout(Duration::from_secs(5))
            // SQLite ignores FK constraints by default — enforce them per-connection
            .pragma("foreign_keys", "ON");

        let pool = SqlitePool::connect_with(opts).await?;
        sqlx::query(SESSIONS_SCHEMA).execute(&pool).await?;

        info!(path = ?path, "database ready");
        Ok(Self { pool })
    }

    /// Register a new session atomically:
    ///   1. Insert the session row.
    ///   2. CREATE TABLE for every declared metric.
    ///
    /// Both steps run inside a single transaction — either the whole session is
    /// visible to the CLI or none of it is. A partial registration is never stored.
    pub async fn register_session(&self, state: &SessionState) -> Result<(), sqlx::Error> {
        let sid = &state.session.session_id;
        let metric_names: Vec<&String> = state.metrics.keys().collect();
        info!(session_id = %sid, metrics = ?metric_names, "registering session");

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "INSERT OR IGNORE INTO sessions \
             (session_id, project_id, name, framework, project_name, project_description, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&state.session.session_id)
        .bind(&state.session.project_id)
        .bind(&state.session.name)
        .bind(format!("{:?}", state.session.framework))
        .bind(&state.project.name)
        .bind(&state.project.description)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            error!(session_id = %sid, error = %e, "failed to insert session row");
            e
        })?;

        for name in state.metrics.keys() {
            let table = sanitize_name(name);
            debug!(session_id = %sid, table = %table, "creating metric table");
            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {table} ( \
                    id         INTEGER PRIMARY KEY AUTOINCREMENT, \
                    session_id TEXT    NOT NULL REFERENCES sessions(session_id), \
                    step       INTEGER NOT NULL, \
                    timestamp  TEXT    NOT NULL, \
                    value      TEXT    NOT NULL \
                )"
            );
            sqlx::query(&sql).execute(&mut *tx).await.map_err(|e| {
                error!(session_id = %sid, table = %table, error = %e, "failed to create metric table");
                e
            })?;
        }

        tx.commit().await.map_err(|e| {
            error!(session_id = %sid, error = %e, "transaction commit failed");
            e
        })?;

        info!(session_id = %sid, "session registered");
        Ok(())
    }

    /// Append one log entry to the metric-specific table for `log.kind`.
    ///
    /// A single INSERT is already atomic in SQLite — no explicit transaction needed.
    /// `log.data` is stored as a JSON object so multi-key entries are preserved exactly.
    pub async fn insert_log(&self, session_id: &str, log: &Log) -> Result<(), sqlx::Error> {
        let table = sanitize_name(&log.kind);
        debug!(session_id = %session_id, kind = %log.kind, step = log.step, "writing log entry");

        let value = serde_json::to_string(&log.data).unwrap_or_else(|_| "{}".to_string());

        sqlx::query(&format!(
            "INSERT INTO {table} (session_id, step, timestamp, value) VALUES (?, ?, ?, ?)"
        ))
        .bind(session_id)
        .bind(log.step as i64)
        .bind(&log.timestamp)
        .bind(value)
        .execute(&self.pool)
        .await
        .map_err(|e| {
            error!(session_id = %session_id, kind = %log.kind, step = log.step, error = %e, "failed to write log entry");
            e
        })?;

        Ok(())
    }
}


// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".rooster").join("rooster.db")
}

/// Produce a safe SQLite table identifier from a metric name.
/// Any character that isn't alphanumeric or `_` is replaced with `_`.
/// Always prefixed with "metric_" so bare digits ("123") stay valid identifiers.
fn sanitize_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    format!("metric_{s}")
}
