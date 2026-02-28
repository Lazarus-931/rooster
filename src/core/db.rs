use std::path::PathBuf;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};

use crate::core::session::{Log, ProjectInfo, SessionInfo, SessionState};


// Only the fixed sessions table lives here.
// Per-metric tables are created dynamically in `create_tables`.
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
        std::fs::create_dir_all(path.parent().unwrap()).ok();

        let opts = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true);

        let pool = SqlitePool::connect_with(opts).await?;
        sqlx::query(SESSIONS_SCHEMA).execute(&pool).await?;

        Ok(Self { pool })
    }

    /// For each metric declared in the session, create a dedicated table:
    ///   metric_{name}(id, session_id, step, timestamp, value TEXT)
    ///
    /// `value` stores the JSON-serialized `data` dict from the Log entry.
    /// Called once after `insert_session`, so the schema is ready before any logs arrive.
    pub async fn create_tables(&self, state: &SessionState) -> Result<(), sqlx::Error> {
        for name in state.metrics.keys() {
            let table = sanitize_name(name);
            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {table} (\
                    id         INTEGER PRIMARY KEY AUTOINCREMENT, \
                    session_id TEXT    NOT NULL, \
                    step       INTEGER NOT NULL, \
                    timestamp  TEXT    NOT NULL, \
                    value      TEXT    NOT NULL, \
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)\
                )"
            );
            sqlx::query(&sql).execute(&self.pool).await?;
        }
        Ok(())
    }

    pub async fn insert_session(&self, session: &SessionInfo, project: &ProjectInfo) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR IGNORE INTO sessions \
             (session_id, project_id, name, framework, project_name, project_description, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, datetime('now'))"
        )
            .bind(&session.session_id)
            .bind(&session.project_id)
            .bind(&session.name)
            .bind(format!("{:?}", session.framework))
            .bind(&project.name)
            .bind(&project.description)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Insert a log entry into the per-metric table for `log.kind`.
    /// `log.data` is JSON-serialized as the `value` column.
    pub async fn insert_log(&self, session_id: &str, log: &Log) -> Result<(), sqlx::Error> {
        let table = sanitize_name(&log.kind);
        let value = serde_json::to_string(&log.data).unwrap_or_else(|_| "{}".to_string());

        sqlx::query(&format!(
            "INSERT INTO {table} (session_id, step, timestamp, value) VALUES (?, ?, ?, ?)"
        ))
        .bind(session_id)
        .bind(log.step as i64)
        .bind(&log.timestamp)
        .bind(value)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}


// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".rooster").join("rooster.db")
}

/// Produce a safe SQLite table identifier from a metric name.
/// Non-alphanumeric/underscore chars are replaced with '_'.
/// Always prefixed with "metric_" so names like "123" remain valid.
fn sanitize_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    format!("metric_{s}")
}
