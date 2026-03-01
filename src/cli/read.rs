use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous};
use sqlx::Row;

use crate::core::session::SessionInfo;


// ── Connection ────────────────────────────────────────────────────────────────

pub struct Reader {
    pool: SqlitePool,
}

impl Reader {
    /// Open a read-only connection to the rooster database.
    ///
    /// Uses WAL mode so reads never block the server's concurrent writes.
    pub async fn connect() -> Result<Self, sqlx::Error> {
        Self::connect_at(default_db_path()).await
    }

    pub async fn connect_at(path: PathBuf) -> Result<Self, sqlx::Error> {
        let opts = SqliteConnectOptions::new()
            .filename(&path)
            .read_only(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(5))
            .pragma("foreign_keys", "ON");

        let pool = SqlitePool::connect_with(opts).await?;
        Ok(Self { pool })
    }

    // ── Session queries ───────────────────────────────────────────────────────

    /// Return all sessions in the database, newest first.
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>, sqlx::Error> {
        let rows = sqlx::query(
            "SELECT session_id, project_id, name, framework FROM sessions ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let sessions = rows
            .into_iter()
            .map(|r| {
                let framework_str: String = r.get("framework");
                SessionInfo {
                    session_id: r.get("session_id"),
                    project_id: r.get("project_id"),
                    name: r.get("name"),
                    framework: parse_framework(&framework_str),
                }
            })
            .collect();

        Ok(sessions)
    }

    /// Return the `SessionInfo` for a single session by ID.
    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionInfo>, sqlx::Error> {
        let row = sqlx::query(
            "SELECT session_id, project_id, name, framework FROM sessions WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| {
            let framework_str: String = r.get("framework");
            SessionInfo {
                session_id: r.get("session_id"),
                project_id: r.get("project_id"),
                name: r.get("name"),
                framework: parse_framework(&framework_str),
            }
        }))
    }

    // ── Metric table queries ──────────────────────────────────────────────────

    /// Return all log entries from a metric's table, ordered by step.
    ///
    /// Each entry is `(step, timestamp, data)` where `data` is the raw JSON
    /// string that was stored — the caller deserializes it as needed.
    pub async fn read_metric(
        &self,
        session_id: &str,
        metric_name: &str,
    ) -> Result<Vec<MetricRow>, sqlx::Error> {
        let table = sanitize_name(metric_name);

        let rows = sqlx::query(&format!(
            "SELECT step, timestamp, value FROM {table} \
             WHERE session_id = ? ORDER BY step ASC"
        ))
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let entries = rows
            .into_iter()
            .map(|r| MetricRow {
                step: r.get::<i64, _>("step") as u64,
                timestamp: r.get("timestamp"),
                value: r.get("value"),
            })
            .collect();

        Ok(entries)
    }

    /// Return the names of all metric tables that exist for a given session.
    ///
    /// These are the table names without the "metric_" prefix, so they match
    /// the names the user declared in `Define(metrics={...})`.
    pub async fn list_metrics(&self, session_id: &str) -> Result<Vec<String>, sqlx::Error> {
        // Find all tables whose name starts with "metric_" and contain a row
        // for this session_id.
        let rows = sqlx::query(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND name LIKE 'metric_%'",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut names = Vec::new();
        for row in rows {
            let table: String = row.get("name");
            // Check that at least one row belongs to this session.
            let count: i64 = sqlx::query_scalar(&format!(
                "SELECT COUNT(*) FROM {table} WHERE session_id = ?"
            ))
            .bind(session_id)
            .fetch_one(&self.pool)
            .await?;

            if count > 0 {
                // Strip the "metric_" prefix to give back the user-facing name.
                let user_name = table.trim_start_matches("metric_").to_string();
                names.push(user_name);
            }
        }

        Ok(names)
    }

    /// Return the latest N rows from a metric table (useful for live tail in the CLI).
    pub async fn tail_metric(
        &self,
        session_id: &str,
        metric_name: &str,
        n: u32,
    ) -> Result<Vec<MetricRow>, sqlx::Error> {
        let table = sanitize_name(metric_name);

        let rows = sqlx::query(&format!(
            "SELECT step, timestamp, value FROM {table} \
             WHERE session_id = ? ORDER BY step DESC LIMIT ?"
        ))
        .bind(session_id)
        .bind(n)
        .fetch_all(&self.pool)
        .await?;

        // Re-order ascending so callers always get oldest → newest.
        let mut entries: Vec<MetricRow> = rows
            .into_iter()
            .map(|r| MetricRow {
                step: r.get::<i64, _>("step") as u64,
                timestamp: r.get("timestamp"),
                value: r.get("value"),
            })
            .collect();

        entries.sort_by_key(|e| e.step);
        Ok(entries)
    }

    /// Read all metrics for a session as a map of metric_name → rows.
    pub async fn read_all_metrics(
        &self,
        session_id: &str,
    ) -> Result<HashMap<String, Vec<MetricRow>>, sqlx::Error> {
        let names = self.list_metrics(session_id).await?;
        let mut out = HashMap::new();
        for name in names {
            let rows = self.read_metric(session_id, &name).await?;
            out.insert(name, rows);
        }
        Ok(out)
    }
}


// ── Types ─────────────────────────────────────────────────────────────────────

/// One row from a per-metric table.
/// `value` is the raw JSON string of `Log.data` — parse with `serde_json` as needed.
pub struct MetricRow {
    pub step: u64,
    pub timestamp: String,
    pub value: String,   // JSON: {"loss": 0.4, ...}
}


// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".rooster").join("rooster.db")
}

fn sanitize_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    format!("metric_{s}")
}

fn parse_framework(s: &str) -> crate::core::session::Framework {
    match s {
        "Jax"        => crate::core::session::Framework::Jax,
        "Pytorch"    => crate::core::session::Framework::Pytorch,
        "Tensorflow" => crate::core::session::Framework::Tensorflow,
        other        => panic!("unknown framework in db: {other}"),
    }
}
