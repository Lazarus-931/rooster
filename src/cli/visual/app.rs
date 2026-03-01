use ratatui::crossterm::event::{Event, KeyCode, KeyEventKind};
use ratatui::widgets::ListState;

use crate::cli::read::{MetricRow, Reader};
use crate::core::session::SessionInfo;


// ── Focus ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum Focus {
    Sessions,
    Metrics,
}


// ── App state ─────────────────────────────────────────────────────────────────

pub struct App {
    // Session panel
    pub sessions:      Vec<SessionInfo>,
    pub session_state: ListState,

    // Metric panel
    pub metrics:      Vec<String>,
    pub metric_state: ListState,

    // Chart data for the selected (session, metric) pair
    // Each point is (step as f64, value as f64)
    pub chart_data: Vec<(f64, f64)>,

    pub focus:      Focus,
    pub status_msg: String,
}

impl App {
    pub fn new() -> Self {
        Self {
            sessions:      Vec::new(),
            session_state: ListState::default(),
            metrics:       Vec::new(),
            metric_state:  ListState::default(),
            chart_data:    Vec::new(),
            focus:         Focus::Sessions,
            status_msg:    "connecting…".to_string(),
        }
    }

    pub fn selected_session_id(&self) -> Option<String> {
        self.session_state
            .selected()
            .and_then(|i| self.sessions.get(i))
            .map(|s| s.session_id.clone())
    }

    pub fn selected_metric_name(&self) -> Option<String> {
        self.metric_state
            .selected()
            .and_then(|i| self.metrics.get(i))
            .cloned()
    }

    /// Pull fresh data from the DB for the currently selected session + metric.
    /// Auto-selects the first session/metric if nothing is selected yet.
    pub async fn refresh(&mut self, reader: &Reader) -> Result<(), sqlx::Error> {
        // ── Sessions ──────────────────────────────────────────────────────────
        let sessions = reader.list_sessions().await?;
        if self.session_state.selected().is_none() && !sessions.is_empty() {
            self.session_state.select(Some(0));
        }
        self.sessions = sessions;

        // ── Metrics for selected session ──────────────────────────────────────
        if let Some(sid) = self.selected_session_id() {
            let metrics = reader.list_metrics(&sid).await?;
            if self.metric_state.selected().is_none() && !metrics.is_empty() {
                self.metric_state.select(Some(0));
            }
            self.metrics = metrics;

            // ── Chart data for selected metric ────────────────────────────────
            if let Some(metric) = self.selected_metric_name() {
                let rows = reader.read_metric(&sid, &metric).await?;
                self.chart_data = parse_chart_data(&rows);
            }
        }

        self.status_msg = "live".to_string();
        Ok(())
    }

    /// Handle a terminal event. Returns `true` if the app should quit.
    pub fn handle_event(&mut self, event: Event) -> bool {
        let Event::Key(key) = event else { return false };
        if key.kind != KeyEventKind::Press { return false; }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Sessions => Focus::Metrics,
                    Focus::Metrics  => Focus::Sessions,
                };
            }
            KeyCode::Up   => self.navigate(-1),
            KeyCode::Down => self.navigate(1),
            _ => {}
        }
        false
    }

    fn navigate(&mut self, delta: i32) {
        match self.focus {
            Focus::Sessions => {
                let n = self.sessions.len();
                if n == 0 { return; }
                let cur = self.session_state.selected().unwrap_or(0) as i32;
                self.session_state.select(Some((cur + delta).rem_euclid(n as i32) as usize));
                // Reset child selections so chart reloads on next refresh
                self.metric_state = ListState::default();
                self.chart_data.clear();
            }
            Focus::Metrics => {
                let n = self.metrics.len();
                if n == 0 { return; }
                let cur = self.metric_state.selected().unwrap_or(0) as i32;
                self.metric_state.select(Some((cur + delta).rem_euclid(n as i32) as usize));
                self.chart_data.clear(); // reloaded on next refresh tick
            }
        }
    }
}


// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract (step, value) pairs from raw DB rows.
///
/// `MetricRow.value` is a JSON object like `{"loss": 0.4}`.
/// We take the first numeric value in the object.
pub fn parse_chart_data(rows: &[MetricRow]) -> Vec<(f64, f64)> {
    rows.iter()
        .filter_map(|row| {
            let v: serde_json::Value = serde_json::from_str(&row.value).ok()?;
            let num = match &v {
                serde_json::Value::Object(map) => map.values().next()?.as_f64()?,
                serde_json::Value::Number(n)   => n.as_f64()?,
                _                              => return None,
            };
            Some((row.step as f64, num))
        })
        .collect()
}
