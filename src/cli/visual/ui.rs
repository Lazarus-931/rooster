use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, List, ListItem, Paragraph},
    Frame,
};

use super::app::{App, Focus};


// ── Top-level layout ──────────────────────────────────────────────────────────
//
//  ┌─────────────────────────────────────────────────────────────┐
//  │  title bar (1 line)                                         │
//  ├──────────────┬──────────────────────────────────────────────┤
//  │  Sessions    │                                              │
//  │  (list)      │        line chart                           │
//  ├──────────────┤                                              │
//  │  Metrics     │                                              │
//  │  (list)      │                                              │
//  ├──────────────┴──────────────────────────────────────────────┤
//  │  status / keybinds bar (1 line)                             │
//  └─────────────────────────────────────────────────────────────┘

pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    let rows = Layout::vertical([
        Constraint::Length(1),   // title
        Constraint::Min(0),      // body
        Constraint::Length(1),   // status
    ])
    .split(area);

    render_title(frame, app, rows[0]);
    render_body(frame, app, rows[1]);
    render_status(frame, app, rows[2]);
}


// ── Title bar ─────────────────────────────────────────────────────────────────

fn render_title(frame: &mut Frame, app: &App, area: Rect) {
    let session_label = app
        .session_state
        .selected()
        .and_then(|i| app.sessions.get(i))
        .map(|s| format!("  {}  •  {:?}", s.name, s.framework))
        .unwrap_or_default();

    let line = Line::from(vec![
        Span::raw(" rooster").bold().fg(Color::Yellow),
        Span::raw(session_label).fg(Color::Gray),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}


// ── Body ──────────────────────────────────────────────────────────────────────

fn render_body(frame: &mut Frame, app: &mut App, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Length(26),  // left: sessions + metrics
        Constraint::Min(0),      // right: chart
    ])
    .split(area);

    render_left(frame, app, cols[0]);
    render_chart(frame, app, cols[1]);
}


// ── Left panel ────────────────────────────────────────────────────────────────

fn render_left(frame: &mut Frame, app: &mut App, area: Rect) {
    let rows = Layout::vertical([
        Constraint::Percentage(40),
        Constraint::Percentage(60),
    ])
    .split(area);

    render_sessions(frame, app, rows[0]);
    render_metrics(frame, app, rows[1]);
}

fn render_sessions(frame: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.focus == Focus::Sessions;
    let border  = if focused { Color::Yellow } else { Color::DarkGray };

    let items: Vec<ListItem> = app.sessions.iter()
        .map(|s| ListItem::new(s.name.clone()))
        .collect();

    let list = List::new(items)
        .block(
            Block::bordered()
                .title("Sessions")
                .border_style(Style::new().fg(border)),
        )
        .highlight_style(Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol("> ");

    frame.render_stateful_widget(list, area, &mut app.session_state);
}

fn render_metrics(frame: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.focus == Focus::Metrics;
    let border  = if focused { Color::Cyan } else { Color::DarkGray };

    let items: Vec<ListItem> = app.metrics.iter()
        .map(|m| ListItem::new(m.clone()))
        .collect();

    let list = List::new(items)
        .block(
            Block::bordered()
                .title("Metrics")
                .border_style(Style::new().fg(border)),
        )
        .highlight_style(Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .highlight_symbol("> ");

    frame.render_stateful_widget(list, area, &mut app.metric_state);
}


// ── Chart ─────────────────────────────────────────────────────────────────────

fn render_chart(frame: &mut Frame, app: &App, area: Rect) {
    let title = app
        .metric_state
        .selected()
        .and_then(|i| app.metrics.get(i))
        .cloned()
        .unwrap_or_else(|| "—".to_string());

    // Nothing to show yet
    if app.chart_data.is_empty() {
        let placeholder = Paragraph::new("No data yet — waiting for logs…")
            .fg(Color::DarkGray)
            .block(
                Block::bordered()
                    .title(title)
                    .border_style(Style::new().fg(Color::DarkGray)),
            );
        frame.render_widget(placeholder, area);
        return;
    }

    let (x_min, x_max, y_min, y_max) = axis_bounds(&app.chart_data);

    let dataset = Dataset::default()
        .name(title.clone())
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(Color::Yellow))
        .data(&app.chart_data);

    let chart = Chart::new(vec![dataset])
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .title("step")
                .bounds([x_min, x_max])
                .labels(vec![
                    Span::raw(format!("{x_min:.0}")),
                    Span::raw(format!("{x_max:.0}")),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("value")
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{y_min:.4}")),
                    Span::raw(format!("{y_max:.4}")),
                ]),
        );

    frame.render_widget(chart, area);
}


// ── Status bar ────────────────────────────────────────────────────────────────

fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let line = Line::from(vec![
        Span::raw(" ↑↓").bold().fg(Color::Yellow),
        Span::raw(" navigate  ").fg(Color::Gray),
        Span::raw("Tab").bold().fg(Color::Yellow),
        Span::raw(" switch panel  ").fg(Color::Gray),
        Span::raw("q").bold().fg(Color::Yellow),
        Span::raw(" quit").fg(Color::Gray),
        Span::raw("  •  ").fg(Color::DarkGray),
        Span::raw(app.status_msg.clone()).fg(Color::Green).italic(),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}


// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute axis bounds with a small y margin so the line never clips the border.
fn axis_bounds(data: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let x_min = data.first().map(|p| p.0).unwrap_or(0.0);
    let x_max = data.last().map(|p| p.0).unwrap_or(1.0);
    let y_min = data.iter().fold(f64::INFINITY,     |a, p| a.min(p.1));
    let y_max = data.iter().fold(f64::NEG_INFINITY, |a, p| a.max(p.1));

    // 5% margin; floor at 0.1 for flat lines
    let margin = ((y_max - y_min).abs() * 0.05).max(0.1);
    (x_min, x_max, y_min - margin, y_max + margin)
}
