mod app;
mod ui;

pub use app::App;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::Event;

use crate::cli::read::Reader;


/// Run the terminal dashboard.
///
/// Opens the DB at `db_path` (or the default `~/.rooster/rooster.db`),
/// then enters a full-screen TUI that live-updates as the server writes.
pub async fn run(db_path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let reader = match db_path {
        Some(p) => Reader::connect_at(p).await?,
        None    => Reader::connect().await?,
    };

    let mut app = App::new();
    app.refresh(&reader).await.ok();   // best-effort — DB may be empty on first launch

    // Dedicate a thread to reading terminal events.
    // `event::read()` blocks until an event arrives so it must not run on the async runtime.
    let (tx, rx) = mpsc::channel::<Event>();
    std::thread::spawn(move || {
        use ratatui::crossterm::event;
        loop {
            match event::read() {
                Ok(ev) => { if tx.send(ev).is_err() { break; } }
                Err(_) => break,
            }
        }
    });

    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut app, &reader, rx).await;
    ratatui::restore();
    result
}


async fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app:      &mut App,
    reader:   &Reader,
    events:   mpsc::Receiver<Event>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_refresh = Instant::now();

    loop {
        // Drain all pending key/mouse events — keeps input feeling instant
        let mut quit = false;
        while let Ok(ev) = events.try_recv() {
            if app.handle_event(ev) { quit = true; }
        }
        if quit { break; }

        // Refresh DB data on a 250ms cadence
        if last_refresh.elapsed() >= Duration::from_millis(250) {
            if let Err(e) = app.refresh(reader).await {
                app.status_msg = format!("db: {e}");
            }
            last_refresh = Instant::now();
        }

        terminal.draw(|f| ui::render(f, app))?;

        // ~60 fps tick — keeps input responsive without burning CPU
        tokio::time::sleep(Duration::from_millis(16)).await;
    }

    Ok(())
}
