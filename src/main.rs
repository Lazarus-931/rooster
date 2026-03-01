mod core;
mod cli;
mod server;

use std::sync::Arc;
use std::time::Duration;

use crate::core::db::Db;

#[tokio::main]
async fn main() {
    match std::env::args().nth(1).as_deref() {
        // `rooster watch` — TUI only, reads from existing DB
        Some("watch") => {
            if let Err(e) = cli::visual::run(None).await {
                eprintln!("rooster watch: {e}");
                std::process::exit(1);
            }
        }

        // `rooster serve` — server only, with tracing logs to stdout
        Some("serve") => {
            tracing_subscriber::fmt::init();
            run_server().await;
        }

        // Default (no subcommand): server in background + TUI in foreground.
        // tracing_subscriber intentionally NOT initialized — stdout logs corrupt ratatui.
        None => {
            tokio::spawn(run_server());
            // Brief pause so the server has time to bind before the TUI connects to the DB.
            tokio::time::sleep(Duration::from_millis(400)).await;
            if let Err(e) = cli::visual::run(None).await {
                eprintln!("rooster: {e}");
                std::process::exit(1);
            }
        }

        Some(unknown) => {
            eprintln!("rooster: unknown command '{unknown}'");
            eprintln!("usage: rooster [serve|watch]");
            std::process::exit(1);
        }
    }
}


async fn run_server() {
    let db = match Db::open().await {
        Ok(db) => Arc::new(db),
        Err(e) => {
            eprintln!("rooster: failed to open database: {e}");
            std::process::exit(1);
        }
    };

    let app = server::router(db);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878")
        .await
        .expect("failed to bind 127.0.0.1:7878");

    // tracing::info so the address only appears when serve mode inits tracing_subscriber.
    // In combined (TUI) mode this is silent — avoids corrupting the terminal.
    tracing::info!("listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app)
        .await
        .expect("server error");
}
