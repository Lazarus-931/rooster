# rooster

A local-first CLI tool for tracking and visualizing ML training runs.

No cloud. No accounts. Runs entirely on your machine.


https://github.com/user-attachments/assets/39115541-53cd-4d89-adbd-6b681b235f59

---

## What it does

- Python client logs metrics from any training script (PyTorch, JAX, TensorFlow)
- Rust backend receives logs over WebSocket and persists them to SQLite
- Terminal dashboard shows live charts as the run progresses

## Stack

| Layer | Technology |
|---|---|
| Client | Python (`pip install -e .`) |
| Server | Rust + Axum + WebSocket |
| Storage | SQLite (WAL mode) at `~/.rooster/rooster.db` |
| Dashboard | Rust + ratatui (Braille charts) |

## Quickstart

**1. Start the dashboard** (opens server + TUI together):
```bash
cargo run
```

**2. Run a training script** in another terminal:
```bash
uv run python example/train_pytorch.py
# or
uv run python example/train_jax.py
```

The dashboard auto-opens when you run either example script.

## Usage in your own training script

```python
from rooster import Define, MetricDef, Collect, Arrange, Send, launch_dashboard

launch_dashboard()   # opens server + TUI in a new terminal window

definition = Define(
    project_name="my_project",
    project_description="...",
    session_name="run_01",
    framework="pytorch",        # "pytorch" | "jax" | "tensorflow"
    metrics={
        "loss":     MetricDef(rate=1),    # log every step
        "accuracy": MetricDef(rate=5),    # log every 5th step
    },
)

collect = Collect(framework="pytorch", metrics=definition.metrics)
arrange = Arrange(definition=definition, collector=collect)
send    = Send()

send.register(arrange)
collect.attach(send, arrange)   # switches to live streaming mode

for step in range(1000):
    loss, accuracy = train_step()
    collect.log({"value": loss},     kind="loss")
    collect.log({"value": accuracy}, kind="accuracy")

send.end(arrange)
```

## CLI commands

```
cargo run           # server + dashboard (default)
cargo run serve     # server only (headless)
cargo run watch     # dashboard only (reads existing DB)
```

## Roadmap

- [ ] Multi-machine support
- [ ] Run comparison view
- [ ] Metric export (CSV / JSON)
- [ ] `rooster` binary via `cargo install`


# AI Disclosure
- This project made use of claude for checking the sql db design, crows diagram and pytorch bugs' I've faced when writing 
training examples.
