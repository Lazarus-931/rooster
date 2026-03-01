# responsible for getting, formatting and sending training information to backend

import datetime
import json
import os
import platform
import shutil
import subprocess
import time
import atexit
import signal
import uuid

import websocket  # websocket-client

import math

import pydantic
from typing import Any, Literal


def launch_dashboard() -> None:
    """Open the rooster server + TUI dashboard in a new terminal window.

    On macOS this opens a new Terminal.app window running ``cargo run``
    (which starts the WebSocket server + TUI together).  On other platforms
    it just prints a reminder to start the server manually.

    Call this once at the top of your training script, before ``Send()``.
    """
    if platform.system() != "Darwin":
        print("rooster: start the dashboard with `cargo run` in another terminal.")
        return

    # Walk up from this file to find the directory that contains Cargo.toml.
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    for _ in range(6):
        if os.path.exists(os.path.join(root, "Cargo.toml")):
            break
        root = os.path.dirname(root)
    else:
        print("rooster: could not find Cargo.toml — start `cargo run` manually.")
        return

    binary = shutil.which("rooster")
    cmd = binary if binary else f"cd {root} && cargo run"

    script = f'tell application "Terminal" to do script "{cmd}"'
    subprocess.Popen(["osascript", "-e", script])

    # Give the server a moment to bind before Send() tries to connect.
    time.sleep(2)


# Scalar type that can survive json.dumps without a custom encoder.
# Maps exactly to what serde_json can deserialize without ambiguity:
#   bool  → JSON true/false  → Rust bool
#   int   → JSON integer     → Rust i64  (clamped to i64 range)
#   float → JSON number      → Rust f64  (nan/inf rejected)
#   str   → JSON string      → Rust String
MetricValue = float | int | str | bool

_I64_MAX = (1 << 63) - 1
_I64_MIN = -(1 << 63)


def _coerce(val: Any) -> MetricValue:
    """Convert any framework scalar (numpy, torch, jax) to a serde_json-safe Python type.

    Raises ValueError for nan/inf — those are not valid JSON numbers and serde rejects them.
    """
    # bool must be checked before int — bool is a subclass of int in Python
    if isinstance(val, bool):
        return val

    if isinstance(val, int):
        # Clamp to i64 range so serde_json can deserialize as i64 without overflow.
        if val > _I64_MAX or val < _I64_MIN:
            return float(val)  # f64 handles the range; precision loss is acceptable
        return val

    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"Metric value {val!r} is not valid JSON. "
                "nan/inf cannot be deserialized by serde_json."
            )
        return val

    if isinstance(val, str):
        return val

    # numpy / torch / jax scalars expose .item() → recurse with the Python native value
    if hasattr(val, "item"):
        return _coerce(val.item())

    # last resort: try float(), then stringify
    try:
        return _coerce(float(val))
    except (TypeError, ValueError):
        return str(val)


WS_URL = "ws://127.0.0.1:7878/ws"


@pydantic.dataclasses.dataclass
class Project:
    name: str
    description: str


@pydantic.dataclasses.dataclass
class TrainingSession:
    project_id: str
    session_id: uuid.UUID
    name: str
    framework: Literal["jax", "pytorch", "tensorflow"]


class Define:
    SUPPORTED_FRAMEWORKS = ("jax", "pytorch", "tensorflow")

    def __init__(
        self,
        project_name: str,
        project_description: str,
        session_name: str,
        framework: Literal["jax", "pytorch", "tensorflow"],
        metrics: dict[str, "MetricDef"],
    ):
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework '{framework}'. Must be one of: {self.SUPPORTED_FRAMEWORKS}"
            )
        if not metrics:
            raise ValueError("At least one metric must be declared in Define(metrics={...})")

        self.project = Project(
            name=project_name,
            description=project_description,
        )

        self.session = TrainingSession(
            project_id=project_name.lower().replace(" ", "_"),
            session_id=uuid.uuid4(),
            name=session_name,
            framework=framework,
        )

        self.metrics: dict[str, MetricDef] = metrics


@pydantic.dataclasses.dataclass
class MetricDef:
    """User declares a metric by name and rate only.
    dtype is inferred automatically from the first logged value — not set by the user.
    """
    rate: int = 1


@pydantic.dataclasses.dataclass
class LogEntry:
    kind: str
    step: int
    timestamp: datetime.datetime
    data: dict[str, MetricValue]
    # dtype is inferred by _coerce and carried implicitly by the MetricValue type.
    # The parser detects it; the server records it in the session's metrics map.
    dtype: str = ""   # populated by Collect.log() — "float", "int", "str", "bool"


def _infer_dtype(val: MetricValue) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    return "str"


class Collect:
    def __init__(
        self,
        framework: Literal["jax", "pytorch", "tensorflow"],
        metrics: dict[str, MetricDef],
    ):
        self.framework = framework
        self.metrics = metrics          # declared metrics — name → MetricDef(rate)
        self.entries: list[LogEntry] = []
        self._step = 0
        self._stream: "tuple[Send, Arrange] | None" = None

    def attach(self, sender: "Send", arranger: "Arrange"):
        """Wire up live streaming. After calling this, every log() sends immediately."""
        self._stream = (sender, arranger)

    def log(self, data: dict[str, Any], kind: str, step: int | None = None):
        # kind must be declared — enforces explicit metric definition
        if kind not in self.metrics:
            raise ValueError(
                f"Metric '{kind}' was not declared. "
                f"Add it to Define(metrics={{...}}) before logging."
            )
        if step is not None:
            self._step = step

        coerced = {k: _coerce(v) for k, v in data.items()}
        # infer dtype from the first value — the parser catches the type automatically
        first_val = next(iter(coerced.values())) if coerced else 0.0
        dtype = _infer_dtype(first_val)

        entry = LogEntry(
            kind=kind,
            step=self._step,
            timestamp=datetime.datetime.utcnow(),
            data=coerced,
            dtype=dtype,
        )
        self._step += 1

        if self._stream is not None:
            sender, arranger = self._stream
            payload = arranger.step_payload(entry)
            if payload is not None:
                sender._send(payload)
        else:
            self.entries.append(entry)

    def parse(self, data: str, kind: str = "metric"):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = {}
            for part in data.split(","):
                part = part.strip()
                if "=" in part:
                    k, _, v = part.partition("=")
                    try:
                        parsed[k.strip()] = float(v.strip())
                    except ValueError:
                        parsed[k.strip()] = v.strip()
        self.log(parsed, kind=kind)

    def as_keras_callback(self):
        if self.framework != "tensorflow":
            raise RuntimeError("Keras callbacks are only available for the 'tensorflow' framework.")
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is not installed.")

        collector = self

        class _RoosterCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                collector.log(data=logs or {}, kind="epoch", step=epoch)

        return _RoosterCallback()

    def wrap_jax_step(self, step_fn):
        if self.framework != "jax":
            raise RuntimeError("wrap_jax_step is only available for the 'jax' framework.")

        collector = self

        def wrapped(*args, **kwargs):
            result = step_fn(*args, **kwargs)
            if isinstance(result, dict):
                collector.log(result, kind="step")
            return result

        return wrapped


class Arrange:
    """Formats LogEntry data into wire-ready payloads.

    Rate is per-metric, taken from the MetricDef declared in Define(metrics={...}).
    No global rate — each metric controls its own send frequency independently.
    """

    def __init__(self, definition: Define, collector: Collect):
        self.definition = definition
        self.collector = collector
        self._counters: dict[str, int] = {}   # per-metric call counter

    def _session_id(self) -> str:
        return str(self.definition.session.session_id)

    def registration_payload(self) -> dict:
        sess = self.definition.session
        proj = self.definition.project
        return {
            "type": "register",
            "project": {
                "name": proj.name,
                "description": proj.description,
            },
            "session": {
                "session_id": str(sess.session_id),
                "project_id": sess.project_id,
                "name": sess.name,
                "framework": sess.framework,
            },
            # Metric schema sent upfront — dtype is absent here, inferred server-side
            # from the first log entry for each metric.
            "metrics": {
                name: {"rate": mdef.rate}
                for name, mdef in self.definition.metrics.items()
            },
        }

    def step_payload(self, entry: LogEntry) -> dict | None:
        """Per-metric rate gate. Returns None when this metric's rate says skip."""
        rate = self.definition.metrics[entry.kind].rate
        count = self._counters.get(entry.kind, 0) + 1
        self._counters[entry.kind] = count
        if count % rate != 0:
            return None
        return {
            "type": "log",
            "session_id": self._session_id(),
            "entries": [
                {
                    "kind": entry.kind,
                    "step": entry.step,
                    "timestamp": entry.timestamp.isoformat(),
                    "dtype": entry.dtype,    # inferred by parser, not declared by user
                    "data": entry.data,
                }
            ],
        }

    def logs_payload(self, entries: list[LogEntry] | None = None) -> dict:
        """Batch payload for all buffered entries (non-streaming flush)."""
        if entries is None:
            entries = self.collector.entries
        return {
            "type": "log",
            "session_id": self._session_id(),
            "entries": [
                {
                    "kind": e.kind,
                    "step": e.step,
                    "timestamp": e.timestamp.isoformat(),
                    "data": e.data,
                }
                for e in entries
            ],
        }

    def end_payload(self) -> dict:
        return {
            "type": "end",
            "session_id": self._session_id(),
        }


class Send:
    """Manages a WebSocket connection to the Rooster backend."""

    def __init__(self, url: str = WS_URL, retries: int = 3):
        self.url = url
        self.retries = retries
        self._ws: websocket.WebSocket | None = None
        self._arranger: "Arrange | None" = None  # set in register(); used by shutdown
        self._ended = False                        # guards against double end-message
        self._connect()
        atexit.register(self._shutdown)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _connect(self):
        for attempt in range(1, self.retries + 1):
            try:
                ws = websocket.WebSocket()
                ws.connect(self.url)
                self._ws = ws
                return
            except Exception as e:
                if attempt == self.retries:
                    raise ConnectionError(
                        f"Could not connect to Rooster backend at {self.url} "
                        f"after {self.retries} attempts: {e}"
                    )
                time.sleep(0.5 * attempt)

    def _send(self, payload: dict):
        """Send payload as a JSON text frame. WebSocket handles message boundaries."""
        data = json.dumps(payload, allow_nan=False)
        if self._ws is None:
            self._connect()
        try:
            self._ws.send(data)
        except Exception:
            self._connect()
            self._ws.send(data)

    def register(self, arranger: "Arrange"):
        # Store so atexit / SIGTERM can send End without the caller doing it manually.
        self._arranger = arranger
        self._send(arranger.registration_payload())

    def flush(self, arranger: "Arrange"):
        """Send all buffered entries and clear the buffer."""
        if not arranger.collector.entries:
            return
        self._send(arranger.logs_payload())
        arranger.collector.entries.clear()

    def end(self, arranger: "Arrange"):
        """Flush remaining buffered entries and send End to the backend.

        Idempotent — safe to call more than once (atexit will call it again).
        """
        if self._ended:
            return
        self._ended = True
        self.flush(arranger)
        self._send(arranger.end_payload())

    def _shutdown(self):
        """Send End (if not already sent) then close the socket.

        Called by atexit on normal exit and by the SIGTERM handler.
        Uses best-effort delivery — if the socket is already gone, we skip silently.
        """
        if not self._ended and self._arranger is not None:
            try:
                self.end(self._arranger)
            except Exception:
                pass  # socket may be dead; the server will fall back to last.json
        self.close()

    def close(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _handle_signal(self, signum, frame):
        self._shutdown()
        # Re-raise with the default handler so the process exits with the right code.
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)
