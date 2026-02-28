# responsible for getting, formatting and sending training information to backend

import datetime
import json
import time
import atexit
import signal
import uuid

import websocket  # websocket-client

import math

import pydantic
from typing import Any, Literal


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
    ):
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework '{framework}'. Must be one of: {self.SUPPORTED_FRAMEWORKS}"
            )

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


@pydantic.dataclasses.dataclass
class MetricRecord:
    step: int
    timestamp: datetime.datetime
    metrics: dict[str, MetricValue]


class Collect:
    def __init__(self, framework: Literal["jax", "pytorch", "tensorflow"]):
        self.framework = framework
        self.records: list[MetricRecord] = []
        self._step = 0
        self._stream: "tuple[Send, Arrange] | None" = None

    def attach(self, sender: "Send", arranger: "Arrange"):
        """Wire up live streaming. After calling this, every log() sends immediately."""
        self._stream = (sender, arranger)

    def log(self, metrics: dict[str, Any], step: int | None = None):
        if step is not None:
            self._step = step
        record = MetricRecord(
            step=self._step,
            timestamp=datetime.datetime.utcnow(),
            metrics={k: _coerce(v) for k, v in metrics.items()},
        )
        self._step += 1
        if self._stream is not None:
            sender, arranger = self._stream
            sender._send(arranger.step_payload(record))
        else:
            self.records.append(record)

    def parse(self, metrics: str):
        try:
            parsed = json.loads(metrics)
        except json.JSONDecodeError:
            parsed = {}
            for part in metrics.split(","):
                part = part.strip()
                if "=" in part:
                    k, _, v = part.partition("=")
                    try:
                        parsed[k.strip()] = float(v.strip())
                    except ValueError:
                        parsed[k.strip()] = v.strip()
        self.log(parsed)

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
                collector.log(metrics=logs or {}, step=epoch)

        return _RoosterCallback()

    def wrap_jax_step(self, step_fn):
        if self.framework != "jax":
            raise RuntimeError("wrap_jax_step is only available for the 'jax' framework.")

        collector = self

        def wrapped(*args, **kwargs):
            result = step_fn(*args, **kwargs)
            if isinstance(result, dict):
                collector.log(result)
            return result

        return wrapped


class Arrange:
    """Serializes Define + Collect data into wire-ready payloads."""

    def __init__(self, definition: Define, collector: Collect):
        self.definition = definition
        self.collector = collector

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
        }

    def step_payload(self, record: MetricRecord) -> dict:
        """Single-record payload for live streaming — one per forward step."""
        return {
            "type": "metrics",
            "session_id": str(self.definition.session.session_id),
            "records": [
                {
                    "step": record.step,
                    "timestamp": record.timestamp.isoformat(),
                    "metrics": record.metrics,
                }
            ],
        }

    def metrics_payload(self, records: list[MetricRecord] | None = None) -> dict:
        if records is None:
            records = self.collector.records
        return {
            "type": "metrics",
            "session_id": str(self.definition.session.session_id),
            "records": [
                {
                    "step": r.step,
                    "timestamp": r.timestamp.isoformat(),
                    "metrics": r.metrics,
                }
                for r in records
            ],
        }

    def end_payload(self) -> dict:
        return {
            "type": "end",
            "session_id": str(self.definition.session.session_id),
        }


class Send:
    """Manages a WebSocket connection to the Rooster backend."""

    def __init__(self, url: str = WS_URL, retries: int = 3):
        self.url = url
        self.retries = retries
        self._ws: websocket.WebSocket | None = None
        self._connect()
        atexit.register(self.close)
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
        self._send(arranger.registration_payload())

    def flush(self, arranger: "Arrange"):
        """Send all pending metric records and clear the collector buffer."""
        if not arranger.collector.records:
            return
        self._send(arranger.metrics_payload())
        arranger.collector.records.clear()

    def end(self, arranger: "Arrange"):
        """Flush remaining metrics and signal end-of-session to the backend."""
        self.flush(arranger)
        self._send(arranger.end_payload())

    def close(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _handle_signal(self, signum, frame):
        self.close()
