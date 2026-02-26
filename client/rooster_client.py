# responsible for getting, formatting and sending training information to backend

import datetime
import socket, json, time, atexit, signal
import socket, json
import uuid

import pydantic
from typing import Literal


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
    metrics: dict


class Collect:
    def __init__(self, framework: Literal["jax", "pytorch", "tensorflow"]):
        self.framework = framework
        self.records: list[MetricRecord] = []
        self._step = 0

    def log(self, metrics: dict, step: int | None = None):
        if step is not None:
            self._step = step
        self.records.append(MetricRecord(
            step=self._step,
            timestamp=datetime.datetime.utcnow(),
            metrics=metrics,
        ))
        self._step += 1

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
    pass


class Send:
    pass

