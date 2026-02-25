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

class Collect:
    def __init__(self, framework):
    pass


class Arrange:
    pass


class Send:
    pass

