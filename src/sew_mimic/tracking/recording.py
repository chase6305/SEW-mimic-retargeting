"""Versioned, streaming JSONL recordings for deterministic offline input replay.

This format stores canonical input after SDK decoding and clock conversion.
Recorded receive times and calibration events are for offline replay only;
they must not be accepted as authority by a live network receiver or driver.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TextIO

import numpy as np

from .model import JointSample, TrackingFrame, TrackingIdentity, _index
from .pipeline import TrackingCalibration

_FORMAT = {
    "type": "header",
    "format": "sew-mimic-tracking",
    "version": 1,
    "units": "metres",
    "orientation": "rotation_matrix_3x3",
    "clock": "receiver_monotonic_seconds",
}
_MAX_RECORD_CHARS = 1024 * 1024


class RecordingError(ValueError):
    """A recording is malformed, unsupported, incomplete or exceeds its limit."""


@dataclass(frozen=True)
class TrackingTick:
    """A control-loop observation time, including ticks with no new input."""

    timestamp: float

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("tick timestamp must be finite")
        object.__setattr__(self, "timestamp", timestamp)


TrackingEvent = TrackingFrame | TrackingCalibration | TrackingTick


def _identity(identity: TrackingIdentity) -> dict:
    return {
        "source_id": identity.source_id,
        "session_id": identity.session_id,
        "space_id": identity.space_id,
        "space_revision": identity.space_revision,
    }


def _encode(event: TrackingEvent) -> dict:
    if isinstance(event, TrackingTick):
        return {"type": "tick", "timestamp": event.timestamp}
    if isinstance(event, TrackingCalibration):
        return {
            "type": "calibration",
            "identity": _identity(event.identity),
            "rotation": event.rotation.tolist(),
            "translation": event.translation.tolist(),
            "left_hand_to_tool": event.left_hand_to_tool.tolist(),
            "right_hand_to_tool": event.right_hand_to_tool.tolist(),
            "tool_length": float(event.tool_length),
        }
    if isinstance(event, TrackingFrame):
        return {
            "type": "frame",
            "identity": _identity(event.identity),
            "sequence": event.sequence,
            "sample_time": event.sample_time,
            "received_time": event.received_time,
            "active": bool(event.active),
            "confidence": event.confidence,
            "joints": {
                name: {
                    "position": None if joint.position is None else joint.position.tolist(),
                    "orientation": (
                        None if joint.orientation is None else joint.orientation.tolist()
                    ),
                    "position_tracked": bool(joint.position_tracked),
                    "orientation_tracked": bool(joint.orientation_tracked),
                }
                for name, joint in event.joints.items()
            },
        }
    raise TypeError("event must be a TrackingFrame, TrackingCalibration or TrackingTick")


def _fields(value: object, names: str, context: str) -> dict:
    if not isinstance(value, dict) or value.keys() != set(names.split()):
        raise ValueError(f"{context} must contain exactly: {names}")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _geometry(value: object, shape: tuple[int, ...], name: str) -> list:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{name} must have shape {shape}")
    if len(shape) == 1:
        return [_number(component, name) for component in value]
    return [_geometry(component, shape[1:], name) for component in value]


def _decode_identity(value: object) -> TrackingIdentity:
    data = _fields(value, "source_id session_id space_id space_revision", "identity")
    return TrackingIdentity(**data)


def _decode(data: dict) -> TrackingEvent:
    kind = data.get("type")
    if kind == "tick":
        _fields(data, "type timestamp", "tick")
        return TrackingTick(_number(data["timestamp"], "timestamp"))
    if kind == "calibration":
        _fields(
            data,
            "type identity rotation translation left_hand_to_tool right_hand_to_tool tool_length",
            "calibration",
        )
        return TrackingCalibration(
            identity=_decode_identity(data["identity"]),
            rotation=_geometry(data["rotation"], (3, 3), "rotation"),
            translation=_geometry(data["translation"], (3,), "translation"),
            left_hand_to_tool=_geometry(data["left_hand_to_tool"], (3, 3), "left_hand_to_tool"),
            right_hand_to_tool=_geometry(data["right_hand_to_tool"], (3, 3), "right_hand_to_tool"),
            tool_length=_number(data["tool_length"], "tool_length"),
        )
    if kind != "frame":
        raise ValueError(f"Unknown event type: {kind!r}")
    _fields(
        data,
        "type identity sequence sample_time received_time active confidence joints",
        "frame",
    )
    if not isinstance(data["joints"], dict):
        raise ValueError("joints must be an object")
    joints = {}
    for name, joint in data["joints"].items():
        _fields(joint, "position orientation position_tracked orientation_tracked", f"joint {name}")
        joints[name] = JointSample(
            None if joint["position"] is None else _geometry(joint["position"], (3,), name),
            None if joint["orientation"] is None else _geometry(joint["orientation"], (3, 3), name),
            joint["position_tracked"],
            joint["orientation_tracked"],
        )
    return TrackingFrame(
        _decode_identity(data["identity"]),
        data["sequence"],
        _number(data["sample_time"], "sample_time"),
        _number(data["received_time"], "received_time"),
        joints,
        data["active"],
        None if data["confidence"] is None else _number(data["confidence"], "confidence"),
    )


def _metadata(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("metadata must map strings to strings")
    return dict(value)


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


class TrackingRecorder:
    """Write canonical events in arrival/control order to a caller-owned stream.

    Use as a context manager to append a completion marker on successful exit.
    A failed write poisons the recorder so a partial stream cannot be marked
    complete. File I/O belongs on a recording worker, outside the control loop.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        metadata: Mapping[str, str] | None = None,
        max_record_chars: int = _MAX_RECORD_CHARS,
    ) -> None:
        self._stream = stream
        self._limit = _index(max_record_chars, "max_record_chars")
        self._count = 0
        self._closed = False
        self._last_tick = -np.inf
        self._write({**_FORMAT, "metadata": _metadata(dict(metadata or {}))})

    def _write(self, record: dict) -> None:
        line = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        if len(line) > self._limit:
            raise RecordingError("record exceeds max_record_chars")
        try:
            if self._stream.write(line) != len(line):
                raise OSError("Incomplete recording write")
        except Exception:
            self._closed = True
            raise

    def write(self, event: TrackingEvent) -> None:
        """Preserve times and ordering, including inactive/duplicate/old frames."""
        if self._closed:
            raise RecordingError("recorder is closed")
        if isinstance(event, TrackingTick) and event.timestamp < self._last_tick:
            raise RecordingError("control tick timestamps must not decrease")
        self._write(_encode(event))
        if isinstance(event, TrackingTick):
            self._last_tick = event.timestamp
        self._count += 1

    def close(self) -> None:
        """Finish the recording and flush; do not close the caller's stream."""
        if not self._closed:
            self._write({"type": "end", "records": self._count})
            self._closed = True
            self._stream.flush()

    def __enter__(self) -> TrackingRecorder:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._closed = True


class TrackingRecordingReader(Iterator[TrackingEvent]):
    """Read bounded records lazily; malformed input stops replay with a line number.

    Timestamps remain in the recorded clock domain. Use TrackingTick timestamps
    as the offline clock; do not refresh measurements with the current clock.
    Fully consume the iterator to verify its completion marker and event count.
    """

    def __init__(self, stream: TextIO, *, max_record_chars: int = _MAX_RECORD_CHARS) -> None:
        self._stream = stream
        self._limit = _index(max_record_chars, "max_record_chars")
        self._line = 0
        self._count = 0
        self._finished = False
        self._failed = False
        self._last_tick = -np.inf
        try:
            header = self._read()
            _fields(header, "type format version units orientation clock metadata", "header")
            if type(header["version"]) is not int or any(
                header[key] != value for key, value in _FORMAT.items()
            ):
                raise ValueError("unsupported recording version or conventions")
            self.metadata = MappingProxyType(_metadata(header["metadata"]))
        except (ValueError, TypeError, OverflowError, RecursionError, OSError) as exc:
            raise RecordingError(f"line {self._line}: {exc}") from exc

    def _read(self) -> dict:
        self._line += 1
        line = self._stream.readline(self._limit + 1)
        if not line:
            raise ValueError("incomplete recording: missing completion marker")
        if len(line) > self._limit:
            raise ValueError("record exceeds max_record_chars")
        data = json.loads(line, object_pairs_hook=_object, parse_constant=_reject_constant)
        if not isinstance(data, dict):
            raise ValueError("record must be an object")
        return data

    def __iter__(self) -> TrackingRecordingReader:
        return self

    def __next__(self) -> TrackingEvent:
        if self._failed:
            raise RecordingError("reader stopped after a malformed record")
        if self._finished:
            raise StopIteration
        try:
            record = self._read()
            if record.get("type") == "end":
                _fields(record, "type records", "completion marker")
                if _index(record["records"], "records") != self._count:
                    raise ValueError("completion marker event count mismatch")
                if self._stream.read(1):
                    raise ValueError("unexpected data after completion marker")
                self._finished = True
            else:
                event = _decode(record)
                if isinstance(event, TrackingTick):
                    if event.timestamp < self._last_tick:
                        raise ValueError("control tick timestamps must not decrease")
                    self._last_tick = event.timestamp
                self._count += 1
                return event
        except (ValueError, TypeError, OverflowError, RecursionError, OSError) as exc:
            self._failed = True
            raise RecordingError(f"line {self._line}: {exc}") from exc
        raise StopIteration
