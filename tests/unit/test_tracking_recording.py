import io
import json
from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import rot
from sew_mimic.tracking import (
    FramePolicy,
    JointSample,
    RecordingError,
    TrackingCalibration,
    TrackingFrame,
    TrackingIdentity,
    TrackingRecorder,
    TrackingRecordingReader,
    TrackingTick,
    TrackingUnavailable,
)


@pytest.fixture
def frame():
    return TrackingFrame(
        TrackingIdentity("quest", "session-1", "local", 3),
        2**60 + 7,
        10.0,
        10.01,
        {
            "left_wrist": JointSample([1, 2, 3], rot([1, 2, 3], 0.8), True, False),
            "left_elbow": JointSample([1, 2, 2]),
            "right_wrist": JointSample(),
        },
    )


def _record(events, **kwargs):
    stream = io.StringIO()
    with TrackingRecorder(stream, **kwargs) as recorder:
        for event in events:
            recorder.write(event)
    return stream.getvalue()


def _changed_record(text, line, change):
    records = [json.loads(item) for item in text.splitlines()]
    change(records[line])
    return "\n".join(json.dumps(item) for item in records) + "\n"


def test_recording_preserves_calibration_identity_quality_and_clock(frame):
    calibration = TrackingCalibration(
        frame.identity,
        rotation=rot([0, 0, 1], 0.4),
        translation=[1, 2, 3],
        left_hand_to_tool=rot([1, 0, 0], 0.7),
        right_hand_to_tool=rot([0, 1, 0], -0.2),
        tool_length=0.17,
    )
    loss = replace(frame, sequence=frame.sequence + 1, active=False, confidence=0.0)
    text = _record([calibration, frame, TrackingTick(10.02), loss], metadata={"operator": "测试"})
    reader = TrackingRecordingReader(io.StringIO(text))
    actual_calibration, actual_frame, tick, actual_loss = list(reader)
    assert reader.metadata == {"operator": "测试"}
    with pytest.raises(TypeError):
        reader.metadata["operator"] = "changed"
    assert actual_calibration.identity == calibration.identity
    for name in ("rotation", "translation", "left_hand_to_tool", "right_hand_to_tool"):
        np.testing.assert_array_equal(getattr(actual_calibration, name), getattr(calibration, name))
    assert actual_calibration.tool_length == calibration.tool_length
    assert actual_frame.identity == frame.identity
    assert actual_frame.sequence == 2**60 + 7
    assert actual_frame.sample_time == frame.sample_time
    assert actual_frame.received_time == frame.received_time
    assert actual_frame.confidence is None
    assert actual_frame.joints.keys() == frame.joints.keys()
    for name, joint in actual_frame.joints.items():
        expected = frame.joints[name]
        for component in ("position", "orientation"):
            if getattr(expected, component) is None:
                assert getattr(joint, component) is None
            else:
                np.testing.assert_array_equal(
                    getattr(joint, component), getattr(expected, component)
                )
                assert not getattr(joint, component).flags.writeable
        assert joint.position_tracked == expected.position_tracked
        assert joint.orientation_tracked == expected.orientation_tracked
    assert tick.timestamp == 10.02
    assert not actual_loss.active
    assert actual_loss.confidence == 0.0


def test_replay_keeps_delayed_frames_stale_instead_of_refreshing_receipt(frame):
    original = replace(frame, sample_time=8.0)
    replayed = next(TrackingRecordingReader(io.StringIO(_record([original]))))
    with pytest.raises(TrackingUnavailable, match="stale"):
        FramePolicy().check(replayed, now=10.02)


def test_recording_preserves_arrival_order_for_duplicates_and_old_packets(frame):
    original = [frame, frame, replace(frame, sequence=0, sample_time=9)]
    replayed = list(TrackingRecordingReader(io.StringIO(_record(original))))
    assert [item.sequence for item in replayed] == [item.sequence for item in original]
    assert [item.sample_time for item in replayed] == [10, 10, 9]


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1"])
def test_reader_rejects_unsupported_or_mistyped_version(version):
    text = _changed_record(_record([]), 0, lambda data: data.update(version=version))
    with pytest.raises(RecordingError, match="line 1.*version"):
        TrackingRecordingReader(io.StringIO(text))


@pytest.mark.parametrize(
    "changes",
    [
        {"units": "centimetres"},
        {"clock": "headset_clock"},
        {"orientation": "xyzw"},
        {"extra": "unknown"},
        {"metadata": {"robot": 4}},
    ],
)
def test_reader_checks_header_conventions_and_metadata(changes):
    text = _changed_record(_record([]), 0, lambda data: data.update(changes))
    with pytest.raises(RecordingError, match="line 1"):
        TrackingRecordingReader(io.StringIO(text))


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(sample_time="10.0"),
        lambda data: data.update(sample_time=True),
        lambda data: data.update(sample_time=float("nan")),
        lambda data: data.update(received_time=float("inf")),
        lambda data: data.update(active=1),
        lambda data: data.update(sequence=2.5),
        lambda data: data.update(confidence=2),
        lambda data: data["identity"].update(space_revision=True),
        lambda data: data["joints"]["left_wrist"].update(position=[True, 1, 2]),
        lambda data: data["joints"]["left_wrist"].update(position=["1", 1, 2]),
        lambda data: data["joints"]["left_wrist"].update(orientation=[[1, 0, 0]]),
        lambda data: data["joints"]["left_wrist"].update(orientation=np.zeros((3, 3)).tolist()),
        lambda data: data["joints"]["right_wrist"].update(position_tracked=True),
        lambda data: data.update(type="command"),
        lambda data: data.update(extra=0),
    ],
)
def test_malformed_event_stops_reader_with_line_number(frame, change):
    text = _changed_record(_record([frame]), 1, change)
    reader = TrackingRecordingReader(io.StringIO(text))
    with pytest.raises(RecordingError, match="line 2"):
        next(reader)
    with pytest.raises(RecordingError, match="stopped"):
        next(reader)


def test_duplicate_json_fields_are_rejected():
    text = _record([TrackingTick(1)])
    text = text.replace('"timestamp":1.0', '"timestamp":1.0,"timestamp":2.0')
    with pytest.raises(RecordingError, match="Duplicate JSON field"):
        list(TrackingRecordingReader(io.StringIO(text)))


@pytest.mark.parametrize("tail", ["", '{"type":', '{"type":"end","records":4}\n'])
def test_truncation_and_wrong_event_count_are_reported(tail):
    lines = _record([TrackingTick(1)]).splitlines(keepends=True)
    reader = TrackingRecordingReader(io.StringIO("".join(lines[:-1]) + tail))
    assert next(reader).timestamp == 1
    with pytest.raises(RecordingError, match="line 3"):
        next(reader)


def test_trailing_data_after_completion_is_rejected():
    with pytest.raises(RecordingError, match="after completion"):
        list(TrackingRecordingReader(io.StringIO(_record([]) + "{}\n")))


def test_decreasing_control_clock_is_rejected_on_write_and_read():
    stream = io.StringIO()
    with TrackingRecorder(stream) as recorder:
        recorder.write(TrackingTick(10))
        with pytest.raises(RecordingError, match="must not decrease"):
            recorder.write(TrackingTick(9))
    assert len(list(TrackingRecordingReader(io.StringIO(stream.getvalue())))) == 1
    text = _changed_record(
        _record([TrackingTick(10), TrackingTick(11)]), 2, lambda data: data.update(timestamp=9)
    )
    with pytest.raises(RecordingError, match="must not decrease"):
        list(TrackingRecordingReader(io.StringIO(text)))


def test_reader_bounds_each_read_and_does_not_load_whole_recording():
    class BoundedStream(io.StringIO):
        def readline(self, size=-1):
            assert 0 < size <= 1025
            return super().readline(size)

        def read(self, size=-1):
            assert size == 1
            return super().read(size)

    text = _record(TrackingTick(i) for i in range(1000))
    reader = TrackingRecordingReader(BoundedStream(text), max_record_chars=1024)
    assert sum(1 for _ in reader) == 1000


def test_oversized_record_is_rejected_before_json_parsing():
    text = _record([]).splitlines(keepends=True)[0] + " " * 1025 + "{}\n"
    reader = TrackingRecordingReader(io.StringIO(text), max_record_chars=1024)
    with pytest.raises(RecordingError, match="max_record_chars"):
        next(reader)
    with pytest.raises(RecordingError, match="max_record_chars"):
        TrackingRecorder(io.StringIO(), metadata={"large": "a" * 1024}, max_record_chars=1024)


def test_context_manager_leaves_stream_open_and_aborted_recording_incomplete():
    stream = io.StringIO()
    with pytest.raises(RuntimeError):
        with TrackingRecorder(stream) as recorder:
            recorder.write(TrackingTick(1))
            raise RuntimeError("acquisition failed")
    assert not stream.closed
    with pytest.raises(RecordingError, match="missing completion marker"):
        list(TrackingRecordingReader(io.StringIO(stream.getvalue())))
    with pytest.raises(RecordingError, match="closed"):
        recorder.write(TrackingTick(2))


def test_partial_write_cannot_append_a_completion_marker():
    class PartialStream(io.StringIO):
        fail = False

        def write(self, text):
            return super().write(text[:-1] if self.fail else text)

    stream = PartialStream()
    recorder = TrackingRecorder(stream)
    stream.fail = True
    with pytest.raises(OSError, match="Incomplete"):
        recorder.write(TrackingTick(1))
    recorder.close()
    assert '"type":"end"' not in stream.getvalue()
    with pytest.raises(RecordingError, match="closed"):
        recorder.write(TrackingTick(2))


def test_read_failure_stops_replay_instead_of_skipping_an_event():
    class FailingStream(io.StringIO):
        fail = False

        def readline(self, size=-1):
            if self.fail:
                raise OSError("recording device disconnected")
            return super().readline(size)

    stream = FailingStream(_record([TrackingTick(1)]))
    reader = TrackingRecordingReader(stream)
    stream.fail = True
    with pytest.raises(RecordingError, match="line 2.*disconnected"):
        next(reader)
    stream.fail = False
    with pytest.raises(RecordingError, match="stopped"):
        next(reader)
