"""Device-neutral tracking inputs; no optional headset SDK dependency."""

from .calibration import (
    CalibrationFitError,
    HandToolRotationFit,
    RigidTransformFit,
    fit_hand_tool_rotation,
    fit_rigid_transform,
)
from .config import TrackingStreamConfig
from .model import JointSample, TrackingFrame, TrackingIdentity
from .motion import TrackingMotionLimits
from .openxr import (
    FB_BODY_JOINT_MAP,
    OPENXR_TO_FLU,
    UNITY_TO_FLU,
    LocationFlags,
    OpenXRAdapter,
    OpenXRJoint,
)
from .pipeline import (
    CalibrationRequired,
    FramePolicy,
    LatestFrameBuffer,
    TrackingCalibration,
    TrackingUnavailable,
)
from .recording import (
    RecordingError,
    TrackingEvent,
    TrackingRecorder,
    TrackingRecordingReader,
    TrackingTick,
)
from .stream import TrackingPoseStream

__all__ = [
    "CalibrationFitError",
    "CalibrationRequired",
    "FB_BODY_JOINT_MAP",
    "FramePolicy",
    "HandToolRotationFit",
    "JointSample",
    "LatestFrameBuffer",
    "LocationFlags",
    "OPENXR_TO_FLU",
    "OpenXRAdapter",
    "OpenXRJoint",
    "RecordingError",
    "RigidTransformFit",
    "TrackingCalibration",
    "TrackingEvent",
    "TrackingFrame",
    "TrackingIdentity",
    "TrackingMotionLimits",
    "TrackingPoseStream",
    "TrackingRecorder",
    "TrackingRecordingReader",
    "TrackingStreamConfig",
    "TrackingTick",
    "TrackingUnavailable",
    "UNITY_TO_FLU",
    "fit_hand_tool_rotation",
    "fit_rigid_transform",
]
