"""Settings used for multiple tests."""

# Default tolerances
TOLERANCES = dict(atol=1e-3, rtol=1e-4)
MILD_TOLERANCES = dict(atol=0.05, rtol=0.05)
STRICT_TOLERANCES = dict(atol=1e-6, rtol=1e-6)

BATCH_DIMS = [[10, 10], [1000]]

REPS = ["4x0n", "4x1n", "10x0n+5x1n+2x2n"]

LOGM2_MEAN_STD = ((0, 1), (0, 0.1), (-3, 1))

from lloca.framesnet.equi_frames import (
    LearnedSO13Frames,
    LearnedRestFrames,
    LearnedPDFrames,
)

FRAMES_PREDICTOR = [
    LearnedSO13Frames,
    LearnedRestFrames,
    LearnedPDFrames,
]
