"""Settings used for multiple tests."""

# Default tolerances
TOLERANCES = dict(atol=1e-3, rtol=1e-4)
MILD_TOLERANCES = dict(atol=0.05, rtol=0.05)
STRICT_TOLERANCES = dict(atol=1e-6, rtol=1e-6)

BATCH_DIMS = [[10, 10], [1000]]
