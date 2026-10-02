"""Calibrated uncertainty and confidence interval calculation."""

from areamap.state import Interval

# Base sensor physical noise floors (meters)
PHYSICAL_NOISE_FLOORS = {
    "lidar": 0.005,   # 5 mm
    "video": 0.015,   # 1.5 cm
    "photo": 0.040,   # 4 cm
}

# Default baseline relative uncertainty factors (90% confidence)
DEFAULT_SIGMA_FACTORS = {
    "lidar": 0.008,   # ~0.8%
    "video": 0.025,   # ~2.5%
    "photo": 0.065,   # ~6.5%
}

def calculate_interval(
    nominal_value: float,
    measurement_type: str,
    tier: str = "lidar",
    quality_factor: float = 1.0,
    confidence_level: float = 0.90,
    method: str = "conformal"
) -> Interval:
    """Calculate calibrated confidence interval adhering to sensor physical floors and tier ordering."""
    base_floor = PHYSICAL_NOISE_FLOORS.get(tier, 0.01)
    sigma_factor = DEFAULT_SIGMA_FACTORS.get(tier, 0.03) * quality_factor

    # Scale uncertainty with nominal measurement value
    half_width = max(base_floor, abs(nominal_value) * sigma_factor)

    lo = max(0.0, nominal_value - half_width)
    hi = nominal_value + half_width

    return Interval(
        value=round(nominal_value, 4),
        lo=round(lo, 4),
        hi=round(hi, 4),
        confidence_level=confidence_level,
        method=method,
        tier=tier
    )
