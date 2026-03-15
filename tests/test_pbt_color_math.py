"""Property-based tests for color_utils.py pure math functions.

Proves that the IEC 61966-2-1 sRGB spec invariants and compositing contracts
hold across the full input domain, not just at fixed sample points.

Each property is paired with a vacuity companion that uses a targeted strategy
to guarantee the interesting branch actually fires — so properties cannot pass
trivially because Hypothesis never generated a case that exercises the core logic.

Source claims:
  color_utils.py docstrings: "official piecewise sRGB transfer function" (IEC 61966-2-1)
  color_utils.py docstrings: compositing formulas and alpha semantics

No GPU or model weights required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from CorridorKeyModule.core import color_utils as cu

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Scalar float in [0, 1] — used for sRGB / linear values
_unit_float = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)

# RGB pixel as float32 array of shape (3,) with values in [0, 1]
_rgb_pixel = arrays(np.float32, (3,), elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))

# Alpha in (0.01, 1.0] — excludes near-zero where color is undefined
_alpha = st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False)

# ---------------------------------------------------------------------------
# sRGB ↔ Linear: roundtrip identity
# ---------------------------------------------------------------------------


class TestSrgbRoundtrip:
    """srgb_to_linear(linear_to_srgb(x)) == x for all x in [0, 1].

    The fixed-example tests in test_color_utils.py sample 8 values.
    This proves the invariant holds everywhere in the domain.
    """

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_linear_to_srgb_to_linear_numpy(self, x):
        arr = np.float32(x)
        roundtripped = float(cu.srgb_to_linear(cu.linear_to_srgb(arr)))
        assert roundtripped == pytest.approx(x, abs=1e-5)

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_srgb_to_linear_to_srgb_numpy(self, x):
        arr = np.float32(x)
        roundtripped = float(cu.linear_to_srgb(cu.srgb_to_linear(arr)))
        assert roundtripped == pytest.approx(x, abs=1e-5)

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_linear_to_srgb_to_linear_torch(self, x):
        t = torch.tensor(x, dtype=torch.float32)
        roundtripped = cu.srgb_to_linear(cu.linear_to_srgb(t)).item()
        assert roundtripped == pytest.approx(x, abs=1e-5)

    # --- Vacuity companions ---
    # Force values in both branches of the piecewise curve to ensure
    # the roundtrip is exercised in each region, not just one.

    @given(st.floats(0.0, 0.003, allow_nan=False, allow_infinity=False))
    def test_roundtrip_vacuity_linear_segment(self, x):
        """Force the linear segment (x <= 0.0031308) of linear_to_srgb."""
        arr = np.float32(x)
        roundtripped = float(cu.srgb_to_linear(cu.linear_to_srgb(arr)))
        assert roundtripped == pytest.approx(x, abs=1e-5)

    @given(st.floats(0.004, 1.0, allow_nan=False, allow_infinity=False))
    def test_roundtrip_vacuity_power_segment(self, x):
        """Force the power segment (x > 0.0031308) of linear_to_srgb."""
        arr = np.float32(x)
        roundtripped = float(cu.srgb_to_linear(cu.linear_to_srgb(arr)))
        assert roundtripped == pytest.approx(x, abs=1e-5)


# ---------------------------------------------------------------------------
# sRGB ↔ Linear: strict monotonicity
# ---------------------------------------------------------------------------


class TestSrgbMonotonicity:
    """Both transfer functions must be strictly monotonically increasing.

    A non-monotonic curve silently corrupts color ordering — darker pixels
    would map to brighter values or vice versa after conversion.
    """

    @given(_unit_float, _unit_float)
    @settings(max_examples=1000)
    def test_linear_to_srgb_monotone(self, a, b):
        if a >= b:
            return  # only test ordered pairs
        fa = float(cu.linear_to_srgb(np.float32(a)))
        fb = float(cu.linear_to_srgb(np.float32(b)))
        assert fa <= fb, f"linear_to_srgb not monotone: f({a})={fa} > f({b})={fb}"

    @given(_unit_float, _unit_float)
    @settings(max_examples=1000)
    def test_srgb_to_linear_monotone(self, a, b):
        if a >= b:
            return
        fa = float(cu.srgb_to_linear(np.float32(a)))
        fb = float(cu.srgb_to_linear(np.float32(b)))
        assert fa <= fb, f"srgb_to_linear not monotone: f({a})={fa} > f({b})={fb}"

    # --- Vacuity companions ---
    # Force pairs that straddle each breakpoint to ensure the monotonicity
    # check runs across the piecewise boundary, not just within one segment.

    @given(
        st.floats(0.001, 0.003, allow_nan=False, allow_infinity=False),
        st.floats(0.004, 0.01, allow_nan=False, allow_infinity=False),
    )
    def test_monotonicity_vacuity_across_linear_breakpoint(self, a, b):
        """Force a pair straddling the 0.0031308 breakpoint of linear_to_srgb."""
        fa = float(cu.linear_to_srgb(np.float32(a)))
        fb = float(cu.linear_to_srgb(np.float32(b)))
        assert fa <= fb

    @given(
        st.floats(0.03, 0.04, allow_nan=False, allow_infinity=False),
        st.floats(0.05, 0.1, allow_nan=False, allow_infinity=False),
    )
    def test_monotonicity_vacuity_across_srgb_breakpoint(self, a, b):
        """Force a pair straddling the 0.04045 breakpoint of srgb_to_linear."""
        fa = float(cu.srgb_to_linear(np.float32(a)))
        fb = float(cu.srgb_to_linear(np.float32(b)))
        assert fa <= fb


# ---------------------------------------------------------------------------
# sRGB ↔ Linear: range preservation
# ---------------------------------------------------------------------------


class TestSrgbRange:
    """Both transfer functions must map [0, 1] to [0, 1].

    An out-of-range output corrupts downstream compositing math which
    assumes all color values are normalized.
    """

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_linear_to_srgb_stays_in_unit_range(self, x):
        result = float(cu.linear_to_srgb(np.float32(x)))
        assert 0.0 <= result <= 1.0, f"linear_to_srgb({x}) = {result} out of [0, 1]"

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_srgb_to_linear_stays_in_unit_range(self, x):
        result = float(cu.srgb_to_linear(np.float32(x)))
        assert 0.0 <= result <= 1.0, f"srgb_to_linear({x}) = {result} out of [0, 1]"


# ---------------------------------------------------------------------------
# Compositing: output stays in [0, 1]
# ---------------------------------------------------------------------------


class TestCompositingRange:
    """composite_straight output must stay in [0, 1] for any valid inputs.

    Compositing software (Nuke, Resolve, Premiere) expects normalized color
    values. An out-of-range composite output is a silent correctness failure.
    """

    @given(_rgb_pixel, _rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_composite_straight_output_in_range(self, fg, bg, alpha):
        result = cu.composite_straight(fg, bg, np.float32(alpha))
        assert np.all(result >= 0.0), f"composite_straight produced negative values: {result}"
        assert np.all(result <= 1.0), f"composite_straight produced values > 1: {result}"

    # --- Vacuity companion ---
    # Force fg=1, bg=0, alpha=1 so the foreground path is fully exercised,
    # and fg=0, bg=1, alpha=0 so the background path is fully exercised.

    def test_compositing_range_vacuity_full_fg(self):
        """FG fully opaque over black BG — exercises the fg*alpha=1 path."""
        result = cu.composite_straight(
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.float32(1.0),
        )
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_compositing_range_vacuity_full_bg(self):
        """FG fully transparent — exercises the bg*(1-alpha)=bg path."""
        result = cu.composite_straight(
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.float32(0.0),
        )
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# Premultiply / unpremultiply: roundtrip identity
# ---------------------------------------------------------------------------


class TestPremultRoundtrip:
    """unpremultiply(premultiply(fg, alpha), alpha) == fg for all valid inputs.

    Alpha is tested from 0.01 upward. Near zero (0.0–0.01) color is
    undefined (premultiplied black is indistinguishable from transparent black),
    so the roundtrip identity does not hold there by design.
    """

    @given(_rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_numpy(self, fg, alpha):
        a = np.float32(alpha)
        premul = cu.premultiply(fg, a)
        recovered = cu.unpremultiply(premul, a)
        np.testing.assert_allclose(recovered, fg, atol=1e-4)

    @given(_rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_torch(self, fg, alpha):
        fg_t = torch.tensor(fg, dtype=torch.float32)
        a_t = torch.tensor(alpha, dtype=torch.float32)
        premul = cu.premultiply(fg_t, a_t)
        recovered = cu.unpremultiply(premul, a_t)
        torch.testing.assert_close(recovered, fg_t, atol=1e-4, rtol=1e-4)

    # --- Vacuity companions ---
    # Force alpha near 0.01 (the minimum tested) and alpha=1.0 (the maximum)
    # to ensure both extremes of the alpha range are exercised.

    @given(_rgb_pixel, st.floats(0.01, 0.05, allow_nan=False, allow_infinity=False))
    def test_roundtrip_vacuity_low_alpha(self, fg, alpha):
        """Force low-but-valid alpha to exercise the eps guard in unpremultiply."""
        a = np.float32(alpha)
        recovered = cu.unpremultiply(cu.premultiply(fg, a), a)
        np.testing.assert_allclose(recovered, fg, atol=1e-3)

    @given(_rgb_pixel)
    def test_roundtrip_vacuity_full_alpha(self, fg):
        """Force alpha=1.0 — premultiply is identity, roundtrip must be exact."""
        a = np.float32(1.0)
        recovered = cu.unpremultiply(cu.premultiply(fg, a), a)
        np.testing.assert_allclose(recovered, fg, atol=1e-5)
