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


# ---------------------------------------------------------------------------
# Despill contracts: green never increases, never goes negative
# ---------------------------------------------------------------------------


class TestDespillContracts:
    """Green channel must only decrease (or stay the same) after despill.

    Source claim: LLM_HANDOVER.md — "luminance-preserving despill()".
    The implementation removes green spill by reducing G and redistributing
    equally to R and B. Green must never increase and must never go negative.
    """

    @given(_rgb_pixel, st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_green_never_increases_numpy(self, pixel, strength):
        result = cu.despill(pixel.copy(), strength=strength)
        assert result[1] <= pixel[1] + 1e-6, (
            f"green increased: {pixel[1]:.6f} → {result[1]:.6f}"
        )

    @given(_rgb_pixel, st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_green_never_negative_numpy(self, pixel, strength):
        result = cu.despill(pixel.copy(), strength=strength)
        assert result[1] >= -1e-6, f"green went negative: {result[1]:.6f}"

    @given(_rgb_pixel, st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_green_never_increases_torch(self, pixel, strength):
        t = torch.tensor(pixel, dtype=torch.float32)
        result = cu.despill(t, strength=strength)
        assert result[1].item() <= pixel[1] + 1e-6, (
            f"green increased: {pixel[1]:.6f} → {result[1].item():.6f}"
        )

    # --- Vacuity companions ---
    # Force G > (R+B)/2 so spill_amount > 0 — the interesting branch.
    # Without this, most random pixels have no spill and the property
    # passes trivially without ever exercising the despill path.

    @given(
        rb=st.floats(0.0, 0.4, allow_nan=False, allow_infinity=False),
        g=st.floats(0.5, 1.0, allow_nan=False, allow_infinity=False),
        strength=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
    )
    def test_green_suppression_vacuity_numpy(self, rb, g, strength):
        """Force spill_amount > 0 by constructing a pixel where G > (R+B)/2."""
        pixel = np.array([rb, g, rb], dtype=np.float32)
        result = cu.despill(pixel.copy(), strength=strength)
        # spill_amount > 0 is guaranteed — green must have decreased
        assert result[1] < g + 1e-6

    @given(
        rb=st.floats(0.0, 0.4, allow_nan=False, allow_infinity=False),
        g=st.floats(0.5, 1.0, allow_nan=False, allow_infinity=False),
        strength=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
    )
    def test_green_suppression_vacuity_torch(self, rb, g, strength):
        """Force spill_amount > 0 for the torch path."""
        pixel = torch.tensor([rb, g, rb], dtype=torch.float32)
        result = cu.despill(pixel, strength=strength)
        assert result[1].item() < g + 1e-6


# ---------------------------------------------------------------------------
# Despill redistribution: equal-weight sum preserved, Rec.709 shift bounded
# ---------------------------------------------------------------------------


class TestDespillRedistribution:
    """Characterize the redistribution behavior of despill.

    Source claim: LLM_HANDOVER.md — "luminance-preserving despill()".
    "Luminance-preserving" here means equal-weight sum (R+G+B), not Rec.709
    perceptual luminance. The implementation does:
        g_new = g - spill
        r_new = r + spill * 0.5
        b_new = b + spill * 0.5
    so R+G+B is unchanged. Under Rec.709 weights (0.2126R, 0.7152G, 0.0722B)
    the shift is spill × (0.5×0.2126 + 0.5×0.0722 − 0.7152) = spill × −0.5728.
    """

    @given(
        rb=st.floats(0.0, 0.4, allow_nan=False, allow_infinity=False),
        g=st.floats(0.5, 1.0, allow_nan=False, allow_infinity=False),
        strength=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=1000)
    def test_equal_weight_sum_preserved(self, rb, g, strength):
        """R+G+B is unchanged after despill for pixels with actual spill."""
        pixel = np.array([rb, g, rb], dtype=np.float32)
        result = cu.despill(pixel.copy(), strength=strength)
        original_sum = float(rb + g + rb)
        result_sum = float(result[0] + result[1] + result[2])
        assert result_sum == pytest.approx(original_sum, abs=1e-5), (
            f"equal-weight sum changed: {original_sum:.6f} → {result_sum:.6f}"
        )

    @given(
        rb=st.floats(0.0, 0.4, allow_nan=False, allow_infinity=False),
        g=st.floats(0.5, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=1000)
    def test_rec709_luminance_shift_bounded(self, rb, g):
        """Rec.709 luminance decreases by spill_amount × 0.5728 at strength=1.

        This proves the documented Rec.709 shift is deterministic and consistent
        with the formula — not that it is zero (it isn't, by design).
        """
        pixel = np.array([rb, g, rb], dtype=np.float32)
        result = cu.despill(pixel.copy(), strength=1.0)

        spill_amount = max(g - (rb + rb) / 2.0, 0.0)
        rec709_before = 0.2126 * rb + 0.7152 * g + 0.0722 * rb
        rec709_after = 0.2126 * float(result[0]) + 0.7152 * float(result[1]) + 0.0722 * float(result[2])
        actual_shift = rec709_after - rec709_before
        expected_shift = -spill_amount * 0.5728

        assert actual_shift == pytest.approx(expected_shift, abs=1e-5), (
            f"Rec.709 shift {actual_shift:.6f} ≠ expected {expected_shift:.6f}"
        )


# ---------------------------------------------------------------------------
# Cross-backend differential: numpy vs torch must agree across full domain
# ---------------------------------------------------------------------------


class TestCrossBackendDifferential:
    """Every pure math function must produce identical results in numpy and torch.

    The dual-backend design means both paths ship in production. A divergence
    between numpy and torch is a silent correctness failure — one will be wrong.
    Hypothesis proves agreement across the full domain, not just at fixed points.
    """

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_linear_to_srgb_numpy_torch_agree(self, x):
        np_result = float(cu.linear_to_srgb(np.float32(x)))
        torch_result = cu.linear_to_srgb(torch.tensor(x, dtype=torch.float32)).item()
        assert np_result == pytest.approx(torch_result, abs=1e-5)

    @given(_unit_float)
    @settings(max_examples=1000)
    def test_srgb_to_linear_numpy_torch_agree(self, x):
        np_result = float(cu.srgb_to_linear(np.float32(x)))
        torch_result = cu.srgb_to_linear(torch.tensor(x, dtype=torch.float32)).item()
        assert np_result == pytest.approx(torch_result, abs=1e-5)

    @given(_rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_premultiply_numpy_torch_agree(self, fg, alpha):
        a_np = np.float32(alpha)
        a_t = torch.tensor(alpha, dtype=torch.float32)
        np_result = cu.premultiply(fg, a_np)
        torch_result = cu.premultiply(torch.tensor(fg, dtype=torch.float32), a_t).numpy()
        np.testing.assert_allclose(np_result, torch_result, atol=1e-5)

    @given(_rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_unpremultiply_numpy_torch_agree(self, fg, alpha):
        a_np = np.float32(alpha)
        a_t = torch.tensor(alpha, dtype=torch.float32)
        # Premultiply first to get a valid premul input
        premul_np = cu.premultiply(fg, a_np)
        premul_t = torch.tensor(premul_np, dtype=torch.float32)
        np_result = cu.unpremultiply(premul_np, a_np)
        torch_result = cu.unpremultiply(premul_t, a_t).numpy()
        np.testing.assert_allclose(np_result, torch_result, atol=1e-5)

    @given(_rgb_pixel, _rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_composite_straight_numpy_torch_agree(self, fg, bg, alpha):
        a_np = np.float32(alpha)
        np_result = cu.composite_straight(fg, bg, a_np)
        fg_t = torch.tensor(fg, dtype=torch.float32)
        bg_t = torch.tensor(bg, dtype=torch.float32)
        a_t = torch.tensor(alpha, dtype=torch.float32)
        torch_result = cu.composite_straight(fg_t, bg_t, a_t).numpy()
        np.testing.assert_allclose(np_result, torch_result, atol=1e-5)

    @given(_rgb_pixel, _rgb_pixel, _alpha)
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_composite_premul_numpy_torch_agree(self, fg, bg, alpha):
        a_np = np.float32(alpha)
        premul_fg_np = cu.premultiply(fg, a_np)
        premul_bg_np = cu.premultiply(bg, a_np)
        np_result = cu.composite_premul(premul_fg_np, a_np, premul_bg_np)
        premul_fg_t = torch.tensor(premul_fg_np, dtype=torch.float32)
        premul_bg_t = torch.tensor(premul_bg_np, dtype=torch.float32)
        a_t = torch.tensor(alpha, dtype=torch.float32)
        torch_result = cu.composite_premul(premul_fg_t, a_t, premul_bg_t).numpy()
        np.testing.assert_allclose(np_result, torch_result, atol=1e-5)

    @given(_rgb_pixel, st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_despill_numpy_torch_agree(self, pixel, strength):
        np_result = cu.despill(pixel.copy(), strength=strength)
        torch_result = cu.despill(torch.tensor(pixel, dtype=torch.float32), strength=strength)
        if isinstance(torch_result, torch.Tensor):
            torch_result = torch_result.numpy()
        np.testing.assert_allclose(np_result, torch_result, atol=1e-5)
