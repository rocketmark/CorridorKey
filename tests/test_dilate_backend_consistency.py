"""Tests documenting the kernel-shape inconsistency between dilate_mask backends.

STATUS: This test documents a KNOWN INCONSISTENCY, not desired behavior.

dilate_mask supports two backends that use structurally different kernels:

1. **Numpy / OpenCV** — cv2.dilate with MORPH_ELLIPSE
   An elliptical structuring element inscribed in the (2r+1)×(2r+1) bounding
   box.  At radius=5 the kernel is a 11×11 ellipse; corner pixels are excluded.

2. **PyTorch** — torch.nn.functional.max_pool2d
   A square (2r+1)×(2r+1) sliding-window maximum.  All pixels in the bounding
   box are included; there is no elliptical masking.

Because the square strictly contains the ellipse, the torch backend always
produces dilation >= the numpy backend at every pixel.  The divergence is
most visible at blob edges and grows with radius: at radius=10, the four
corners of the torch result extend ~2 pixels beyond the numpy result.

**Why this may be intentional:**
The numpy path is used for matte post-processing (clean_matte, apply_garbage_matte)
where the elliptical shape produces a visually smoother edge.  The torch path is
used inside the model inference loop where max_pool2d is the natural, GPU-friendly
primitive.  Changing either would alter matte edge behavior.

**If you unify the backends**, these tests will tell you something changed.
Remove or update them accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from CorridorKeyModule.core import color_utils as cu

# ---------------------------------------------------------------------------
# Document the divergence
# ---------------------------------------------------------------------------


class TestDilateMaskBackendInconsistency:
    """Assert that the two backends produce DIFFERENT results at non-trivial
    radii — documenting the inconsistency so it isn't accidentally removed.
    """

    def test_backends_differ_at_blob_edge(self):
        """numpy (ellipse) and torch (square) must produce different output.

        A single bright pixel in a black image — at radius=5 the two kernels
        produce visibly different shapes.  If this test fails, the backends
        have been unified; remove this test.
        """
        mask_np = np.zeros((32, 32), dtype=np.float32)
        mask_np[16, 16] = 1.0
        mask_t = torch.zeros(32, 32, dtype=torch.float32)
        mask_t[16, 16] = 1.0

        np_result = cu.dilate_mask(mask_np, radius=5)
        torch_result = cu.dilate_mask(mask_t, radius=5).numpy()

        assert not np.allclose(np_result, torch_result), (
            "numpy and torch dilate_mask now produce identical output — "
            "has the kernel inconsistency been fixed? If so, remove this test."
        )

    def test_torch_is_superset_of_numpy(self):
        """torch (square) output >= numpy (ellipse) output at every pixel.

        The square kernel contains the ellipse, so max_pool2d activates
        at every pixel where cv2.dilate activates, plus the corner pixels
        that the ellipse excludes.  This is a structural property, not a
        numerical tolerance.
        """
        mask_np = np.zeros((64, 64), dtype=np.float32)
        mask_np[32, 32] = 1.0
        mask_t = torch.zeros(64, 64, dtype=torch.float32)
        mask_t[32, 32] = 1.0

        np_result = cu.dilate_mask(mask_np, radius=7)
        torch_result = cu.dilate_mask(mask_t, radius=7).numpy()

        assert np.all(torch_result >= np_result - 1e-6), (
            "torch result is not a superset of numpy result — "
            "the square-vs-ellipse relationship has changed."
        )

    def test_backends_agree_at_radius_zero(self):
        """radius=0 is a noop in both backends — they must agree exactly."""
        mask_np = np.random.default_rng(42).random((32, 32)).astype(np.float32)
        mask_t = torch.tensor(mask_np)

        np_result = cu.dilate_mask(mask_np.copy(), radius=0)
        torch_result = cu.dilate_mask(mask_t, radius=0)

        np.testing.assert_array_equal(np_result, mask_np)
        np.testing.assert_array_equal(torch_result.numpy(), mask_np)


# ---------------------------------------------------------------------------
# Quantify the divergence
# ---------------------------------------------------------------------------


class TestDilateMaskDivergenceMagnitude:
    """Quantify how many pixels differ between backends at various radii.

    These tests serve as documentation — if someone changes the dilation
    implementation, these show exactly where the drift happens and by how much.
    """

    @pytest.mark.parametrize(
        "radius,expected_min_differing_pixels",
        [
            # At small radii the ellipse and square are very similar —
            # only the 4 corner pixels of the bounding box differ.
            (1, 0),   # 3×3: ellipse == square (both fill the 3×3 box)
            (2, 1),   # 5×5: ellipse cuts 4 corners, square keeps them
            (5, 10),  # 11×11: corner exclusion grows with radius
            (10, 40), # 21×21: significant corner exclusion
        ],
    )
    def test_divergence_scales_with_radius(self, radius, expected_min_differing_pixels):
        """More pixels should differ as radius grows."""
        mask_np = np.zeros((64, 64), dtype=np.float32)
        mask_np[32, 32] = 1.0
        mask_t = torch.zeros(64, 64, dtype=torch.float32)
        mask_t[32, 32] = 1.0

        np_result = cu.dilate_mask(mask_np, radius=radius)
        torch_result = cu.dilate_mask(mask_t, radius=radius).numpy()

        differing = int(np.sum(np.abs(torch_result - np_result) > 1e-6))
        assert differing >= expected_min_differing_pixels, (
            f"radius={radius}: expected >= {expected_min_differing_pixels} differing pixels, "
            f"got {differing}. The backends may have been unified — check the implementation."
        )

    def test_worst_case_divergence_is_bounded(self):
        """The pixel-count divergence at large radius stays within a known range.

        Informational — documents the scale of the backend difference so
        reviewers can judge whether it matters for their use case.
        """
        mask_np = np.zeros((128, 128), dtype=np.float32)
        mask_np[64, 64] = 1.0
        mask_t = torch.zeros(128, 128, dtype=torch.float32)
        mask_t[64, 64] = 1.0

        np_result = cu.dilate_mask(mask_np, radius=15)
        torch_result = cu.dilate_mask(mask_t, radius=15).numpy()

        differing = int(np.sum(np.abs(torch_result - np_result) > 1e-6))

        # At radius=15 (31×31 kernel) the square activates 961 pixels; the
        # inscribed ellipse activates ~729 pixels.  The corner exclusion is
        # ~232 pixels — about 32% of the numpy (ellipse) area.  This matches
        # the theoretical ratio: square / ellipse = 4/π ≈ 1.27 at large radii.
        total_activated = int(np.sum(np_result > 1e-6))
        assert differing > 0, "Expected non-zero divergence at radius=15"
        assert 0.20 < differing / total_activated < 0.50, (
            f"Divergence ({differing} px) is {differing/total_activated:.1%} of dilated area "
            f"({total_activated} px) — outside the expected 20–50% band for ellipse vs square."
        )
