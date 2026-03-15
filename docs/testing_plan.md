# CorridorKey Testing Plan

## Guiding Principle

The goal is not to find gaps to fill — it is to prove that the explicit design decisions and documented properties in this codebase actually hold. The developers have made deliberate choices (the gamma inconsistency in third-party paths, the MPS fallback behavior) that are already documented and tested. We are not here to second-guess those. We are here to write tests that confirm the things the documentation *claims* are true.

---

## Test Targets

Three sources contain explicit correctness claims:

- **`LLM_HANDOVER.md`** — documents the EXR pipeline, the `srgb_to_linear` requirement (with a Bug History note), and calls out `despill()` as "luminance-preserving"
- **`README.md`** — states "Resolution Independent" as a product feature bullet
- **`color_utils.py` docstrings** — describe the IEC 61966-2-1 piecewise sRGB spec, compositing formulas, and alpha semantics

Each item in the Work Log below maps directly to one of those claims.

---

## Test Organization

```
tests/
  test_color_utils.py                 # fixed-example unit tests for color_utils
  test_inference_engine.py            # pipeline integration tests (mocked model)
  test_pbt_color_math.py              # Hypothesis properties:
                                      #   - sRGB roundtrip, monotonicity, range
                                      #   - compositing range, premult roundtrip
                                      #   - despill contracts + redistribution characterization
                                      #   - vacuity companions for each property
                                      #   - numpy vs torch differential properties
  test_inference_assembly.py          # resolution independence shape tests
  test_gamma_consistency.py           # documents intentional gamma 2.2 divergence in third-party paths
  test_dilate_backend_consistency.py  # documents ellipse (numpy) vs square (torch) kernel divergence
  test_pbt_backend_resolution.py      # property-based backend resolution priority tests
  test_pbt_dep_preservation.py        # property-based dependency preservation tests
  test_clip_manager.py                # clip discovery and asset validation
```

All tests run without GPU or model weights.

---

## Current Coverage

`test_color_utils.py` and `test_inference_engine.py` together cover the existing fixed-example tests well:

| Source | Test Class | What It Proves |
|---|---|---|
| `test_color_utils.py` | `TestSrgbLinearConversion` | Roundtrip at fixed values, breakpoint continuity, negative clamping |
| `test_color_utils.py` | `TestPremultiply` | Roundtrip, α=0 → zero, α=1 → identity |
| `test_color_utils.py` | `TestCompositing` | Straight/premul equivalence, α=0/1 boundary cases |
| `test_color_utils.py` | `TestDespill` | Mode behavior, strength=0 noop, partial green, numpy/torch parity |
| `test_color_utils.py` | `TestCleanMatte` | Large blob kept, small blob removed, 3D shape preserved, boundary conditions |
| `test_color_utils.py` | `TestDilateMask` | radius=0 noop, expansion, shape preservation |
| `test_color_utils.py` | `TestApplyGarbageMatte` | None passthrough, zeros outside region, broadcast |
| `test_color_utils.py` | `TestRgbToYuv` | Rec.601 known values, three layout branches |
| `test_inference_engine.py` | `TestProcessFrameOutputs` | Output keys, shapes, dtype, alpha/FG in [0,1] |
| `test_inference_engine.py` | `TestProcessFrameColorSpace` | sRGB/linear input paths, uint8 normalization, model called once |
| `test_inference_engine.py` | `TestProcessFramePostProcessing` | Despill toggle, despeckle toggle, processed is 4-channel RGBA |
| `test_gamma_consistency.py` | `TestGammaInconsistency` | Documents intentional gamma 2.2 divergence in third-party paths |

The fixed-example tests cover specific known values well. Their limitation is coverage: a fixed test at `alpha=0.6` proves nothing about `alpha=0.6000001`. For pure math functions with continuous input domains, property-based testing with Hypothesis closes this gap by generating thousands of inputs automatically.

The functions that benefit from PBT are the four pure math functions in `color_utils.py`: `linear_to_srgb`/`srgb_to_linear`, `premultiply`/`unpremultiply`, `composite_straight`/`composite_premul`, and `despill`. The structural tests (`clean_matte`, `dilate_mask`, `apply_garbage_matte`, `rgb_to_yuv`) test discrete behavior against known specs — fixed examples are the right tool there.

---

## What We Are Not Testing (and Why)

| Item | Reason |
|---|---|
| Gamma 2.2 in VideoMaMa / GVM paths | Explicitly documented in `test_gamma_consistency.py` as intentional. Those models were likely trained on gamma 2.2 data. `test_gamma_consistency.py` already proves the inconsistency exists and quantifies it. |
| MPS vs CUDA vs CPU numerical parity | `README.md` explicitly documents that MPS has unsupported ops and may silently fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK=1`. Device-dependent behavior is a known, managed reality. A strict parity test would assert a guarantee the developers have not made. |
| Temporal consistency across frames | Requires GPU + reference footage. The README notes this is a new release and doesn't claim frame-to-frame consistency as a tested property. |
| Alpha hint inversion / statistical soundness | Requires GPU inference on real data. The README explicitly says *"Naturally, I have not tested everything."* These are aspirational; not properties the documentation claims to guarantee today. |

---

## Work Log

All items are GPU-free and complete.

| # | Work | Tied to Documented Claim | Upstream action |
|---|---|---|---|
| 1 | Fix `test_processed_is_linear_premul_rgba` — expected value used `0.6**2.2 * 0.8` (gamma 2.2 approximation) with `atol=1e-2`. The gap between gamma 2.2 and piecewise sRGB at `FG=0.6` is `~0.005`, well within that tolerance — a regression back to `x**2.2` would have passed undetected. Tightened to `cu.srgb_to_linear(np.float32(0.6)) * 0.8` + `atol=1e-4`. | `LLM_HANDOVER.md` Bug History: *"Do not apply a pure mathematical Gamma 2.2 curve."* | PR submitted upstream |
| 2 | Resolution independence shape tests (`test_inference_assembly.py`) — verify all four outputs match input spatial dimensions at multiple resolutions: `64×128`, `300×200` (non-square, non-power-of-two), `1×1` (degenerate). Tests the resize-up → Lanczos4 resize-down path without a GPU. | `README.md`: *"Resolution Independent: The engine dynamically scales inference to handle 4K plates."* | New file, fork only |
| 3 | Core Hypothesis properties + vacuity companions (`test_pbt_color_math.py`) — sRGB roundtrip, strict monotonicity, range preservation, compositing output in [0,1], premult roundtrip. Each property paired with a targeted vacuity strategy that forces the interesting branch to fire. | `color_utils.py` IEC 61966-2-1 spec + docstrings | New file, fork only |
| 4 | Despill properties + vacuity + cross-backend differential (`test_pbt_color_math.py`) — green never increases, never negative, equal-weight sum preserved, Rec.709 shift bounded at `spill × 0.5728`. Vacuity companion forces `G > limit`. Cross-backend differential (numpy vs torch) for all 7 pure math functions across full domain. | `LLM_HANDOVER.md`: *"luminance-preserving `despill()`"* + dual-backend design | New tests, fork only |
| 5 | `clean_matte` boundary conditions (`test_color_utils.py`) — all-zero input, all-opaque input, idempotency on connected-components core (`dilation=0, blur_size=0`). | `README.md`: *"Auto-Cleanup: morphological cleanup system."* | PR submitted upstream |
| 6 | `dilate_mask` backend consistency (`test_dilate_backend_consistency.py`) — documents and quantifies ellipse (numpy/cv2) vs square (torch/max_pool2d) kernel divergence. Proves torch is a strict superset, divergence scales with radius, and is ~32% at radius=15 (matching theoretical `4/π` ratio). | Behavioral finding — not documented upstream | New file, fork only |
| 7 | Fix `test_despill_strength_variants_dont_crash` — the fixture returned uniform gray (R=G=B=0.6), so `spill_amount=0` always and the despill branch never ran. The test also contradicted itself: docstring claimed results should differ, assertion checked they were equal. Replaced with inline green-heavy mock (R=0.2, G=0.8, B=0.2) and directional assertion on the green channel mean. | Vacuous test audit | PR submitted upstream as nikopueringer/CorridorKey#179 |

---

## Findings

Behaviors discovered while writing tests that are not documented upstream.

| Finding | Where | Description | Upstream action |
|---|---|---|---|
| `clean_matte` is not idempotent at default settings | `color_utils.py:250`, `test_color_utils.py::TestCleanMatte::test_idempotent` | The dilation + Gaussian blur post-processing expands surviving blobs' feathered edges on every call — there is no fixed point. Running cleanup twice on the same matte produces slightly different output. The connected-components core (`dilation=0, blur_size=0`) IS idempotent. Not a bug for the primary use case (one call per frame in batch), but a latent trap for refinement loops. No docstring warning exists. | PR submitted — docstring addition to `clean_matte` in `color_utils.py` |
| `dilate_mask` numpy and torch backends use structurally different kernels | `color_utils.py:146`, `tests/test_dilate_backend_consistency.py` | numpy uses `cv2.dilate` with `MORPH_ELLIPSE`; torch uses `max_pool2d` (square kernel). The square strictly contains the ellipse, so torch dilation ≥ numpy at every pixel. At radius=15, torch activates 961 pixels vs numpy's 729 — a 32% difference matching the theoretical `4/π` ratio for square vs inscribed circle. Not documented anywhere in the codebase. | Pending — considering a docstring addition to `dilate_mask` in `color_utils.py` |
| `test_despill_strength_variants_dont_crash` was vacuous and internally contradictory | `tests/test_inference_engine.py::TestProcessFramePostProcessing` | `mock_greenformer` returns uniform gray (R=G=B=0.6), so `spill_amount=0` always — the despill branch never ran. The docstring claimed results should differ at strength 0.0 vs 1.0, but the assertion checked they were equal. Both passed whether despill was working or deleted. | Fixed — inline green-heavy mock forces `spill_amount > 0`; directional assertion on green channel mean. PR submitted upstream as nikopueringer/CorridorKey#179. |
