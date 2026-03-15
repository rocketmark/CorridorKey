# CorridorKey Testing Plan

## Guiding Principle

The goal is not to find gaps to fill — it is to prove that the explicit design decisions and documented properties in this codebase actually hold. The developers have made deliberate choices (the gamma inconsistency in third-party paths, the MPS fallback behavior) that are already documented and tested. We are not here to second-guess those. We are here to write tests that confirm the things the documentation *claims* are true.

---

## What the Documentation Claims (Our Test Targets)

Three sources contain explicit correctness claims:

- **`LLM_HANDOVER.md`** — documents the EXR pipeline, the `srgb_to_linear` requirement (with a Bug History note), and calls out `despill()` as "luminance-preserving"
- **`README.md`** — states "Resolution Independent" as a product feature bullet
- **`color_utils.py` docstrings** — describe the IEC 61966-2-1 piecewise sRGB spec, compositing formulas, and alpha semantics

Each of the items below maps directly to one of those claims.

---

## Current Coverage

`test_color_utils.py` and `test_inference_engine.py` together cover the existing fixed-example tests well:

| Source | Test Class | What It Proves |
|---|---|---|
| `test_color_utils.py` | `TestSrgbLinearConversion` | Roundtrip at fixed values, breakpoint continuity, negative clamping |
| `test_color_utils.py` | `TestPremultiply` | Roundtrip, α=0 → zero, α=1 → identity |
| `test_color_utils.py` | `TestCompositing` | Straight/premul equivalence, α=0/1 boundary cases |
| `test_color_utils.py` | `TestDespill` | Mode behavior, strength=0 noop, partial green, numpy/torch parity |
| `test_color_utils.py` | `TestCleanMatte` | Large blob kept, small blob removed, 3D shape preserved |
| `test_color_utils.py` | `TestDilateMask` | radius=0 noop, expansion, shape preservation |
| `test_color_utils.py` | `TestApplyGarbageMatte` | None passthrough, zeros outside region, broadcast |
| `test_color_utils.py` | `TestRgbToYuv` | Rec.601 known values, three layout branches |
| `test_inference_engine.py` | `TestProcessFrameOutputs` | Output keys, shapes, dtype, alpha/FG in [0,1] |
| `test_inference_engine.py` | `TestProcessFrameColorSpace` | sRGB/linear input paths, uint8 normalization, model called once |
| `test_inference_engine.py` | `TestProcessFramePostProcessing` | Despill toggle, despeckle toggle, processed is 4-channel RGBA |
| `test_gamma_consistency.py` | `TestGammaInconsistency` | Documents intentional gamma 2.2 divergence in third-party paths |

---

## What to Add

### 1. Fix the existing EXR pipeline test — it proves the wrong thing

**Source claim:** `LLM_HANDOVER.md`, Critical Dataflow Properties, with an explicit Bug History note:
> *"Do not apply a pure mathematical Gamma 2.2 curve; use the piecewise real sRGB transfer functions defined in `color_utils.py`."*

**Current state:** `test_processed_is_linear_premul_rgba` pins the expected value to `0.6**2.2 * 0.8` — which is the gamma 2.2 approximation the handover doc explicitly says not to use. The test is asserting the wrong thing.

**Fix:** Replace the hardcoded constant with a call through `srgb_to_linear` directly:

```python
expected_rgb = cu.srgb_to_linear(np.float32(0.6)) * 0.8
np.testing.assert_allclose(rgb, expected_rgb, atol=1e-4)
```

This proves the documented chain — `srgb_to_linear → premultiply → concat` — is correctly wired in the actual pipeline, not just that the output is numerically close to a manually computed approximation. If someone replaces `srgb_to_linear` with `x**2.2` in inference_engine.py, this test will catch it. The current version will not.

**File:** `tests/test_inference_engine.py` (edit existing test, not a new file)

---

### 2. Prove "Resolution Independent" holds

**Source claim:** `README.md`, Features section:
> *"Resolution Independent: The engine dynamically scales inference to handle 4K plates while predicting using its native 2048×2048 high-fidelity backbone."*

**What to prove:** Given the same mock model, `process_frame` must return outputs whose spatial dimensions exactly match the input — at multiple resolutions, without transposing or padding incorrectly.

**Tests to write (`test_inference_assembly.py`):**
- Feed a `64×128` input, verify all four outputs are `64×128`
- Feed a `300×200` input (non-square, non-power-of-two), verify all four outputs are `300×200`
- Feed a `1×1` input (degenerate case), verify it doesn't crash and returns `1×1`

These test the resize-up → Lanczos4 resize-down path without a GPU. The mock model always returns a `2048×2048` tensor regardless of actual content; the pipeline is responsible for getting the dimensions right.

---

### 3. Prove the despill redistribution behavior (characterize, don't assert a false invariant)

**Source claim:** `LLM_HANDOVER.md`:
> *"Pure math functions for digital compositing. Crucial: Pay attention to... luminance-preserving `despill()`."*

**What "luminance-preserving" means here:** The implementation transfers removed green energy equally to R and B (`r += spill*0.5`, `b += spill*0.5`). This preserves the *equal-weight sum* `R+G+B`, not Rec.709 perceptual luminance. Under Rec.709 weights (0.2126 R, 0.7152 G, 0.0722 B), the shift is `spill × −0.5728` — luminance decreases when green is removed.

**Tests to write (`test_pbt_color_math.py`):**
- **Equal-weight sum is preserved:** For any pixel where despill runs, `R_new + G_new + B_new == R + G + B` within float tolerance. This is what the implementation actually guarantees.
- **Rec.709 luminance shift is documented and bounded:** Assert that the shift magnitude is `≈ spill_amount × 0.5728`, within a small tolerance. This proves the behavior is deterministic and consistent with the formula, not that it's zero.

This approach proves what the code actually does rather than testing against a false invariant. If someone later changes the redistribution weights, these tests will catch it.

---

### 4. Prove the IEC 61966-2-1 sRGB spec holds across the full input domain

**Source claim:** `color_utils.py` docstrings:
> *"Converts Linear to sRGB using the official piecewise sRGB transfer function."*
> *"Converts sRGB to Linear using the official piecewise sRGB transfer function."*

**Current gap:** The existing tests use fixed examples. The piecewise spec has invariants that must hold everywhere in [0, 1], not just at the sampled points.

**Tests to write (`test_pbt_color_math.py`) using Hypothesis:**

- **Roundtrip identity over full domain** — `@given(st.floats(0.0, 1.0))`: `srgb_to_linear(linear_to_srgb(x)) ≈ x` and `linear_to_srgb(srgb_to_linear(x)) ≈ x` within float32 tolerance. Proves the functions are true inverses, not just approximately so at a few points.
- **Strict monotonicity** — for any `a < b` in [0, 1], `linear_to_srgb(a) < linear_to_srgb(b)`. A non-monotonic transfer function would silently corrupt color ordering and is one of the failure modes a curve inversion introduces.
- **Range preservation** — output of `linear_to_srgb` and `srgb_to_linear` stays in [0, 1] for any input in [0, 1].

Also cover the compositing and premult functions:
- **Compositing output stays in [0, 1]** — `@given(fg, bg, alpha all in [0,1])`: `composite_straight(fg, bg, alpha)` output is in [0, 1]. This is the downstream invariant that compositing software depends on.
- **Premult roundtrip over full domain** — `@given(fg in [0,1]^3, alpha in [0.01, 1.0])`: `unpremultiply(premultiply(fg, alpha), alpha) ≈ fg`. Alpha is excluded near zero because color is undefined there.
- **Despill never increases green, never produces negative green** — `@given(image in [0,1]^3, strength in [0,1])`: `result[1] <= input[1]` and `result[1] >= 0.0`. These are the two unconditional contracts the despill operation makes.

---

### 5. Prove `clean_matte` behaves as documented at its boundaries

**Source claim:** `README.md`, Features:
> *"Auto-Cleanup: Includes a morphological cleanup system to automatically prune any tracking markers or tiny background features that slip through CorridorKey's detection."*

**Current gap:** The existing tests prove it works in the middle of its operating range (large blob kept, small blob removed). The boundaries are untested.

**Tests to add (`test_color_utils.py`):**
- **All-zero matte:** `clean_matte(np.zeros(...))` returns all zeros without crashing. The connected-components pass on an empty matte must not produce spurious output.
- **All-opaque matte:** `clean_matte(np.ones(...))` preserves the region (the single large component is above any reasonable threshold).
- **Idempotency:** Run `clean_matte` on an already-clean output; the second pass should equal the first within float tolerance. If the cleanup is not stable, repeated passes would degrade mattes in a batch pipeline.

---

## What We Are Not Testing (and Why)

| Item | Reason |
|---|---|
| Gamma 2.2 in VideoMaMa / GVM paths | Explicitly documented in `test_gamma_consistency.py` as intentional. Those models were likely trained on gamma 2.2 data. `test_gamma_consistency.py` already proves the inconsistency exists and quantifies it. |
| MPS vs CUDA vs CPU numerical parity | `README.md` explicitly documents that MPS has unsupported ops and may silently fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK=1`. Device-dependent behavior is a known, managed reality. A strict parity test would assert a guarantee the developers have not made. |
| Temporal consistency across frames | Requires GPU + reference footage. The README notes this is a new release and doesn't claim frame-to-frame consistency as a tested property. |
| Alpha hint inversion / statistical soundness | Requires GPU inference on real data. The README explicitly says *"Naturally, I have not tested everything."* These are aspirational; not properties the documentation claims to guarantee today. |

---

## Priority Order

| Priority | Test | Tied to Documented Claim | Effort |
|---|---|---|---|
| 1 | Fix `test_processed_is_linear_premul_rgba` to use `srgb_to_linear` | LLM_HANDOVER.md Bug History | Trivial (edit one line) |
| 2 | Resolution independence shape tests | README "Resolution Independent" feature | Low |
| 3 | Hypothesis roundtrip + monotonicity for sRGB | `color_utils.py` docstrings cite IEC spec | Medium |
| 4 | Hypothesis compositing range + premult roundtrip + despill invariants | `color_utils.py` docstrings | Medium |
| 5 | Despill redistribution characterization (equal-weight sum, Rec.709 shift) | LLM_HANDOVER.md "luminance-preserving" | Low |
| 6 | `clean_matte` boundary conditions (empty, full, idempotency) | README "Auto-Cleanup" | Low |

Priority 1 is a bug fix in the existing test suite. Priorities 2–6 are new tests proving documented claims. All six are GPU-free.

---

## Test Organization

```
tests/
  test_color_utils.py              # existing — add clean_matte boundary tests
  test_inference_engine.py         # existing — fix test_processed_is_linear_premul_rgba
  test_pbt_color_math.py           # NEW — Hypothesis property tests for sRGB spec,
                                   #        compositing invariants, despill contracts,
                                   #        and despill redistribution characterization
  test_inference_assembly.py       # NEW — resolution independence shape tests
  test_gamma_consistency.py        # existing — keep as-is, intentional design documented
  test_pbt_backend_resolution.py   # existing — keep as-is
  test_pbt_dep_preservation.py     # existing — keep as-is
```

All tests run without GPU or model weights.
