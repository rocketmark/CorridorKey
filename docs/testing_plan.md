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

The fixed-example tests cover specific known values well. Their limitation is coverage: a fixed test at `alpha=0.6` proves nothing about `alpha=0.6000001`. For pure math functions with continuous input domains, property-based testing with Hypothesis closes this gap by generating thousands of inputs automatically.

The functions that benefit from PBT are the four pure math functions in `color_utils.py`: `linear_to_srgb`/`srgb_to_linear`, `premultiply`/`unpremultiply`, `composite_straight`/`composite_premul`, and `despill`. The structural tests (`clean_matte`, `dilate_mask`, `apply_garbage_matte`, `rgb_to_yuv`) test discrete behavior against known specs — fixed examples are the right tool there.

---

## What to Add

### 1. Fix the existing EXR pipeline test — it proves the wrong thing 

**Source claim:** `LLM_HANDOVER.md`, Critical Dataflow Properties, with an explicit Bug History note:
> *"Do not apply a pure mathematical Gamma 2.2 curve; use the piecewise real sRGB transfer functions defined in `color_utils.py`."*

**Was:** `test_processed_is_linear_premul_rgba` pinned the expected value to `0.6**2.2 * 0.8` — the gamma 2.2 approximation — with `atol=1e-2`. The gap between gamma 2.2 and piecewise sRGB at `FG=0.6` is `~0.005`, well within that tolerance. A regression back to `x**2.2` would have passed undetected.

**Fixed:** Expected value now calls `cu.srgb_to_linear(np.float32(0.6)) * 0.8` with `atol=1e-4`. Submitted upstream as PR against `nikopueringer/CorridorKey`.

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

---

### 4. Prove the IEC 61966-2-1 sRGB spec holds across the full input domain

**Source claim:** `color_utils.py` docstrings:
> *"Converts Linear to sRGB using the official piecewise sRGB transfer function."*
> *"Converts sRGB to Linear using the official piecewise sRGB transfer function."*

**Current limitation of fixed tests:** The existing tests sample 8 specific values and 2 breakpoints. The piecewise spec has invariants that must hold everywhere in [0, 1].

**Tests to write (`test_pbt_color_math.py`) using Hypothesis:**

#### Properties

- **Roundtrip identity over full domain** — `@given(st.floats(0.0, 1.0))`: `srgb_to_linear(linear_to_srgb(x)) ≈ x` and `linear_to_srgb(srgb_to_linear(x)) ≈ x` within float32 tolerance.
- **Strict monotonicity** — for any `a < b` in [0, 1], `linear_to_srgb(a) < linear_to_srgb(b)`. A non-monotonic transfer function silently corrupts color ordering.
- **Range preservation** — output of both functions stays in [0, 1] for any input in [0, 1].
- **Compositing output stays in [0, 1]** — `@given(fg, bg, alpha all in [0,1])`: `composite_straight(fg, bg, alpha)` output is in [0, 1]. This is the invariant compositing software depends on.
- **Premult roundtrip over full domain** — `@given(fg in [0,1]^3, alpha in [0.01, 1.0])`: `unpremultiply(premultiply(fg, alpha), alpha) ≈ fg`. Alpha excluded near zero because color is undefined there.
- **Despill never increases green, never produces negative green** — `@given(image in [0,1]^3, strength in [0,1])`: `result[1] <= input[1]` and `result[1] >= 0.0`.

#### Property vacuity tests

Each property above needs a companion test that forces the interesting case to actually fire — proving the property isn't passing vacuously because the relevant branch is never reached.

Without this, a naive Hypothesis strategy for despill will generate inputs where `G < (R+B)/2` most of the time (spill_amount is always zero), so "never increases green" passes trivially without ever exercising the spill path. The vacuity companion uses a targeted strategy that forces the interesting case:

```python
# Vacuity: force spill_amount > 0 by constructing inputs where G > limit
@given(
    rb=st.floats(0.0, 0.4),
    g=st.floats(0.5, 1.0),   # forces G > (R+B)/2
    strength=st.floats(0.0, 1.0),
)
def test_despill_green_suppression_vacuity(rb, g, strength):
    """Verify the green-suppression property fires on actual spill pixels."""
    img = np.array([[rb, g, rb]], dtype=np.float32)
    result = despill(img, strength=strength)
    # spill_amount > 0 is guaranteed by construction — assert it actually ran
    assert result[0, 1] < g or strength == 0.0
```

Write a vacuity companion for each property in `test_pbt_color_math.py`.

#### Cross-backend differential tests (numpy vs torch)

Every pure math function in `color_utils.py` has both a numpy and a torch path. The existing tests check parity at a handful of specific inputs. Hypothesis proves the two backends agree across the full domain:

```python
@given(st.floats(0.0, 1.0))
def test_srgb_numpy_torch_agree(x):
    np_result = float(cu.linear_to_srgb(np.float32(x)))
    torch_result = cu.linear_to_srgb(torch.tensor(x, dtype=torch.float32)).item()
    assert np_result == pytest.approx(torch_result, abs=1e-5)
```

Write cross-backend differential properties for: `linear_to_srgb`, `srgb_to_linear`, `premultiply`, `unpremultiply`, `composite_straight`, `composite_premul`, and `despill`.

---

### 5. Prove `clean_matte` behaves as documented at its boundaries

**Source claim:** `README.md`, Features:
> *"Auto-Cleanup: Includes a morphological cleanup system to automatically prune any tracking markers or tiny background features that slip through CorridorKey's detection."*

**Current gap:** The existing tests prove it works in the middle of its operating range (large blob kept, small blob removed). The boundaries are untested.

**Tests to add (`test_color_utils.py`):**
- **All-zero matte:** `clean_matte(np.zeros(...))` returns all zeros without crashing. The connected-components pass on an empty matte must not produce spurious output.
- **All-opaque matte:** `clean_matte(np.ones(...))` preserves the region (the single large component is above any reasonable threshold).
- **Idempotency:** Run `clean_matte` on an already-clean output; the second pass should equal the first within float tolerance. If cleanup is not stable, repeated passes degrade mattes in a batch pipeline. **Finding:** The dilation+blur post-processing is intentionally not idempotent — each pass expands the surviving blob's feathered edge further. Idempotency holds only for the connected-components core (`dilation=0, blur_size=0`). The test uses those parameters and the docstring records the finding.

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

| # | Test | Tied to Documented Claim | Effort | Done |
|---|---|---|---|---|
| 1 | Fix `test_processed_is_linear_premul_rgba` to use `srgb_to_linear` — `0.6**2.2` was the disallowed gamma 2.2 approximation; `atol=1e-2` was too loose to catch a regression. Tightened to `srgb_to_linear()` + `atol=1e-4`. PR submitted upstream. | LLM_HANDOVER.md Bug History | Trivial | ✓ |
| 2 | Resolution independence shape tests — verify all four outputs match input spatial dimensions at multiple resolutions including non-square and degenerate cases. | README "Resolution Independent" | Low | ✓ |
| 3 | Core Hypothesis properties + vacuity companions — sRGB roundtrip, strict monotonicity, range preservation, compositing output in [0,1], premult roundtrip. Each property paired with a targeted vacuity strategy that forces the interesting branch to fire. | `color_utils.py` IEC spec + docstrings | Medium | ✓ |
| 4 | Despill properties + vacuity + cross-backend differential — green never increases, never negative, equal-weight sum preserved, Rec.709 shift bounded at `spill × 0.5728`. Vacuity companion forces `G > limit`. Cross-backend differential (numpy vs torch) for all 7 pure math functions across full domain. | LLM_HANDOVER.md "luminance-preserving" + dual-backend design | Medium | ✓ |
| 5 | `clean_matte` boundary conditions — all-zero input, all-opaque input, idempotency on already-clean output. | README "Auto-Cleanup" | Low | ✓ |

All items are GPU-free.

---

## Findings

Behaviors discovered while writing tests that are not documented upstream.

| Finding | Where | Description | Upstream action |
|---|---|---|---|
| `clean_matte` is not idempotent at default settings | `color_utils.py:250`, `test_color_utils.py::TestCleanMatte::test_idempotent` | The dilation + Gaussian blur post-processing expands surviving blobs' feathered edges on every call — there is no fixed point. Running cleanup twice on the same matte produces slightly different output. The connected-components core (`dilation=0, blur_size=0`) IS idempotent. Not a bug for the primary use case (one call per frame in batch), but a latent trap for refinement loops. No docstring warning exists. | PR submitted — docstring addition to `clean_matte` in `color_utils.py` |
| `dilate_mask` numpy and torch backends use structurally different kernels | `color_utils.py:146`, `tests/test_dilate_backend_consistency.py` | numpy uses `cv2.dilate` with `MORPH_ELLIPSE`; torch uses `max_pool2d` (square kernel). The square strictly contains the ellipse, so torch dilation ≥ numpy at every pixel. At radius=15, torch activates 961 pixels vs numpy's 729 — a 32% difference matching the theoretical `4/π` ratio for square vs inscribed circle. Not documented anywhere in the codebase. | Pending — considering a docstring addition to `dilate_mask` in `color_utils.py` |

---

## Test Organization

```
tests/
  test_color_utils.py              # existing — add clean_matte boundary tests (item 8)
  test_inference_engine.py         # existing — EXR pipeline test fixed (item 1)
  test_pbt_color_math.py           # NEW — Hypothesis properties (items 3–7):
                                   #   - sRGB roundtrip, monotonicity, range
                                   #   - compositing range, premult roundtrip
                                   #   - despill contracts + redistribution characterization
                                   #   - vacuity companions for each property
                                   #   - numpy vs torch differential properties
  test_inference_assembly.py       # NEW — resolution independence shape tests (item 2)
  test_gamma_consistency.py        # existing — keep as-is, intentional design documented
  test_dilate_backend_consistency.py  # NEW — documents ellipse (numpy) vs square (torch) kernel divergence
  test_pbt_backend_resolution.py   # existing — keep as-is
  test_pbt_dep_preservation.py     # existing — keep as-is
```

All tests run without GPU or model weights.
