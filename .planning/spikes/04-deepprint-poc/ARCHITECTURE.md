# DeepPrint Pre-trained Model — Architecture Spec

Source: `best_model.pyt` (TF-Hub-style zip containing a PyTorch state_dict)
downloaded from the Google Drive referenced in the upstream
`tim-rohwedder/fixed-length-fingerprint-extractors` README.

## Identified model

**`DeepPrint_TexMinu_512`** — texture + minutia branches, total
embedding 512-D (256 texture + 256 minutia).

**Note**: the upstream filename suggested `DeepPrint_Tex_512` (texture
only) but the state_dict contains BOTH branches. The texture branch
outputs 256-D (not 512) and the minutia branch outputs 256-D, so the
combined embedding is 512-D. This is more useful for our use case
because the minutia branch carries spatial info we can decode back
to perito-friendly minutiae.

## Input

- **Grayscale, 1 channel**
- **Size 299×299**
- **Normalization**: `mean=[0.5], std=[0.5]` (single channel, [-1, 1])
  (NOT ImageNet RGB normalization)

## Architecture (from state_dict key analysis)

### Stem (`stem.features.0..5`)

```
features.0: BasicConv2d(1, 32, k=3, s=2)
features.1: BasicConv2d(32, 32, k=3, s=1)
features.2: BasicConv2d(32, 64, k=3, s=1, p=1)
features.3: Mixed_3a     (branch0 + branch1, 3 sub-branches)
features.4: Mixed_4a     (branch0 2 sub + branch1 4 sub)
features.5: Mixed_5a     (single conv 320 channels, k=1)
```

Output: 384 channels, reduced spatial dim.

### Texture branch (`texture_branch._0_block`..`_2_block`, `_6_linear`)

```
_0_block: 4 Inception_A + 1 Reduction_A
_1_block: 7 Inception_B + 1 Reduction_B
_2_block: 3 Inception_C
_3_avg_pool2d: kernel=8
_4_flatten
_5_dropout(0.2)
_6_linear: 1536 → 256
L2-normalize(dim=1)
```

Output: **256-D L2-normalized**.

### Minutia stem (`minutia_stem.features.0..5`)

```
features.0..5: 6 Inception_A blocks (no Reduction between)
```

Output: 384 channels, same spatial dim as stem output.

### Minutia embedding (`minutia_embedding._0_block`, `_4_linear`)

```
_0_block: 4 conv layers (384→768, 768→768, 768→896, 896→1024, each with stride/padding)
_1_max_pool2d: kernel=9
_2_flatten
_3_dropout(0.2)
_4_linear: 1024 → 256
L2-normalize(dim=1)
```

Output: **256-D L2-normalized**.

### Minutia map (`minutia_map.features.0..3`)

```
features.0: ConvTranspose2d(384, 128, k=3, s=2)
features.1: Conv2d(128, 128, k=7, s=1)
features.2: ConvTranspose2d(128, 32, k=3, s=2)
features.3: Conv2d(32, 6, k=3, s=1)
```

Output: 6-channel map, spatial upsampled. **Crops to 128×128** by
removing last row/col (`[:, :, :-1, :-1]`).

The 6 channels are likely: 3 for minutia probability (ending, bifurcation, other)
× 2 for orientation (sin, cos) — or similar 6-d representation.
(This is the auxiliary task; we use it for the perito's
visualizable minutiae in the spike.)

### Heads (training-only, ignore at inference)

```
texture_logits.0: Linear(256, 8000) + Dropout(0.2)
minutia_logits.0: Linear(256, 8000)
```

`8000` = number of SFinGe synthetic subjects in the training set.
We **never** use these at inference; we keep them in the model
class only so `load_state_dict(strict=True)` passes.

## Key naming convention

The state_dict uses **`branch0` / `branch1`** (no underscore) for the
branches inside Inception blocks, AND **`branch1_0`, `branch1_1a`,
`branch1_1b`, `branch2_0`..`branch2_3a`, `branch2_3b`** for the
sub-branches inside Inception_C blocks.

This is **different from the tim-rohwedder upstream code** which uses
`branch_0`, `branch_1` (with underscore). The state_dict is the
ground truth — our code must match it.

## Total parameter count

- Stem: 66 params
- Texture branch: 830 params
- Minutia stem: 252 params
- Minutia embedding: 10 params
- Minutia map: 8 params
- Heads: 4 params (2 + 2)
- **Total: ~1170 tensors**

File size: 875 MB (float32 weights, large because of all the BN
running_mean / running_var tensors).

## Inference path

```python
# At inference (eval mode):
x = stem(input)                # (B, 384, 35, 35)
tex = texture_branch(x)         # (B, 256), L2-normalized
minu = minutia_stem(x)         # (B, 384, 35, 35)
minu_emb = minutia_embedding(minu)  # (B, 256), L2-normalized
minu_map = minutia_map(minu)   # (B, 6, 128, 128), for perito UI
embedding = concat([tex, minu_emb], dim=1)  # (B, 512)
```

For texture-only POC, use just `tex` (256-D). For minutia decoding
(to show to the perito), use `minu_map` (the 6-channel heatmap).
