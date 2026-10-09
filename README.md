# Early Access of ECCD Code on CosyVoice Model

This repository contains the inference code accompanying the paper:

> *Experience-Calibrated Contrastive Decoding for Mitigating Hallucinations in LM-Based Text-to-Speech*

---

## Files

| File | Description |
|------|-------------|
| `eccd_decoding.py` | Core ECCD module: scoring function, inference classes, and batch generation functions for both SeedTTS-Eval and CV3-Eval. |
| `run_seedtts_eval_batch.py` | Example batch inference script for SeedTTS-Eval (`test-zh`, `test-en`, `test-hard`). |
| `run_cv3_eval_batch.py` | Example batch inference script for CV3-Eval (multilingual zero-shot). |
| `eccd_cosyvoice_environment.yml` | Exported conda environment listing the Python version and all conda/pip dependencies used to run the reported experiments. |

---

## Method Overview

ECCD is a training-free decoding method that contrasts two predictions from the same speech LM:

- **Expert distribution** `p_E(x_i) = p(x_i | x_<i, a, T_p, T_t)` Full-information prediction with all text conditions.
- **Amateur distribution** `p_A(x_i) = p(x_i | x_<i, a)` Text-ablated prediction that serves as an *experience proxy*.

At each step, the **plausible set** `V_head = TopK_k(p_E)` restricts where the alignment contrast is applied. The ECCD next-token score (Eq. 6 in the paper) is:

```
S_ECCD(x_i) = log p_E(x_i) + alpha * (1 - C_i) * [ log(p_E / p_A) ]_+
                for x_i in V_head
            = log p_E(x_i)
                for x_i not in V_head
```

where:

- `alpha` — alignment enhancement strength.
- `C_i = sum_{x in V_head} p_A(x)` — **Experience Compatibility Coefficient** (ECC), bounded to [0, 1]. It measures how much the experience proxy supports the expert-defined plausible set.
- `[ · ]_+ = max(0, ·)` — positive-only enhancement.

Two design mechanisms:

1. **Experience preservation** — `log p_E` is retained as an expert anchor so the alignment contrast *modifies* rather than *replaces* the full-information score.
2. **Experience calibration** — `(1 - C_i)` adapts the enhancement strength: when the experience proxy already supports the plausible set (`C_i` large), the correction is attenuated to preserve the existing information balance; when it does not (`C_i` small), the correction is strengthened.

### Analysis metrics (recorded in per-step stats)

| Field | Paper symbol | Description |
|-------|-------------|-------------|
| `alignment_influence` | `I_i` | `D_KL(p_E ‖ p_A)`, distribution-level alignment influence (Eq. 4). |
| `alignment_gain` | `G_i` | `log p_E(x̂) − log p_A(x̂)`, decision-level alignment gain for the selected token (Eq. 5). |
| `experience_compatibility` | `C_i` | `sum_{x in V_head} p_A(x)`, Experience Compatibility Coefficient (Eq. 7). |
| `plausible_set_size` | `|V_head|` | Number of candidates in the plausible set. |
| `affected_token_num` | — | Tokens in `V_head` where `p_E > p_A`. |
| `expert_mass_in_plausible_set` | — | `sum_{x in V_head} p_E(x)`. |
| `topn_overlap_ratio` | — | Top-n overlap between expert and amateur distributions. |
| `total_variation_distance` | — | `0.5 * sum |p_E − p_A|`. |
| `abs_alignment_gain` | — | `p_E(x̂) − p_A(x̂)` for the selected token. |
| `selected_token_id` | `x̂_i` | The token selected at this step. |

---

## Ablation Variants

The `variant` parameter in `eccd_next_token_score()` and the batch functions controls the scoring formulation. This corresponds to the component ablation in Table 3 of the paper.

| `variant` | Score within `V_head` | Score outside | Description |
|-----------|----------------------|---------------|-------------|
| `vanilla_cd` | `log(p_E / p_A)` | `−∞` | Vanilla contrastive decoding (Eq. 5). |
| `expert_anchor` | `log p_E + log(p_E / p_A)` | `log p_E` | + Experience preservation (expert anchor). |
| `positive_only` | `log p_E + [log(p_E / p_A)]_+` | `log p_E` | + Positive-only enhancement. |
| `eccd` | `log p_E + α(1−C_i)[log(p_E/p_A)]_+` | `log p_E` | + `C_i` calibration (default, full ECCD, Eq. 6). |
| `eccd_I` | `log p_E + α·I_i·[log(p_E/p_A)]_+` | `log p_E` | `I_i` calibration (ECCD_I variant). |

---

## Requirements

This code runs on top of the CosyVoice codebase. It imports from:

- `cosyvoice.cli.cosyvoice.AutoModel`
- `cosyvoice.utils.common.nucleus_sampling`
- `modelscope.snapshot_download`

Optional dependency for Japanese with CosyVoice3:

- `pykakasi` — converts Japanese kanji/hiragana to space-separated katakana.

The code supports both **CosyVoice2** and **CosyVoice3**. No modifications to the CosyVoice model code are needed.

### Environment

`eccd_cosyvoice_environment.yml` is a conda environment export captured from the runtime used for the reported experiments. To recreate an equivalent environment:

```bash
conda env create -f eccd_cosyvoice_environment.yml
conda activate cosyvoice
```

Note that some packages (e.g., `ttsfrd`) are internal frontend dependencies of CosyVoice and may not be available on public package indices; the CosyVoice codebase should be installed first following its own instructions.

---

## Usage

### 1. SeedTTS-Eval batch inference

Edit the configuration section in `run_seedtts_eval_batch.py`:

```python
model_name = "CosyVoice2"       # or "CosyVoice3"
exp_name  = "eccd_alpha1_k25"

use_eccd  = True
k         = 25                  # plausible set size
alpha     = 1.0                  # enhancement strength
variant   = 'eccd'              # ablation variant
```

Then run:

```bash
cd /path/to/CosyVoice
python ECCD_CODE_APPENDIX/run_seedtts_eval_batch.py
```

For the **native decoding** baseline, set `use_eccd = False`. For the **lower-temperature** baseline, set `use_eccd = False` and
`lt_tau = 0.875` (or `0.75`).

### 2. CV3-Eval batch inference

Edit the configuration section in `run_cv3_eval_batch.py`:

```python
model_name = "CosyVoice3"
exp_name  = "eccd_alpha1_k25"

k         = 25
alpha     = 1.0
variant   = 'eccd'
```

Then run:

```bash
python ECCD_CODE_APPENDIX/run_cv3_eval_batch.py
```

CosyVoice2 only supports `zh`, `en`, `ja`, `ko`; unsupported languages are skipped automatically.

---

## Key Parameters

| Parameter | Default | Paper symbol | Description |
|-----------|---------|--------------|-------------|
| `k` | `25` | `k` | Plausible set size `|V_head|`. |
| `alpha` | `1.0` | `α` | Alignment enhancement strength. |
| `variant` | `'eccd'` | — | Ablation variant (see table above). |
| `temperature` | `1.0` | — | Softmax temperature for ECCD scores (1.0 = no sharpening). |
| `sampling` | `25` | — | Top-k for nucleus sampling. |
| `strict_rep` | `False` | — | Strict repetition penalty (ban recent tokens). |
| `expert_only` | `False` | — | Restrict `V_head` to tokens where `p_E > p_A`. |
| `random_sample` | `False` | — | Multinomial sampling instead of nucleus. |
| `max_token_text_ratio` | `20` | — | Maximum generation length / text length. |
| `min_token_text_ratio` | `2` | — | Minimum generation length / text length. |

The paper uses `alpha = 1, k = 25` as the fixed default across all experiments (see Section 4.1, "Implementation details").

---

## Output Structure

```
<ECCD_OUTPUT_ROOT>/<model_name>/<exp_name>/<lang>/<manifest>/
├── gen_audio/
│   ├── gen_<prompt_wav>_to_<utt_id>.wav      # SeedTTS-Eval naming
│   └── ...
└── gen_stats/
    ├── gen_<prompt_wav>_to_<utt_id>_stats.json  # per-step stats
    └── console_log_<timestamp>.txt
```

For CV3-Eval, generated audio is named `{uttid}.wav` and stats are named `{uttid}_stats.json`, matching the CV3-Eval evaluation convention.

---

## Per-Step Stats JSON Format

When `save_stats = True`, each sample's per-step statistics are saved as a JSON list. Each element corresponds to one decoding step:

```json
[
  {
    "plausible_set_size": 25,
    "affected_token_num": 18,
    "experience_compatibility": 0.73,
    "expert_mass_in_plausible_set": 0.95,
    "topn_overlap_ratio": 0.72,
    "alignment_influence": 0.88,
    "total_variation_distance": 0.42,
    "alignment_gain": 1.24,
    "abs_alignment_gain": 0.15,
    "selected_token_id": 4023
  },
  ...
]
```

These stats support the mechanism analysis in Section 4.2 of the paper, including the temporal profiles of `I_i` and `G_i`, the first-error onset comparison, and the complete-region averages.

## Acknowledgements

We gratefully acknowledge the following open-source projects that this work builds upon:

- [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)

We thank all authors for their contributions to the community.

## Citation

If you find this work useful, please cite our paper:

```tex
@article{liu2026experience,
  title={Experience-Calibrated Contrastive Decoding for Mitigating Hallucinations in LM-Based Text-to-Speech},
  author={Liu, Chenlin and Fang, Minghui and Bi, Zhonghao and Su, Zekai and Wang, Rong and Han, Jiqing},
  journal={arXiv preprint arXiv:2608.00722},
  year={2026}
}
```

## Contact

For questions or collaborations, feel free to reach out via email: [chenlin.liu@stu.hit.edu.cn](mailto:chenlin.liu@stu.hit.edu.cn).
