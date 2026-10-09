#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example batch inference script for CV3-Eval.

Generates audio for all entries in the specified CV3-Eval zero-shot
subsets using ECCD.  Evaluation is not included here; use the generated
audio with a separate evaluator.

Only zero-shot tasks are covered:
  - zero_shot               (multilingual zero-shot voice cloning)
  - zero_shot_clone_10speak (10-speaker zero-shot voice cloning)

Usage:
  python run_cv3_eval_batch.py
  # Adjust the configuration section below before running.
"""

import os
import sys
import gc
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.')

from datetime import datetime

import torch

from eccd_decoding import run_batch_eccd_inference_cv3eval


class Tee:
    """Redirect stdout to both terminal and a log file."""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# =============================================================================
# Configuration - adjust before running
# =============================================================================

# --- CV3-Eval zero-shot tasks ---
# Each entry: (cv3_task, cv3_lang, eval_lang)
#   - cv3_task:  CV3-Eval task subdir (e.g. zero_shot, zero_shot_clone_10speak)
#   - cv3_lang:  language subdir under the task (e.g. zh, en, hard_zh, ja, ...)
#   - eval_lang: language used by ASR for WER computation during evaluation.
CV3_DATASETS = [
    # Standard zero-shot languages (500 entries each)
    ("zero_shot", "en", "en"),
    ("zero_shot", "zh", "zh"),
    ("zero_shot", "ja", "ja"),
    ("zero_shot", "ko", "ko"),
    # Languages supported by CosyVoice3 but not CosyVoice2
    ("zero_shot", "fr", "fr"),
    ("zero_shot", "de", "de"),
    ("zero_shot", "it", "it"),
    ("zero_shot", "es", "es"),
    ("zero_shot", "ru", "ru"),
    # Hard subsets (Chinese 60 / English 64 entries)
    ("zero_shot", "hard_zh", "hard_zh"),
    ("zero_shot", "hard_en", "hard_en"),
]

CV2_SUPPORTED_LANG = ['en', 'zh', 'ja', 'ko']

# --- Model ---
model_name = "CosyVoice3"            # "CosyVoice2" | "CosyVoice3"

MODEL_DIRS = {
    "CosyVoice2": "pretrained_models/CosyVoice2-0.5B",
    "CosyVoice3": "pretrained_models/Fun-CosyVoice3-0.5B",
}

# --- Experiment ---
exp_name = "eccd_alpha1_k25"        # experiment name (used as sub-folder)

# --- ECCD parameters ---
k = 25                 # plausible set size |V_head|
alpha = 1.0             # alignment enhancement strength
variant = 'eccd'        # 'eccd' | 'vanilla_cd' | 'expert_anchor' | 'positive_only' | 'eccd_I'
temperature = 1.0       # softmax temperature for ECCD scores (1.0 = no sharpening)

# --- Sampling parameters ---
sampling = 25           # top-k for nucleus sampling
strict_rep = False      # True = strict repetition penalty
rep_win_size = 10       # window size for strict repetition penalty
random_sample = False   # True = multinomial sampling; False = nucleus sampling
expert_only = False     # True = restrict V_head to tokens where p_E > p_A

# --- Generation length ---
max_token_text_ratio = 20
min_token_text_ratio = 2

# --- Save per-step stats for mechanism analysis ---
save_stats = True


# =============================================================================
# Derived paths
# =============================================================================

CV3_EVAL_ROOT = "/data/local/training_data/CV3-Eval"
ECCD_OUTPUT_ROOT = "/data/local/training_data/Constrative_Decoding/ECCD/evaluation_cv3eval/"


def build_paths(cv3_task, cv3_lang):
    """Build all paths for a given (cv3_task, cv3_lang) pair."""
    cv3_data_dir = os.path.join(CV3_EVAL_ROOT, "data", cv3_task, cv3_lang)
    eval_target_path = os.path.join(ECCD_OUTPUT_ROOT, model_name)
    gen_audio_save_path = os.path.join(
        eval_target_path, exp_name, cv3_task, cv3_lang, "gen_audio")
    stats_save_path = os.path.join(
        eval_target_path, exp_name, cv3_task, cv3_lang, "gen_stats")
    return cv3_data_dir, gen_audio_save_path, stats_save_path


# =============================================================================
# Main
# =============================================================================

def run_single_dataset(cv3_task, cv3_lang, eval_lang):
    """Run ECCD generation for a single CV3-Eval subset."""
    # CosyVoice2 only supports zh/en/ja/ko; skip unsupported languages.
    if model_name == "CosyVoice2":
        lang_suffix = cv3_lang[-2:]
        if lang_suffix not in CV2_SUPPORTED_LANG:
            print(f"CosyVoice2 doesn't support lang {lang_suffix.upper()}, SKIP.")
            return

    cv3_data_dir, gen_audio_save_path, stats_save_path = build_paths(cv3_task, cv3_lang)

    os.makedirs(stats_save_path, exist_ok=True)
    log_filename = os.path.join(
        stats_save_path, f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_filename)
    sys.stdout = tee

    print(f"{'='*60}")
    print(f"ECCD Inference Pipeline (CV3-Eval)")
    print(f"  Model:      {model_name}")
    print(f"  CV3 Task:   {cv3_task}")
    print(f"  CV3 Lang:   {cv3_lang}")
    print(f"  Eval Lang:  {eval_lang}  (used for ASR/WER)")
    print(f"  Experiment: {exp_name}")
    print(f"  k={k}, alpha={alpha}, variant={variant}, temperature={temperature}")
    print(f"  sampling={sampling}, strict_rep={strict_rep}, expert_only={expert_only}")
    print(f"  random_sample={random_sample}")
    print(f"{'='*60}")

    os.makedirs(gen_audio_save_path, exist_ok=True)

    generated, failed = run_batch_eccd_inference_cv3eval(
        model_dir=MODEL_DIRS[model_name],
        model_name=model_name,
        cv3_data_dir=cv3_data_dir,
        output_dir=gen_audio_save_path,
        sampling=sampling,
        k=k, alpha=alpha, variant=variant, temperature=temperature,
        strict_rep=strict_rep, rep_win_size=rep_win_size,
        expert_only=expert_only, random_sample=random_sample,
        max_token_text_ratio=max_token_text_ratio,
        min_token_text_ratio=min_token_text_ratio,
        stats_dir=stats_save_path if save_stats else None,
    )

    print(f"\nGeneration summary: {len(generated)} succeeded, {len(failed)} failed")
    print(f"\n{'='*60}")
    print(f"CV3-Eval [{cv3_task}/{cv3_lang}] complete!")
    print(f"  Audio:  {gen_audio_save_path}")
    print(f"  Stats:  {stats_save_path}")
    print(f"{'='*60}")

    # Explicit cleanup before the next subset.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.stdout = tee.stdout
    tee.close()


def main():
    overall_start = datetime.now()
    print(f"{'#'*60}")
    print(f"# Multi-Subset ECCD Inference (CV3-Eval)")
    print(f"# Model:      {model_name}")
    print(f"# Datasets:   {len(CV3_DATASETS)}")
    print(f"# Experiment: {exp_name}")
    print(f"{'#'*60}\n")

    results_summary = []
    for i, (cv3_task, cv3_lang, eval_lang) in enumerate(CV3_DATASETS, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(CV3_DATASETS)}] Starting: {cv3_task}/{cv3_lang} (eval_lang={eval_lang})")
        print(f"{'#'*60}")
        run_single_dataset(cv3_task, cv3_lang, eval_lang)
        results_summary.append(f"{cv3_task}/{cv3_lang}")

    overall_end = datetime.now()
    print(f"\n{'#'*60}")
    print(f"# All CV3-Eval subsets complete!")
    print(f"# Total time: {overall_end - overall_start}")
    print(f"# Subsets run: {', '.join(results_summary)}")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()
