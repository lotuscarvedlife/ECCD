#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example batch inference script for SeedTTS-Eval.

Generates audio for all entries in the specified SeedTTS-Eval test sets
(test-zh, test-en, test-hard) using ECCD.  Evaluation is not included here;
use the generated audio with a separate evaluator.

Usage:
  python run_seedtts_eval_batch.py
  # Adjust the configuration section below before running.
"""

import os
import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.')

from datetime import datetime

from eccd_decoding import run_batch_eccd_inference


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

# --- Datasets ---
# Each entry: (lang, manifest_fn)
#   - lang:        used for ASR language selection during evaluation.
#   - manifest_fn: SeedTTS-Eval meta.lst filename under EVAL_ROOT/<lang>/.
DATASETS = [
    ("zh", "hardcase.lst"),
    # ("zh", "meta.lst"),
    # ("en", "meta.lst"),
]

# --- Model ---
model_name = "CosyVoice2"            # "CosyVoice2" | "CosyVoice3"

MODEL_DIRS = {
    "CosyVoice2": "pretrained_models/CosyVoice2-0.5B",
    "CosyVoice3": "pretrained_models/Fun-CosyVoice3-0.5B",
}

# --- Experiment ---
exp_name = "eccd_alpha1_k25"        # experiment name (used as sub-folder)
# exp_name = "native_baseline"
# exp_name = "lt_tau0.875"

# --- Decoding method ---
# Set to True for ECCD, False for native decoding (or lower-temperature baseline).
use_eccd = True

# --- ECCD parameters ---
k = 25                 # plausible set size |V_head|
alpha = 1.0             # alignment enhancement strength
variant = 'eccd'        # 'eccd' | 'vanilla_cd' | 'expert_anchor' | 'positive_only' | 'eccd_I'
temperature = 1.0       # softmax temperature for ECCD scores (1.0 = no sharpening)

# --- Native / lower-temperature baseline parameters ---
lt_tau = 1.0            # lower-temperature baseline: set < 1.0 (e.g. 0.75 or 0.875)

# --- Sampling parameters ---
sampling = 25           # top-k for nucleus sampling
strict_rep = False      # True = strict repetition penalty (ban recent tokens)
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

EVAL_ROOT = "/data/local/training_data/seedtts_testset/"
ECCD_OUTPUT_ROOT = "/data/local/training_data/Constrative_Decoding/ECCD/evaluation/"


def build_paths(lang, manifest_fn):
    """Build all paths for a given (lang, manifest_fn) pair."""
    evaluation_path = os.path.join(EVAL_ROOT, lang)
    eval_target_path = os.path.join(ECCD_OUTPUT_ROOT, model_name)
    manifest_path = os.path.join(evaluation_path, manifest_fn)
    gen_audio_save_path = os.path.join(
        eval_target_path, exp_name, lang, manifest_fn[:-4], "gen_audio")
    stats_save_path = os.path.join(
        eval_target_path, exp_name, lang, manifest_fn[:-4], "gen_stats")
    return manifest_path, gen_audio_save_path, stats_save_path


# =============================================================================
# Main
# =============================================================================

def run_single_dataset(lang, manifest_fn):
    """Run ECCD (or native) generation for a single (lang, manifest_fn) pair."""
    manifest_path, gen_audio_save_path, stats_save_path = build_paths(lang, manifest_fn)

    os.makedirs(stats_save_path, exist_ok=True)
    log_filename = os.path.join(
        stats_save_path, f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_filename)
    sys.stdout = tee

    print(f"{'='*60}")
    print(f"ECCD Inference Pipeline (SeedTTS-Eval)")
    print(f"  Model:      {model_name}")
    print(f"  Dataset:    {manifest_fn} ({lang})")
    print(f"  Experiment: {exp_name}")
    print(f"  Method:     {'ECCD' if use_eccd else 'Native' + (' (LT tau={})'.format(lt_tau) if lt_tau != 1.0 else '')}")
    if use_eccd:
        print(f"  k={k}, alpha={alpha}, variant={variant}, temperature={temperature}")
    print(f"  sampling={sampling}, strict_rep={strict_rep}, expert_only={expert_only}")
    print(f"  random_sample={random_sample}")
    print(f"{'='*60}")

    os.makedirs(gen_audio_save_path, exist_ok=True)

    if use_eccd:
        generated, failed = run_batch_eccd_inference(
            model_dir=MODEL_DIRS[model_name],
            model_name=model_name,
            meta_lst_path=manifest_path,
            output_dir=gen_audio_save_path,
            sampling=sampling,
            k=k, alpha=alpha, variant=variant, temperature=temperature,
            strict_rep=strict_rep, rep_win_size=rep_win_size,
            expert_only=expert_only, random_sample=random_sample,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            stats_dir=stats_save_path if save_stats else None,
        )
    else:
        # Native decoding or lower-temperature baseline.
        from eccd_decoding import run_batch_native_inference
        generated, failed = run_batch_native_inference(
            model_dir=MODEL_DIRS[model_name],
            model_name=model_name,
            meta_lst_path=manifest_path,
            output_dir=gen_audio_save_path,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            temperature=lt_tau,
        )

    print(f"\nGeneration summary: {len(generated)} succeeded, {len(failed)} failed")
    print(f"\n{'='*60}")
    print(f"Dataset [{lang}/{manifest_fn}] complete!")
    print(f"  Audio:  {gen_audio_save_path}")
    print(f"  Stats:  {stats_save_path}")
    print(f"{'='*60}")

    sys.stdout = tee.stdout
    tee.close()


def main():
    overall_start = datetime.now()
    print(f"{'#'*60}")
    print(f"# Multi-Dataset ECCD Inference (SeedTTS-Eval)")
    print(f"# Model:      {model_name}")
    print(f"# Datasets:   {len(DATASETS)}")
    print(f"# Experiment: {exp_name}")
    print(f"{'#'*60}\n")

    results_summary = []
    for i, (lang, manifest_fn) in enumerate(DATASETS, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(DATASETS)}] Starting: {lang}/{manifest_fn}")
        print(f"{'#'*60}")
        run_single_dataset(lang, manifest_fn)
        results_summary.append(f"{lang}/{manifest_fn}")

    overall_end = datetime.now()
    print(f"\n{'#'*60}")
    print(f"# All datasets complete!")
    print(f"# Total time: {overall_end - overall_start}")
    print(f"# Datasets run: {', '.join(results_summary)}")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()
