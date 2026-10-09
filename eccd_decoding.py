#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experience-Calibrated Contrastive Decoding (ECCD) for LM-based Text-to-Speech.

Implements the ECCD method described in:
  "Experience-Calibrated Contrastive Decoding for Mitigating Hallucinations
   in LM-Based Text-to-Speech"

Core ideas (see paper Section 3):
  - Expert distribution  p_E(x_i) = p(x_i | x_{<i}, a, T_p, T_t)
        Full-information prediction that combines alignment and experience.
  - Amateur distribution p_A(x_i) = p(x_i | x_{<i}, a)
        Text-ablated prediction that serves as an experience proxy.
  - Plausible set  V_head = TopK_k(p_E)
        Expert-defined candidate set recomputed at every step.
  - Experience Compatibility Coefficient (ECC)
        C_i = sum_{x in V_head} p_A(x)        in [0, 1]
        Set-level measure of how much the experience proxy supports the
        expert-defined plausible set.
  - Alignment influence  I_i = D_KL(p_E || p_A)
        Distribution-level measure of how strongly text conditions reshape
        the candidate distribution (Eq. 4).
  - Alignment gain  G_i = log p_E(x_hat) - log p_A(x_hat)
        Decision-level measure of alignment support for the selected token
        (Eq. 5).

ECCD next-token score (Eq. 6):
    S_ECCD(x_i) = log p_E(x_i) + alpha * (1 - C_i) * [ log(p_E/p_A) ]_+
                  for x_i in V_head
                = log p_E(x_i)
                  for x_i not in V_head

The two design mechanisms are:
  - Experience preservation: log p_E is retained as an expert anchor so the
    alignment contrast modifies rather than replaces the full-information score.
  - Experience calibration:  (1 - C_i) adapts the enhancement strength to the
    set-level compatibility between expert and experience.

Ablation variants are supported via the ``variant`` argument:
  - 'vanilla_cd'   : vanilla contrastive decoding (Eq. 5 in the paper).
  - 'expert_anchor' : + expert anchor (experience preservation).
  - 'positive_only' : + positive-only enhancement.
  - 'eccd'          : + C_i calibration  (default, full ECCD, Eq. 6).
  - 'eccd_I'        : + I_i calibration  (ECCD_I variant).

Supported base models: CosyVoice2 (Qwen2LM) and CosyVoice3 (CosyVoice3LM).
"""

import os
import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.')

import json
import time
import torch
import torch.nn.functional as F
import torchaudio
import uuid as uuid_lib
import traceback
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import nucleus_sampling
from modelscope import snapshot_download
import logging

# -----------------------------------------------------------------------------
# Japanese kana conversion (optional, required by CosyVoice3 for Japanese)
# -----------------------------------------------------------------------------
try:
    import pykakasi
    JAPANESE_CONVERT_AVAILABLE = True
    _kakasi = pykakasi.kakasi()
except ImportError:
    JAPANESE_CONVERT_AVAILABLE = False
    _kakasi = None


def to_katakana_spaced(text):
    """Convert Japanese kanji/hiragana to space-separated katakana.

    CosyVoice3 requires Japanese input to be space-separated katakana.
    """
    result = _kakasi.convert(text)
    return ' '.join([item['kana'] for item in result])


# =============================================================================
# ECCD Scoring
# =============================================================================

def compute_plausible_set(log_p_E, k=25):
    """Build the plausible set V_head as the top-k candidates under p_E.

    Args:
        log_p_E: [vocab_size] expert log-probabilities.
        k:       plausible set size.

    Returns:
        Boolean mask of shape [vocab_size] selecting V_head.
    """
    probs_E = torch.exp(log_p_E)
    _, topk_indices = torch.topk(probs_E, k=k, dim=-1)
    mask_head = torch.zeros_like(probs_E, dtype=torch.bool)
    mask_head[topk_indices] = True
    return mask_head


def compute_alignment_influence(log_p_E, log_p_A):
    """Alignment influence  I_i = D_KL(p_E || p_A)  (Eq. 4).

    Distribution-level measure of how strongly the text conditions reshape
    the candidate distribution relative to the text-ablated reference.
    """
    probs_E = torch.exp(log_p_E)
    return (probs_E * (log_p_E - log_p_A)).sum().item()


def compute_experience_compatibility(log_p_A, mask_head):
    """Experience Compatibility Coefficient  C_i = sum_{x in V_head} p_A(x)
    (Eq. 7).

    Measures how much probability mass the experience proxy assigns to the
    expert-defined plausible set.  Bounded to [0, 1].
    """
    probs_A = torch.exp(log_p_A)
    return probs_A[mask_head].sum().item()


def eccd_next_token_score(
    log_p_E,
    log_p_A,
    k=25,
    alpha=1.0,
    variant='eccd',
    temperature=1.0,
    expert_only=False,
):
    """Compute the ECCD next-token distribution and per-step analysis stats.

    Args:
        log_p_E:     [vocab_size] expert  log-probabilities  (full-information).
        log_p_A:     [vocab_size] amateur log-probabilities  (text-ablated).
        k:           plausible set size |V_head|.
        alpha:       alignment enhancement strength.
        variant:     ablation variant (see module docstring).
        temperature: softmax temperature applied to the final scores.
                     Defaults to 1.0 (no sharpening); the lower-temperature
                     baseline is handled separately in NativeInference.
        expert_only: if True, restrict V_head to tokens where p_E > p_A.
                     Not used by the default ECCD formulation.

    Returns:
        probs_ECCD: [vocab_size] ECCD probability distribution.
        step_stats: dict of per-step statistics for mechanism analysis.
    """
    if log_p_E.dim() == 2:
        log_p_E = log_p_E.squeeze(0)
    if log_p_A.dim() == 2:
        log_p_A = log_p_A.squeeze(0)

    probs_E = torch.exp(log_p_E)
    probs_A = torch.exp(log_p_A)

    # --- Plausible set V_head = TopK_k(p_E) ---
    mask_head = compute_plausible_set(log_p_E, k=k)
    if expert_only:
        mask_head = mask_head & (log_p_E > log_p_A)

    n = mask_head.sum().item()

    # --- Top-n overlap ratio between expert and amateur (supplementary stat) ---
    if n > 0:
        expert_topn = mask_head.nonzero().squeeze().tolist()
        if n == 1:
            expert_topn = [expert_topn]
        amateur_topn = torch.topk(probs_A, k=n).indices.tolist()
        overlap = len(set(expert_topn) & set(amateur_topn))
        topn_overlap_ratio = overlap / n
    else:
        topn_overlap_ratio = 0.0

    # --- Alignment influence  I_i = D_KL(p_E || p_A) ---
    alignment_influence = compute_alignment_influence(log_p_E, log_p_A)

    # --- Experience Compatibility Coefficient  C_i ---
    experience_compatibility = compute_experience_compatibility(log_p_A, mask_head)

    # --- Total Variation Distance (supplementary stat) ---
    total_variation_distance = (0.5 * torch.abs(probs_E - probs_A).sum()).item()

    step_stats = {
        'plausible_set_size': n,
        'affected_token_num': (mask_head & (log_p_E > log_p_A)).sum().item(),
        'experience_compatibility': experience_compatibility,        # C_i
        'expert_mass_in_plausible_set': probs_E[mask_head].sum().item(),
        'topn_overlap_ratio': topn_overlap_ratio,
        'alignment_influence': alignment_influence,                  # I_i
        'total_variation_distance': total_variation_distance,
    }

    # --- Log-ratio  log(p_E / p_A)  over the plausible set ---
    log_ratio = log_p_E[mask_head] - log_p_A[mask_head]

    # --- Apply variant-specific scoring ---
    if variant == 'vanilla_cd':
        # Vanilla CD (Eq. 5):  S = log(p_E/p_A) in V_head,  -inf outside.
        scores = torch.full_like(log_p_E, -float('inf'))
        scores[mask_head] = log_ratio
    elif variant == 'expert_anchor':
        # +Expert anchor:  S = log p_E + log(p_E/p_A) in V_head,  log p_E outside.
        scores = log_p_E.clone()
        scores[mask_head] = log_p_E[mask_head] + log_ratio
    elif variant == 'positive_only':
        # +Positive-only:  S = log p_E + [log(p_E/p_A)]_+ in V_head,  log p_E outside.
        scores = log_p_E.clone()
        scores[mask_head] = log_p_E[mask_head] + torch.clamp(log_ratio, min=0)
    elif variant == 'eccd':
        # Full ECCD (Eq. 6):
        #   S = log p_E + alpha * (1 - C_i) * [log(p_E/p_A)]_+ in V_head,
        #   S = log p_E                                          outside.
        scores = log_p_E.clone()
        enhancement = alpha * (1.0 - experience_compatibility) * log_ratio
        scores[mask_head] = log_p_E[mask_head] + torch.clamp(enhancement, min=0)
    elif variant == 'eccd_I':
        # ECCD_I: replace (1 - C_i) with I_i as the calibration signal.
        scores = log_p_E.clone()
        enhancement = alpha * alignment_influence * log_ratio
        scores[mask_head] = log_p_E[mask_head] + torch.clamp(enhancement, min=0)
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Expected one of: "
            "'vanilla_cd', 'expert_anchor', 'positive_only', 'eccd', 'eccd_I'."
        )

    if temperature != 1.0:
        scores = scores / temperature

    probs_ECCD = F.softmax(scores, dim=-1)
    return probs_ECCD, step_stats


# =============================================================================
# Sampling utilities
# =============================================================================

def strict_ras_sampling(weighted_scores, decoded_tokens, sampling,
                        top_p=0.8, top_k=25, win_size=10):
    """Strict repetition-penalty sampling.

    Unconditionally masks out every token that appeared in the recent window,
    then performs nucleus sampling on the remaining distribution.  Compared
    with the original ``ras_sampling`` (which only re-samples when repetition
    exceeds a threshold), this version bans all recently generated tokens
    before sampling.
    """
    if len(decoded_tokens) > 0:
        recent_tokens = decoded_tokens[-win_size:]
        for token_id in set(recent_tokens):
            weighted_scores[token_id] = -float('inf')
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=sampling)
    return top_ids


# =============================================================================
# ECCD inference (dual-model: expert + text-ablated amateur)
# =============================================================================

class ECCDInference:
    """Experience-Calibrated Contrastive Decoding inference.

    The expert branch receives the full text conditions (T_p + T_t) and the
    amateur branch receives the text-ablated condition.  Both branches share
    the same realized history (teacher-forced), so their predictions differ
    operationally only in the availability of text conditions.
    """

    def __init__(self, expert_lm, amateur_lm, model_type='cosyvoice2',
                 sampling=25, strict_rep=False, rep_win_size=10,
                 expert_only=False, random_sample=False):
        """
        Args:
            expert_lm:   LM with full information     (cosyvoice.model.llm).
            amateur_lm:  LM with text conditions ablated.
            model_type:  'cosyvoice2' or 'cosyvoice3'.
            sampling:    top-k for nucleus sampling.
            strict_rep:  if True, use strict repetition penalty.
            rep_win_size: window size for strict repetition penalty.
            expert_only: if True, restrict V_head to tokens where p_E > p_A.
            random_sample: if True, use multinomial sampling instead of nucleus.
        """
        self.expert_lm = expert_lm
        self.amateur_lm = amateur_lm
        self.model_type = model_type
        self.sampling = sampling
        self.strict_rep = strict_rep
        self.rep_win_size = rep_win_size
        self.expert_only = expert_only
        self.random_sample = random_sample
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_sos_task_emb(self, lm_model):
        """Return (sos_emb, task_id_emb) for the given LM."""
        if lm_model.__class__.__name__ == 'CosyVoice3LM':
            sos_emb = lm_model.speech_embedding.weight[lm_model.sos].reshape(1, 1, -1)
            task_id_emb = lm_model.speech_embedding.weight[lm_model.task_id].reshape(1, 1, -1)
        else:  # Qwen2LM (CosyVoice2)
            sos_emb = lm_model.llm_embedding.weight[lm_model.sos].reshape(1, 1, -1)
            task_id_emb = lm_model.llm_embedding.weight[lm_model.task_id].reshape(1, 1, -1)
        return sos_emb, task_id_emb

    def _build_lm_input(self, lm_model, text_emb, prompt_speech_token):
        """Build the LM input sequence: [sos, text, task_id, prompt_speech]."""
        sos_emb, task_id_emb = self._get_sos_task_emb(lm_model)
        if prompt_speech_token.shape[1] != 0:
            prompt_speech_token_emb = lm_model.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, lm_model.llm_input_size, dtype=text_emb.dtype
            ).to(text_emb.device)
        return torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

    @torch.inference_mode()
    def inference(
        self,
        # Expert inputs (full: prompt_text + target_text)
        expert_text, expert_text_len,
        expert_prompt_text, expert_prompt_text_len,
        # Amateur inputs (text-ablated: no prompt_text content)
        amateur_text, amateur_text_len,
        amateur_prompt_text, amateur_prompt_text_len,
        # Shared prompt speech token (expert), optionally separate for amateur
        prompt_speech_token, prompt_speech_token_len,
        embedding,
        amateur_prompt_speech_token=None,
        amateur_prompt_speech_token_len=None,
        # Generation length ratios
        max_token_text_ratio=20,
        min_token_text_ratio=2,
        # ECCD parameters
        k=25,
        alpha=1.0,
        variant='eccd',
        temperature=1.0,
        # Analysis and latency bookkeeping
        stats=None,
        timing=None,
    ):
        """Run ECCD inference and return the generated token list.

        Args:
            k:           plausible set size |V_head|.
            alpha:       alignment enhancement strength.
            variant:     ablation variant (see module docstring).
            temperature: softmax temperature (default 1.0).
            stats:       if provided (a list), per-step stats dicts are appended
                         in-place for mechanism analysis.
            timing:      if provided (a dict), populated with 'lm_time' and
                         'num_tokens' for latency measurement.

        Returns:
            out_tokens: list of generated token IDs.
        """
        original_text_len = expert_text_len.item()
        min_len = int(original_text_len * min_token_text_ratio)
        max_len = int(original_text_len * max_token_text_ratio)

        # ===== Expert: prompt_text + target_text =====
        expert_text_concat = torch.concat([expert_prompt_text, expert_text], dim=1)
        expert_text_emb = self.expert_lm.llm.model.model.embed_tokens(expert_text_concat)
        if self.expert_lm.__class__.__name__ == 'CosyVoice3LM':
            assert 151646 in expert_text_concat, \
                '<|endofprompt|> not in expert text, check input!'

        # ===== Amateur: text-ablated (no prompt_text content) =====
        if amateur_prompt_text.shape[1] > 0:
            amateur_text_concat = torch.concat([amateur_prompt_text, amateur_text], dim=1)
        else:
            amateur_text_concat = amateur_text
        amateur_text_emb = self.amateur_lm.llm.model.model.embed_tokens(amateur_text_concat)
        if self.amateur_lm.__class__.__name__ == 'CosyVoice3LM':
            assert 151646 in amateur_text_concat, \
                '<|endofprompt|> not in amateur text, check input!'

        if amateur_prompt_speech_token is None:
            amateur_prompt_speech_token = prompt_speech_token

        expert_lm_input = self._build_lm_input(
            self.expert_lm, expert_text_emb, prompt_speech_token
        )
        amateur_lm_input = self._build_lm_input(
            self.amateur_lm, amateur_text_emb, amateur_prompt_speech_token
        )

        print(f"Expert LM input length:   {expert_lm_input.shape[1]}")
        print(f"Amateur LM input length:  {amateur_lm_input.shape[1]}")

        # ===== Autoregressive generation loop =====
        out_tokens = []
        expert_cache = None
        amateur_cache = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lm_start = time.time()

        for i in range(max_len):
            # --- Expert forward: p_E(x_i | x_<i, a, T_p, T_t) ---
            expert_masks = torch.tril(
                torch.ones((1, expert_lm_input.shape[1], expert_lm_input.shape[1]),
                           device=expert_lm_input.device)
            ).to(torch.bool)
            expert_y_pred, expert_cache = self.expert_lm.llm.forward_one_step(
                expert_lm_input, masks=expert_masks, cache=expert_cache
            )
            expert_logp = self.expert_lm.llm_decoder(expert_y_pred[:, -1]).log_softmax(dim=-1)

            # --- Amateur forward: p_A(x_i | x_<i, a) ---
            amateur_masks = torch.tril(
                torch.ones((1, amateur_lm_input.shape[1], amateur_lm_input.shape[1]),
                           device=amateur_lm_input.device)
            ).to(torch.bool)
            amateur_y_pred, amateur_cache = self.amateur_lm.llm.forward_one_step(
                amateur_lm_input, masks=amateur_masks, cache=amateur_cache
            )
            amateur_logp = self.amateur_lm.llm_decoder(amateur_y_pred[:, -1]).log_softmax(dim=-1)

            # --- ECCD next-token score ---
            log_p_E = expert_logp.squeeze(0)    # [vocab_size]
            log_p_A = amateur_logp.squeeze(0)   # [vocab_size]

            probs_ECCD, step_stats = eccd_next_token_score(
                log_p_E, log_p_A,
                k=k, alpha=alpha,
                variant=variant, temperature=temperature,
                expert_only=self.expert_only,
            )
            modified_logp = torch.log(probs_ECCD + 1e-10)

            # --- Token selection ---
            if self.random_sample:
                if i < min_len:
                    modified_logp[self.expert_lm.speech_token_size] = -float('inf')
                top_ids = modified_logp.softmax(dim=-1).multinomial(1).item()
            elif self.strict_rep:
                if i < min_len:
                    modified_logp[self.expert_lm.speech_token_size] = -float('inf')
                top_ids = strict_ras_sampling(
                    modified_logp, out_tokens, self.sampling,
                    win_size=self.rep_win_size,
                )
            else:
                # Native nucleus sampling with conditional re-sampling (ras_sampling).
                top_ids = self.expert_lm.sampling_ids(
                    modified_logp,
                    out_tokens,
                    self.sampling,
                    ignore_eos=True if i < min_len else False,
                )

            # --- Decision-level alignment gain  G_i = log p_E(x_hat) - log p_A(x_hat) ---
            step_stats['alignment_gain'] = (log_p_E[top_ids] - log_p_A[top_ids]).item()
            step_stats['abs_alignment_gain'] = (
                torch.exp(log_p_E[top_ids]) - torch.exp(log_p_A[top_ids])
            ).item()
            step_stats['selected_token_id'] = top_ids

            if stats is not None:
                stats.append(step_stats)

            if top_ids in self.expert_lm.stop_token_ids:
                break

            out_tokens.append(top_ids)

            # --- Teacher-force the selected token into both branches ---
            expert_lm_input = self.expert_lm.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            amateur_lm_input = self.amateur_lm.speech_embedding.weight[top_ids].reshape(1, 1, -1)

            if (i + 1) % 50 == 0:
                print(f"  Step {i + 1}/{max_len}, generated {len(out_tokens)} tokens")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lm_end = time.time()

        if timing is not None:
            timing['lm_time'] = lm_end - lm_start
            timing['num_tokens'] = len(out_tokens)

        return out_tokens


# =============================================================================
# Native decoding baseline (single expert model, no contrastive decoding)
# =============================================================================

class NativeInference:
    """Native single-model TTS inference.

    Mirrors the expert branch of :class:`ECCDInference` but performs only a
    single forward pass per step and samples via the model's native
    ``sampling_ids`` (ras_sampling).  An optional ``temperature`` supports the
    lower-temperature baseline (tau in {0.75, 0.875}) discussed in the paper.
    """

    def __init__(self, expert_lm, model_type='cosyvoice2', sampling=25):
        """
        Args:
            expert_lm:  LM with full information (cosyvoice.model.llm).
            model_type: 'cosyvoice2' or 'cosyvoice3'.
            sampling:   top-k for nucleus sampling.
        """
        self.expert_lm = expert_lm
        self.model_type = model_type
        self.sampling = sampling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_sos_task_emb(self, lm_model):
        if lm_model.__class__.__name__ == 'CosyVoice3LM':
            sos_emb = lm_model.speech_embedding.weight[lm_model.sos].reshape(1, 1, -1)
            task_id_emb = lm_model.speech_embedding.weight[lm_model.task_id].reshape(1, 1, -1)
        else:
            sos_emb = lm_model.llm_embedding.weight[lm_model.sos].reshape(1, 1, -1)
            task_id_emb = lm_model.llm_embedding.weight[lm_model.task_id].reshape(1, 1, -1)
        return sos_emb, task_id_emb

    def _build_lm_input(self, lm_model, text_emb, prompt_speech_token):
        sos_emb, task_id_emb = self._get_sos_task_emb(lm_model)
        if prompt_speech_token.shape[1] != 0:
            prompt_speech_token_emb = lm_model.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, lm_model.llm_input_size, dtype=text_emb.dtype
            ).to(text_emb.device)
        return torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

    @torch.inference_mode()
    def inference(
        self,
        expert_text, expert_text_len,
        expert_prompt_text, expert_prompt_text_len,
        prompt_speech_token, prompt_speech_token_len,
        embedding,
        max_token_text_ratio=20,
        min_token_text_ratio=2,
        temperature=1.0,
        timing=None,
    ):
        """Run native single-model inference.

        Args:
            temperature: logit temperature applied before sampling.
                         ``1.0`` recovers the native decoder; values < 1.0
                         implement the lower-temperature baseline.
            timing:      if provided (a dict), populated with 'lm_time' and
                         'num_tokens'.

        Returns:
            out_tokens: list of generated token IDs.
        """
        original_text_len = expert_text_len.item()
        min_len = int(original_text_len * min_token_text_ratio)
        max_len = int(original_text_len * max_token_text_ratio)

        expert_text_concat = torch.concat([expert_prompt_text, expert_text], dim=1)
        expert_text_emb = self.expert_lm.llm.model.model.embed_tokens(expert_text_concat)
        if self.expert_lm.__class__.__name__ == 'CosyVoice3LM':
            assert 151646 in expert_text_concat, \
                '<|endofprompt|> not in expert text, check input!'

        expert_lm_input = self._build_lm_input(
            self.expert_lm, expert_text_emb, prompt_speech_token
        )
        print(f"Native LM input length: {expert_lm_input.shape[1]}")

        out_tokens = []
        expert_cache = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lm_start = time.time()

        for i in range(max_len):
            expert_masks = torch.tril(
                torch.ones((1, expert_lm_input.shape[1], expert_lm_input.shape[1]),
                           device=expert_lm_input.device)
            ).to(torch.bool)
            expert_y_pred, expert_cache = self.expert_lm.llm.forward_one_step(
                expert_lm_input, masks=expert_masks, cache=expert_cache
            )
            expert_logp = self.expert_lm.llm_decoder(expert_y_pred[:, -1]).log_softmax(dim=-1)

            if temperature != 1.0:
                expert_logp = expert_logp / temperature

            top_ids = self.expert_lm.sampling_ids(
                expert_logp.squeeze(0),
                out_tokens,
                self.sampling,
                ignore_eos=True if i < min_len else False,
            )

            if top_ids in self.expert_lm.stop_token_ids:
                break

            out_tokens.append(top_ids)
            expert_lm_input = self.expert_lm.speech_embedding.weight[top_ids].reshape(1, 1, -1)

            if (i + 1) % 50 == 0:
                print(f"  Step {i + 1}/{max_len}, generated {len(out_tokens)} tokens")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lm_end = time.time()

        if timing is not None:
            timing['lm_time'] = lm_end - lm_start
            timing['num_tokens'] = len(out_tokens)

        return out_tokens


# =============================================================================
# Data loading (SeedTTS-Eval meta.lst format)
# =============================================================================

def load_meta_lst_entry(meta_lst_path, index=0):
    """Load a single entry from a SeedTTS-Eval meta.lst file.

    Format: ``utt_id|prompt_text|prompt_wav|target_text``

    Returns:
        (utt_id, prompt_text, prompt_wav_abspath, target_text)
    """
    meta_dir = os.path.dirname(os.path.abspath(meta_lst_path))
    with open(meta_lst_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if index < 0 or index >= len(lines):
        raise ValueError(f"Index {index} out of range, meta.lst has {len(lines)} entries")
    parts = lines[index].strip().split('|')
    utt_id = parts[0]
    prompt_text = parts[1]
    prompt_wav = os.path.join(meta_dir, parts[2])
    target_text = parts[3] if len(parts) > 3 else ''
    return utt_id, prompt_text, prompt_wav, target_text


# =============================================================================
# Text-condition construction helpers
# =============================================================================

def _build_expert_prompt_text(prompt_text_str, model_type):
    """Expert prompt_text: full text condition (with CV3 instruction prefix)."""
    if model_type == 'cosyvoice3':
        return "You are a helpful assistant.<|endofprompt|>" + prompt_text_str
    return prompt_text_str


def _build_amateur_prompt_text(model_type):
    """Amateur prompt_text: text-ablated (CV3 still needs the instruction prefix)."""
    if model_type == 'cosyvoice3':
        return "You are a helpful assistant.<|endofprompt|>"
    return ""


def _resolve_model_type(model_dir, model_name):
    """Determine model_type ('cosyvoice2' or 'cosyvoice3') from dir/name."""
    if 'CosyVoice3' in model_dir or model_name == 'CosyVoice3':
        return 'cosyvoice3'
    if 'CosyVoice2' in model_dir or model_name == 'CosyVoice2':
        return 'cosyvoice2'
    raise ValueError(f"Unknown model type for {model_dir}")


# =============================================================================
# Batch ECCD inference — SeedTTS-Eval (meta.lst format)
# =============================================================================

def run_batch_eccd_inference(
    model_dir, model_name, meta_lst_path, output_dir,
    sampling=25,
    k=25, alpha=1.0, variant='eccd', temperature=1.0,
    strict_rep=False, rep_win_size=10,
    expert_only=False, random_sample=False,
    max_token_text_ratio=20, min_token_text_ratio=2,
    stats_dir=None,
    skip_existing=True, timings=None, max_samples=None,
):
    """Batch ECCD inference for SeedTTS-Eval meta.lst datasets.

    Two model instances are loaded once and reused across all samples:
      - Expert:  receives prompt_text + target_text + prompt_audio.
      - Amateur: text-ablated (no prompt_text content) + prompt_audio.

    Generated audio naming follows the evaluator convention:
        ``gen_{prompt_wav_noext}_to_{utt_id}.wav``

    Args:
        k:       plausible set size |V_head|.
        alpha:   alignment enhancement strength.
        variant: ablation variant.
        stats_dir: if set, per-step stats are saved as JSON per sample.
        timings:   if provided, per-sample timing dicts are appended in-place.

    Returns:
        (generated_paths, failed_indices)
    """
    print(f"\n{'='*60}")
    print(f"Batch ECCD Inference for {model_name}")
    print(f"{'='*60}")

    if not os.path.exists(model_dir):
        model_dir = snapshot_download(model_dir)
    model_type = _resolve_model_type(model_dir, model_name)

    # --- Load expert and amateur model instances ---
    print("Loading expert model...")
    expert_cosyvoice = AutoModel(model_dir=model_dir)
    print("Loading amateur model...")
    amateur_cosyvoice = AutoModel(model_dir=model_dir)
    print("Both models loaded successfully.")

    # --- Read manifest ---
    meta_dir = os.path.dirname(os.path.abspath(meta_lst_path))
    with open(meta_lst_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    print(f"Total {len(lines)} entries in {meta_lst_path}")

    dual_cd = ECCDInference(
        expert_lm=expert_cosyvoice.model.llm,
        amateur_lm=amateur_cosyvoice.model.llm,
        model_type=model_type,
        sampling=sampling,
        strict_rep=strict_rep,
        rep_win_size=rep_win_size,
        expert_only=expert_only,
        random_sample=random_sample,
    )

    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    failed_indices = []
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    effective_lines = lines if max_samples is None else lines[:max_samples]
    if max_samples is not None:
        print(f"Limiting to first {len(effective_lines)}/{len(lines)} samples (max_samples={max_samples})")

    for idx, line in enumerate(tqdm(effective_lines, desc="Generating audio")):
        parts = line.split('|')
        utt_id = parts[0]
        prompt_text_str = parts[1]
        prompt_wav = os.path.join(meta_dir, parts[2])
        target_text_str = parts[3] if len(parts) > 3 else ''

        prompt_audio_fn = os.path.basename(parts[2])
        new_audio_fn = utt_id + ".wav"
        gen_audio_fn = f"gen_{prompt_audio_fn[:-4]}_to_{new_audio_fn[:-4]}.wav"
        gen_audio_path = os.path.join(output_dir, gen_audio_fn)

        if skip_existing and os.path.exists(gen_audio_path):
            print(f"  [{idx}] Skipping {gen_audio_fn} (already exists)")
            generated_paths.append(gen_audio_path)
            continue

        print(f"\n  [{idx}/{len(effective_lines)}] {utt_id}: {target_text_str[:40]}...")
        sample_timing = {'utt_id': utt_id}

        try:
            expert_prompt_text_str = _build_expert_prompt_text(prompt_text_str, model_type)
            amateur_prompt_text_str = _build_amateur_prompt_text(model_type)

            expert_model_input = expert_cosyvoice.frontend.frontend_zero_shot(
                tts_text=target_text_str,
                prompt_text=expert_prompt_text_str,
                prompt_wav=prompt_wav,
                resample_rate=expert_cosyvoice.sample_rate,
                zero_shot_spk_id='',
            )
            amateur_model_input = amateur_cosyvoice.frontend.frontend_zero_shot(
                tts_text="",
                prompt_text=amateur_prompt_text_str,
                prompt_wav=prompt_wav,
                resample_rate=amateur_cosyvoice.sample_rate,
                zero_shot_spk_id='',
            )

            sample_stats = [] if stats_dir else None

            tokens = dual_cd.inference(
                stats=sample_stats,
                expert_text=expert_model_input['text'],
                expert_text_len=expert_model_input['text_len'],
                expert_prompt_text=expert_model_input.get(
                    'prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                expert_prompt_text_len=expert_model_input.get(
                    'prompt_text_len', torch.tensor([0], dtype=torch.int32)),
                amateur_text=amateur_model_input['text'],
                amateur_text_len=amateur_model_input['text_len'],
                amateur_prompt_text=amateur_model_input.get(
                    'prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                amateur_prompt_text_len=amateur_model_input.get(
                    'prompt_text_len', torch.tensor([0], dtype=torch.int32)),
                prompt_speech_token=expert_model_input.get(
                    'llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                prompt_speech_token_len=expert_model_input.get(
                    'llm_prompt_speech_token_len', torch.tensor([0], dtype=torch.int32)),
                embedding=expert_model_input.get('llm_embedding', torch.zeros(0, 192)),
                amateur_prompt_speech_token=amateur_model_input.get(
                    'llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                amateur_prompt_speech_token_len=amateur_model_input.get(
                    'llm_prompt_speech_token_len', torch.tensor([0], dtype=torch.int32)),
                max_token_text_ratio=max_token_text_ratio,
                min_token_text_ratio=min_token_text_ratio,
                k=k, alpha=alpha, variant=variant, temperature=temperature,
                timing=sample_timing,
            )
            print(f"    Generated {len(tokens)} audio tokens")

            if stats_dir:
                stats_json_path = os.path.join(
                    stats_dir, gen_audio_fn.replace('.wav', '_stats.json'))
                with open(stats_json_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_stats, f, indent=2, ensure_ascii=False)

            # --- token2wav by expert ---
            uuid_str = str(uuid_lib.uuid4())
            with expert_cosyvoice.model.lock:
                expert_cosyvoice.model.tts_speech_token_dict[uuid_str] = tokens
                expert_cosyvoice.model.llm_end_dict[uuid_str] = True
                expert_cosyvoice.model.hift_cache_dict[uuid_str] = None
            tts_speech_token = torch.tensor([tokens], dtype=torch.int32)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            token2wav_start = time.time()
            tts_speech = expert_cosyvoice.model.token2wav(
                token=tts_speech_token,
                prompt_token=expert_model_input['flow_prompt_speech_token'],
                prompt_feat=expert_model_input['prompt_speech_feat'],
                embedding=expert_model_input['flow_embedding'],
                token_offset=0, uuid=uuid_str, finalize=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            token2wav_end = time.time()

            torchaudio.save(gen_audio_path, tts_speech.cpu(), expert_cosyvoice.sample_rate)
            audio_duration = tts_speech.shape[1] / expert_cosyvoice.sample_rate
            print(f"    Saved: {gen_audio_path} ({audio_duration:.2f}s)")
            generated_paths.append(gen_audio_path)

            token2wav_time = token2wav_end - token2wav_start
            lm_time = sample_timing.get('lm_time', 0.0)
            num_tokens = sample_timing.get('num_tokens', len(tokens))
            total_time = lm_time + token2wav_time
            sample_timing['token2wav_time'] = token2wav_time
            sample_timing['audio_duration'] = audio_duration
            sample_timing['total_time'] = total_time
            sample_timing['tps'] = (num_tokens / lm_time) if lm_time > 0 else 0.0
            sample_timing['rtf'] = (total_time / audio_duration) if audio_duration > 0 else 0.0
            if timings is not None:
                timings.append(sample_timing)

            with expert_cosyvoice.model.lock:
                expert_cosyvoice.model.tts_speech_token_dict.pop(uuid_str, None)
                expert_cosyvoice.model.llm_end_dict.pop(uuid_str, None)
                expert_cosyvoice.model.hift_cache_dict.pop(uuid_str, None)

        except Exception as e:
            print(f"    ERROR at index {idx}: {e}")
            traceback.print_exc()
            failed_indices.append(idx)

    del dual_cd, amateur_cosyvoice, expert_cosyvoice
    torch.cuda.empty_cache()

    print(f"\nBatch generation complete: {len(generated_paths)}/{len(effective_lines)} saved, "
          f"{len(failed_indices)} failed")
    return generated_paths, failed_indices


# =============================================================================
# Batch native inference — SeedTTS-Eval (meta.lst format)
# =============================================================================

def run_batch_native_inference(
    model_dir, model_name, meta_lst_path, output_dir,
    sampling=25,
    max_token_text_ratio=20, min_token_text_ratio=2,
    temperature=1.0,
    skip_existing=True, timings=None, max_samples=None,
):
    """Batch native decoding baseline for SeedTTS-Eval meta.lst datasets.

    Loads a single expert model and performs native nucleus sampling.  When
    ``temperature < 1.0`` this implements the lower-temperature baseline
    (tau in {0.75, 0.875}) discussed in the paper.

    Returns:
        (generated_paths, failed_indices)
    """
    print(f"\n{'='*60}")
    print(f"Batch Native Inference for {model_name}")
    print(f"{'='*60}")

    if not os.path.exists(model_dir):
        model_dir = snapshot_download(model_dir)
    model_type = _resolve_model_type(model_dir, model_name)

    print("Loading expert model...")
    expert_cosyvoice = AutoModel(model_dir=model_dir)
    print("Expert model loaded successfully.")

    meta_dir = os.path.dirname(os.path.abspath(meta_lst_path))
    with open(meta_lst_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    print(f"Total {len(lines)} entries in {meta_lst_path}")

    baseline = NativeInference(
        expert_lm=expert_cosyvoice.model.llm,
        model_type=model_type,
        sampling=sampling,
    )

    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    failed_indices = []

    effective_lines = lines if max_samples is None else lines[:max_samples]
    if max_samples is not None:
        print(f"Limiting to first {len(effective_lines)}/{len(lines)} samples (max_samples={max_samples})")

    for idx, line in enumerate(tqdm(effective_lines, desc="Generating audio (native)")):
        parts = line.split('|')
        utt_id = parts[0]
        prompt_text_str = parts[1]
        prompt_wav = os.path.join(meta_dir, parts[2])
        target_text_str = parts[3] if len(parts) > 3 else ''

        prompt_audio_fn = os.path.basename(parts[2])
        new_audio_fn = utt_id + ".wav"
        gen_audio_fn = f"gen_{prompt_audio_fn[:-4]}_to_{new_audio_fn[:-4]}.wav"
        gen_audio_path = os.path.join(output_dir, gen_audio_fn)

        if skip_existing and os.path.exists(gen_audio_path):
            print(f"  [{idx}] Skipping {gen_audio_fn} (already exists)")
            generated_paths.append(gen_audio_path)
            continue

        print(f"\n  [{idx}/{len(effective_lines)}] {utt_id}: {target_text_str[:40]}...")
        sample_timing = {'utt_id': utt_id}

        try:
            expert_prompt_text_str = _build_expert_prompt_text(prompt_text_str, model_type)

            expert_model_input = expert_cosyvoice.frontend.frontend_zero_shot(
                tts_text=target_text_str,
                prompt_text=expert_prompt_text_str,
                prompt_wav=prompt_wav,
                resample_rate=expert_cosyvoice.sample_rate,
                zero_shot_spk_id='',
            )

            tokens = baseline.inference(
                expert_text=expert_model_input['text'],
                expert_text_len=expert_model_input['text_len'],
                expert_prompt_text=expert_model_input.get(
                    'prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                expert_prompt_text_len=expert_model_input.get(
                    'prompt_text_len', torch.tensor([0], dtype=torch.int32)),
                prompt_speech_token=expert_model_input.get(
                    'llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                prompt_speech_token_len=expert_model_input.get(
                    'llm_prompt_speech_token_len', torch.tensor([0], dtype=torch.int32)),
                embedding=expert_model_input.get('llm_embedding', torch.zeros(0, 192)),
                max_token_text_ratio=max_token_text_ratio,
                min_token_text_ratio=min_token_text_ratio,
                temperature=temperature,
                timing=sample_timing,
            )
            print(f"    Generated {len(tokens)} audio tokens")

            uuid_str = str(uuid_lib.uuid4())
            with expert_cosyvoice.model.lock:
                expert_cosyvoice.model.tts_speech_token_dict[uuid_str] = tokens
                expert_cosyvoice.model.llm_end_dict[uuid_str] = True
                expert_cosyvoice.model.hift_cache_dict[uuid_str] = None
            tts_speech_token = torch.tensor([tokens], dtype=torch.int32)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            token2wav_start = time.time()
            tts_speech = expert_cosyvoice.model.token2wav(
                token=tts_speech_token,
                prompt_token=expert_model_input['flow_prompt_speech_token'],
                prompt_feat=expert_model_input['prompt_speech_feat'],
                embedding=expert_model_input['flow_embedding'],
                token_offset=0, uuid=uuid_str, finalize=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            token2wav_end = time.time()

            torchaudio.save(gen_audio_path, tts_speech.cpu(), expert_cosyvoice.sample_rate)
            audio_duration = tts_speech.shape[1] / expert_cosyvoice.sample_rate
            print(f"    Saved: {gen_audio_path} ({audio_duration:.2f}s)")
            generated_paths.append(gen_audio_path)

            token2wav_time = token2wav_end - token2wav_start
            lm_time = sample_timing.get('lm_time', 0.0)
            num_tokens = sample_timing.get('num_tokens', len(tokens))
            total_time = lm_time + token2wav_time
            sample_timing['token2wav_time'] = token2wav_time
            sample_timing['audio_duration'] = audio_duration
            sample_timing['total_time'] = total_time
            sample_timing['tps'] = (num_tokens / lm_time) if lm_time > 0 else 0.0
            sample_timing['rtf'] = (total_time / audio_duration) if audio_duration > 0 else 0.0
            if timings is not None:
                timings.append(sample_timing)

            with expert_cosyvoice.model.lock:
                expert_cosyvoice.model.tts_speech_token_dict.pop(uuid_str, None)
                expert_cosyvoice.model.llm_end_dict.pop(uuid_str, None)
                expert_cosyvoice.model.hift_cache_dict.pop(uuid_str, None)

        except Exception as e:
            print(f"    ERROR at index {idx}: {e}")
            traceback.print_exc()
            failed_indices.append(idx)

    del baseline, expert_cosyvoice
    torch.cuda.empty_cache()

    print(f"\nBatch native generation complete: {len(generated_paths)}/{len(effective_lines)} saved, "
          f"{len(failed_indices)} failed")
    return generated_paths, failed_indices


# =============================================================================
# Batch ECCD inference — CV3-Eval native format
# =============================================================================

def run_batch_eccd_inference_cv3eval(
    model_dir, model_name, cv3_data_dir, output_dir,
    sampling=25,
    k=25, alpha=1.0, variant='eccd', temperature=1.0,
    strict_rep=False, rep_win_size=10,
    expert_only=False, random_sample=False,
    max_token_text_ratio=20, min_token_text_ratio=2,
    stats_dir=None,
):
    """Batch ECCD inference for CV3-Eval native dataset structure.

    Reads from a CV3-Eval language subdir (e.g. ``data/zero_shot/zh/``)
    containing:
        - text:           ``<uttid> <target_text>``
        - prompt_text:    ``<uttid> <prompt_text>``
        - prompt_wav.scp: ``<uttid> <prompt_wav_relpath>``

    Generated audio is named ``{uttid}.wav`` to match the CV3-Eval evaluation
    convention.

    Args:
        cv3_data_dir: e.g. ``/data/local/training_data/CV3-Eval/data/zero_shot/zh``
        output_dir:   directory to save generated wavs.
        k:             plausible set size |V_head|.
        alpha:         alignment enhancement strength.
        variant:       ablation variant.
        stats_dir:     if set, per-step stats are saved as JSON per sample.

    Returns:
        (generated_paths, failed_uttids)
    """
    print(f"\n{'='*60}")
    print(f"Batch ECCD for CV3-Eval: {cv3_data_dir} (model={model_name})")
    print(f"{'='*60}")

    if not os.path.exists(model_dir):
        model_dir = snapshot_download(model_dir)
    model_type = _resolve_model_type(model_dir, model_name)

    # --- Load CV3-Eval native files ---
    text_path = os.path.join(cv3_data_dir, "text")
    prompt_text_path = os.path.join(cv3_data_dir, "prompt_text")
    prompt_wav_scp_path = os.path.join(cv3_data_dir, "prompt_wav.scp")
    for p in (text_path, prompt_text_path, prompt_wav_scp_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CV3-Eval file missing: {p}")

    # CV3-Eval root is three levels up from cv3_data_dir:
    #   data/zero_shot/zh  ->  CV3-Eval repo root
    cv3_root = os.path.abspath(os.path.join(cv3_data_dir, "..", "..", ".."))
    lang = os.path.basename(cv3_data_dir.rstrip('/'))[-2:]

    def _read_kv(path):
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                uttid, val = line.split(maxsplit=1)
                d[uttid] = val
        return d

    text_map = _read_kv(text_path)
    prompt_text_map = _read_kv(prompt_text_path)
    prompt_wav_map = _read_kv(prompt_wav_scp_path)

    common_uttids = [
        uid for uid in text_map.keys()
        if uid in prompt_text_map and uid in prompt_wav_map
    ]
    print(f"Total {len(common_uttids)} valid entries in {cv3_data_dir}")

    # --- Load expert and amateur model instances ---
    print("Loading expert model...")
    expert_cosyvoice = AutoModel(model_dir=model_dir)
    print("Loading amateur model...")
    amateur_cosyvoice = AutoModel(model_dir=model_dir)
    print("Both models loaded successfully.")

    dual_cd = ECCDInference(
        expert_lm=expert_cosyvoice.model.llm,
        amateur_lm=amateur_cosyvoice.model.llm,
        model_type=model_type,
        sampling=sampling,
        strict_rep=strict_rep,
        rep_win_size=rep_win_size,
        expert_only=expert_only,
        random_sample=random_sample,
    )

    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    failed_uttids = []
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    for idx, uttid in enumerate(tqdm(common_uttids, desc="Generating audio")):
        target_text_str = text_map[uttid]
        prompt_text_str = prompt_text_map[uttid]
        prompt_wav_rel = prompt_wav_map[uttid]

        # CosyVoice3 requires Japanese input as space-separated katakana.
        if model_type == 'cosyvoice3' and lang == 'ja':
            assert JAPANESE_CONVERT_AVAILABLE, \
                "pykakasi not installed; Japanese text cannot be converted to katakana."
            target_text_str = to_katakana_spaced(target_text_str)
            prompt_text_str = to_katakana_spaced(prompt_text_str)

        # Resolve prompt wav absolute path.
        if os.path.isabs(prompt_wav_rel):
            prompt_wav = prompt_wav_rel
        else:
            prompt_wav = os.path.join(cv3_root, prompt_wav_rel)
            if not os.path.exists(prompt_wav):
                alt = os.path.join(cv3_data_dir, prompt_wav_rel)
                if os.path.exists(alt):
                    prompt_wav = alt

        gen_audio_path = os.path.join(output_dir, f"{uttid}.wav")

        if os.path.exists(gen_audio_path):
            print(f"  [{idx}] Skipping {uttid} (already exists)")
            generated_paths.append(gen_audio_path)
            continue

        print(f"\n  [{idx}/{len(common_uttids)}] {uttid}: {target_text_str[:40]}...")

        try:
            expert_prompt_text_str = _build_expert_prompt_text(prompt_text_str, model_type)
            amateur_prompt_text_str = _build_amateur_prompt_text(model_type)

            expert_model_input = expert_cosyvoice.frontend.frontend_zero_shot(
                tts_text=target_text_str,
                prompt_text=expert_prompt_text_str,
                prompt_wav=prompt_wav,
                resample_rate=expert_cosyvoice.sample_rate,
                zero_shot_spk_id='',
            )
            amateur_model_input = amateur_cosyvoice.frontend.frontend_zero_shot(
                tts_text="",
                prompt_text=amateur_prompt_text_str,
                prompt_wav=prompt_wav,
                resample_rate=amateur_cosyvoice.sample_rate,
                zero_shot_spk_id='',
            )

            sample_stats = [] if stats_dir else None

            tokens = dual_cd.inference(
                stats=sample_stats,
                expert_text=expert_model_input['text'],
                expert_text_len=expert_model_input['text_len'],
                expert_prompt_text=expert_model_input.get(
                    'prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                expert_prompt_text_len=expert_model_input.get(
                    'prompt_text_len', torch.tensor([0], dtype=torch.int32)),
                amateur_text=amateur_model_input['text'],
                amateur_text_len=amateur_model_input['text_len'],
                amateur_prompt_text=amateur_model_input.get(
                    'prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                amateur_prompt_text_len=amateur_model_input.get(
                    'prompt_text_len', torch.tensor([0], dtype=torch.int32)),
                prompt_speech_token=expert_model_input.get(
                    'llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                prompt_speech_token_len=expert_model_input.get(
                    'llm_prompt_speech_token_len', torch.tensor([0], dtype=torch.int32)),
                embedding=expert_model_input.get('llm_embedding', torch.zeros(0, 192)),
                amateur_prompt_speech_token=amateur_model_input.get(
                    'llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                amateur_prompt_speech_token_len=amateur_model_input.get(
                    'llm_prompt_speech_token_len', torch.tensor([0], dtype=torch.int32)),
                max_token_text_ratio=max_token_text_ratio,
                min_token_text_ratio=min_token_text_ratio,
                k=k, alpha=alpha, variant=variant, temperature=temperature,
            )
            print(f"    Generated {len(tokens)} audio tokens")

            if stats_dir:
                stats_json_path = os.path.join(stats_dir, f"{uttid}_stats.json")
                with open(stats_json_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_stats, f, indent=2, ensure_ascii=False)

            # --- token2wav by expert ---
            uuid_str = str(uuid_lib.uuid4())
            with expert_cosyvoice.model.lock:
                expert_cosyvoice.model.tts_speech_token_dict[uuid_str] = tokens
                expert_cosyvoice.model.llm_end_dict[uuid_str] = True
                expert_cosyvoice.model.hift_cache_dict[uuid_str] = None
            tts_speech_token = torch.tensor([tokens], dtype=torch.int32)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            token2wav_start = time.time()
            tts_speech = expert_cosyvoice.model.token2wav(
                token=tts_speech_token,
                prompt_token=expert_model_input['flow_prompt_speech_token'],
                prompt_feat=expert_model_input['prompt_speech_feat'],
                embedding=expert_model_input['flow_embedding'],
                token_offset=0, uuid=uuid_str, finalize=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            token2wav_end = time.time()

            torchaudio.save(gen_audio_path, tts_speech.cpu(), expert_cosyvoice.sample_rate)
            audio_duration = tts_speech.shape[1] / expert_cosyvoice.sample_rate
            print(f"    Saved: {gen_audio_path} ({audio_duration:.2f}s)")
            generated_paths.append(gen_audio_path)

            with expert_cosyvoice.model.lock:
                expert_cosyvoice.model.tts_speech_token_dict.pop(uuid_str, None)
                expert_cosyvoice.model.llm_end_dict.pop(uuid_str, None)
                expert_cosyvoice.model.hift_cache_dict.pop(uuid_str, None)

        except Exception as e:
            print(f"    ERROR at {uttid}: {e}")
            traceback.print_exc()
            failed_uttids.append(uttid)

    del dual_cd, amateur_cosyvoice, expert_cosyvoice
    torch.cuda.empty_cache()

    print(f"\nBatch generation complete: {len(generated_paths)}/{len(common_uttids)} saved, "
          f"{len(failed_uttids)} failed")
    return generated_paths, failed_uttids
