"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
)
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 3, 5, 7],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""

    cfg = resolving_view(server_args)

    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={cfg.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={cfg.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if resolved_view(server_args).enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
        )
    if cfg.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if cfg.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def _load_adaptive_config(
    cfg_path: str | None,
) -> tuple[dict, dict[int, dict]]:
    """Load and validate adaptive config.

    Uses ``DEFAULT_ADAPTIVE_CONFIG`` when *cfg_path* is ``None``.
    """
    if cfg_path is not None:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_ADAPTIVE_CONFIG

    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config must contain at least one integer-string "
            'BS key, e.g. {"1": {"candidate_steps": [1,3,7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
) -> list[int]:
    """Union of every BS slot's candidate steps; sizes the runtime buffers."""
    _, bs_entries = _load_adaptive_config(cfg_path)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


class AdaptiveStepSlot:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - num_steps can be selected from different candidate sets on different batch_sizes
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self.ema_accept_len = (
                1 - self.ema_alpha
            ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # EMA ceiling: only caps downward — never blocks step-ups, so the
        # system can explore higher steps and let the EMA catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False


class AdaptiveSpeculativeParams:
    """Routes ``batch_size`` to the correct per-BS slot.

    A slot is a per-BS configuration of adaptive step selection.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
    ):
        cfg, bs_entries = _load_adaptive_config(cfg_path)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = AdaptiveStepSlot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        return self._route(batch_size).current_steps

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        step_time_ms: float = 0.0,
    ) -> int | None:
        """Feed verify results to the matching BS slot's EMA.

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``).
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """Map *batch_size* → pad to CUDA-graph BS → closest slot."""
        return self._slots[
            self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]


# ============================================================================
# DFLASH Adaptive Block Size (MAB-ABS — Marginal-Accept-Benefit Adaptive Block Size)
# ============================================================================


class PositionalAcceptHistogram:
    """Batch-aggregated per-position conditional acceptance rates with EMA.

    At block_size=B, each request's commit_len ∈ [1, B] (1=only bonus, B=all accepted).
    For draft position i ∈ [1, B-1]: reach[i] += 1 if verified, pass[i] += 1 if accepted.
    p_i = EMA(pass[i] / reach[i]).
    """

    def __init__(self, max_b: int, ema_alpha: float = 0.1, prior: float = 0.5):
        self.max_b = max_b
        self.alpha = ema_alpha
        self.p = [prior] * (max_b + 1)
        self.w = [0.0] * (max_b + 1)

    def update(self, commit_lens: list[int], verify_len: int) -> None:
        if not commit_lens:
            return
        a = self.alpha
        reach = [0.0] * (self.max_b + 1)
        pas = [0.0] * (self.max_b + 1)
        for al in commit_lens:
            for i in range(1, verify_len):
                reach[i] += 1
                if i < al:
                    pas[i] += 1
                else:
                    break
        for i in range(1, self.max_b + 1):
            if reach[i] > 0:
                ratio = pas[i] / reach[i]
                self.p[i] = ratio if self.w[i] < 1e-6 else (1 - a) * self.p[i] + a * ratio
                self.w[i] = (1 - a) * self.w[i] + a
            else:
                self.w[i] = (1 - a) * self.w[i]

    def prob(self, i: int) -> float | None:
        return self.p[i] if self.w[i] > 0.05 else None

    def fit_decay(self) -> tuple[float, float]:
        """Fit observed positional rates to p_i = p0 * r^(i-1)."""
        xs, ys, ws = [], [], []
        for i in range(1, self.max_b):
            p = self.prob(i)
            if p is not None and p > 1e-3:
                xs.append(float(i - 1))
                ys.append(math.log(p))
                ws.append(self.w[i])

        if not xs:
            return self.p[0], 0.85
        if len(xs) == 1:
            return min(1.0, math.exp(ys[0])), 0.85

        sw = sum(ws)
        mx = sum(x * w for x, w in zip(xs, ws)) / sw
        my = sum(y * w for y, w in zip(ys, ws)) / sw
        var = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws)) / sw
        cov = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sw
        if var < 1e-9:
            return min(1.0, math.exp(my)), 0.85

        slope = cov / var
        intercept = my - slope * mx
        return min(1.0, math.exp(intercept)), min(1.0, math.exp(slope))

    def expected_accept(
        self,
        target_b: int,
        current_b: int,
        extrapolation_discount: float,
    ) -> float:
        """Predict E[commit_len] for *target_b*, including the bonus token."""
        p0, ratio = self.fit_decay()
        accept = 1.0
        cumulative = 1.0
        for i in range(1, target_b):
            p = self.prob(i) if i < current_b else None
            if p is None:
                p = p0 * (ratio ** (i - 1)) * extrapolation_discount
            cumulative *= min(1.0, max(0.0, p))
            accept += cumulative
        return accept


class StepCostModel:
    """Per-block_size latency EMA + linear extrapolation for unseen sizes.

    Stores latency EMA per block_size (not global regression) because once the
    controller converges, block_size stops changing and a global least-squares
    would see zero x-variance and fail to solve for slope b.
    """

    def __init__(self, ema_alpha: float = 0.1, b_prior: float = 2.14):
        self.alpha = ema_alpha
        self.b_prior = b_prior
        self._t: dict[int, float] = {}
        self._w: dict[int, float] = {}

    def update(self, block_size: int, step_time_ms: float) -> None:
        a = self.alpha
        b = int(block_size)
        if b not in self._t:
            self._t[b] = float(step_time_ms)
            self._w[b] = a
        else:
            self._t[b] = (1 - a) * self._t[b] + a * float(step_time_ms)
            self._w[b] = (1 - a) * self._w[b] + a

    def params(self) -> tuple[float, float]:
        pts = [
            (float(b), t, self._w[b])
            for b, t in self._t.items()
            if self._w[b] > 1e-3
        ]
        if not pts:
            return 0.0, self.b_prior
        if len(pts) == 1:
            x, y, _ = pts[0]
            return max(0.0, y - self.b_prior * x), self.b_prior

        sw = sum(w for _, _, w in pts)
        mx = sum(x * w for x, _, w in pts) / sw
        my = sum(y * w for _, y, w in pts) / sw
        var = sum(w * (x - mx) ** 2 for x, _, w in pts) / sw
        cov = sum(w * (x - mx) * (y - my) for x, y, w in pts) / sw
        b = max(1e-3, cov / var) if var > 1e-9 else self.b_prior
        return max(0.0, my - b * mx), b

    def predict(self, block_size: int) -> float:
        b = int(block_size)
        # 0.4 = one exploration dwell (decision_interval=5 obs at alpha=0.1
        # gives weight 0.41): a directly measured cost always beats the linear
        # extrapolation, whose intercept misprices the off-linear B=0 arm.
        if b in self._t and self._w[b] > 0.4:
            return self._t[b]
        a_, b_ = self.params()
        return a_ + b_ * block_size




class GlobalLinearCostModel:
    """Shared T(bs, B) = F + c * (bs * B) model, fitted online.

    Replaces per-B cost EMAs in the decision: every observed step contributes a
    (x = bs*B, y = T) sample regardless of which B it ran at, so the estimate
    for a candidate never goes stale while another one is active (the frozen-EMA
    failure mode that kept the controller pinned on a small block after one bad
    phase). Recency-weighted recursive least squares over a bounded ring.
    """

    def __init__(self, ring: int = 256, gamma: float = 0.995):
        self.ring = ring
        self.gamma = gamma
        self.xs: list[float] = []
        self.ys: list[float] = []

    def add(self, x: float, y: float) -> None:
        self.xs.append(float(x))
        self.ys.append(float(y))
        if len(self.xs) > self.ring:
            self.xs.pop(0)
            self.ys.pop(0)

    def params(self) -> tuple[float, float]:
        """Weighted least squares (F, c); falls back to (min-y, 0)."""
        n = len(self.xs)
        if n < 4:
            return (min(self.ys) if self.ys else 0.0, 0.0)
        sw = sx = sy = sxx = sxy = 0.0
        for k in range(n):
            w = self.gamma ** (n - 1 - k)
            x, y = self.xs[k], self.ys[k]
            sw += w
            sx += w * x
            sy += w * y
            sxx += w * x * x
            sxy += w * x * y
        var = sxx / sw - (sx / sw) ** 2
        if var < 1e-9:
            return (sy / sw, 0.0)
        c = (sxy / sw - (sx / sw) * (sy / sw)) / var
        return (sy / sw - c * (sx / sw), c)

    def predict(self, x: float) -> float:
        f, c = self.params()
        return max(1e-3, f + c * x)

    def last_sample_age(self, x: float) -> int:
        """Steps since a sample with this x value was added (ring steps max)."""
        for k in range(len(self.xs) - 1, -1, -1):
            if abs(self.xs[k] - x) < 0.5:
                return len(self.xs) - 1 - k
        return len(self.xs) + 1


class MabAbsSlot:
    """MAB-ABS decision policy: adaptive block_size via marginal benefit criterion.

    Selects the candidate B minimizing T(B) / E[commit_len(B)].  The initial
    block size is a hard ceiling; adaptation only moves downward from it.
    """

    def __init__(self, max_block_size: int, cfg: dict, cost: GlobalLinearCostModel):
        candidates = sorted(
            {
                int(block_size)
                for block_size in cfg["candidate_block_sizes"]
                if int(block_size) <= max_block_size
            }
        )
        candidates.append(max_block_size)
        self.candidates = sorted(set(candidates))
        self.max_b = max_block_size

        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.decision_interval = cfg.get("decision_interval", 5)
        # At bs~1 the predicted T(B) difference between candidates is <2%, so a
        # thin band flips the argmax on noise alone, graph swaps storm, and the
        # overlap pipeline hits rank divergence (observed: NCCL broadcast
        # timeout after 9 switches in one minute). 10% + a post-switch dwell
        # keeps switches to genuinely structural changes.
        self.hysteresis = cfg.get("hysteresis", 0.10)
        self.switch_dwell_decisions = cfg.get("switch_dwell_decisions", 10)
        # Anti-flap: a switch needs the same argmax preference on N consecutive
        # decisions. Decode bursts run hundreds of batches per second, so
        # decision-counted dwell alone can pass in under a second.
        self.switch_confirm_votes = cfg.get("switch_confirm_votes", 3)
        self._confirm_candidate: int | None = None
        self._confirm_votes = 0
        self._decisions_since_switch = 10_000
        # Higher than the old 0.9: the periodic explorer refreshes real data for
        # positions beyond the running block, so extrapolation no longer needs
        # to double-penalize larger candidates.
        self.extrapolation_discount = cfg.get("extrapolation_discount", 0.98)
        # Every N decisions, force one probe of the stalest candidate. This is
        # the rested-bandit cure for the frozen-cost trap: a candidate measured
        # once during a bad phase would otherwise never be retried.
        self.explore_every = cfg.get("explore_every", 20)

        self.current_b = max_block_size
        self.hist = PositionalAcceptHistogram(max_b=self.max_b)
        # Shared across slots: T(bs,B) has one shape regardless of slot.
        self.cost = cost
        self._last_bs = 1
        self.batch_count = 0
        self.decision_count = 0

    def observe(self, commit_lens: list[int], step_time_ms: float, batch_size: int = 1) -> None:
        # B=0 is the "speculation off" candidate: every step accepts exactly
        # the single sampled token, so the positional histogram has nothing to
        # learn; only the plain-decode cost is tracked (x = 0).
        if self.current_b != 0:
            self.hist.update(commit_lens, self.current_b)
        self.cost.add(self._last_bs * self.current_b, step_time_ms)
        self._last_bs = max(1, int(batch_size))
        self.batch_count += 1

    def maybe_decide(self) -> int | None:
        if self.batch_count <= self.warmup_batches:
            return None
        if (self.batch_count - self.warmup_batches) % self.decision_interval != 0:
            return None
        self.decision_count += 1
        self._decisions_since_switch += 1
        if self._decisions_since_switch < self.switch_dwell_decisions:
            return None

        # Periodic explorer: probe the stalest other candidate so every arm's
        # cost and (for larger blocks) accept data stay fresh. Without this a
        # candidate measured once in a bad phase is never revisited.
        if self.explore_every and self.decision_count % self.explore_every == 0:
            others = [c for c in self.candidates if c != self.current_b]
            if others:
                nb = max(
                    others,
                    key=lambda c: self.cost.last_sample_age(self._last_bs * c),
                )
                self.current_b = nb
                self._decisions_since_switch = 0
                return nb

        # Structural cost/benefit: T(bs,B) = F + c*bs*B fitted from ALL recent
        # steps (never stale), accept from the shared positional histogram.
        # Bidirectional by construction: whichever B minimizes ms/token wins,
        # regardless of whether it is larger or smaller than the current one.
        evals: dict[int, float] = {}
        for cand in self.candidates:
            if cand == 0:
                evals[0] = self.cost.predict(0)
                continue
            acc = self.hist.expected_accept(
                cand,
                self.current_b,
                self.extrapolation_discount,
            )
            if acc < 1e-6:
                continue
            evals[cand] = self.cost.predict(self._last_bs * cand) / acc
        if not evals:
            return None

        best_b = min(evals, key=evals.get)
        if best_b == self.current_b:
            self._confirm_candidate = None
            self._confirm_votes = 0
            return None
        cur_tpot = evals.get(self.current_b)
        warranted = cur_tpot is None or evals[best_b] < cur_tpot * (
            1.0 - self.hysteresis
        )
        if not warranted:
            self._confirm_candidate = None
            self._confirm_votes = 0
            return None
        # Require the same preference on consecutive decisions before acting.
        if best_b != self._confirm_candidate:
            self._confirm_candidate = best_b
            self._confirm_votes = 1
            return None
        self._confirm_votes += 1
        if self._confirm_votes < self.switch_confirm_votes:
            return None
        self.current_b = best_b
        self._decisions_since_switch = 0
        self._confirm_candidate = None
        self._confirm_votes = 0
        return best_b


class AdaptiveDFlashParams:
    """Routes batch_size to the correct per-BS block_size slot (MAB-ABS)."""

    # 0 in a candidate list is the "speculation off" arm: the worker runs a
    # plain (non-speculative) decode step for that slot. The cost model decides
    # when it wins -- typically high batch size, where the batched plain step
    # amortizes weights far better than bs*block-wide verify.
    DEFAULT_CONFIG: dict[str, dict] = {
        "1": {"candidate_block_sizes": [4, 8]},
        "8": {"candidate_block_sizes": [4, 8]},
        "16": {"candidate_block_sizes": [4, 8]},
        "32": {"candidate_block_sizes": [4, 8]},
        "64": {"candidate_block_sizes": [4, 8]},
    }

    def __init__(self, max_block_size: int, cfg_path: str | None = None):
        if cfg_path is not None:
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = self.DEFAULT_CONFIG

        bs_entries: dict[int, dict] = {}
        for key, entry in cfg.items():
            if not key.isdigit():
                continue
            sizes = entry.get("candidate_block_sizes")
            if not sizes:
                continue
            bs_entries[int(key)] = entry

        if not bs_entries:
            bs_entries = {1: {"candidate_block_sizes": [4, 6, 8]}}

        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, MabAbsSlot] = {}
        self._cuda_graph_bs: list[int] | None = None
        # Activation floor: slots routed below this batch size are locked to the
        # initial (trained) block — the MAB is inert there. At bs below the
        # weight-stream floor the step time is flat in B, so switching only
        # truncates accept (measured: accept 4.0->3.28, TPOT +34%); the gate
        # makes adaptation a guaranteed no-op until concurrency reaches the
        # regime where T(B) actually slopes. JSON-configurable per slot or via
        # __defaults__: {"min_adaptive_bs": 16}.
        self._min_adaptive_bs = int(
            cfg.get("__defaults__", {}).get("min_adaptive_bs", 16)
        )
        # One cost model shared by all slots: T(bs,B) = F + c*bs*B has a single
        # shape, and pooling samples across slots converges far faster.
        self._shared_cost = GlobalLinearCostModel()
        for bs, entry in sorted(bs_entries.items()):
            slot_cfg = {**cfg.get("__defaults__", {}), **entry}
            if bs < self._min_adaptive_bs:
                slot_cfg["candidate_block_sizes"] = [max_block_size]
            self._slots[bs] = MabAbsSlot(
                max_block_size=max_block_size,
                cfg=slot_cfg,
                cost=self._shared_cost,
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveDFlashParams(MAB-ABS) initialized: "
            f"block_size={first_slot.current_b}, "
            f"candidate_block_sizes={first_slot.candidates}",
        )

    @cached_property
    def candidate_block_sizes(self) -> list[int]:
        return sorted({s for p in self._slots.values() for s in p.candidates})

    # --- AdaptiveController compatibility interface ---------------------
    # AdaptiveController drives the generic spec axis as "steps"; for DFLASH we
    # reuse that axis to carry block_size (draft_tokens_from_steps = identity,
    # wired in the controller). The methods below alias the block_size API onto
    # the step-shaped names the controller expects.

    @property
    def candidate_steps(self) -> list[int]:
        """Candidate block_sizes exposed under the controller's step axis."""
        return self.candidate_block_sizes

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """BS buckets that can reach this block_size at runtime (for capture pruning)."""
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidates
        ]

    def get_steps_for_batch(self, batch_size: int) -> int:
        """Alias of get_block_size_for_batch under the controller's step axis."""
        return self.get_block_size_for_batch(batch_size)

    def get_block_size_for_batch(self, batch_size: int) -> int:
        return self._route(batch_size).current_b

    def set_block_size_for_batch(
        self, batch_size: int, block_size: int
    ) -> None:
        """Install a decision made by TP rank 0 into this rank's routed slot."""
        self._route(batch_size).current_b = int(block_size)

    def on_verify_complete(
        self, accept_lens: list[int], batch_size: int, step_time_ms: float = 0.0
    ) -> int | None:
        slot = self._route(batch_size)
        slot.observe(accept_lens, step_time_ms, batch_size=batch_size)
        return slot.maybe_decide()

    def _route(self, batch_size: int) -> MabAbsSlot:
        return self._slots[self._find_closest_bs(batch_size)]

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]
