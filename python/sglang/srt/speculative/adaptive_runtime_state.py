from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    The draft and verify resources are required by adaptive workers. Algorithms
    with a draft-extend stage can also populate its optional resources.
    Switching adaptive steps swaps the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "DecodeCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "DecodeCudaGraphRunner | None"

    # -- DFLASH-specific per-block_size resources (None for EAGLE) --
    # DFLASH runs draft+verify in one worker; the draft CUDA graph embeds the
    # conv `position & (block_size-1)` alignment, so BOTH draft and target
    # graphs must be recaptured per block_size. These extra resources are
    # rebuilt per state and swapped atomically on apply.
    dflash_draft_sampler: object = None  # _DflashDraftSampler / _SelectorDraftSampler
    dflash_block_pos_offsets: "object" = None  # torch.Tensor [block_size]
    dflash_draft_block_spec_info: object = None  # verify spec_info for target graph
    dflash_fused_kv_helper: object = None  # FusedKVMaterializeHelper (max_position_hint)


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveSpecPolicy(Protocol):
    """Policy interface used by AdaptiveController to select runtime states."""

    @property
    def candidate_steps(self) -> list[int]: ...

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None: ...

    def get_steps_for_batch(self, batch_size: int) -> int: ...

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None: ...

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        config_path: str | None = None,
        params=None,
        draft_tokens_from_steps=None,
    ):
        """AdaptiveController.

        *params* overrides the default ``AdaptiveSpeculativeParams`` (EAGLE).
        DFLASH passes ``AdaptiveDFlashParams`` (block_size axis) instead.
        *draft_tokens_from_steps* maps a candidate step → its draft-token width;
        EAGLE's chain layout uses ``steps + 1`` (default), DFLASH's block layout
        uses identity (block_size == draft_tokens).
        """
        self.worker = worker
        self.params = (
            params
            if params is not None
            else AdaptiveSpeculativeParams(
                initial_steps=worker.speculative_num_steps,
                cfg_path=config_path,
            )
        )
        self._draft_tokens_from_steps = (
            draft_tokens_from_steps
            if draft_tokens_from_steps is not None
            else (lambda steps: steps + 1)
        )
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=self._draft_tokens_from_steps(steps),
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def activate_step(self, speculative_num_steps: int) -> None:
        """Activate a decision made outside :meth:`on_verify_complete`.

        DFLASH TP uses rank 0 to make the block-size decision and broadcasts
        the result before every rank activates the same pre-built state.
        """
        self._activate(speculative_num_steps)

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        step_time_ms: float = 0.0,
    ) -> None:
        """Feed verify results; switch runtime state if the policy requests it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size, step_time_ms=step_time_ms
        )
        if new_step is not None:
            self._activate(new_step)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
