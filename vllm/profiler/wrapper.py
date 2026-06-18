# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import Literal

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger

logger = init_logger(__name__)


class WorkerProfiler(ABC):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        self._delay_iters = profiler_config.delay_iterations
        if self._delay_iters > 0:
            logger.info_once(
                "GPU profiling will start "
                f"{self._delay_iters} steps after start_profile."
            )

        self._max_iters = profiler_config.max_iterations
        if self._max_iters > 0:
            logger.info_once(
                "GPU profiling will stop "
                f"after {self._max_iters} worker steps, "
                "or when stop_profile is received."
            )

        # Track when the profiler gets triggered by start_profile
        self._active_iteration_count = 0
        self._active = False

        # Track when the profiler is actually running
        self._profiling_for_iters = 0
        self._running = False

        # Cross-rank coordination of the profiler start/stop toggle.
        # cudaProfilerStart/Stop are host-side toggles; if ranks flip them at
        # different points relative to in-flight collectives (e.g. MoE all2all
        # or the DP all-reduce) a multi-GPU run can wedge. Subclasses whose
        # toggle is collective-sensitive (CUDA / nsys) opt in by setting this.
        # Only the counter-driven transitions (delayed start / max-iteration
        # stop) are coordinated, since those are iteration-aligned across DP
        # ranks; the immediate start_profile / explicit stop_profile paths are
        # not aligned and are intentionally left uncoordinated.
        self._coordinate_toggle = False

    @abstractmethod
    def _start(self) -> None:
        """Start the profiler."""
        pass

    @abstractmethod
    def _stop(self) -> None:
        """Stop the profiler."""
        pass

    def _toggle_coordination_enabled(self) -> bool:
        # Escape hatch: VLLM_PROFILER_TOGGLE_SYNC=0/1 forces the behavior off/on.
        override = os.environ.get("VLLM_PROFILER_TOGGLE_SYNC")
        if override is not None:
            return override == "1"
        return self._coordinate_toggle

    def _coordinate_toggle_rendezvous(self, tag: str) -> None:
        """Quiesce the device and rendezvous across all ranks that share
        deadlock-prone collectives, so the profiler toggle happens at the same
        step boundary on every rank.

        This is best-effort: any failure to resolve a process group degrades to
        a purely local toggle rather than risking a hang. It must only be called
        from iteration-aligned (counter-driven) transitions so that every rank
        reaches the rendezvous on the same iteration.
        """
        if not self._toggle_coordination_enabled():
            return
        try:
            import torch.distributed as dist

            if not (dist.is_available() and dist.is_initialized()):
                return

            from vllm.distributed.parallel_state import (
                get_dp_group,
                get_ep_group,
                get_world_group,
            )

            # Pick the broadest group that spans the ranks sharing the
            # collectives we must not interrupt: EP (MoE all2all) first, then
            # DP (padding all-reduce), then the local world (TP/PP). All ranks
            # resolve the same group because these sizes are global config.
            group = None
            for getter in (get_ep_group, get_dp_group, get_world_group):
                try:
                    coord = getter()
                except Exception:
                    continue
                if coord is not None and coord.world_size > 1:
                    group = coord.device_group
                    break
            if group is None:
                return

            torch.cuda.synchronize()
            if dist.get_backend(group) == "nccl":
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier(group=group)
            torch.cuda.synchronize()
            logger.info_once("Profiler toggle (%s) coordinated across ranks.", tag)
        except Exception as e:
            logger.warning(
                "Profiler toggle (%s) cross-rank coordination failed; "
                "toggling locally instead: %s",
                tag,
                e,
            )

    def _call_start(self, coordinate: bool = False) -> None:
        """Call _start with error handling but no safeguards."""
        if coordinate:
            self._coordinate_toggle_rendezvous("start")
        try:
            self._start()
            self._running = True  # Only mark as running if start succeeds
        except Exception as e:
            logger.warning("Failed to start profiler: %s", e)

    def _call_stop(self, coordinate: bool = False) -> None:
        """Call _stop with error handling but no safeguards."""
        if coordinate:
            self._coordinate_toggle_rendezvous("stop")
        try:
            self._stop()
            logger.info_once("Profiler stopped successfully.")
        except Exception as e:
            logger.warning("Failed to stop profiler: %s", e)
        self._running = False  # Always mark as not running, assume stop worked

    def start(self) -> None:
        """Attempt to start the profiler, accounting for delayed starts."""
        if self._active:
            logger.debug(
                "start_profile received when profiler is already active. "
                "Ignoring request."
            )
            return
        self._active = True
        if self._delay_iters == 0:
            self._call_start()

    def step(self) -> None:
        """Update the profiler state at each worker step,
        to handle delayed starts and max iteration limits."""
        if not self._active:
            return

        self._active_iteration_count += 1

        if (
            not self._running
            and self._delay_iters > 0
            and self._active_iteration_count == self._delay_iters
        ):
            logger.info_once("Starting profiler after delay...")
            # Counter-driven start: every rank trips on the same iteration, so
            # coordinate the toggle to avoid a cross-rank cudaProfilerStart
            # cascade wedging in-flight collectives.
            self._call_start(coordinate=True)

        # Call profiler step for schedule-based profiling
        # Only count iterations where data is actually recorded (not warmup)
        if self._running and self._profiler_step():
            self._profiling_for_iters += 1

        if (
            self._max_iters > 0
            and self._running
            and self._profiling_for_iters > self._max_iters
        ):
            # Automatically stop the profiler after max iters
            # will be marked as not running, but leave as active so that stop
            # can clean up properly
            logger.info_once("Max profiling iterations reached. Stopping profiler...")
            # Counter-driven stop: iteration-aligned across ranks, so coordinate.
            self._call_stop(coordinate=True)
            return

    def _profiler_step(self) -> bool:
        """Called each step when profiler is running.
        Override in subclasses to handle schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        return True

    def stop(self) -> None:
        """Attempt to stop the profiler, accounting for overlapped calls."""
        if not self._active:
            logger.debug(
                "stop_profile received when profiler is not active. Ignoring request."
            )
            return
        self._active = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0

        if self._running:
            self._call_stop()

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        logger.info_once("Shutting down profiler")
        if self._running:
            self.stop()

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return nullcontext()


TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
}


class TorchProfilerWrapper(WorkerProfiler):
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity],
        on_trace_ready: Callable[[torch.profiler.profile], None] | None = None,
    ) -> None:
        super().__init__(profiler_config)

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        torch_profiler_trace_dir = profiler_config.torch_profiler_dir
        if local_rank in (None, 0):
            logger.info_once(
                "Torch profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                profiler_config.torch_profiler_record_shapes,
                profiler_config.torch_profiler_with_memory,
                profiler_config.torch_profiler_with_stack,
                profiler_config.torch_profiler_with_flops,
            )

        # Determine trace handler: use custom handler if provided,
        # otherwise default to tensorboard trace handler
        if on_trace_ready is not None:
            trace_handler = on_trace_ready
        else:
            trace_handler = torch.profiler.tensorboard_trace_handler(
                torch_profiler_trace_dir,
                worker_name=worker_name,
                use_gzip=profiler_config.torch_profiler_use_gzip,
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1

        # Create profiler schedule if warmup or wait iterations are configured
        profiler_schedule = None
        if profiler_config.warmup_iterations > 0 or profiler_config.wait_iterations > 0:
            profiler_schedule = torch.profiler.schedule(
                skip_first=0,
                wait=profiler_config.wait_iterations,
                warmup=profiler_config.warmup_iterations,
                active=profiler_config.active_iterations,
                repeat=1,
            )
            if local_rank in (None, 0):
                logger.info_once(
                    "Profiler schedule configured: wait=%d, warmup=%d, active=%d",
                    profiler_config.wait_iterations,
                    profiler_config.warmup_iterations,
                    profiler_config.active_iterations,
                )

        self.profiler = torch.profiler.profile(
            activities=[TorchProfilerActivityMap[activity] for activity in activities],
            schedule=profiler_schedule,
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=trace_handler,
        )

        # Track if we're using a schedule (need to call step())
        self._uses_schedule = profiler_schedule is not None
        self._warmup_iterations = profiler_config.warmup_iterations
        # Subtract 1 because profiler.start() already consumes step 0
        # (WAIT or WARMUP), so only wait + warmup - 1 non-active steps
        # remain to be advanced through via profiler.step() calls.
        self._warmup_steps_remaining = max(
            profiler_config.wait_iterations + profiler_config.warmup_iterations - 1,
            0,
        )

    def _build_profiler_table(
        self,
        sort_key: str,
        row_limit: int | None = None,
    ) -> str:
        if row_limit is None:  # use profiler default row limit of 100
            return self.profiler.key_averages().table(sort_by=sort_key)
        return self.profiler.key_averages().table(
            sort_by=sort_key,
            row_limit=row_limit,
        )

    def _write_profiler_table(self, rank: int, table: str) -> None:
        profiler_dir = self.profiler_config.torch_profiler_dir

        # Skip file write for URI paths (gs://, s3://, etc.)
        # as standard file I/O doesn't work with URI schemes
        if not _is_uri_path(profiler_dir):
            profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
            with open(profiler_out_file, "w") as f:
                print(table, file=f)

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

        profiler_config = self.profiler_config
        rank = self.local_rank
        if profiler_config.torch_profiler_dump_cuda_time_total:
            table = self._build_profiler_table(sort_key="self_cuda_time_total")
            self._write_profiler_table(rank, table)

            # only print profiler results on rank 0
            if rank == 0:
                print(table)

        if self.dump_cpu_time_total:
            table = self._build_profiler_table(
                sort_key="self_cpu_time_total", row_limit=50
            )
            self._write_profiler_table(rank, table)

            # only print profiler results on rank 0
            if rank == 0:
                print(table)

    @override
    def _profiler_step(self) -> bool:
        """Call profiler.step() when using schedule-based profiling.

        Returns:
            True if the step was an active profiling step (data recorded),
            False if the step was a warmup step (data discarded).
        """
        if self._uses_schedule:
            self.profiler.step()
            # Track warmup steps - only count active steps toward max_iterations
            if self._warmup_steps_remaining > 0:
                self._warmup_steps_remaining -= 1
                return False
        return True

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)


class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        super().__init__(profiler_config)
        # Note: lazy import to avoid dependency issues if CUDA is not available.
        import torch.cuda.profiler as cuda_profiler

        self._cuda_profiler = cuda_profiler
        # cudaProfilerStart/Stop is collective-sensitive under DP/EP; coordinate
        # the counter-driven toggle across ranks (nsys capture path).
        self._coordinate_toggle = True

    @override
    def _start(self) -> None:
        self._cuda_profiler.start()

    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()

    @override
    def annotate_context_manager(self, name: str):
        return torch.cuda.nvtx.range(name)
