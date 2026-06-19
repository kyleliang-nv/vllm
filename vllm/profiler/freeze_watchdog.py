"""Freeze watchdog for diagnosing the nsys capture-under-load deadlock.

This module is a *diagnostic-only* instrument. It is gated entirely by the env
var ``VLLM_FREEZE_WATCHDOG=1``; when that is unset (the default) every public
entrypoint is a no-op and the module adds zero runtime overhead.

Why this exists
---------------
On GB300 DEP decode (Kimi-K2.5 NVFP4, async-scheduling + mp + one-sided
flashinfer NVLink all2all), engaging ``cudaProfilerStart`` mid-stream under full
``gen=8`` load wedges all DP ranks simultaneously. The process then sits idle
until ``shm_broadcast`` logs its 60s warnings and finally the ``sample_tokens``
RPC times out (~5 min) and the engine is killed.

Prior attempts only had Python tracebacks captured at *teardown* (on SIGKILL),
by which point the real-work ranks had already unwound out of CUDA and showed
"no traceback" -> we could never name the stuck collective.

This watchdog instead captures stacks *at hang time*:
  * a per-process iteration heartbeat (``beat()``) is touched every engine step,
  * a daemon thread stays dormant until stepping has started (so it never
    perturbs cudagraph capture / warmup -- the failure mode misun hit when
    arming faulthandler too early), and
  * once steps STOP advancing for ``VLLM_FREEZE_WATCHDOG_STALL_SECS`` it captures
    three independent layers, each best-effort:
        1. faulthandler.dump_traceback(all_threads=True)  -- own Python stacks,
           no external deps. Captured at hang time, so a rank wedged in a
           CUDA/C++ collective still shows the Python call-site that entered it.
        2. ``py-spy dump --native --pid <pid>``            -- Python + native C
           frames (names the NVSHMEM / DeepGEMM / cuda frame the thread blocks
           in). Run over EVERY local python pid.
        3. ``gdb -p <pid> -batch thread apply all bt``      -- pure native
           backtrace as a fallback when py-spy is unavailable.

Layers 2/3 sweep every local python pid, so a single firing process dumps the
whole node. A per-node, per-attempt sentinel keeps exactly one process doing the
(expensive) node-wide py-spy/gdb sweep while every process still writes its own
faulthandler dump.

Env knobs
---------
  VLLM_FREEZE_WATCHDOG=1               enable (required).
  VLLM_FREEZE_WATCHDOG_DIR=<path>      output dir. STRONGLY recommend a Lustre
                                       path so dumps survive the job; defaults
                                       to /tmp/vllm_freeze_diag (node-local).
  VLLM_FREEZE_WATCHDOG_STALL_SECS=75   idle seconds before declaring a freeze.
                                       Keep < the shm_broadcast 60s * a couple
                                       windows and well under VLLM_RPC_TIMEOUT.
  VLLM_FREEZE_WATCHDOG_MAX_DUMPS=2     how many dump rounds (2 => one confirming
                                       round ~CONFIRM_GAP later proves "wedged"
                                       vs "merely slow").
  VLLM_FREEZE_WATCHDOG_CONFIRM_GAP=30  seconds between dump rounds.
  VLLM_FREEZE_WATCHDOG_MIN_STEPS=50    require at least this many steps before
                                       arming the stall check (extra warmup
                                       guard on top of "stepping has started").
  VLLM_FREEZE_WATCHDOG_TOOL_TIMEOUT=60 per-tool subprocess timeout (s).
  VLLM_FREEZE_WATCHDOG_PYSPY=1         enable py-spy layer (default 1).
  VLLM_FREEZE_WATCHDOG_GDB=1           enable gdb layer (default 1).
"""

from __future__ import annotations

import faulthandler
import os
import socket
import subprocess
import sys
import threading
import time

_ENABLED = os.environ.get("VLLM_FREEZE_WATCHDOG", "") == "1"

_armed = False
_arm_lock = threading.Lock()

# Heartbeat state. Plain attribute writes on a module global are atomic enough
# under the GIL for our coarse (seconds-scale) timing; no lock on the hot path.
_last_beat_monotonic: float | None = None
_steps_seen = 0


def _log(msg: str) -> None:
    sys.stderr.write(f"[freeze-watchdog] {msg}\n")
    sys.stderr.flush()


def beat() -> None:
    """Mark forward progress. Cheap; safe to call every engine iteration."""
    if not _ENABLED:
        return
    global _last_beat_monotonic, _steps_seen
    _last_beat_monotonic = time.monotonic()
    _steps_seen += 1


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _diag_dir() -> str:
    d = os.environ.get("VLLM_FREEZE_WATCHDOG_DIR") or "/tmp/vllm_freeze_diag"
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:  # pragma: no cover - defensive
        _log(f"could not create diag dir {d!r}: {e}; falling back to /tmp")
        d = "/tmp"
    return d


def _run_to_file(cmd: list[str], path: str, timeout: float) -> None:
    """Run ``cmd``, append stdout+stderr to ``path``. Never raises."""
    try:
        with open(path, "ab") as f:
            f.write(f"\n$ {' '.join(cmd)}\n".encode())
            f.flush()
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            f.write(f"\n[exit {proc.returncode}]\n".encode())
    except subprocess.TimeoutExpired:
        try:
            with open(path, "ab") as f:
                f.write(f"\n[TIMEOUT after {timeout}s running {cmd[0]}]\n".encode())
        except OSError:
            pass
    except FileNotFoundError:
        try:
            with open(path, "ab") as f:
                f.write(f"\n[tool not found: {cmd[0]}]\n".encode())
        except OSError:
            pass
    except Exception as e:  # pragma: no cover - defensive
        _log(f"tool {cmd[0]} failed: {e}")


def _local_python_pids() -> list[int]:
    """All python pids on this node (vllm/dynamo engine + worker procs)."""
    pids: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return [os.getpid()]
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            continue
        low = cmdline.lower()
        if (
            "python" in low
            or "vllm" in low
            or "dynamo" in low
            or "enginecore" in low
        ):
            pids.append(pid)
    # Always include ourselves even if the heuristic missed us.
    me = os.getpid()
    if me not in pids:
        pids.append(me)
    return sorted(set(pids))


def _capture(role: str, rank: int, idle: float, attempt: int) -> None:
    diag = _diag_dir()
    host = socket.gethostname()
    pid = os.getpid()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tool_timeout = _env_float("VLLM_FREEZE_WATCHDOG_TOOL_TIMEOUT", 60.0)

    base = os.path.join(
        diag, f"freeze_{host}_r{rank}_{role}_pid{pid}_a{attempt}_{stamp}"
    )
    _log(
        f"FREEZE detected (role={role} rank={rank} pid={pid} host={host} "
        f"idle={idle:.0f}s steps={_steps_seen} attempt={attempt}). "
        f"Dumping stacks to {base}.*"
    )

    # Layer 1: own Python stacks (no external deps, every process does this).
    fh_path = f"{base}.faulthandler.txt"
    try:
        with open(fh_path, "w") as f:
            f.write(
                f"# freeze-watchdog faulthandler dump\n"
                f"# host={host} pid={pid} rank={rank} role={role}\n"
                f"# idle={idle:.1f}s steps_seen={_steps_seen} attempt={attempt}\n"
                f"# time={stamp}\n\n"
            )
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
    except Exception as e:  # pragma: no cover - defensive
        _log(f"faulthandler dump failed: {e}")

    # Layers 2/3: node-wide native sweep. Exactly one process per (host,attempt)
    # wins the sentinel and runs the expensive py-spy/gdb sweep over all pids.
    sentinel = os.path.join(diag, f".nodesweep_{host}_a{attempt}.lock")
    won = False
    try:
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{host} pid={pid} {stamp}\n".encode())
        os.close(fd)
        won = True
    except FileExistsError:
        won = False
    except OSError:
        won = False

    if not won:
        return

    want_pyspy = os.environ.get("VLLM_FREEZE_WATCHDOG_PYSPY", "1") == "1"
    want_gdb = os.environ.get("VLLM_FREEZE_WATCHDOG_GDB", "1") == "1"

    sweep_path = f"{base}.nodesweep.txt"
    pids = _local_python_pids()
    try:
        with open(sweep_path, "w") as f:
            f.write(
                f"# freeze-watchdog node-wide native sweep\n"
                f"# host={host} winner_pid={pid} attempt={attempt} time={stamp}\n"
                f"# local python pids: {pids}\n"
            )
    except OSError:
        pass

    _run_to_file(["nvidia-smi"], sweep_path, tool_timeout)

    for target in pids:
        if want_pyspy:
            _run_to_file(
                ["py-spy", "dump", "--pid", str(target), "--native"],
                f"{base}.pyspy_pid{target}.txt",
                tool_timeout,
            )
        if want_gdb:
            _run_to_file(
                [
                    "gdb",
                    "-p",
                    str(target),
                    "-batch",
                    "-ex",
                    "set pagination off",
                    "-ex",
                    "thread apply all bt",
                ],
                f"{base}.gdb_pid{target}.txt",
                tool_timeout,
            )
    _log(f"node-wide sweep complete -> {base}.*")


def _watch_loop(role: str, rank: int) -> None:
    stall = _env_float("VLLM_FREEZE_WATCHDOG_STALL_SECS", 75.0)
    max_dumps = _env_int("VLLM_FREEZE_WATCHDOG_MAX_DUMPS", 2)
    confirm_gap = _env_float("VLLM_FREEZE_WATCHDOG_CONFIRM_GAP", 30.0)
    min_steps = _env_int("VLLM_FREEZE_WATCHDOG_MIN_STEPS", 50)
    poll = 5.0

    _log(
        f"armed (role={role} rank={rank} pid={os.getpid()}); "
        f"stall={stall:.0f}s max_dumps={max_dumps} min_steps={min_steps} "
        f"dir={_diag_dir()}"
    )

    dumps_done = 0
    while dumps_done < max_dumps:
        time.sleep(poll)
        lb = _last_beat_monotonic
        # Dormant until stepping has actually started (past warmup / cudagraph
        # capture). This is the key guard that avoids perturbing capture.
        if lb is None or _steps_seen < min_steps:
            continue
        idle = time.monotonic() - lb
        if idle < stall:
            continue
        _capture(role, rank, idle, attempt=dumps_done + 1)
        dumps_done += 1
        if dumps_done < max_dumps:
            time.sleep(confirm_gap)

    _log("watchdog finished (max dumps reached)")


def arm(role: str = "worker", rank: int = -1) -> None:
    """Start the watchdog thread once per process. No-op unless enabled."""
    if not _ENABLED:
        return
    global _armed
    with _arm_lock:
        if _armed:
            return
        _armed = True
    # Make the Python interpreter dump on hard faults too (cheap, complementary).
    try:
        faulthandler.enable(all_threads=True)
    except Exception:  # pragma: no cover - defensive
        pass
    t = threading.Thread(
        target=_watch_loop,
        args=(role, rank),
        name="vllm-freeze-watchdog",
        daemon=True,
    )
    t.start()
