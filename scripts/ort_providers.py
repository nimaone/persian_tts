# Shared smart execution-provider selection for the ONNX engines (torch-free).
#
# The engines request CUDA (onnxruntime-gpu — fastest on NVIDIA Maxwell+) or
# DirectML (onnxruntime-directml — any DX12 GPU: NVIDIA/AMD/Intel) when that
# build offers them, then MEASURE a one-step micro-benchmark against the CPU:
# weak or small-VRAM GPUs lose to a good CPU on these autoregressive loops,
# so the winner is measured, not assumed. Two traps handled here:
#   - ORT silently falls back to CPU when a provider is missing — the session's
#     get_providers() is the ground truth, not the requested list;
#   - provider builds can throw (old GPU / OOM / missing DLLs) — caught and
#     treated as "CPU wins".
# PARSIGO_EP=cpu|cuda|dml forces a provider; 'auto' (default) benches.
import os
import time

import numpy as np
import onnxruntime as ort

_GPU_ORDER = ("CUDAExecutionProvider", "DmlExecutionProvider")
_FORCED = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}
_LABELS = {"CPUExecutionProvider": "CPU",
           "CUDAExecutionProvider": "GPU (CUDA)",
           "DmlExecutionProvider": "GPU (DirectML)"}


def smart_providers(model_path, opts, feeds, log=print):
    """Pick the fastest execution provider for `model_path` (usually the
    engine's main autoregressive step graph). `feeds` must be one
    representative step's inputs. Returns the provider LIST for
    InferenceSession(..., providers=...) — always contains CPU as fallback."""
    avail = ort.get_available_providers()
    want = os.environ.get("PARSIGO_EP", "auto").lower()
    if want == "cpu":
        return ["CPUExecutionProvider"]
    if want in _FORCED:
        if _FORCED[want] in avail:
            log(f"execution provider (forced): {_FORCED[want]}")
            return [_FORCED[want], "CPUExecutionProvider"]
        log(f"execution provider (forced): CPU — {_FORCED[want]} is not in "
            f"this onnxruntime build ({avail})")
        return ["CPUExecutionProvider"]

    gpu_names = [g for g in _GPU_ORDER if g in avail]
    if not gpu_names:
        return ["CPUExecutionProvider"]      # CPU-only wheel: as before

    t_cpu = None
    best_t, best_name = None, None
    for name in gpu_names:
        t_gpu = _bench_one(model_path, opts, feeds, name, log)
        if t_gpu is None:
            log(f"execution provider: CPU ({name} unavailable)")
            continue
        if t_cpu is None:
            t_cpu = _bench_one(model_path, opts, feeds,
                               "CPUExecutionProvider", log)
        log(f"[{name}] GPU {t_gpu:.3f}s vs CPU {t_cpu:.3f}s per step")
        if t_cpu is not None and t_gpu < t_cpu and \
                (best_t is None or t_gpu < best_t):
            best_t, best_name = t_gpu, name
    if best_name:
        log(f"execution provider: {best_name}")
        return [best_name, "CPUExecutionProvider"]
    log("execution provider: CPU")
    return ["CPUExecutionProvider"]


def device_label(prov_list):
    """Friendly badge for /api/config — CPU / GPU (CUDA) / GPU (DirectML)."""
    return _LABELS.get(prov_list[0], prov_list[0])


def _bench_one(model_path, opts, feeds, provider, log):
    """min of two runs of one step; None when the provider cannot serve."""
    try:
        s = ort.InferenceSession(str(model_path), opts,
                                 providers=[provider, "CPUExecutionProvider"])
        # ORT silently falls back to CPU when a provider is missing — the
        # activated list is the ground truth, not our request
        if provider != "CPUExecutionProvider" and \
                provider not in s.get_providers():
            log(f"[{provider}] not active in this build "
                f"(activated: {s.get_providers()[0]})")
            del s
            return None
        best = None
        for _ in range(2):
            t0 = time.perf_counter()
            s.run(None, feeds)
            best = min(best or 1e9, time.perf_counter() - t0)
        del s
        return best
    except Exception as e:   # old GPU / OOM / missing DLLs
        log(f"[{provider}] session failed: {type(e).__name__}: {str(e)[:120]}")
        return None
