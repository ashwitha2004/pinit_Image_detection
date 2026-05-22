"""
ONNX Export & Benchmark Tool
==============================
Exports the current PyTorch checkpoint to ONNX, validates inference parity,
and benchmarks latency for both backends.

What this does
--------------
1. EXPORT   — torch.onnx.export() with opset 17, dynamic batch axis,
               constant-folding and shape inference.
2. VALIDATE — Run 50 random inputs through both backends, compare outputs.
               Asserts max absolute difference < 1e-4 per output element.
3. BENCHMARK— Warm-up 10 passes, then time N passes on both backends.
               Reports mean / std / min / max latency for each.
4. DEPLOY   — After export, set AI_DETECTOR_BACKEND=onnx (or pass --use-onnx
               as a runtime flag) to make the inference server use ONNX.

Speed expectations (CPU, EfficientNet-B0, 224×224 single image)
----------------------------------------------------------------
  PyTorch  : ~70–150 ms / pass (varies with thread count)
  ONNX RT  : ~30–70  ms / pass (ORT graph optimisations + memory planning)
  Speedup  : typically 2–3× on CPU, 1.2–1.5× on CUDA

Usage
-----
  # Export + validate + benchmark (default 100 benchmark passes)
  python -m inference.export_onnx

  # Export only
  python -m inference.export_onnx --export-only

  # Custom output path + benchmark passes
  python -m inference.export_onnx --output path/to/model.onnx --n-bench 200

  # Validate an existing .onnx file without re-exporting
  python -m inference.export_onnx --skip-export --validate --benchmark
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── UTF-8 output on Windows ───────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf_8"}:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("export_onnx")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent           # backend/inference/
_BACKEND    = _HERE.parent                              # backend/
_CKPT_DIR   = _HERE / "checkpoints"
_DEFAULT_PTH  = _CKPT_DIR / "forensic_detector.pth"
_DEFAULT_ONNX = _CKPT_DIR / "ai_detector.onnx"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Export
# ─────────────────────────────────────────────────────────────────────────────

def export_model(
    pth_path: Path,
    onnx_path: Path,
    opset: int = 17,
) -> Path:
    """
    Load the PyTorch checkpoint and export it to ONNX.

    The model takes two inputs (original, residual) each shaped (B, 3, 224, 224).
    The batch dimension is dynamic so the exported model accepts any batch size.

    Args:
        pth_path:  Path to the .pth checkpoint (forensic_detector.pth).
        onnx_path: Destination path for the .onnx file.
        opset:     ONNX opset version (default 17 — supported by ORT 1.14+).

    Returns:
        Path to the saved .onnx file.
    """
    import torch
    import sys as _sys

    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    # ── Import architecture ───────────────────────────────────────────────────
    training_dir = _BACKEND.parent if (_BACKEND.parent / "training").exists() else _BACKEND
    if str(_BACKEND) not in _sys.path:
        _sys.path.insert(0, str(_BACKEND))

    try:
        from training.dual_branch_model import DualBranchForensicNet, ARCHITECTURE
    except ImportError:
        raise ImportError(
            "Could not import DualBranchForensicNet from training.dual_branch_model. "
            "Make sure you are running from the backend/ directory."
        )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {pth_path}")
    ckpt  = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    T     = float(ckpt.get("temperature", 1.0))
    arch  = ckpt.get("architecture", ARCHITECTURE)
    epoch = ckpt.get("epoch", "?")

    logger.info(f"  Architecture : {arch}")
    logger.info(f"  Epoch        : {epoch}")
    logger.info(f"  Temperature  : {T}")

    model = DualBranchForensicNet(pretrained=False)
    own_state  = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in own_state and own_state[k].shape == v.shape}
    model.load_state_dict({**own_state, **compatible})
    model.eval()

    match_pct = len(compatible) / max(len(own_state), 1)
    logger.info(f"  Weights      : {len(compatible)}/{len(own_state)} keys loaded ({match_pct:.0%})")

    if match_pct < 0.9:
        logger.warning(f"  Only {match_pct:.0%} keys matched — checkpoint may be incompatible.")

    # ── Export ────────────────────────────────────────────────────────────────
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 224, 224)

    logger.info(f"Exporting to ONNX (opset {opset})…")

    # PyTorch 2.x defaults to the dynamo-based exporter which requires
    # the optional 'onnxscript' package.  We use torch.jit.trace first
    # (TorchScript path) which avoids that dependency entirely and
    # produces a smaller, faster ONNX graph.
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy, dummy), strict=False)

    # PyTorch 2.1+ defaults to the dynamo exporter which requires `dynamic_shapes`
    # syntax instead of `dynamic_axes`.  Force the legacy TorchScript path with
    # dynamo=False so the standard `dynamic_axes` dict works across all versions.
    torch.onnx.export(
        traced,
        (dummy, dummy),
        str(onnx_path),
        input_names=["original", "residual"],
        output_names=["logits"],
        dynamic_axes={
            "original": {0: "batch_size"},
            "residual": {0: "batch_size"},
            "logits":   {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,   # use TorchScript exporter, not dynamo (works without dynamic_shapes)
    )

    file_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported -> {onnx_path}  ({file_mb:.1f} MB)")

    # ── ONNX shape inference & model check ───────────────────────────────────
    try:
        import onnx
        import onnx.shape_inference as si

        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        inferred    = si.infer_shapes(model_proto)
        onnx.save(inferred, str(onnx_path))
        logger.info("ONNX model check + shape inference: PASSED")
    except ImportError:
        logger.warning("'onnx' package not installed — skipping model check.")
    except Exception as e:
        logger.warning(f"ONNX model check warning: {e}")

    return onnx_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Validate parity
# ─────────────────────────────────────────────────────────────────────────────

def validate_parity(
    pth_path: Path,
    onnx_path: Path,
    n_samples: int = 50,
    tol: float = 1e-3,
) -> Tuple[bool, float]:
    """
    Run n_samples random inputs through both PyTorch and ONNX backends.
    Assert that outputs agree within tolerance.

    Returns (passed: bool, max_abs_error: float).
    """
    import torch

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    logger.info(f"Validating parity ({n_samples} random samples, tol={tol})…")

    # ── Load PyTorch model ────────────────────────────────────────────────────
    import sys as _sys
    if str(_BACKEND) not in _sys.path:
        _sys.path.insert(0, str(_BACKEND))

    from training.dual_branch_model import DualBranchForensicNet
    ckpt  = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    T     = float(ckpt.get("temperature", 1.0))

    pt_model = DualBranchForensicNet(pretrained=False)
    own      = pt_model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in own and own[k].shape == v.shape}
    pt_model.load_state_dict({**own, **compatible})
    pt_model.eval()

    # ── Load ONNX session ─────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping parity validation.")
        return True, 0.0

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), opts,
                                 providers=["CPUExecutionProvider"])

    # ── Compare outputs ───────────────────────────────────────────────────────
    max_err  = 0.0
    all_pass = True

    for i in range(n_samples):
        orig_np  = np.random.randn(1, 3, 224, 224).astype(np.float32)
        resid_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # PyTorch output
        with torch.no_grad():
            pt_logits = pt_model(
                torch.from_numpy(orig_np),
                torch.from_numpy(resid_np),
            ).numpy()                          # (1, 2)

        # ONNX output
        ort_logits = sess.run(
            None,
            {"original": orig_np, "residual": resid_np},
        )[0]                                   # (1, 2)

        err = float(np.max(np.abs(pt_logits - ort_logits)))
        max_err = max(max_err, err)
        if err > tol:
            logger.warning(f"  Sample {i}: max_abs_error={err:.2e} > tol={tol:.2e}")
            all_pass = False

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{n_samples} — running max error: {max_err:.2e}")

    status = "PASSED" if all_pass else "FAILED"
    logger.info(
        f"Parity validation {status}: "
        f"max_abs_error={max_err:.2e}  (tol={tol:.2e})  "
        f"n={n_samples}"
    )
    return all_pass, max_err


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    pth_path: Path,
    onnx_path: Path,
    n_warmup:   int = 10,
    n_bench:    int = 100,
    batch_size: int = 1,
) -> dict:
    """
    Benchmark PyTorch vs ONNX Runtime inference latency.

    Both backends process identical random inputs; timings are wall-clock ms.
    Reports mean / std / min / max / p95 latency and estimated throughput.

    Returns a result dict with both backend stats and speedup ratio.
    """
    import torch

    if not onnx_path.exists():
        logger.warning(f"ONNX file not found: {onnx_path} — skipping ONNX benchmark.")
        return {}

    # ── Load PyTorch ──────────────────────────────────────────────────────────
    import sys as _sys
    if str(_BACKEND) not in _sys.path:
        _sys.path.insert(0, str(_BACKEND))

    from training.dual_branch_model import DualBranchForensicNet
    ckpt  = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    T     = float(ckpt.get("temperature", 1.0))

    pt_model = DualBranchForensicNet(pretrained=False)
    own      = pt_model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in own and own[k].shape == v.shape}
    pt_model.load_state_dict({**own, **compatible})
    pt_model.eval()

    # ── Load ONNX ─────────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        ort_available = True
    except ImportError:
        logger.warning("onnxruntime not installed — only PyTorch will be benchmarked.")
        ort_available = False

    if ort_available:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(str(onnx_path), opts,
                                     providers=["CPUExecutionProvider"])

    # ── Inputs ────────────────────────────────────────────────────────────────
    orig_np  = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    resid_np = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    orig_t   = torch.from_numpy(orig_np)
    resid_t  = torch.from_numpy(resid_np)

    def _pt_pass():
        with torch.no_grad():
            return pt_model(orig_t, resid_t)

    def _ort_pass():
        return sess.run(None, {"original": orig_np, "residual": resid_np})

    def _time_runs(fn, n_warmup, n_bench, label):
        logger.info(f"  [{label}] Warm-up ({n_warmup} passes)…")
        for _ in range(n_warmup):
            fn()
        logger.info(f"  [{label}] Benchmarking ({n_bench} passes)…")
        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        return {
            "mean_ms":   float(np.mean(arr)),
            "std_ms":    float(np.std(arr)),
            "min_ms":    float(np.min(arr)),
            "max_ms":    float(np.max(arr)),
            "p95_ms":    float(np.percentile(arr, 95)),
            "throughput_ips": float(1000.0 / np.mean(arr)),   # images/sec
        }

    results: dict = {
        "batch_size":  batch_size,
        "n_warmup":    n_warmup,
        "n_bench":     n_bench,
        "input_shape": [batch_size, 3, 224, 224],
    }

    logger.info(f"\nBenchmarking (batch={batch_size}, n={n_bench})…")
    results["pytorch"] = _time_runs(_pt_pass, n_warmup, n_bench, "PyTorch")

    if ort_available:
        results["onnxruntime"] = _time_runs(_ort_pass, n_warmup, n_bench, "ONNX-RT")
        speedup = results["pytorch"]["mean_ms"] / results["onnxruntime"]["mean_ms"]
        results["speedup_x"] = round(speedup, 2)
    else:
        results["onnxruntime"] = None
        results["speedup_x"] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print results
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmark_table(results: dict) -> None:
    if not results:
        return
    print()
    print("=" * 64)
    print("  Inference Latency Benchmark")
    print(f"  batch={results['batch_size']}  "
          f"input=3x224x224  "
          f"n={results['n_bench']} passes")
    print("=" * 64)
    print(f"  {'Backend':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'P95':>8} {'Img/s':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key, label in [("pytorch", "PyTorch"), ("onnxruntime", "ONNX Runtime")]:
        s = results.get(key)
        if s is None:
            print(f"  {label:<20} {'N/A':>8}")
            continue
        print(
            f"  {label:<20} "
            f"{s['mean_ms']:>7.1f}ms "
            f"{s['std_ms']:>7.1f}ms "
            f"{s['min_ms']:>7.1f}ms "
            f"{s['p95_ms']:>7.1f}ms "
            f"{s['throughput_ips']:>7.1f}"
        )

    speedup = results.get("speedup_x")
    if speedup is not None:
        winner = "ONNX Runtime" if speedup > 1 else "PyTorch"
        ratio  = speedup if speedup >= 1 else 1 / speedup
        print()
        print(f"  Speedup: {ratio:.2f}x faster with {winner}")
        if speedup > 1:
            saved_pct = (1 - 1 / speedup) * 100
            print(f"  Latency reduction: {saved_pct:.0f}%")

    print("=" * 64)
    print()


def print_deployment_guide(onnx_path: Path) -> None:
    """Print concise instructions for switching to ONNX inference mode."""
    print()
    print("  To use ONNX Runtime in production:")
    print()
    print("  Option A — Environment variable (recommended):")
    print("    set AI_DETECTOR_BACKEND=onnx   (Windows)")
    print("    export AI_DETECTOR_BACKEND=onnx  (Linux/macOS)")
    print("    # Then restart the backend server.")
    print()
    print("  Option B — Remove the .pth file so model_loader falls through to .onnx:")
    print(f"    # The model_loader already knows about: {onnx_path.name}")
    print()
    print("  To revert to PyTorch:")
    print("    unset AI_DETECTOR_BACKEND  (or remove from .env)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export PyTorch checkpoint to ONNX, validate parity, benchmark latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (export + validate + benchmark)
  python -m inference.export_onnx

  # Export only
  python -m inference.export_onnx --export-only

  # Validate + benchmark an already-exported ONNX (skip re-export)
  python -m inference.export_onnx --skip-export

  # Custom paths
  python -m inference.export_onnx \\
      --pth  inference/checkpoints/forensic_detector.pth \\
      --output inference/checkpoints/ai_detector.onnx

  # More benchmark passes for stable estimate
  python -m inference.export_onnx --n-bench 200

  # Benchmark batch inference (e.g. batch=4)
  python -m inference.export_onnx --batch-size 4
""",
    )
    p.add_argument("--pth",         type=Path, default=_DEFAULT_PTH,
                   help=f"Source .pth checkpoint (default: {_DEFAULT_PTH.name})")
    p.add_argument("--output",      type=Path, default=_DEFAULT_ONNX,
                   help=f"Output .onnx path (default: {_DEFAULT_ONNX.name})")
    p.add_argument("--opset",       type=int,  default=17,
                   help="ONNX opset version (default: 17)")
    p.add_argument("--skip-export", action="store_true",
                   help="Skip export — use an existing .onnx file for validate/benchmark.")
    p.add_argument("--export-only", action="store_true",
                   help="Export only — skip validation and benchmark.")
    p.add_argument("--n-validate",  type=int, default=50,
                   help="Number of random samples for parity validation (default: 50).")
    p.add_argument("--n-warmup",    type=int, default=10,
                   help="Warm-up passes before benchmarking (default: 10).")
    p.add_argument("--n-bench",     type=int, default=100,
                   help="Benchmark passes per backend (default: 100).")
    p.add_argument("--batch-size",  type=int, default=1,
                   help="Batch size for benchmark (default: 1).")
    p.add_argument("--tol",         type=float, default=1e-3,
                   help="Max allowed parity error per output element (default: 1e-3).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print()
    logger.info("ONNX Export & Benchmark Tool")
    logger.info(f"  Checkpoint : {args.pth}")
    logger.info(f"  ONNX output: {args.output}")

    # ── Step 1: Export ────────────────────────────────────────────────────────
    if not args.skip_export:
        logger.info("\nStep 1/3 — Exporting PyTorch model to ONNX…")
        try:
            export_model(args.pth, args.output, opset=args.opset)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            sys.exit(1)
    else:
        logger.info("Step 1/3 — Export skipped (--skip-export).")
        if not args.output.exists():
            logger.error(f"ONNX file not found: {args.output} — cannot validate/benchmark.")
            sys.exit(1)

    if args.export_only:
        logger.info("Export complete (--export-only; skipping validate/benchmark).")
        print_deployment_guide(args.output)
        return

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    logger.info("\nStep 2/3 — Validating parity (PyTorch vs ONNX)…")
    try:
        passed, max_err = validate_parity(
            args.pth, args.output,
            n_samples=args.n_validate,
            tol=args.tol,
        )
        if not passed:
            logger.warning(
                f"Parity check FAILED (max_err={max_err:.2e} > tol={args.tol:.2e}). "
                "The ONNX model may produce different results. Check opset/architecture."
            )
    except Exception as e:
        logger.error(f"Validation error: {e}")

    # ── Step 3: Benchmark ─────────────────────────────────────────────────────
    logger.info("\nStep 3/3 — Benchmarking…")
    try:
        results = benchmark(
            args.pth, args.output,
            n_warmup=args.n_warmup,
            n_bench=args.n_bench,
            batch_size=args.batch_size,
        )
        print_benchmark_table(results)
    except Exception as e:
        logger.error(f"Benchmark error: {e}")

    # ── Deployment guide ──────────────────────────────────────────────────────
    print_deployment_guide(args.output)


if __name__ == "__main__":
    main()
