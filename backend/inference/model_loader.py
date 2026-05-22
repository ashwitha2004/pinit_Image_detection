"""
AI Detection Model Loader
──────────────────────────────────────────────────────────────────────────────
Lazy-loads the correct classifier architecture from the first available
checkpoint.  Architecture is auto-detected from checkpoint metadata so
both production (EfficientNet dual-branch) and debug (MobileNetV3-Small
single-branch) checkpoints load correctly.

Priority order for checkpoint discovery:
  1. backend/inference/checkpoints/forensic_detector.pth  ← trained model
  2. backend/inference/checkpoints/ai_detector.pth
  3. backend/inference/checkpoints/ai_detector.onnx
  4. backend/outputs/checkpoints/best_model.pth

Architecture detection order (for .pth checkpoints):
  1. ckpt["architecture"] metadata field  ← set by train.py
  2. State-dict key inspection (first linear layer shape)
  3. Default → EfficientNet-B0 dual-branch

Applies temperature scaling from the checkpoint for calibrated probabilities.
Falls back gracefully when torch is not installed (returns None).
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level cache ────────────────────────────────────────────────────────
_model_cache:          Optional[object] = None
_onnx_cache:           Optional[object] = None
_device_cache:         Optional[str]    = None
_temperature_cache:    float            = 1.0
_load_time:            Optional[float]  = None
_has_trained_weights:  bool             = False   # True only when a checkpoint was loaded
_loaded_checkpoint:    Optional[str]    = None    # path of the loaded checkpoint
_architecture_cache:   Optional[str]    = None    # actual arch string from checkpoint metadata
_model_lock                             = threading.Lock()

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_TRAINING_DIR = _HERE.parent / "training"

CHECKPOINT_PATHS = [
    _HERE / "checkpoints" / "forensic_detector.pth",   # trained binary model (priority)
    _HERE / "checkpoints" / "ai_detector.pth",
    _HERE / "checkpoints" / "ai_detector.onnx",
    _HERE.parent / "outputs" / "checkpoints" / "best_model.pth",
]

MODEL_VERSION = "dual_branch_efficientnet_b0_v2"
CLASS_NAMES   = ["REAL_CAMERA", "AI_GENERATED"]

# Architecture name constants (mirrors training/dual_branch_model.py)
_ARCH_EFFICIENTNET = "dual_branch_efficientnet_b0_v2"
_ARCH_DEBUG_MOBILE = "single_branch_mobilenetv3_small_debug"

# Human-readable display names keyed by architecture constant
_ARCH_DISPLAY = {
    _ARCH_EFFICIENTNET: "EfficientNet-B0 Dual-Branch",
    _ARCH_DEBUG_MOBILE: "MobileNetV3-Small Single-Branch (debug)",
}

# First classifier linear layer input dims — used for state-dict sniffing
# when the checkpoint has no "architecture" metadata field
_ARCH_SNIFF = {
    # shape of classifier.1.weight  →  architecture key
    (512, 2560): _ARCH_EFFICIENTNET,   # 1280 * 2 features
    (128, 576):  _ARCH_DEBUG_MOBILE,   # 576 MobileNetV3-Small features
}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture import (shared with training/)
# ─────────────────────────────────────────────────────────────────────────────

def _import_model_classes() -> dict:
    """
    Return {arch_key: model_class} for every known architecture.

    Tries to import from training/dual_branch_model.py first.
    Falls back to inline definitions when the training module is unavailable
    (e.g. inference-only deployment).

    Always returns at least {_ARCH_EFFICIENTNET: <class>}.
    Returns both keys when DebugSingleBranchNet is available.
    """
    classes: dict = {}

    # ── Primary: import from training package ─────────────────────────────────
    try:
        if str(_TRAINING_DIR.parent) not in sys.path:
            sys.path.insert(0, str(_TRAINING_DIR.parent))
        from training.dual_branch_model import (
            DualBranchForensicNet,
            DebugSingleBranchNet,
            ARCHITECTURE,
            ARCHITECTURE_DEBUG,
        )
        classes[ARCHITECTURE]       = DualBranchForensicNet
        classes[ARCHITECTURE_DEBUG] = DebugSingleBranchNet
        logger.debug("[ModelLoader] Imported model classes from training package.")
        return classes
    except ImportError:
        logger.debug("[ModelLoader] training package not importable; using inline fallbacks.")

    # ── Fallback: inline EfficientNet-B0 dual-branch ─────────────────────────
    try:
        import torch
        import torch.nn as nn

        def _build_effnet(pretrained: bool):
            try:
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                net = efficientnet_b0(weights=weights)
            except (ImportError, TypeError):
                from torchvision.models import efficientnet_b0
                net = efficientnet_b0(pretrained=pretrained)
            return nn.Sequential(net.features, net.avgpool)

        class _FallbackDualBranchNet(nn.Module):
            def __init__(self, pretrained: bool = True, dropout: float = 0.3):
                super().__init__()
                self.backbone   = _build_effnet(pretrained)
                self.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(1280 * 2, 512),
                    nn.GELU(),
                    nn.Dropout(p=dropout * 0.67),
                    nn.Linear(512, 128),
                    nn.GELU(),
                    nn.Linear(128, 2),
                )
                for layer in self.classifier:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        nn.init.zeros_(layer.bias)

            def forward(self, original, residual):
                feat_orig  = self.backbone(original).flatten(1)
                feat_resid = self.backbone(residual).flatten(1)
                return self.classifier(torch.cat([feat_orig, feat_resid], dim=1))

        classes[_ARCH_EFFICIENTNET] = _FallbackDualBranchNet

    except Exception as exc:
        logger.warning(f"[ModelLoader] EfficientNet inline fallback failed: {exc}")

    # ── Fallback: inline MobileNetV3-Small single-branch ─────────────────────
    try:
        import torch.nn as nn

        def _build_mobilenet(pretrained: bool):
            try:
                from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
                net = mobilenet_v3_small(weights=weights)
            except (ImportError, TypeError):
                from torchvision.models import mobilenet_v3_small
                net = mobilenet_v3_small(pretrained=pretrained)
            return nn.Sequential(net.features, net.avgpool)

        class _FallbackDebugNet(nn.Module):
            def __init__(self, pretrained: bool = True, dropout: float = 0.2):
                super().__init__()
                self.backbone   = _build_mobilenet(pretrained)
                self.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(576, 128),
                    nn.GELU(),
                    nn.Linear(128, 2),
                )
                for layer in self.classifier:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        nn.init.zeros_(layer.bias)

            def forward(self, original, residual=None):
                return self.classifier(self.backbone(original).flatten(1))

        classes[_ARCH_DEBUG_MOBILE] = _FallbackDebugNet

    except Exception as exc:
        logger.debug(f"[ModelLoader] MobileNet inline fallback failed: {exc}")

    return classes


def _sniff_arch_from_state(state: dict) -> Optional[str]:
    """
    Detect architecture from state_dict key shapes when metadata is absent.
    Checks the first linear layer in the classifier head.
    Returns architecture key or None if unrecognised.
    """
    w = state.get("classifier.1.weight")
    if w is None:
        return None
    try:
        shape = tuple(w.shape)
        return _ARCH_SNIFF.get(shape)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper classes (uniform .predict() interface)
# ─────────────────────────────────────────────────────────────────────────────

class DualBranchEfficientNet:
    """PyTorch model wrapper with temperature-calibrated predictions."""

    def __init__(self, torch_model, device: str, temperature: float = 1.0):
        self.model       = torch_model
        self.device      = device
        self.temperature = max(temperature, 0.05)

    def predict(self, original_tensor, residual_tensor) -> Tuple[float, float]:
        """Returns (ai_probability, camera_probability) — calibrated softmax."""
        import torch

        self.model.eval()
        with torch.no_grad():
            orig  = _to_torch(original_tensor).to(self.device)
            resid = _to_torch(residual_tensor).to(self.device)
            logits = self.model(orig, resid)            # (1, 2)
            probs  = torch.softmax(logits / self.temperature, dim=1)[0]

        camera_p = float(probs[0].cpu())
        ai_p     = float(probs[1].cpu())
        return ai_p, camera_p

    def update_temperature(self, T: float):
        self.temperature = max(T, 0.05)


class OnnxBranchModel:
    """ONNX Runtime inference wrapper."""

    def __init__(self, session, temperature: float = 1.0):
        self.session     = session
        self.temperature = max(temperature, 0.05)
        self.input_names = [inp.name for inp in session.get_inputs()]

    def predict(self, original_tensor, residual_tensor) -> Tuple[float, float]:
        orig_np  = _to_numpy(original_tensor)
        resid_np = _to_numpy(residual_tensor)
        feeds    = {self.input_names[0]: orig_np, self.input_names[1]: resid_np}
        logits   = self.session.run(None, feeds)[0][0]     # (2,)

        logits = logits / self.temperature
        exp    = np.exp(logits - np.max(logits))
        probs  = exp / exp.sum()
        return float(probs[1]), float(probs[0])  # (ai, camera)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(t) -> np.ndarray:
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.array(t, dtype=np.float32)


def _to_torch(t):
    import torch
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_numpy(np.array(t, dtype=np.float32))


def _try_load_onnx(path: Path, temperature: float = 1.0) -> Optional[OnnxBranchModel]:
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(path), opts, providers=providers)
        logger.info(f"[ModelLoader] ONNX loaded: {path}")
        return OnnxBranchModel(sess, temperature)
    except Exception as e:
        logger.debug(f"[ModelLoader] ONNX load failed ({path}): {e}")
        return None


def _try_load_pth(
    path: Path,
    device: str,
    model_classes: dict,
) -> Tuple[Optional[object], float, Optional[str]]:
    """
    Load a .pth checkpoint, auto-detecting the correct architecture.

    Detection order:
      1. ckpt["architecture"] metadata key  (set by train.py since v2)
      2. State-dict shape sniffing          (for older checkpoints)
      3. Default to EfficientNet dual-branch

    Returns (model_wrapper, temperature, arch_key).
    Returns (None, 1.0, None) on failure.
    """
    try:
        import torch
        ckpt  = torch.load(str(path), map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        T     = float(ckpt.get("temperature", 1.0))

        # ── Step 1: read architecture from metadata ──────────────────────────
        arch = ckpt.get("architecture")

        # ── Step 2: sniff from state-dict if not in metadata ─────────────────
        if arch is None or arch not in model_classes:
            sniffed = _sniff_arch_from_state(state)
            if sniffed and sniffed in model_classes:
                logger.info(
                    f"[ModelLoader] arch metadata='{arch}' not recognised; "
                    f"inferred '{sniffed}' from state-dict shapes."
                )
                arch = sniffed

        # ── Step 3: default to EfficientNet ──────────────────────────────────
        if arch is None or arch not in model_classes:
            if arch:
                logger.warning(
                    f"[ModelLoader] Unknown architecture '{arch}'. "
                    f"Known: {list(model_classes.keys())}. Defaulting to EfficientNet."
                )
            arch = _ARCH_EFFICIENTNET

        if arch not in model_classes:
            logger.error("[ModelLoader] No model class available. Cannot load checkpoint.")
            return None, 1.0, None

        model_cls = model_classes[arch]
        logger.info(
            f"[ModelLoader] Loading {path.name}  "
            f"arch={arch}  epoch={ckpt.get('epoch')}  T={T:.4f}"
        )

        # ── Instantiate and load weights ──────────────────────────────────────
        model     = model_cls(pretrained=False).to(device)
        own_state = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in own_state and own_state[k].shape == v.shape
        }

        match_pct = len(compatible) / max(len(own_state), 1)
        if match_pct < 0.5:
            logger.warning(
                f"[ModelLoader] Only {len(compatible)}/{len(own_state)} keys matched "
                f"({match_pct:.0%}). Checkpoint may be incompatible with '{arch}'."
            )

        model.load_state_dict({**own_state, **compatible})
        model.eval()

        if len(compatible) < 5:
            logger.warning("[ModelLoader] Too few matching keys; treating as untrained.")
            T = 1.0

        logger.info(
            f"[ModelLoader] Loaded {len(compatible)}/{len(own_state)} keys "
            f"({match_pct:.0%} match) from {path.name}"
        )
        return DualBranchEfficientNet(model, device, T), T, arch

    except Exception as exc:
        logger.warning(f"[ModelLoader] Failed to load {path}: {exc}")
        return None, 1.0, None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint_order() -> list:
    """
    Return the checkpoint search list, optionally reordered by the
    AI_DETECTOR_BACKEND environment variable.

    AI_DETECTOR_BACKEND=onnx  — prefer the .onnx checkpoint over .pth
                                (useful for cloud deployment where ONNX RT
                                 is 2–3× faster on CPU).
    AI_DETECTOR_BACKEND=pytorch (default) — standard priority order.

    Set in shell:
        export AI_DETECTOR_BACKEND=onnx    (Linux/macOS)
        set    AI_DETECTOR_BACKEND=onnx    (Windows)
    Or in backend/.env:
        AI_DETECTOR_BACKEND=onnx
    """
    import os
    backend_pref = os.environ.get("AI_DETECTOR_BACKEND", "pytorch").strip().lower()

    if backend_pref == "onnx":
        # Move .onnx paths to the front, keep .pth paths as fallback
        onnx_paths = [p for p in CHECKPOINT_PATHS if p.suffix == ".onnx"]
        pth_paths  = [p for p in CHECKPOINT_PATHS if p.suffix != ".onnx"]
        ordered    = onnx_paths + pth_paths
        logger.info(
            "[ModelLoader] AI_DETECTOR_BACKEND=onnx — ONNX Runtime preferred over PyTorch."
        )
        return ordered

    return list(CHECKPOINT_PATHS)   # default priority


def get_model() -> Tuple[Optional[object], Optional[str]]:
    """
    Return (model_wrapper, device).
    model_wrapper has .predict(orig_tensor, resid_tensor) → (ai_prob, cam_prob).
    Returns (None, "none") when torch/onnxruntime are unavailable.
    Thread-safe; loads once per process.

    Backend selection:
        Set env var AI_DETECTOR_BACKEND=onnx to prefer ONNX Runtime.
        Default is PyTorch (.pth checkpoint takes priority).
    """
    global _model_cache, _onnx_cache, _device_cache, _temperature_cache, _load_time, \
           _has_trained_weights, _loaded_checkpoint, _architecture_cache

    if _model_cache is not None or _onnx_cache is not None:
        return _model_cache or _onnx_cache, _device_cache

    with _model_lock:
        if _model_cache is not None or _onnx_cache is not None:
            return _model_cache or _onnx_cache, _device_cache

        t0            = time.time()
        model_classes = _import_model_classes()

        for cp in _resolve_checkpoint_order():
            if not cp.exists():
                continue

            if cp.suffix == ".onnx":
                onnx_m = _try_load_onnx(cp)
                if onnx_m:
                    _onnx_cache          = onnx_m
                    _device_cache        = "onnx"
                    _has_trained_weights = True
                    _loaded_checkpoint   = str(cp)
                    _architecture_cache  = _ARCH_EFFICIENTNET   # ONNX export assumed EfficientNet
                    _load_time           = time.time() - t0
                    return _onnx_cache, _device_cache

            elif cp.suffix == ".pth" and model_classes:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

                wrapper, T, arch = _try_load_pth(cp, device, model_classes)
                if wrapper is not None:
                    _model_cache         = wrapper
                    _device_cache        = device
                    _temperature_cache   = T
                    _has_trained_weights = True
                    _loaded_checkpoint   = str(cp)
                    _architecture_cache  = arch
                    _load_time           = time.time() - t0
                    logger.info(
                        f"[ModelLoader] Ready on {device} in {_load_time:.2f}s  "
                        f"checkpoint={cp.name}  arch={arch}"
                    )
                    return _model_cache, _device_cache

        # No checkpoint found — build fresh EfficientNet (random head, ImageNet backbone)
        effnet_cls = model_classes.get(_ARCH_EFFICIENTNET)
        if effnet_cls is not None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model  = effnet_cls(pretrained=True).to(device)
                model.eval()
                _model_cache         = DualBranchEfficientNet(model, device, 1.0)
                _device_cache        = device
                _has_trained_weights = False   # ImageNet backbone only — no forensic training
                _architecture_cache  = _ARCH_EFFICIENTNET
                _load_time           = time.time() - t0
                logger.warning(
                    "[ModelLoader] No trained checkpoint found. "
                    "Using ImageNet-pretrained EfficientNet-B0 + random head. "
                    "Run backend/run_training.bat (or .sh) to train the model."
                )
                return _model_cache, _device_cache
            except ImportError:
                pass

        logger.warning("[ModelLoader] torch not available. Forensic-only mode.")
        _device_cache = "none"
        _load_time    = time.time() - t0
        return None, _device_cache


def get_temperature() -> float:
    """Return the temperature used for the current model."""
    get_model()   # ensure loaded
    return _temperature_cache


def get_model_info() -> dict:
    model, device = get_model()
    T = _temperature_cache

    # Resolve human-readable architecture display name
    arch_display = _ARCH_DISPLAY.get(
        _architecture_cache or "",
        _architecture_cache or "unknown",
    )

    return {
        "model_version":       MODEL_VERSION,
        "architecture":        arch_display,
        "architecture_key":    _architecture_cache or "unknown",
        "classes":             CLASS_NAMES,
        "device":              device or "none",
        "loaded":              model is not None,
        "has_trained_weights": _has_trained_weights,
        "checkpoint":          (
            Path(_loaded_checkpoint).name if _loaded_checkpoint else None
        ),
        "checkpoint_path":     _loaded_checkpoint,
        "temperature":         round(T, 4),
        "load_time_s":         round(_load_time, 3) if _load_time else None,
        "backend": (
            "onnxruntime" if isinstance(model, OnnxBranchModel)
            else "pytorch"  if isinstance(model, DualBranchEfficientNet)
            else "none"
        ),
    }


def export_to_onnx(output_path: Optional[Path] = None) -> Path:
    """Export current PyTorch model to ONNX. Requires torch."""
    import torch
    model_wrapper, device = get_model()
    if not isinstance(model_wrapper, DualBranchEfficientNet):
        raise RuntimeError("PyTorch model must be loaded before ONNX export.")

    output_path = output_path or (_HERE / "checkpoints" / "ai_detector.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 224, 224, device=device)
    model_wrapper.model.eval()

    torch.onnx.export(
        model_wrapper.model,
        (dummy, dummy),
        str(output_path),
        input_names=["original", "residual"],
        output_names=["logits"],
        dynamic_axes={"original": {0: "batch"}, "residual": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"[ModelLoader] ONNX exported → {output_path}")
    return output_path
