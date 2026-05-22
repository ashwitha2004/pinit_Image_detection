"""
Hybrid AI Detection API Routes
================================
POST /api/inference/detect-ai           — multipart file upload
POST /api/inference/detect-ai-base64   — base64 JSON body (frontend-friendly)
POST /api/inference/evaluate-model     — evaluation metrics on local test directory
GET  /api/inference/model-info         — loaded model metadata
GET  /api/inference/health             — service health check
GET  /api/inference/confusion-matrix   — live confusion-matrix stats (since restart)
"""

import base64
import logging
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Body, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .ai_detector import get_detector
from .fusion import FusionResult, build_fusion_input, fuse
from .model_loader import get_model_info, CHECKPOINT_PATHS

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/inference",
    tags=["inference"],
)

# ── In-memory confusion matrix (reset on restart) ────────────────────────────
_cm_stats: Dict[str, int] = defaultdict(int)
#   Keys: "total", "ai_predicted", "camera_predicted",
#         "true_ai", "true_camera",              (when ground-truth provided)
#         "TP", "TN", "FP", "FN"

# ── Response schemas ──────────────────────────────────────────────────────────

class BranchScores(BaseModel):
    cnn_score:      float = Field(..., description="AI probability — RGB image branch")
    residual_score: float = Field(..., description="AI probability — noise residual branch")
    fft_score:      float = Field(..., description="AI probability — FFT frequency branch")
    forensic_score: float = Field(..., description="AI probability — heuristic forensic")
    metadata_score: float = Field(..., description="AI probability — EXIF metadata")


class ResidualStats(BaseModel):
    residual_mean_abs:  float
    residual_std:       float
    residual_kurtosis:  float
    channel_correlation: float


class FFTStats(BaseModel):
    hf_ratio:         float
    spectral_entropy: float
    radial_falloff:   float
    fft_ai_proxy:     Optional[float] = None
    cnn_ai_proxy:     Optional[float] = None
    resid_ai_proxy:   Optional[float] = None


class CalibrationInfo(BaseModel):
    temperature:         float
    has_trained_weights: bool
    heuristic_proxy:     bool
    model_backend:       str   # "pytorch" | "onnxruntime" | "none"


class DetectAIResponse(BaseModel):
    """Unified three-branch hybrid detection response."""
    # Final probabilities
    ai_probability:     float = Field(..., ge=0, le=1)
    camera_probability: float = Field(..., ge=0, le=1)

    # Per-layer confidence
    dl_confidence:       float
    forensic_confidence: float
    fusion_confidence:   float

    # Dominant reasons
    dominant_signals: List[str]

    # Raw forensic signals (legacy compatibility)
    forensic_signals: Dict[str, Any]

    # Per-branch AI probabilities
    branch_scores: BranchScores

    # Weights used in fusion (for debug panel)
    fusion_weights: Dict[str, float]

    # Residual + FFT diagnostics
    residual_stats: Optional[ResidualStats] = None
    fft_stats:      Optional[FFTStats]      = None

    # Calibration / model state
    calibration: CalibrationInfo

    # Meta
    model_version:      str
    dl_available:       bool
    device_used:        str
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    model_version:       str
    architecture:        str
    architecture_key:    Optional[str]  = None
    classes:             List[str]
    device:              str
    loaded:              bool
    has_trained_weights: bool           = False
    checkpoint:          Optional[str]  = None
    load_time_s:         Optional[float]
    backend:             str


class ConfusionMatrixResponse(BaseModel):
    total_predictions: int
    ai_predicted:      int
    camera_predicted:  int
    # Only populated when ground-truth labels were supplied via evaluate endpoint
    TP: int; TN: int; FP: int; FN: int
    accuracy:   Optional[float]
    precision:  Optional[float]
    recall:     Optional[float]
    f1:         Optional[float]


# ── Helpers ───────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _validate_upload(image: UploadFile) -> str:
    if not image.filename:
        raise HTTPException(400, "No filename provided.")
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")
    if image.size is not None and image.size > MAX_FILE_SIZE:
        raise HTTPException(400, "File exceeds 10 MB limit.")
    return ext


_FORENSIC_SKIP = {
    "metadata_detected":       False,
    "camera_probability":      0.5,
    "ai_probability":          0.5,
    "screenshot_probability":  0.0,
    "prediction":              "DL-primary",
    "forensic_confidence_pct": 0.0,
}


async def _run_forensic_analysis(temp_path: Path, skip: bool = False) -> Dict[str, Any]:
    """
    Run the legacy heuristic ForensicService.

    When ``skip=True`` (trained DL model is active) the forensic analysis is
    skipped entirely.  The fusion layer zeroes out the 'frequency' and
    'metadata' weights for trained models, so the heuristic signals carry no
    weight in the final verdict.  Skipping avoids 60-400 s of pure-Python
    pixel analysis that would otherwise dominate request latency.
    """
    if skip:
        logger.debug("[DetectAI] Forensic analysis skipped (trained DL model — heuristic weight = 0)")
        return _FORENSIC_SKIP

    try:
        from forensic.service import ForensicService
        svc    = ForensicService()
        result = await svc.analyze_image(temp_path)
        signals = result.signals
        return {
            "metadata_detected":      signals.metadata_detected,
            "camera_probability":     signals.camera_probability,
            "ai_probability":         signals.ai_probability,
            "screenshot_probability": signals.screenshot_probability,
            "prediction":             result.prediction,
            "forensic_confidence_pct": result.confidence,
        }
    except Exception as e:
        logger.warning(f"[DetectAI] Forensic service error: {e}")
        return {
            "metadata_detected":      False,
            "camera_probability":     0.5,
            "ai_probability":         0.5,
            "screenshot_probability": 0.0,
            "prediction":             "Unknown",
            "forensic_confidence_pct": 0.0,
        }


def _build_response(
    dl_result,
    fusion_out: FusionResult,
    forensic_signals: Dict[str, Any],
    proc_ms: float,
) -> DetectAIResponse:
    """Assemble the unified response object."""
    # Update in-memory confusion matrix stats
    _cm_stats["total"] += 1
    if fusion_out.ai_probability >= 0.5:
        _cm_stats["ai_predicted"] += 1
    else:
        _cm_stats["camera_predicted"] += 1

    info = get_model_info()

    resid_stats = None
    if dl_result.residual_stats:
        try:
            resid_stats = ResidualStats(**{
                k: dl_result.residual_stats[k]
                for k in ResidualStats.__fields__
                if k in dl_result.residual_stats
            })
        except Exception:
            pass

    fft_stats = None
    if getattr(dl_result, "fft_stats", None):
        try:
            fft_stats = FFTStats(**{
                k: dl_result.fft_stats[k]
                for k in FFTStats.__fields__
                if k in dl_result.fft_stats
            })
        except Exception:
            pass

    return DetectAIResponse(
        ai_probability=     round(fusion_out.ai_probability,     4),
        camera_probability= round(fusion_out.camera_probability, 4),
        dl_confidence=       round(fusion_out.dl_confidence,       4),
        forensic_confidence= round(fusion_out.forensic_confidence, 4),
        fusion_confidence=   round(fusion_out.fusion_confidence,   4),
        dominant_signals=    fusion_out.dominant_signals,
        forensic_signals=    forensic_signals,
        branch_scores=BranchScores(
            cnn_score=      round(fusion_out.signal_breakdown.get("cnn_score",      0.5), 4),
            residual_score= round(fusion_out.signal_breakdown.get("residual_score", 0.5), 4),
            fft_score=      round(fusion_out.signal_breakdown.get("fft_score",      0.5), 4),
            forensic_score= round(fusion_out.signal_breakdown.get("forensic_score", 0.5), 4),
            metadata_score= round(fusion_out.signal_breakdown.get("metadata_score", 0.5), 4),
        ),
        fusion_weights={k: round(v, 4) for k, v in fusion_out.weights_used.items()},
        residual_stats=resid_stats,
        fft_stats=fft_stats,
        calibration=CalibrationInfo(
            temperature=         info.get("temperature", 1.0),
            has_trained_weights= getattr(dl_result, "has_trained_weights", False),
            heuristic_proxy=     getattr(dl_result, "heuristic_proxy",     False),
            model_backend=       info.get("backend", "none"),
        ),
        model_version=      dl_result.model_version,
        dl_available=        dl_result.dl_available,
        device_used=         dl_result.device_used,
        processing_time_ms=  round(proc_ms, 2),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/detect-ai", response_model=DetectAIResponse)
async def detect_ai(
    image: UploadFile = File(..., description="Image to analyze (JPEG/PNG/WebP, max 10 MB)"),
):
    """
    Three-branch hybrid AI detection pipeline (multipart file upload).

    Weight split (when trained checkpoint present):
      DL model  70% → RGB branch 35% + residual 20% + FFT 15%
      Frequency 20% → classical heuristic spectral analysis
      Metadata  10% → EXIF reliability
    """
    ext = _validate_upload(image)
    t0  = time.time()
    tmp = None
    try:
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=ext)
        tmp = Path(tmp_str)
        import os
        os.close(tmp_fd)

        content = await image.read()
        tmp.write_bytes(content)

        detector         = get_detector()
        dl_result        = detector.analyze(tmp)

        # Skip slow heuristic forensic analysis when trained DL weights are
        # present — fusion layer gives them 0% weight anyway (DL-only profile).
        skip_forensic = getattr(dl_result, "has_trained_weights", False)
        forensic_signals = await _run_forensic_analysis(tmp, skip=skip_forensic)

        fusion_in  = build_fusion_input(dl_result, forensic_signals)
        fusion_out = fuse(fusion_in)

        return _build_response(dl_result, fusion_out, forensic_signals, (time.time() - t0) * 1000)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DetectAI] Unhandled error: {e}", exc_info=True)
        raise HTTPException(500, f"Hybrid detection failed: {e}")
    finally:
        if tmp and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


class Base64DetectRequest(BaseModel):
    """JSON body for base64-encoded image detection."""
    image_base64: str = Field(..., description="Data URL or raw base64 string")
    filename:     str = Field(default="image.jpg", description="Filename hint for extension detection")
    # Optional ground-truth label for confusion-matrix tracking (client-provided)
    ground_truth: Optional[str] = Field(default=None, description="'ai' | 'camera' (optional)")


@router.post("/detect-ai-base64", response_model=DetectAIResponse)
async def detect_ai_base64(req: Base64DetectRequest = Body(...)):
    """
    Three-branch hybrid AI detection pipeline — JSON base64 input.

    Accepts a data URL (data:image/jpeg;base64,...) or raw base64 string.
    Designed for the frontend VerifyProof page which works with base64 images.

    Ground-truth label (optional) — when provided the result is counted
    in the server-side confusion matrix accessible via GET /confusion-matrix.
    """
    t0 = time.time()

    # Decode base64
    b64 = req.image_base64
    if b64.startswith("data:"):
        # Strip data URL header
        try:
            b64 = b64.split(",", 1)[1]
        except IndexError:
            raise HTTPException(400, "Malformed data URL.")

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 payload.")

    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, "Image exceeds 10 MB limit.")

    # Determine extension from filename
    ext = Path(req.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"   # safe fallback

    tmp = None
    try:
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=ext)
        tmp = Path(tmp_str)
        import os
        os.close(tmp_fd)
        tmp.write_bytes(img_bytes)

        detector         = get_detector()
        dl_result        = detector.analyze(tmp)

        skip_forensic = getattr(dl_result, "has_trained_weights", False)
        forensic_signals = await _run_forensic_analysis(tmp, skip=skip_forensic)

        fusion_in  = build_fusion_input(dl_result, forensic_signals)
        fusion_out = fuse(fusion_in)

        # Update confusion matrix when ground truth is provided
        if req.ground_truth in ("ai", "camera"):
            predicted_ai = fusion_out.ai_probability >= 0.5
            actual_ai    = req.ground_truth == "ai"
            _cm_stats["true_ai"     if actual_ai else "true_camera"] += 1
            if predicted_ai and actual_ai:     _cm_stats["TP"] += 1
            elif not predicted_ai and not actual_ai: _cm_stats["TN"] += 1
            elif predicted_ai and not actual_ai:     _cm_stats["FP"] += 1
            else:                                    _cm_stats["FN"] += 1

        return _build_response(dl_result, fusion_out, forensic_signals, (time.time() - t0) * 1000)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DetectAIBase64] Unhandled error: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {e}")
    finally:
        if tmp and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


@router.get("/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix():
    """
    Return the live confusion matrix accumulated since the last server restart.

    Ground-truth labels must be supplied by the client via the
    ``ground_truth`` field in ``/detect-ai-base64`` requests.
    """
    TP = _cm_stats.get("TP", 0)
    TN = _cm_stats.get("TN", 0)
    FP = _cm_stats.get("FP", 0)
    FN = _cm_stats.get("FN", 0)
    labeled = TP + TN + FP + FN

    acc  = (TP + TN) / labeled  if labeled > 0 else None
    prec = TP / (TP + FP)       if (TP + FP) > 0 else None
    rec  = TP / (TP + FN)       if (TP + FN) > 0 else None
    f1   = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else None

    return ConfusionMatrixResponse(
        total_predictions= _cm_stats.get("total", 0),
        ai_predicted=      _cm_stats.get("ai_predicted", 0),
        camera_predicted=  _cm_stats.get("camera_predicted", 0),
        TP=TP, TN=TN, FP=FP, FN=FN,
        accuracy=  round(acc,  4) if acc  is not None else None,
        precision= round(prec, 4) if prec is not None else None,
        recall=    round(rec,  4) if rec  is not None else None,
        f1=        round(f1,   4) if f1   is not None else None,
    )


# ── Failure Reporting ─────────────────────────────────────────────────────────

class ReportFailureRequest(BaseModel):
    """JSON body for POST /report-failure."""
    image_base64:      str   = Field(..., description="Data URL or raw base64 of the image")
    filename:          str   = Field(default="image.jpg", description="Original filename hint")
    predicted_label:   str   = Field(..., description="Model prediction: 'ai' | 'camera'")
    ai_probability:    float = Field(..., ge=0.0, le=1.0, description="Model AI probability (0-1)")
    camera_probability: float = Field(..., ge=0.0, le=1.0, description="Model camera probability (0-1)")
    confidence:        float = Field(default=0.0, ge=0.0, le=100.0, description="Fusion confidence pct (0-100)")
    correction_label:  Optional[str] = Field(
        default=None,
        description="User-supplied ground truth: 'ai' | 'camera' | null (unreviewed)"
    )


class ReportFailureResponse(BaseModel):
    success:    bool
    message:    str
    saved_path: Optional[str] = None
    subfolder:  Optional[str] = None


_VALID_LABELS = {"ai", "camera"}


@router.post("/report-failure", response_model=ReportFailureResponse)
async def report_failure(req: ReportFailureRequest = Body(...)):
    """
    Accept a user-reported incorrect prediction and persist the image for
    future hard-negative retraining.

    The image is stored in collect_failures/ with the following routing:
      correction_label == "ai"     → collect_failures/ai/
      correction_label == "camera" → collect_failures/real/
      correction_label is None     → collect_failures/unreviewed/

    Each image is deduplicated by SHA-256 hash so reporting the same image
    multiple times is safe.  A JSON sidecar is written alongside each image
    with prediction metadata for the retraining script.

    This endpoint does NOT touch inference logic or existing datasets.
    """
    # ── Validate labels ───────────────────────────────────────────────────────
    if req.predicted_label not in _VALID_LABELS:
        raise HTTPException(
            400,
            f"predicted_label must be one of {_VALID_LABELS}, got '{req.predicted_label}'",
        )
    if req.correction_label is not None and req.correction_label not in _VALID_LABELS:
        raise HTTPException(
            400,
            f"correction_label must be one of {_VALID_LABELS} or null, got '{req.correction_label}'",
        )

    # ── Decode base64 ─────────────────────────────────────────────────────────
    b64 = req.image_base64
    if b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except IndexError:
            raise HTTPException(400, "Malformed data URL — missing comma separator.")

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 payload — could not decode.")

    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"Image exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit.")

    # ── Extension ─────────────────────────────────────────────────────────────
    ext = Path(req.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"

    # ── Save ──────────────────────────────────────────────────────────────────
    try:
        from .failure_logger import save_failure, get_collection_stats

        saved_path = save_failure(
            img_bytes=         img_bytes,
            ext=               ext,
            predicted_label=   req.predicted_label,
            ai_probability=    req.ai_probability,
            camera_probability= req.camera_probability,
            confidence=        req.confidence,
            correction_label=  req.correction_label,
        )

        # Determine which subfolder was used for the response message
        subfolder = (
            "ai"         if req.correction_label == "ai"     else
            "real"       if req.correction_label == "camera" else
            "unreviewed"
        )
        stats = get_collection_stats()
        logger.info(
            f"[ReportFailure] Saved to collect_failures/{subfolder}. "
            f"Collection totals: {stats}"
        )

        return ReportFailureResponse(
            success=    True,
            message=    "Report saved. Thank you — this image will improve future accuracy.",
            saved_path= saved_path,
            subfolder=  subfolder,
        )

    except Exception as e:
        logger.error(f"[ReportFailure] Failed to save image: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to save report: {e}")


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return information about the currently loaded DL model."""
    info = get_model_info()
    return ModelInfoResponse(**{k: info[k] for k in ModelInfoResponse.__fields__ if k in info})


@router.get("/health")
async def health():
    """Quick health check for the inference service."""
    info = get_model_info()
    return {
        "status":              "healthy",
        "service":             "hybrid_ai_detection_v2",
        "dl_available":        info["loaded"],
        "has_trained_weights": info.get("has_trained_weights", False),
        "temperature":         info.get("temperature", 1.0),
        "device":              info["device"],
        "model_version":       info["model_version"],
        "backend":             info["backend"],         # "pytorch" | "onnxruntime" | "none"
        "checkpoint":          info.get("checkpoint"),  # filename of loaded checkpoint
        "branches":            ["RGB", "residual", "FFT"],
        "fusion_spec":         "70% DL / 20% frequency / 10% metadata",
    }


# ── Evaluation endpoint ───────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    real_dirs:      List[str] = Field(default=[], description="Dirs of real camera images")
    ai_dirs:        List[str] = Field(default=[], description="Dirs of AI-generated images")
    root_dirs:      List[str] = Field(default=[], description="Root dirs with real/ and ai/ sub-folders")
    csv_files:      List[str] = Field(default=[], description="CSV manifest files")
    batch_size:     int       = Field(default=32, ge=1, le=128)
    max_fpr_target: float     = Field(default=0.05, ge=0.0, le=0.5)


class EvaluateResponse(BaseModel):
    checkpoint:               str
    architecture:             str
    temperature:              float
    device:                   str
    accuracy:                 float
    precision:                float
    recall:                   float
    f1:                       float
    roc_auc:                  float
    average_precision:        float
    false_positive_rate:      float
    false_negative_rate:      float
    expected_calibration_error: float
    confusion_matrix:         List[List[int]]
    TP: int; TN: int; FP: int; FN: int
    total_samples:            int
    real_samples:             int
    ai_samples:               int
    per_class:                Dict[str, Any]
    optimal_threshold:        Dict[str, float]


@router.post("/evaluate-model", response_model=EvaluateResponse)
async def evaluate_model(req: EvaluateRequest = Body(...)):
    """
    Run full evaluation metrics on a local test dataset.

    Requires a trained checkpoint at
    backend/inference/checkpoints/forensic_detector.pth.
    """
    if not any([req.real_dirs, req.ai_dirs, req.root_dirs, req.csv_files]):
        raise HTTPException(400, "Provide at least one of: real_dirs, ai_dirs, root_dirs, csv_files")

    ckpt_path = None
    for cp in CHECKPOINT_PATHS:
        if cp.suffix == ".pth" and cp.exists():
            ckpt_path = cp
            break

    if ckpt_path is None:
        raise HTTPException(
            503,
            "No trained checkpoint found. Train first with backend/training/train.py, "
            "then place the .pth in backend/inference/checkpoints/forensic_detector.pth"
        )

    try:
        _backend = Path(__file__).resolve().parent.parent
        if str(_backend) not in sys.path:
            sys.path.insert(0, str(_backend))
        from training.evaluate import evaluate as run_evaluate
    except ImportError as e:
        raise HTTPException(500, f"Evaluation module unavailable: {e}")

    try:
        result = run_evaluate(
            checkpoint_path=str(ckpt_path),
            real_dirs=req.real_dirs,
            ai_dirs=req.ai_dirs,
            root_dirs=req.root_dirs,
            csv_files=req.csv_files,
            batch_size=req.batch_size,
            max_fpr_target=req.max_fpr_target,
        )
    except Exception as e:
        logger.error(f"[Evaluate] Error: {e}", exc_info=True)
        raise HTTPException(500, f"Evaluation failed: {e}")

    # Mirror labelled results into the live confusion matrix
    for key in ("TP", "TN", "FP", "FN"):
        _cm_stats[key] += result.get(key, 0)
    _cm_stats["total"] += result.get("total_samples", 0)

    return EvaluateResponse(**{k: result[k] for k in EvaluateResponse.__fields__})
