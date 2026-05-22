"""
Failure Logger
==============
Persists user-reported wrong predictions to disk for future
hard-negative retraining.

Folder layout (relative to backend/):
    collect_failures/
        ai/           — confirmed AI images (user corrected a "camera" prediction)
        real/         — confirmed real camera images (user corrected an "ai" prediction)
        unreviewed/   — no correction label supplied (needs human review)

Each saved image is accompanied by a JSON sidecar with full metadata so the
retraining script can reconstruct per-image context without touching the DB.

Duplicate detection:
    A SHA-256 prefix (first 16 hex chars) of the raw image bytes is included
    in the filename.  Before saving, we scan the target subfolder for any
    existing file with the same hash prefix and skip the write if found.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Root of the failure-collection directory — sits alongside inference/ inside backend/
_COLLECT_ROOT = Path(__file__).parent.parent / "collect_failures"

_SUBFOLDERS = ("ai", "real", "unreviewed")


def _ensure_dirs() -> None:
    """Create subdirectories on first use (idempotent)."""
    for sub in _SUBFOLDERS:
        (_COLLECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def _image_hash(img_bytes: bytes) -> str:
    """Return a 16-char hex prefix of the SHA-256 digest."""
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def _is_duplicate(subfolder: Path, img_hash: str) -> Optional[Path]:
    """
    Return the path to an already-saved file with this hash, or None.
    Checks all three subfolders so the same image reported twice (possibly
    with different corrections) is still deduplicated globally.
    """
    for sub in _SUBFOLDERS:
        for existing in (_COLLECT_ROOT / sub).glob(f"*{img_hash}*"):
            if existing.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                return existing
    return None


def save_failure(
    img_bytes: bytes,
    ext: str,
    predicted_label: str,           # "ai" | "camera"
    ai_probability: float,
    camera_probability: float,
    confidence: float,
    correction_label: Optional[str] = None,  # "ai" | "camera" | None
) -> str:
    """
    Save a reported image and its prediction metadata for retraining.

    Args:
        img_bytes:         Raw image bytes (JPEG / PNG / WebP).
        ext:               File extension including leading dot (e.g. ".jpg").
        predicted_label:   What the model predicted — "ai" or "camera".
        ai_probability:    Model AI probability (0–1).
        camera_probability: Model camera probability (0–1).
        confidence:        Fusion confidence percentage (0–100).
        correction_label:  User-supplied ground truth — "ai", "camera", or None.

    Returns:
        Absolute path of the saved image file.
    """
    _ensure_dirs()

    # ── Determine target subfolder ────────────────────────────────────────────
    if correction_label == "ai":
        subfolder = "ai"
    elif correction_label == "camera":
        subfolder = "real"
    else:
        subfolder = "unreviewed"

    # ── Duplicate guard ───────────────────────────────────────────────────────
    img_hash = _image_hash(img_bytes)
    existing = _is_duplicate(_COLLECT_ROOT / subfolder, img_hash)
    if existing:
        logger.info(
            f"[FailureLogger] Duplicate skipped — already saved as {existing.name}"
        )
        return str(existing)

    # ── File paths ────────────────────────────────────────────────────────────
    ts_str   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem     = f"{ts_str}_{img_hash}"
    img_path = _COLLECT_ROOT / subfolder / f"{stem}{ext}"
    meta_path = _COLLECT_ROOT / subfolder / f"{stem}.json"

    # ── Write image ───────────────────────────────────────────────────────────
    img_path.write_bytes(img_bytes)

    # ── Write JSON sidecar ────────────────────────────────────────────────────
    meta = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "predicted_label":    predicted_label,
        "correction_label":   correction_label,
        "ai_probability":     round(ai_probability,     4),
        "camera_probability": round(camera_probability, 4),
        "confidence_pct":     round(confidence,         2),
        "image_hash_prefix":  img_hash,
        "saved_subfolder":    subfolder,
        "image_file":         img_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(
        f"[FailureLogger] Saved → collect_failures/{subfolder}/{img_path.name} "
        f"(predicted={predicted_label}, correction={correction_label}, "
        f"ai={ai_probability:.1%}, cam={camera_probability:.1%}, "
        f"conf={confidence:.1f}%)"
    )
    return str(img_path)


def get_collection_stats() -> dict:
    """
    Return a summary of images currently stored in collect_failures/.
    Used by the health/status endpoint.
    """
    _ensure_dirs()
    stats = {}
    total = 0
    for sub in _SUBFOLDERS:
        # Count image files only (not JSON sidecars or .gitkeep)
        count = sum(
            1 for f in (_COLLECT_ROOT / sub).iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        stats[sub] = count
        total += count
    stats["total"] = total
    return stats
