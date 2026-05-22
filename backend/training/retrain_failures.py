"""
Hard-Negative Retraining Pipeline
===================================
Scans collect_failures/, merges human-reviewed images into the main datasets,
then optionally launches a full retraining run against the combined dataset.

Workflow
--------
1. Scan collect_failures/ai/    — confirmed AI images
         collect_failures/real/ — confirmed real camera images
         collect_failures/unreviewed/ — not labelled yet (skipped unless --include-unreviewed)

2. Deduplicate each candidate against datasets/ai/ and datasets/real/ using
   SHA-256 of the raw image bytes.

3. Copy (never move) deduplicated images into the target dataset folder with a
   "failure_" prefix so they are always identifiable.

4. Write a human-readable + machine-readable retraining summary log to
   collect_failures/retrain_log_<timestamp>.json

5. If --retrain is passed, invoke the existing training.train script on the
   updated dataset automatically.

Usage
-----
  # Dry-run — see what WOULD be merged without touching datasets
  python -m training.retrain_failures --dry-run

  # Merge reviewed failures into datasets
  python -m training.retrain_failures

  # Merge AND immediately retrain (CPU-safe flags; remove --debug-small-model for production)
  python -m training.retrain_failures --retrain --epochs 10 --batch-size 8

  # Include unreviewed images (use with caution)
  python -m training.retrain_failures --include-unreviewed --dry-run

Rules
-----
- NEVER deletes or modifies collect_failures/ source images.
- NEVER overwrites existing dataset images.
- Skips duplicates detected by SHA-256 full-file hash.
- Produces a timestamped JSON log for audit trail.
- Zero changes to inference logic or existing checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf_8"}:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retrain_failures")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent          # backend/training/
_BACKEND        = _HERE.parent                             # backend/
_COLLECT_ROOT   = _BACKEND / "collect_failures"
_DATASET_REAL   = _BACKEND / "datasets" / "real"
_DATASET_AI     = _BACKEND / "datasets" / "ai"
_CHECKPOINTS    = _BACKEND / "inference" / "checkpoints"

IMG_EXTS: set = {".jpg", ".jpeg", ".png", ".webp"}

# Label-to-dataset mapping
_LABEL_TO_DIR: Dict[str, Path] = {
    "ai":   _DATASET_AI,
    "real": _DATASET_REAL,
}

# collect_failures subfolder → dataset label
_SUBFOLDER_TO_LABEL: Dict[str, str] = {
    "ai":         "ai",
    "real":       "real",
    "unreviewed": "unreviewed",  # handled separately
}


# ─────────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Return full SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_hash_set(directory: Path) -> Set[str]:
    """
    Build a set of SHA-256 hashes for all image files in *directory*.
    Used to detect duplicates before copying.
    This is cached per-call; call once per dataset dir per run.
    """
    if not directory.exists():
        return set()
    hashes: Set[str] = set()
    img_files = [f for f in directory.iterdir() if f.suffix.lower() in IMG_EXTS]
    total = len(img_files)
    if total == 0:
        return hashes
    logger.info(f"  Building hash index for {directory.name}/ ({total} images)…")
    for i, fp in enumerate(img_files):
        try:
            hashes.add(_sha256(fp))
        except Exception as e:
            logger.debug(f"  Hash error {fp.name}: {e}")
        if (i + 1) % 500 == 0:
            logger.info(f"    {i + 1}/{total} hashed…")
    logger.info(f"  Hash index ready: {len(hashes)} entries.")
    return hashes


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)


def _next_failure_stem(dest_dir: Path, label: str, index: int) -> str:
    """
    Return a unique filename stem like 'failure_ai_00042'.
    The prefix makes these images easily identifiable in the dataset.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"failure_{label}_{ts}_{index:05d}"


def _collect_failure_images(subfolder: str) -> List[Tuple[Path, Optional[dict]]]:
    """
    Return all image files in collect_failures/<subfolder>/ paired with their
    JSON sidecar metadata (or None if no sidecar).
    Skips .gitkeep and non-image files.
    """
    src = _COLLECT_ROOT / subfolder
    if not src.exists():
        return []
    pairs: List[Tuple[Path, Optional[dict]]] = []
    for fp in sorted(src.iterdir()):
        if fp.suffix.lower() not in IMG_EXTS:
            continue
        meta_path = fp.with_suffix(".json")
        meta: Optional[dict] = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        pairs.append((fp, meta))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Core merge logic
# ─────────────────────────────────────────────────────────────────────────────

def merge_failures(
    dry_run: bool = False,
    include_unreviewed: bool = False,
) -> dict:
    """
    Scan collect_failures/, deduplicate, and copy reviewed images to datasets/.

    Returns a summary dict with counts for the log.
    """
    _DATASET_REAL.mkdir(parents=True, exist_ok=True)
    _DATASET_AI.mkdir(parents=True, exist_ok=True)

    # Pre-build hash sets for deduplication (done once — expensive but accurate)
    logger.info("Building deduplication hash indexes…")
    hash_real = _build_hash_set(_DATASET_REAL)
    hash_ai   = _build_hash_set(_DATASET_AI)
    # Combined hash set — prevents an image from being added twice even if it
    # sits in the wrong collect_failures subfolder
    seen_hashes: Set[str] = hash_real | hash_ai

    summary: dict = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "dry_run":    dry_run,
        "subfolders": {},
        "totals": {
            "scanned":    0,
            "copied":     0,
            "skipped_duplicate": 0,
            "skipped_unreviewed": 0,
            "errors":     0,
        },
        "dataset_before": {
            "real": _count_images(_DATASET_REAL),
            "ai":   _count_images(_DATASET_AI),
        },
    }

    # ── Process reviewed subfolders ───────────────────────────────────────────
    for subfolder, label in _SUBFOLDER_TO_LABEL.items():
        if label == "unreviewed":
            if not include_unreviewed:
                logger.info(
                    f"  Skipping collect_failures/unreviewed/ "
                    f"(use --include-unreviewed to process these)"
                )
                continue
            # Unreviewed images have no confirmed label — we cannot safely
            # assign them. Log and skip each one.
            pairs = _collect_failure_images(subfolder)
            summary["totals"]["scanned"] += len(pairs)
            summary["totals"]["skipped_unreviewed"] += len(pairs)
            if pairs:
                logger.warning(
                    f"  {len(pairs)} unreviewed image(s) in collect_failures/unreviewed/ "
                    f"— cannot merge without a confirmed label. "
                    f"Please manually move them to collect_failures/ai/ or collect_failures/real/ "
                    f"and re-run."
                )
            summary["subfolders"][subfolder] = {
                "label": "unreviewed",
                "scanned": len(pairs),
                "copied": 0,
                "skipped_unreviewed": len(pairs),
                "skipped_duplicate": 0,
                "errors": 0,
            }
            continue

        dest_dir = _LABEL_TO_DIR[label]
        pairs    = _collect_failure_images(subfolder)
        sub_summary = {
            "label":              label,
            "scanned":            len(pairs),
            "copied":             0,
            "skipped_duplicate":  0,
            "skipped_unreviewed": 0,
            "errors":             0,
            "copied_files":       [],
        }

        if not pairs:
            logger.info(f"  collect_failures/{subfolder}/ — empty, nothing to merge.")
            summary["subfolders"][subfolder] = sub_summary
            continue

        logger.info(
            f"  Processing collect_failures/{subfolder}/ "
            f"→ datasets/{dest_dir.name}/  ({len(pairs)} image(s))"
        )

        # Running index for unique filenames (start after existing dataset files)
        file_index = _count_images(dest_dir)

        for src_img, meta in pairs:
            summary["totals"]["scanned"] += 1
            try:
                img_hash = _sha256(src_img)
            except Exception as e:
                logger.warning(f"    Cannot read {src_img.name}: {e}")
                sub_summary["errors"] += 1
                summary["totals"]["errors"] += 1
                continue

            # Duplicate check against ALL dataset images
            if img_hash in seen_hashes:
                logger.debug(f"    SKIP (duplicate) {src_img.name}")
                sub_summary["skipped_duplicate"] += 1
                summary["totals"]["skipped_duplicate"] += 1
                continue

            # Build destination path
            stem = _next_failure_stem(dest_dir, label, file_index)
            ext  = src_img.suffix.lower()
            if ext not in IMG_EXTS:
                ext = ".jpg"
            dest_path = dest_dir / f"{stem}{ext}"

            if dry_run:
                logger.info(f"    [DRY-RUN] Would copy: {src_img.name} → {dest_path.name}")
                sub_summary["copied"] += 1
                summary["totals"]["copied"] += 1
                seen_hashes.add(img_hash)  # prevent phantom duplicates in dry-run output
                file_index += 1
                continue

            try:
                shutil.copy2(str(src_img), str(dest_path))
                seen_hashes.add(img_hash)  # mark as seen so next loop does not re-add it
                file_index += 1
                sub_summary["copied"] += 1
                sub_summary["copied_files"].append(dest_path.name)
                summary["totals"]["copied"] += 1
                logger.info(
                    f"    COPIED {src_img.name} → datasets/{dest_dir.name}/{dest_path.name}"
                    + (f"  (pred={meta.get('predicted_label')}, "
                       f"corr={meta.get('correction_label')}, "
                       f"ai={meta.get('ai_probability', 0):.1%})" if meta else "")
                )
            except Exception as e:
                logger.error(f"    ERROR copying {src_img.name}: {e}")
                sub_summary["errors"] += 1
                summary["totals"]["errors"] += 1

        logger.info(
            f"  {subfolder}: copied={sub_summary['copied']}  "
            f"dup={sub_summary['skipped_duplicate']}  "
            f"err={sub_summary['errors']}"
        )
        summary["subfolders"][subfolder] = sub_summary

    # ── Dataset state after merge ─────────────────────────────────────────────
    summary["dataset_after"] = {
        "real": _count_images(_DATASET_REAL),
        "ai":   _count_images(_DATASET_AI),
    }
    delta_real = summary["dataset_after"]["real"] - summary["dataset_before"]["real"]
    delta_ai   = summary["dataset_after"]["ai"]   - summary["dataset_before"]["ai"]
    summary["dataset_delta"] = {"real": delta_real, "ai": delta_ai}

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def write_log(summary: dict) -> Path:
    """Write a JSON + human-readable text log to collect_failures/."""
    _COLLECT_ROOT.mkdir(parents=True, exist_ok=True)
    ts_str   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = _COLLECT_ROOT / f"retrain_log_{ts_str}.json"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Log written → {log_path}")
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    dr = " [DRY-RUN]" if summary["dry_run"] else ""
    print()
    print("=" * 60)
    print(f"  Hard-Negative Merge Summary{dr}")
    print("=" * 60)

    tot = summary["totals"]
    print(f"  Scanned           : {tot['scanned']}")
    print(f"  Copied to datasets: {tot['copied']}")
    print(f"  Duplicates skipped: {tot['skipped_duplicate']}")
    print(f"  Unreviewed skipped: {tot['skipped_unreviewed']}")
    print(f"  Errors            : {tot['errors']}")
    print()

    before = summary["dataset_before"]
    after  = summary["dataset_after"]
    delta  = summary["dataset_delta"]
    print(f"  datasets/real/ : {before['real']} -> {after['real']}  (+{delta['real']})")
    print(f"  datasets/ai/   : {before['ai']}   -> {after['ai']}    (+{delta['ai']})")

    print()
    print("  Per-subfolder:")
    for sub, info in summary["subfolders"].items():
        status = "copied" if info["copied"] else "empty/skipped"
        print(
            f"    collect_failures/{sub:<14} "
            f"scanned={info['scanned']}  "
            f"copied={info['copied']}  "
            f"dup={info.get('skipped_duplicate', 0)}  "
            f"({status})"
        )
    print("=" * 60)
    print()

    if summary["totals"]["copied"] == 0 and not summary["dry_run"]:
        print("  Nothing was merged.")
        print("  To populate collect_failures/, use the 'Report Wrong Detection'")
        print("  button in the Verification Report screen and then re-run this script.")
    elif summary["totals"]["copied"] > 0 and not summary["dry_run"]:
        print("  Merge complete!")
        print()
        print("  Next step — retrain on the expanded dataset:")
        print()
        print("    cd backend")
        print("    python -m training.retrain_failures --retrain --epochs 15 --batch-size 8")
        print()
        print("  Or run training directly:")
        print()
        print("    python -m training.train \\")
        print(f"        --real-dirs datasets/real \\")
        print(f"        --ai-dirs   datasets/ai \\")
        print(f"        --output    inference/checkpoints \\")
        print(f"        --epochs 15 --batch-size 8")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Optional automatic retraining
# ─────────────────────────────────────────────────────────────────────────────

def trigger_retrain(
    epochs: int,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    extra_args: List[str],
) -> int:
    """
    Call training.train as a subprocess so it runs in the same Python
    environment but does not import into this module's memory.

    Returns the process exit code (0 = success).
    """
    cmd = [
        sys.executable, "-m", "training.train",
        "--real-dirs", str(_DATASET_REAL),
        "--ai-dirs",   str(_DATASET_AI),
        "--output",    str(_CHECKPOINTS),
        "--epochs",    str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
    ]
    if max_samples is not None:
        cmd += ["--max-train-samples", str(max_samples)]
    cmd += extra_args

    logger.info("Launching retraining…")
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info(f"  Epochs: {epochs}  Batch: {batch_size}  Workers: {num_workers}")

    result = subprocess.run(cmd, cwd=str(_BACKEND))
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Hard-negative retraining pipeline.\n"
            "Merges human-reviewed misclassified images from collect_failures/ "
            "into datasets/ and optionally triggers retraining."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be merged (safe — no files written)
  python -m training.retrain_failures --dry-run

  # Merge reviewed failures into datasets
  python -m training.retrain_failures

  # Merge and retrain for 15 epochs
  python -m training.retrain_failures --retrain --epochs 15 --batch-size 8

  # Include unreviewed images (label them yourself first!)
  python -m training.retrain_failures --include-unreviewed

  # Quick sanity-check retrain on CPU (small model, 1 epoch)
  python -m training.retrain_failures --retrain --epochs 1 --batch-size 4 \\
      --num-workers 0 --max-samples 200 --debug-small-model
""",
    )

    # Merge options
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview the merge without copying any files.",
    )
    p.add_argument(
        "--include-unreviewed", action="store_true",
        help=(
            "Also process collect_failures/unreviewed/ — use only after manually "
            "confirming those images are labelled correctly."
        ),
    )

    # Retrain options
    p.add_argument(
        "--retrain", action="store_true",
        help="After merging, immediately run training.train on the updated dataset.",
    )
    p.add_argument("--epochs",      type=int, default=15,
                   help="Training epochs when --retrain is set (default: 15).")
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Batch size when --retrain is set (default: 8).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default: 0 — safe for Windows CPU).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap training samples per class (useful for quick debug runs).")
    p.add_argument("--debug-small-model", action="store_true",
                   help="Pass --debug-small-model to train.py (MobileNetV3-Small, fast on CPU).")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print()
    logger.info("Hard-Negative Retraining Pipeline")
    logger.info(f"  collect_failures/ : {_COLLECT_ROOT}")
    logger.info(f"  datasets/real/    : {_DATASET_REAL}")
    logger.info(f"  datasets/ai/      : {_DATASET_AI}")
    if args.dry_run:
        logger.info("  MODE: DRY-RUN — no files will be written")

    # ── Step 1: scan & merge ──────────────────────────────────────────────────
    logger.info("\nStep 1/2 — Scanning and merging failure images…")
    summary = merge_failures(
        dry_run=args.dry_run,
        include_unreviewed=args.include_unreviewed,
    )

    # ── Step 2: write log ─────────────────────────────────────────────────────
    if not args.dry_run:
        log_path = write_log(summary)
        summary["log_path"] = str(log_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(summary)

    # ── Step 3: optional retrain ──────────────────────────────────────────────
    if args.retrain:
        if args.dry_run:
            logger.warning("--retrain is ignored in dry-run mode.")
        elif summary["totals"]["copied"] == 0:
            logger.warning(
                "No new images were merged — skipping retrain. "
                "Run without --dry-run after populating collect_failures/."
            )
        else:
            logger.info("\nStep 2/2 — Launching retraining on expanded dataset…")
            extra = ["--debug-small-model"] if args.debug_small_model else []
            rc = trigger_retrain(
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                extra_args=extra,
            )
            if rc == 0:
                logger.info("Retraining completed successfully.")
            else:
                logger.error(f"Retraining exited with code {rc}.")
                sys.exit(rc)
    else:
        logger.info("Step 2/2 — Retrain skipped (pass --retrain to enable).")


if __name__ == "__main__":
    main()
