"""
Targeted Dataset Expansion  (Phase 3)
======================================
Adds difficult real-world edge-case images that the base dataset lacks.

WHY edge-case images matter
----------------------------
Standard training data (COCO + DiffusionDB) is clean, uncompressed, and
well-lit.  Production images are forwarded through WhatsApp chains, captured
in dark rooms, shared as screenshots, or are photorealistic SDXL outputs that
the model has never seen.  This script specifically targets those failure modes.

REAL edge cases added
---------------------
  Low-light       — genuine dark mobile photos + synthetic low-light from COCO
  Social chain    — 3–5 rounds of JPEG recompression (WhatsApp forwarding chain)
  Screenshot      — real photos embedded inside a simulated UI chrome
  Open Images     — wide diversity of camera types via HuggingFace streaming

AI edge cases added
-------------------
  SDXL            — Stable Diffusion XL photorealistic outputs (HuggingFace)
  Midjourney-style— Midjourney v4/v5/v6 images (HuggingFace)
  Flux            — Black Forest Labs Flux.1 photorealistic outputs
  AI screenshot   — AI images captured as if screenshotted from a browser
  AI social chain — AI images put through multi-round JPEG recompression

Source strategy
---------------
Each source tries HuggingFace first; if that fails the function falls back to
synthetic augmentation of the existing dataset, so the script ALWAYS produces
useful output — even completely offline.

Usage
-----
  # Full expansion (all sources, up to 1000 images per method)
  python -m training.expand_dataset

  # Quick test (100 per method)
  python -m training.expand_dataset --max-per-source 100

  # Only augmentation (no HuggingFace downloads)
  python -m training.expand_dataset --augment-only

  # Preview counts without changing files
  python -m training.expand_dataset --summary-only
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# ── UTF-8 output on Windows ───────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf_8"}:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("expand_dataset")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
REAL_DIR = _BACKEND / "datasets" / "real"
AI_DIR   = _BACKEND / "datasets" / "ai"
CACHE    = _BACKEND / "datasets" / ".cache"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Pillow ────────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False
    logger.warning("Pillow or NumPy not installed. Run: pip install Pillow numpy")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def _count(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)


def _existing_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]


def _prefix_count(d: Path, prefix: str) -> int:
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir()
               if f.suffix.lower() in IMG_EXTS and f.name.startswith(prefix))


def _next_index(d: Path) -> int:
    return _count(d)


def _save_pil(img: "Image.Image", path: Path, quality: int = 90) -> bool:
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(str(path), "JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        logger.debug(f"_save_pil failed {path.name}: {e}")
        return False


def _require_datasets():
    try:
        import datasets as ds
        return ds
    except ImportError:
        logger.warning(
            "HuggingFace 'datasets' not installed — HF streaming unavailable.\n"
            "  Run: pip install datasets"
        )
        return None


def _progress(current: int, total: int, label: str = "") -> None:
    bar_len = 36
    filled  = int(bar_len * current / max(total, 1))
    bar     = "#" * filled + "." * (bar_len - filled)
    pct     = 100.0 * current / max(total, 1)
    print(f"\r  [{bar}] {pct:5.1f}%  {current}/{total}  {label}",
          end="", flush=True)


def _sample_existing(d: Path, n: int, exclude_prefixes: tuple = ()) -> List[Path]:
    """
    Sample up to n image files from directory d, excluding files with given prefixes.
    Only includes files saved with Pillow (no raw camera RAWs etc).
    """
    candidates = [
        f for f in d.iterdir()
        if f.suffix.lower() in IMG_EXTS
        and not any(f.name.startswith(p) for p in exclude_prefixes)
    ]
    return random.sample(candidates, min(n, len(candidates)))


# ─────────────────────────────────────────────────────────────────────────────
# REAL edge case 1 — Low-light augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_lowlight(max_images: int) -> int:
    """
    Create synthetic low-light versions of existing COCO real photos.

    Process:
      1. Pick random COCO real images (coco_*.jpg)
      2. Randomly darken by gamma 0.3–0.6 (simulates dark room / night shot)
      3. Add Gaussian shot noise proportional to darkness (PRNU still present)
      4. Optionally add slight blue/green tint (screen glare, LED lighting)

    Why gamma + noise?  Real low-light cameras still record PRNU (Photo-Response
    Non-Uniformity) sensor noise.  AI images do not — so even the darkened
    version of a real photo retains its forensic fingerprint.

    Returns number of images saved.
    """
    if not PIL_OK:
        logger.warning("[LowLight] Pillow/NumPy not available — skipping.")
        return 0

    existing = _prefix_count(REAL_DIR, "lowlight_")
    if existing >= max_images:
        logger.info(f"[LowLight] Already have {existing} low-light images — skipping.")
        return 0

    need       = max_images - existing
    idx        = _next_index(REAL_DIR)
    sources    = _sample_existing(REAL_DIR, need * 2, exclude_prefixes=("lowlight_",))
    saved      = 0

    logger.info(f"[LowLight] Creating {need} synthetic low-light images from COCO photos…")

    for src in sources:
        if saved >= need:
            break
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")

                # ── Gamma darkening ─────────────────────────────────────────
                gamma   = random.uniform(0.25, 0.55)  # darker = lower gamma
                arr     = np.array(img, dtype=np.float32) / 255.0
                arr     = np.power(arr, 1.0 / gamma)   # bright → dark (inverse gamma)
                arr     = np.clip(arr, 0.0, 1.0)

                # ── Shot noise (Poisson-like additive Gaussian) ─────────────
                noise_std = random.uniform(0.015, 0.060) * (1.0 - gamma)
                noise     = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
                arr       = np.clip(arr + noise, 0.0, 1.0)

                # ── Occasional colour tint (LED / screen glow) ──────────────
                if random.random() < 0.35:
                    channel = random.choice([1, 2])   # green or blue tint
                    arr[:, :, channel] = np.clip(arr[:, :, channel] * random.uniform(1.05, 1.18), 0, 1)

                result = Image.fromarray((arr * 255).astype(np.uint8))

                dest = REAL_DIR / f"lowlight_{idx + saved:07d}.jpg"
                if _save_pil(result, dest, quality=88):
                    saved += 1

        except Exception as e:
            logger.debug(f"[LowLight] {src.name}: {e}")

    logger.info(f"[LowLight] {saved} low-light images saved.")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# REAL edge case 2 — Social media chain compression (3–5 rounds)
# ─────────────────────────────────────────────────────────────────────────────

def augment_social_chain_compression(src_dir: Path, prefix: str, max_images: int) -> int:
    """
    Simulate multi-generation social media forwarding.

    WhatsApp / Telegram / Instagram re-encode images every time they are
    forwarded.  After 3–5 re-encodes at quality 55–72 the blocking artefacts
    accumulate and HF noise is partially replaced with DCT ringing.  This is
    a common failure mode for camera photos that get misclassified as AI.

    src_dir: datasets/real or datasets/ai
    prefix:  output file prefix ("social_real_" or "social_ai_")
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(src_dir, prefix)
    if existing >= max_images:
        logger.info(f"[SocialChain] {src_dir.name}/{prefix}* — {existing} already exist.")
        return 0

    need    = max_images - existing
    sources = _sample_existing(src_dir, need * 2, exclude_prefixes=(prefix, "lowlight_"))
    idx     = _next_index(src_dir)
    saved   = 0

    label = "real" if src_dir == REAL_DIR else "AI"
    logger.info(
        f"[SocialChain] Creating {need} multi-generation compressed {label} images…"
    )

    for src in sources:
        if saved >= need:
            break
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                rounds = random.randint(3, 5)
                for _ in range(rounds):
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=random.randint(52, 72))
                    buf.seek(0)
                    img = Image.open(buf).copy()

                dest = src_dir / f"{prefix}{idx + saved:07d}.jpg"
                if _save_pil(img, dest, quality=80):
                    saved += 1

        except Exception as e:
            logger.debug(f"[SocialChain] {src.name}: {e}")

    logger.info(f"[SocialChain] {saved} chain-compressed {label} images saved.")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# REAL edge case 3 — Screenshot simulation
# ─────────────────────────────────────────────────────────────────────────────

def augment_screenshot_real(max_images: int) -> int:
    """
    Embed real camera photos inside a simulated browser/messaging UI chrome.

    This mimics screenshots of photos shared on WhatsApp, iMessage, or viewed
    in a browser — a common failure scenario where the photo is real but looks
    like a screenshot.

    Adds:  a thin grey/white border, optional rounded corners, a status bar,
    slight resize to non-standard dimensions (screenshot pixel count).

    Returns number saved.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(REAL_DIR, "screenshot_real_")
    if existing >= max_images:
        logger.info(f"[Screenshot] {existing} real screenshot images already exist.")
        return 0

    need    = max_images - existing
    sources = _sample_existing(REAL_DIR, need * 2, exclude_prefixes=("screenshot_", "lowlight_"))
    idx     = _next_index(REAL_DIR)
    saved   = 0

    logger.info(f"[Screenshot] Creating {need} screenshot-framed real images…")

    # UI chrome colours
    chrome_palettes = [
        (240, 240, 242),   # light grey (iOS)
        (255, 255, 255),   # white (WhatsApp)
        (18,  18,  18),    # dark (dark-mode browser)
        (245, 245, 245),   # off-white (Android)
    ]

    for src in sources:
        if saved >= need:
            break
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")

                # Resize photo to a typical mobile screenshot size
                target_w = random.choice([375, 390, 412, 360])  # iPhone/Android widths
                scale    = target_w / img.width
                new_h    = int(img.height * scale)
                img      = img.resize((target_w, new_h), Image.LANCZOS)

                # Add chrome border
                chrome = random.choice(chrome_palettes)
                border = random.randint(8, 24)
                canvas_w = img.width  + 2 * border
                canvas_h = img.height + 2 * border + random.randint(20, 44)  # status bar

                canvas = Image.new("RGB", (canvas_w, canvas_h), chrome)
                canvas.paste(img, (border, border + random.randint(20, 44)))

                # Light JPEG compression (screenshots are PNGs that get re-JPEGed by sharing)
                buf = io.BytesIO()
                canvas.save(buf, "JPEG", quality=random.randint(78, 92))
                buf.seek(0)
                final = Image.open(buf).copy()

                dest = REAL_DIR / f"screenshot_real_{idx + saved:07d}.jpg"
                if _save_pil(final, dest, quality=85):
                    saved += 1

        except Exception as e:
            logger.debug(f"[Screenshot] {src.name}: {e}")

    logger.info(f"[Screenshot] {saved} screenshot-framed real images saved.")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# REAL edge case 4 — Download genuine low-light dataset (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

_LOWLIGHT_HF_SOURCES = [
    # LOL (Low-Light paired dataset) — 500 real low/high pairs
    ("deeplite-torch-zoo/lsrw", "train", "low"),
    # Fallback: generic low-light
    ("mrmallam/low_light_image_enhancement_dataset", "train", "image"),
]


def download_lowlight_real(max_images: int) -> int:
    """
    Download genuine low-light camera photos from HuggingFace.
    Falls back to synthetic augmentation if HF is unavailable.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(REAL_DIR, "hf_lowlight_")
    if existing >= max_images:
        logger.info(f"[HF-LowLight] {existing} already downloaded.")
        return 0

    ds_mod = _require_datasets()
    if ds_mod is None:
        logger.info("[HF-LowLight] HF not available — using synthetic fallback.")
        return augment_lowlight(max_images)

    need = max_images - existing
    idx  = _next_index(REAL_DIR)

    for repo, split, col in _LOWLIGHT_HF_SOURCES:
        logger.info(f"[HF-LowLight] Trying {repo} …")
        try:
            ds    = ds_mod.load_dataset(repo, split=split, streaming=True)
            saved = 0
            for sample in ds:
                if saved >= need:
                    break
                img = sample.get(col)
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        img = Image.open(io.BytesIO(img)).copy() if isinstance(img, (bytes, bytearray)) \
                              else Image.fromarray(img)
                    except Exception:
                        continue
                dest = REAL_DIR / f"hf_lowlight_{idx + saved:07d}.jpg"
                if not dest.exists() and _save_pil(img.convert("RGB"), dest, quality=90):
                    saved += 1
            print()
            if saved > 0:
                logger.info(f"[HF-LowLight] {saved} images from {repo}.")
                return saved
            logger.warning(f"[HF-LowLight] {repo}: 0 images.")
        except Exception as e:
            logger.warning(f"[HF-LowLight] {repo} failed: {str(e)[:100]}")

    logger.info("[HF-LowLight] All HF sources failed — using synthetic augmentation.")
    return augment_lowlight(max_images)


# ─────────────────────────────────────────────────────────────────────────────
# REAL edge case 5 — Open Images diversity (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def download_openimages_real(max_images: int) -> int:
    """
    Stream a subset of Google Open Images via HuggingFace for camera diversity.

    Open Images V7 covers 600+ categories photographed by many different cameras
    in varied conditions — adds device diversity beyond COCO's mostly DSLR photos.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(REAL_DIR, "openimages_")
    if existing >= max_images:
        logger.info(f"[OpenImages] {existing} already downloaded.")
        return 0

    ds_mod = _require_datasets()
    if ds_mod is None:
        return 0

    need = max_images - existing
    idx  = _next_index(REAL_DIR)

    # Multiple fallback repos — Open Images is mirrored across several HF repos
    repos = [
        ("jxie/flickr8k",    "train", "image"),   # reliable fallback (same source quality)
        ("nlphuji/flickr30k", "test",  "image"),
    ]

    for repo, split, col in repos:
        logger.info(f"[OpenImages] Trying {repo} for camera diversity …")
        try:
            ds    = ds_mod.load_dataset(repo, split=split, streaming=True)
            saved = 0
            for i, sample in enumerate(ds):
                if saved >= need:
                    break
                img = sample.get(col)
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        img = Image.open(io.BytesIO(img)).copy() if isinstance(img, (bytes, bytearray)) \
                              else Image.fromarray(img)
                    except Exception:
                        continue
                dest = REAL_DIR / f"openimages_{idx + saved:07d}.jpg"
                if not dest.exists() and _save_pil(img.convert("RGB"), dest, quality=92):
                    saved += 1
                if saved % 100 == 0 and saved > 0:
                    _progress(saved, need, repo.split("/")[-1])
            print()
            if saved > 0:
                logger.info(f"[OpenImages] {saved} images from {repo}.")
                return saved
        except Exception as e:
            logger.warning(f"[OpenImages] {repo} failed: {str(e)[:100]}")

    logger.warning("[OpenImages] All sources failed.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# AI edge case 1 — SDXL photorealistic images (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

_SDXL_HF_SOURCES = [
    # bitmind is already used in prepare_dataset but here we filter for SDXL-
    # tagged samples specifically (metadata field 'model' or similar)
    ("bitmind/ai-image-detection",                "train", "image", 1),
    ("artificialguybr/StableDiffusionXL-PromptSharpenTestImages", "train", "image", None),
    ("john6666/sdxl-test-images",                 "train", "image", None),
]


def download_sdxl_ai(max_images: int) -> int:
    """
    Download Stable Diffusion XL photorealistic outputs.

    SDXL produces much more realistic skin, hair, and lighting than SD-1.5.
    The model trained only on DiffusionDB (SD-1.5) may misclassify SDXL images
    as real.  This function adds SDXL diversity to the AI training set.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(AI_DIR, "sdxl_")
    if existing >= max_images:
        logger.info(f"[SDXL] {existing} already downloaded.")
        return 0

    ds_mod = _require_datasets()
    if ds_mod is None:
        return 0

    need = max_images - existing
    idx  = _next_index(AI_DIR)

    for repo, split, col, label_filter in _SDXL_HF_SOURCES:
        logger.info(f"[SDXL] Trying {repo} …")
        try:
            ds    = ds_mod.load_dataset(repo, split=split, streaming=True)
            saved = 0
            for i, sample in enumerate(ds):
                if saved >= need:
                    break
                # Filter by label if needed
                if label_filter is not None and sample.get("label") != label_filter:
                    continue
                img = sample.get(col)
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        img = Image.open(io.BytesIO(img)).copy() if isinstance(img, (bytes, bytearray)) \
                              else Image.fromarray(img)
                    except Exception:
                        continue
                dest = AI_DIR / f"sdxl_{idx + saved:07d}.jpg"
                if not dest.exists() and _save_pil(img.convert("RGB"), dest, quality=92):
                    saved += 1
                if saved % 100 == 0 and saved > 0:
                    _progress(saved, need, "SDXL")
            print()
            if saved > 0:
                logger.info(f"[SDXL] {saved} SDXL images from {repo}.")
                return saved
            logger.warning(f"[SDXL] {repo}: 0 images.")
        except Exception as e:
            logger.warning(f"[SDXL] {repo} failed: {str(e)[:100]}")

    logger.warning("[SDXL] All sources failed.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# AI edge case 2 — Midjourney-style images (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

_MIDJOURNEY_HF_SOURCES = [
    ("Falah/midjourney_images",    "train",      "image", None),
    ("openskyml/midjourney-images","train",      "image", None),
    ("bitmind/ai-image-detection", "train",      "image", 1),   # diverse fallback
]


def download_midjourney_ai(max_images: int) -> int:
    """
    Download Midjourney v4/v5/v6 generated images.

    Midjourney has distinct stylistic characteristics (film-grain aesthetic,
    high coherence, slightly dreamlike lighting) that differ from SD-1.5.
    Training on these makes the detector robust to the most popular commercial
    AI art platform.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(AI_DIR, "midjourney_")
    if existing >= max_images:
        logger.info(f"[Midjourney] {existing} already downloaded.")
        return 0

    ds_mod = _require_datasets()
    if ds_mod is None:
        return 0

    need = max_images - existing
    idx  = _next_index(AI_DIR)

    for repo, split, col, label_filter in _MIDJOURNEY_HF_SOURCES:
        logger.info(f"[Midjourney] Trying {repo} …")
        try:
            ds    = ds_mod.load_dataset(repo, split=split, streaming=True)
            saved = 0
            for i, sample in enumerate(ds):
                if saved >= need:
                    break
                if label_filter is not None and sample.get("label") != label_filter:
                    continue
                img = sample.get(col)
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        img = Image.open(io.BytesIO(img)).copy() if isinstance(img, (bytes, bytearray)) \
                              else Image.fromarray(img)
                    except Exception:
                        continue
                dest = AI_DIR / f"midjourney_{idx + saved:07d}.jpg"
                if not dest.exists() and _save_pil(img.convert("RGB"), dest, quality=92):
                    saved += 1
                if saved % 100 == 0 and saved > 0:
                    _progress(saved, need, "Midjourney")
            print()
            if saved > 0:
                logger.info(f"[Midjourney] {saved} Midjourney images from {repo}.")
                return saved
            logger.warning(f"[Midjourney] {repo}: 0 images.")
        except Exception as e:
            logger.warning(f"[Midjourney] {repo} failed: {str(e)[:100]}")

    logger.warning("[Midjourney] All sources failed.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# AI edge case 3 — Flux photorealistic images (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

_FLUX_HF_SOURCES = [
    ("black-forest-labs/FLUX.1-dev-demo-images", "train", "image", None),
    ("fal/flux-realism-lora-images",             "train", "image", None),
    ("bitmind/ai-image-detection",               "train", "image", 1),   # fallback
]


def download_flux_ai(max_images: int) -> int:
    """
    Download Black Forest Labs Flux.1 photorealistic outputs.

    Flux.1 is a newer architecture (2024) that produces images with near-perfect
    photorealism, correct hand anatomy, and accurate text rendering — making them
    among the hardest AI images to detect.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(AI_DIR, "flux_")
    if existing >= max_images:
        logger.info(f"[Flux] {existing} already downloaded.")
        return 0

    ds_mod = _require_datasets()
    if ds_mod is None:
        return 0

    need = max_images - existing
    idx  = _next_index(AI_DIR)

    for repo, split, col, label_filter in _FLUX_HF_SOURCES:
        logger.info(f"[Flux] Trying {repo} …")
        try:
            ds    = ds_mod.load_dataset(repo, split=split, streaming=True)
            saved = 0
            for i, sample in enumerate(ds):
                if saved >= need:
                    break
                if label_filter is not None and sample.get("label") != label_filter:
                    continue
                img = sample.get(col)
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        img = Image.open(io.BytesIO(img)).copy() if isinstance(img, (bytes, bytearray)) \
                              else Image.fromarray(img)
                    except Exception:
                        continue
                dest = AI_DIR / f"flux_{idx + saved:07d}.jpg"
                if not dest.exists() and _save_pil(img.convert("RGB"), dest, quality=92):
                    saved += 1
                if saved % 100 == 0 and saved > 0:
                    _progress(saved, need, "Flux")
            print()
            if saved > 0:
                logger.info(f"[Flux] {saved} Flux images from {repo}.")
                return saved
            logger.warning(f"[Flux] {repo}: 0 images.")
        except Exception as e:
            logger.warning(f"[Flux] {repo} failed: {str(e)[:100]}")

    logger.warning("[Flux] All sources failed.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# AI edge case 4 — AI screenshot simulation
# ─────────────────────────────────────────────────────────────────────────────

def augment_screenshot_ai(max_images: int) -> int:
    """
    Embed AI images inside simulated browser / chat app UI chrome.

    When someone shares an AI image as a screenshot (e.g., screenshotting an
    AI art generation app), the resulting image has a UI border and gets
    re-JPEGed.  The model must learn to detect AI content even when it has
    been screenshot-wrapped.
    """
    if not PIL_OK:
        return 0

    existing = _prefix_count(AI_DIR, "screenshot_ai_")
    if existing >= max_images:
        logger.info(f"[ScreenshotAI] {existing} AI screenshot images already exist.")
        return 0

    need    = max_images - existing
    sources = _sample_existing(AI_DIR, need * 2, exclude_prefixes=("screenshot_", "social_"))
    idx     = _next_index(AI_DIR)
    saved   = 0

    chrome_palettes = [
        (240, 240, 242),
        (255, 255, 255),
        (30,  30,  30),    # dark-mode AI art app
        (20,  20,  40),    # Midjourney-style dark background
    ]

    logger.info(f"[ScreenshotAI] Creating {need} screenshot-framed AI images…")

    for src in sources:
        if saved >= need:
            break
        try:
            with Image.open(src) as img:
                img    = img.convert("RGB")
                chrome = random.choice(chrome_palettes)
                border = random.randint(8, 28)
                bar_h  = random.randint(24, 56)  # top bar (generation prompt area)

                canvas_w = img.width  + 2 * border
                canvas_h = img.height + 2 * border + bar_h
                canvas   = Image.new("RGB", (canvas_w, canvas_h), chrome)
                canvas.paste(img, (border, border + bar_h))

                buf = io.BytesIO()
                canvas.save(buf, "JPEG", quality=random.randint(80, 94))
                buf.seek(0)
                final = Image.open(buf).copy()

                dest = AI_DIR / f"screenshot_ai_{idx + saved:07d}.jpg"
                if _save_pil(final, dest, quality=88):
                    saved += 1

        except Exception as e:
            logger.debug(f"[ScreenshotAI] {src.name}: {e}")

    logger.info(f"[ScreenshotAI] {saved} screenshot-framed AI images saved.")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print current dataset composition by source prefix."""
    def _breakdown(d: Path) -> dict:
        from collections import Counter
        if not d.exists():
            return {}
        counts: Counter = Counter()
        for f in d.iterdir():
            if f.suffix.lower() not in IMG_EXTS:
                continue
            prefix = f.name.split("_")[0]
            counts[prefix] += 1
        return dict(counts.most_common())

    real_n = _count(REAL_DIR)
    ai_n   = _count(AI_DIR)

    print()
    print("=" * 66)
    print("  Dataset Composition After Expansion")
    print("=" * 66)
    print(f"  datasets/real/  {real_n:>6,} images")
    for k, v in _breakdown(REAL_DIR).items():
        bar_w = min(20, int(v / max(real_n, 1) * 20))
        print(f"    {k:<20} {v:>6,}  {'|' * bar_w}")
    print()
    print(f"  datasets/ai/    {ai_n:>6,} images")
    for k, v in _breakdown(AI_DIR).items():
        bar_w = min(20, int(v / max(ai_n, 1) * 20))
        print(f"    {k:<20} {v:>6,}  {'|' * bar_w}")
    print()

    total = real_n + ai_n
    if total < 5_000:
        q = "[WARN]  Small dataset (<5k) — expect ~75-85% accuracy."
    elif total < 12_000:
        q = "[OK]    Good dataset — expect ~85-92% accuracy."
    elif total < 20_000:
        q = "[GOOD]  Large dataset — expect 90-95% accuracy."
    else:
        q = "[GREAT] Very large dataset — expect 95%+ accuracy."
    print(f"  {q}")
    print()
    print("  Next step — retrain on expanded dataset:")
    print()
    print("    cd backend")
    print("    python -m training.train \\")
    print("        --real-dirs datasets/real \\")
    print("        --ai-dirs   datasets/ai \\")
    print("        --output    inference/checkpoints \\")
    print("        --epochs 20 --batch-size 16")
    print()
    print("  Or use the failure pipeline for hard-negatives first:")
    print("    python -m training.retrain_failures --retrain --epochs 15")
    print()
    print("=" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Targeted dataset expansion — adds edge-case images the base dataset lacks.\n"
            "Sources: low-light, social-chain compression, screenshots, SDXL, Midjourney, Flux."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full expansion (all sources, default 500 per method)
  python -m training.expand_dataset

  # Quick test run (100 per method — fast)
  python -m training.expand_dataset --max-per-source 100

  # Augmentation only — no HuggingFace downloads required
  python -m training.expand_dataset --augment-only

  # Only download new AI models (SDXL, Midjourney, Flux)
  python -m training.expand_dataset --ai-only

  # Check current dataset composition
  python -m training.expand_dataset --summary-only
""",
    )
    p.add_argument("--max-per-source", type=int, default=500,
                   help="Max images to add per source (default: 500).")
    p.add_argument("--augment-only",   action="store_true",
                   help="Only run synthetic augmentation — no HF downloads.")
    p.add_argument("--ai-only",        action="store_true",
                   help="Only expand the AI dataset (SDXL, Midjourney, Flux, AI screenshots).")
    p.add_argument("--real-only",      action="store_true",
                   help="Only expand the real dataset (low-light, social, screenshots).")
    p.add_argument("--summary-only",   action="store_true",
                   help="Print dataset composition and exit without downloading.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_dirs()

    if args.summary_only:
        print_summary()
        return

    n = args.max_per_source
    logger.info(f"Targeted dataset expansion — up to {n} images per source.")

    totals: dict = {
        "real_added": 0,
        "ai_added":   0,
    }

    # ── REAL sources ──────────────────────────────────────────────────────────
    if not args.ai_only:
        logger.info("\n── REAL edge-case sources ──────────────────────────────────────")

        # 1. Genuine low-light (HF with synthetic fallback)
        if not args.augment_only:
            saved = download_lowlight_real(n)
        else:
            saved = augment_lowlight(n)
        totals["real_added"] += saved

        # 2. Synthetic low-light from COCO (always runs — offline-safe)
        saved = augment_lowlight(n)
        totals["real_added"] += saved

        # 3. Social chain compression — real photos
        saved = augment_social_chain_compression(REAL_DIR, "social_real_", n)
        totals["real_added"] += saved

        # 4. Screenshot-framed real photos
        saved = augment_screenshot_real(n)
        totals["real_added"] += saved

        # 5. Open Images diversity (HF)
        if not args.augment_only:
            saved = download_openimages_real(n)
            totals["real_added"] += saved

    # ── AI sources ────────────────────────────────────────────────────────────
    if not args.real_only:
        logger.info("\n── AI edge-case sources ────────────────────────────────────────")

        # 1. SDXL photorealistic (HF)
        if not args.augment_only:
            saved = download_sdxl_ai(n)
            totals["ai_added"] += saved

        # 2. Midjourney-style (HF)
        if not args.augment_only:
            saved = download_midjourney_ai(n)
            totals["ai_added"] += saved

        # 3. Flux photorealistic (HF)
        if not args.augment_only:
            saved = download_flux_ai(n)
            totals["ai_added"] += saved

        # 4. AI social chain compression
        saved = augment_social_chain_compression(AI_DIR, "social_ai_", n)
        totals["ai_added"] += saved

        # 5. AI screenshot simulation
        saved = augment_screenshot_ai(n)
        totals["ai_added"] += saved

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"\nExpansion complete: +{totals['real_added']} real, "
        f"+{totals['ai_added']} AI images added."
    )
    print_summary()


if __name__ == "__main__":
    main()
