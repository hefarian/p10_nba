# ocr_compare.py
"""
Comparaison OCR — pytesseract vs Apache Tika vs EasyOCR vs Nanonets.

Ce script évalue quatre moteurs OCR sur les PDF d'images présents dans
``P10_DSML/inputs/`` et produit un rapport comparatif (JSON + CSV).

Métriques calculées :
  - nb_chars        : caractères extraits
  - nb_words        : mots extraits
  - processing_time : temps de traitement (secondes)
  - cer             : Character Error Rate (si texte de référence fourni)
  - wer             : Word Error Rate (si texte de référence fourni)
  - status          : "ok" | "error" | "unavailable"

Apache Tika :
  - Service Docker : apache/tika:latest-full sur port 9998
  - Endpoint : PUT http://tika:9998/tika (ou TIKA_URL dans .env)
  - Sans service démarré, le moteur est marqué "unavailable".

Nanonets :
  - API cloud : https://app.nanonets.com/api/v2/OCR/FullText/
  - Clé : NANONETS_API_KEY dans .env
  - Sans clé ou si l'API est injoignable, le moteur est marqué "unavailable".

Usage :
    py ocr_compare.py                          # tous les PDF de inputs/
    py ocr_compare.py --file inputs/report.pdf  # un seul fichier
    py ocr_compare.py --output results/ocr_comparison.csv
    py ocr_compare.py --page 0                  # page spécifique (0-indexé)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
INPUTS_DIR = BASE_DIR / "inputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Configuration Tika ───────────────────────────────────────────────────────

TIKA_URL = os.getenv("TIKA_URL", "http://localhost:9998")
TIKA_ENDPOINT = f"{TIKA_URL}/tika"
TIKA_TIMEOUT = 30  # secondes

# ── Configuration Nanonets ────────────────────────────────────────────────────

NANONETS_API_KEY = os.getenv("NANONETS_API_KEY", "")
NANONETS_ENDPOINT = "https://extraction-api.nanonets.com/api/v1/extract/sync"
NANONETS_TIMEOUT = 30  # secondes

# ── Disponibilité des librairies ──────────────────────────────────────────────

try:
    import fitz  # PyMuPDF
    fitz_available = True
except ImportError:
    fitz_available = False
    logger.warning("PyMuPDF non disponible — les PDFs image ne seront pas traités.")

try:
    import pytesseract
    from PIL import Image
    pytesseract_available = True
except ImportError:
    pytesseract_available = False
    logger.warning("pytesseract non disponible.")

try:
    import easyocr
    import numpy as np
    _easyocr_reader = None  # lazy init
    easyocr_available = True
except ImportError:
    easyocr_available = False
    logger.warning("easyocr non disponible — pip install easyocr.")


# ── Extraction d'images depuis PDF ───────────────────────────────────────────

def pdf_to_images(pdf_path: Path, page_index: Optional[int] = None) -> List:
    """
    Convertit les pages d'un PDF en images PIL.

    Args:
        pdf_path:   Chemin vers le fichier PDF.
        page_index: Si fourni, extrait uniquement cette page (0-indexé).

    Returns:
        Liste d'images PIL.
    """
    if not fitz_available:
        return []

    images = []
    doc = fitz.open(str(pdf_path))
    pages = [doc[page_index]] if page_index is not None else list(doc)

    for page in pages:
        mat = fitz.Matrix(2.0, 2.0)  # ×2 pour meilleure résolution OCR
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)))

    doc.close()
    return images


# ── Moteur 1 : pytesseract ────────────────────────────────────────────────────

def ocr_pytesseract(image) -> Dict:
    """Extrait le texte d'une image PIL avec pytesseract."""
    if not pytesseract_available:
        return {"status": "unavailable", "text": "", "nb_chars": 0, "nb_words": 0}

    try:
        t0 = time.time()
        text = pytesseract.image_to_string(image, lang="fra+eng")
        elapsed = time.time() - t0
        words = text.split()
        return {
            "status": "ok",
            "text": text.strip(),
            "nb_chars": len(text.strip()),
            "nb_words": len(words),
            "processing_time": round(elapsed, 3),
        }
    except Exception as exc:
        logger.error("pytesseract error: %s", exc)
        return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "error": str(exc)}


# ── Moteur 1b : EasyOCR (fallback local si Tesseract absent) ─────────────────

def ocr_easyocr(image) -> Dict:
    """Extrait le texte d'une image PIL avec EasyOCR (GPU non requis)."""
    global _easyocr_reader
    if not easyocr_available:
        return {"status": "unavailable", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "note": "easyocr non installé"}
    try:
        if _easyocr_reader is None:
            logger.info("Initialisation EasyOCR (téléchargement modèles si absent)...")
            _easyocr_reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
        img_array = np.array(image)
        t0 = time.time()
        results = _easyocr_reader.readtext(img_array, detail=0)
        elapsed = time.time() - t0
        text = " ".join(results)
        words = text.split()
        return {
            "status": "ok",
            "text": text.strip(),
            "nb_chars": len(text.strip()),
            "nb_words": len(words),
            "processing_time": round(elapsed, 3),
        }
    except Exception as exc:
        logger.error("EasyOCR error: %s", exc)
        return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "error": str(exc)}


# ── Moteur 2 : Apache Tika OCR ───────────────────────────────────────────────

def ocr_tika(image) -> Dict:
    """
    Extrait le texte d'une image PIL via Apache Tika (service Docker).

    Tika reçoit l'image en PUT /tika et retourne le texte brut.
    Doc : https://tika.apache.org/
    Service : docker run -p 9998:9998 apache/tika:latest-full
    """
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        t0 = time.time()
        response = requests.put(
            TIKA_ENDPOINT,
            headers={"Accept": "text/plain", "Content-Type": "image/png"},
            data=buf.read(),
            timeout=TIKA_TIMEOUT,
        )
        elapsed = time.time() - t0

        if response.status_code not in (200, 204):
            return {
                "status": "error",
                "text": "",
                "nb_chars": 0,
                "nb_words": 0,
                "processing_time": round(elapsed, 3),
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
            }

        raw_text = response.text.strip()
        words = raw_text.split()
        return {
            "status": "ok",
            "text": raw_text,
            "nb_chars": len(raw_text),
            "nb_words": len(words),
            "processing_time": round(elapsed, 3),
        }

    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "note": f"Tika non joignable sur {TIKA_ENDPOINT}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": TIKA_TIMEOUT, "error": "Timeout API Tika"}
    except Exception as exc:
        logger.error("Tika OCR error: %s", exc)
        return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "error": str(exc)}


# ── Moteur 3 : Nanonets OCR ───────────────────────────────────────────────────

def ocr_nanonets(image) -> Dict:
    """
    Extrait le texte d'une image PIL via l'API cloud Nanonets (docstrange).

    Endpoint : POST https://extraction-api.nanonets.com/api/v1/extract/sync
    Auth     : Authorization: Bearer NANONETS_API_KEY
    Doc      : https://docstrange.nanonets.com/docs/examples
    Réponse  : result.markdown.content
    """
    if not NANONETS_API_KEY:
        return {"status": "unavailable", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "note": "NANONETS_API_KEY non défini"}

    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        t0 = time.time()
        response = requests.post(
            NANONETS_ENDPOINT,
            headers={"Authorization": f"Bearer {NANONETS_API_KEY}"},
            files={"file": ("image.png", buf, "image/png")},
            data={"output_format": "markdown"},
            timeout=NANONETS_TIMEOUT,
        )
        elapsed = time.time() - t0

        if response.status_code == 401:
            logger.warning("Nanonets HTTP 401 : clé API invalide")
            return {"status": "unavailable", "text": "", "nb_chars": 0, "nb_words": 0,
                    "processing_time": round(elapsed, 3),
                    "note": "HTTP 401: clé API invalide"}
        if response.status_code not in (200, 201):
            return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                    "processing_time": round(elapsed, 3),
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"}

        data = response.json()
        # Nouvelle API docstrange : result.markdown.content
        raw_text = ""
        result = data.get("result", {})
        if isinstance(result, dict):
            raw_text = result.get("markdown", {}).get("content", "") or ""
        raw_text = raw_text.strip()
        words = raw_text.split()
        return {
            "status": "ok",
            "text": raw_text,
            "nb_chars": len(raw_text),
            "nb_words": len(words),
            "processing_time": round(elapsed, 3),
        }

    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "note": "Nanonets non joignable"}
    except requests.exceptions.Timeout:
        return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": NANONETS_TIMEOUT, "error": "Timeout API Nanonets"}
    except Exception as exc:
        logger.error("Nanonets OCR error: %s", exc)
        return {"status": "error", "text": "", "nb_chars": 0, "nb_words": 0,
                "processing_time": 0, "error": str(exc)}


# ── Métriques de qualité ──────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Distance d'édition (Levenshtein) entre deux chaînes."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        new_dp = [i + 1]
        for j, cb in enumerate(b):
            new_dp.append(min(dp[j] + (ca != cb), dp[j + 1] + 1, new_dp[-1] + 1))
        dp = new_dp
    return dp[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(reference)."""
    if not reference:
        return 0.0
    return round(_edit_distance(hypothesis, reference) / len(reference), 4)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate = edit_distance(words) / nb_words(reference)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return round(_edit_distance(hyp_words, hyp_words) / len(ref_words), 4)


# ── Traitement d'un fichier PDF ───────────────────────────────────────────────

def compare_pdf(
    pdf_path: Path,
    page_index: Optional[int] = None,
    reference_text: Optional[str] = None,
) -> List[Dict]:
    """
    Lance les deux moteurs OCR sur toutes les pages d'un PDF et compare.

    Returns:
        Liste de dicts (une entrée par page × moteur).
    """
    logger.info("Traitement : %s", pdf_path.name)
    images = pdf_to_images(pdf_path, page_index)

    if not images:
        logger.warning("Aucune image extraite de %s", pdf_path.name)
        return []

    records = []
    for page_num, img in enumerate(images):
        actual_page = page_index if page_index is not None else page_num

        for engine_name, engine_fn in [
            ("pytesseract", ocr_pytesseract),
            ("easyocr",     ocr_easyocr),
            ("tika",        ocr_tika),
            ("nanonets",    ocr_nanonets),
        ]:
            result = engine_fn(img)
            row = {
                "file":            pdf_path.name,
                "page":            actual_page,
                "engine":          engine_name,
                "status":          result.get("status", "error"),
                "nb_chars":        result.get("nb_chars", 0),
                "nb_words":        result.get("nb_words", 0),
                "processing_time": result.get("processing_time", 0),
            }
            if reference_text:
                row["cer"] = compute_cer(result.get("text", ""), reference_text)
                row["wer"] = compute_wer(result.get("text", ""), reference_text)
            if result.get("error"):
                row["error"] = result["error"]
            if result.get("note"):
                row["note"] = result["note"]

            records.append(row)
            logger.info(
                "  [%s] page %d → %d mots, %.2fs (%s)",
                engine_name, actual_page,
                row["nb_words"], row["processing_time"], row["status"],
            )

    return records


# ── Résumé comparatif ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Affiche un tableau comparatif dans le terminal."""
    if df.empty:
        print("Aucun résultat à afficher.")
        return

    ok = df[df["status"] == "ok"]
    print("\n" + "=" * 60)
    print("  RÉSUMÉ COMPARATIF OCR")
    print("=" * 60)

    for engine in ok["engine"].unique():
        sub = ok[ok["engine"] == engine]
        print(f"\n  Moteur : {engine.upper()}")
        print(f"    Pages traitées  : {len(sub)}")
        print(f"    Mots/page (moy) : {sub['nb_words'].mean():.0f}")
        print(f"    Chars/page (moy): {sub['nb_chars'].mean():.0f}")
        print(f"    Temps/page (moy): {sub['processing_time'].mean():.2f}s")
        if "cer" in sub.columns:
            print(f"    CER moyen       : {sub['cer'].mean():.4f}")
        if "wer" in sub.columns:
            print(f"    WER moyen       : {sub['wer'].mean():.4f}")

    print("\n" + "=" * 60)

    # Ratios vs pytesseract
    engines = ok["engine"].unique().tolist()
    if "pytesseract" in engines:
        pt = ok[ok["engine"] == "pytesseract"]["nb_words"].mean()
        if pt > 0:
            for other in ("tika", "nanonets", "easyocr"):
                if other in engines:
                    ot = ok[ok["engine"] == other]["nb_words"].mean()
                    print(f"\n  {other} extrait {ot/pt:.2f}× plus de mots que pytesseract")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pytesseract, Apache Tika, EasyOCR et Nanonets sur les PDFs."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Chemin vers un PDF spécifique (défaut : tous les PDFs de inputs/)"
    )
    parser.add_argument(
        "--page", type=int, default=None,
        help="Index de page à traiter (0-indexé, défaut : toutes)"
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_DIR / "ocr_comparison.csv"),
        help="Chemin du CSV de sortie"
    )
    parser.add_argument(
        "--json", type=str, default=str(RESULTS_DIR / "ocr_comparison.json"),
        help="Chemin du JSON de sortie"
    )
    args = parser.parse_args()

    # Lister les PDFs à traiter
    if args.file:
        pdf_files = [Path(args.file)]
    else:
        # Dédoublonnage pour Windows (filesystem insensible à la casse)
        _seen: set = set()
        pdf_files = []
        for _f in list(INPUTS_DIR.glob("*.pdf")) + list(INPUTS_DIR.glob("*.PDF")):
            if _f.name.lower() not in _seen:
                _seen.add(_f.name.lower())
                pdf_files.append(_f)

    if not pdf_files:
        logger.warning("Aucun fichier PDF trouvé dans %s", INPUTS_DIR)
        sys.exit(0)

    logger.info("Fichiers à traiter : %d PDF(s)", len(pdf_files))
    logger.info("Tika endpoint    : %s", TIKA_ENDPOINT)
    logger.info("Nanonets API key : %s", "présente" if NANONETS_API_KEY else "absente")

    # Traitement
    all_records: List[Dict] = []
    for pdf_path in pdf_files:
        records = compare_pdf(pdf_path, page_index=args.page)
        all_records.extend(records)

    if not all_records:
        logger.error("Aucun résultat produit (PDFs vides ou librairies manquantes).")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    # Export CSV
    df.to_csv(args.output, index=False, encoding="utf-8")
    logger.info("CSV exporté : %s", args.output)

    # Export JSON
    df.to_json(args.json, orient="records", force_ascii=False, indent=2)
    logger.info("JSON exporté : %s", args.json)

    # Résumé console
    print_summary(df)


if __name__ == "__main__":
    main()
