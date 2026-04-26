# evaluate_ragas.py
"""
Script d'évaluation automatisé du système RAG avec le framework RAGAS.

Usage:
    py evaluate_ragas.py
    py evaluate_ragas.py --output results/eval_avant.json

Ce script :
  1. Charge le jeu de questions métiers NBA (simples, complexes, bruitées)
  2. Appelle le pipeline RAG pour obtenir réponses + contextes
  3. Calcule les métriques RAGAS (faithfulness, answer_relevancy, context_precision)
  4. Exporte un rapport comparatif JSON + tableau CSV

Notes :
  - Les métriques RAGAS nécessitent un LLM juge. Ici on utilise Mistral via l'adapter OpenAI-compat.
  - Si RAGAS_USE_SENTENCES_TRANSFORMERS=1 dans .env, on utilise un embedder local (plus lent mais gratuit).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Fix conflit OpenMP entre PyTorch et FAISS sur Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from dotenv import load_dotenv

# Ajouter le répertoire courant au path si nécessaire
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports locaux
# ---------------------------------------------------------------------------
try:
    from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
    from utils.vector_store import VectorStoreManager
    from utils.pipeline import EvalSample, RAGQuery, RAGResponse, RetrievedChunk
except ImportError as e:
    logger.error(f"Erreur d'import : {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Imports RAGAS
# ---------------------------------------------------------------------------
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning(
        "RAGAS non installé. Exécutez : pip install ragas>=0.2.0 datasets"
    )
    RAGAS_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("'datasets' non installé. Exécutez : pip install datasets")
    DATASETS_AVAILABLE = False

try:
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings  # type: ignore
    LANGCHAIN_MISTRAL_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatMistralAI  # type: ignore
        from langchain_mistralai import MistralAIEmbeddings  # type: ignore
        LANGCHAIN_MISTRAL_AVAILABLE = True
    except ImportError:
        LANGCHAIN_MISTRAL_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logger.error("mistralai non installé.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Jeu de tests métiers NBA
# ---------------------------------------------------------------------------

TEST_CASES: List[EvalSample] = [
    # --- Questions simples ---
    EvalSample(
        question="Quel joueur a marqué le plus de points au total cette saison ?",
        ground_truth="Shai Gilgeous-Alexander est le meilleur marqueur de la saison régulière avec 2485 points.",
        category="simple",
        answer="", contexts=[],
    ),
    EvalSample(
        question="Quel est le pourcentage moyen aux lancers francs des joueurs NBA cette saison ?",
        ground_truth="Le pourcentage moyen aux lancers francs des joueurs NBA est d'environ 80%.",
        category="simple",
        answer="", contexts=[],
    ),
    EvalSample(
        question="Qui est le meilleur intercepteur (STL) de la saison régulière ?",
        ground_truth="Le joueur avec le plus d'interceptions est parmi les meneurs de jeu défensifs.",
        category="simple",
        answer="", contexts=[],
    ),
    # --- Questions complexes ---
    EvalSample(
        question=(
            "Quel joueur a le meilleur pourcentage à 3 points parmi ceux "
            "qui tirent au moins 3 fois par match ?"
        ),
        ground_truth=(
            "Parmi les joueurs tirant au moins 3 tentatives par match à 3 points, "
            "le meilleur pourcentage dépasse généralement 42%."
        ),
        category="complex",
        answer="", contexts=[],
    ),
    EvalSample(
        question=(
            "Compare les statistiques de rebonds (OREB et DREB) des joueurs "
            "de l'équipe OKC par rapport à l'équipe DEN cette saison."
        ),
        ground_truth=(
            "L'équipe DEN est dominée par Nikola Jokić qui cumule des statistiques "
            "de rebonds exceptionnelles (889 rebonds totaux)."
        ),
        category="complex",
        answer="", contexts=[],
    ),
    EvalSample(
        question=(
            "Quelle équipe a le meilleur ratio victoires/matchs joués "
            "parmi les équipes de conférence Ouest ?"
        ),
        ground_truth="L'équipe OKC présente un excellent bilan avec 63 victoires sur 76 matchs.",
        category="complex",
        answer="", contexts=[],
    ),
    # --- Questions bruitées ---
    EvalSample(
        question="shai gilgeous c'est qui exactement et c'est quoi ses stats???",
        ground_truth=(
            "Shai Gilgeous-Alexander est le meneur de l'OKC Thunder, "
            "meilleur marqueur de la saison avec 2485 points, 51.9% au tir."
        ),
        category="noisy",
        answer="", contexts=[],
    ),
    EvalSample(
        question="meilleur rebondeur genre top 3 joueurs ?? genre les pivots?",
        ground_truth=(
            "Les meilleurs rebondeurs sont généralement les pivots. "
            "Nikola Jokić de Denver domine avec 889 rebonds."
        ),
        category="noisy",
        answer="", contexts=[],
    ),
    EvalSample(
        question="ekip qui gagne le plus en 2024 2025 nba saison reguliere",
        ground_truth="OKC Thunder est l'une des meilleures équipes de la saison régulière 2024-2025.",
        category="noisy",
        answer="", contexts=[],
    ),
]

# ---------------------------------------------------------------------------
# Fonctions RAG
# ---------------------------------------------------------------------------

def call_rag_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: MistralClient,
    k: int = SEARCH_K,
) -> tuple[str, List[str]]:
    """Appelle le pipeline RAG et retourne (réponse_llm, contextes)."""

    # 1. Validation de la requête
    query = RAGQuery(question=question, k=k)

    # 2. Recherche dans le vector store
    try:
        results = vector_store.search(query.question, k=query.k)
    except Exception as e:
        logger.warning(f"Recherche vector store échouée pour '{question}': {e}")
        results = []

    # 3. Construction du contexte
    contexts = [r["text"] for r in results]
    context_str = "\n\n---\n\n".join(
        f"Source: {r['metadata'].get('source', 'Inconnue')}\nContenu: {r['text']}"
        for r in results
    ) if results else "Aucune information trouvée dans la base de connaissances."

    # 4. Appel LLM
    system_prompt = f"""Tu es un analyste NBA expert. Réponds uniquement en te basant sur le contexte fourni.
Si le contexte ne contient pas l'information, dis-le clairement.

CONTEXTE:
{context_str}

QUESTION: {question}

RÉPONSE:"""

    try:
        response = mistral_client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content if response.choices else "Pas de réponse."
    except Exception as e:
        logger.error(f"Erreur LLM pour '{question}': {e}")
        answer = f"Erreur lors de la génération de la réponse: {e}"

    # 5. Validation de la sortie
    try:
        validated = RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=r.get("id", "unknown"),
                    text=r["text"],
                    source=r["metadata"].get("source", "unknown"),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ],
            had_context=bool(results),
        )
        return validated.answer, contexts
    except Exception as e:
        logger.warning(f"Validation RAGResponse échouée: {e}")
        return answer, contexts


# ---------------------------------------------------------------------------
# Évaluation RAGAS
# ---------------------------------------------------------------------------

def build_ragas_dataset(samples: List[EvalSample]) -> Optional["Dataset"]:
    """Construit le Dataset RAGAS depuis les échantillons évalués."""
    if not DATASETS_AVAILABLE:
        return None

    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth or "" for s in samples],
    }
    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset: "Dataset", api_key: str) -> Optional[Dict]:
    """Lance l'évaluation RAGAS et retourne les scores."""
    if not RAGAS_AVAILABLE:
        logger.error("RAGAS n'est pas disponible.")
        return None

    metrics = [faithfulness, answer_relevancy, context_precision]

    # Configurer le LLM juge RAGAS
    if LANGCHAIN_MISTRAL_AVAILABLE:
        try:
            langchain_llm = ChatMistralAI(
                mistral_api_key=api_key,
                model="mistral-small-latest",
                temperature=0,
            )
            langchain_embeddings = MistralAIEmbeddings(
                mistral_api_key=api_key,
                model="mistral-embed",
            )
            llm_wrapper = LangchainLLMWrapper(langchain_llm)
            emb_wrapper = LangchainEmbeddingsWrapper(langchain_embeddings)

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                raise_exceptions=False,
            )
            return result
        except Exception as e:
            logger.error(f"Erreur évaluation RAGAS (Mistral): {e}")

    # Fallback : sentence-transformers pour les embeddings
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        emb_wrapper = LangchainEmbeddingsWrapper(emb)

        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy],  # faithfulness/precision nécessitent LLM juge
            embeddings=emb_wrapper,
            raise_exceptions=False,
        )
        return result
    except Exception as e:
        logger.error(f"Erreur évaluation RAGAS (HuggingFace fallback): {e}")
        return None


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def build_report(
    samples: List[EvalSample],
    ragas_result: Optional[Dict],
) -> pd.DataFrame:
    """Construit un tableau de résultats par question."""

    rows = []
    for s in samples:
        row = {
            "Catégorie": s.category,
            "Question": s.question[:80] + "..." if len(s.question) > 80 else s.question,
            "Réponse (extrait)": s.answer[:100] + "..." if len(s.answer) > 100 else s.answer,
            "Nb contextes": len(s.contexts),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ajout des scores RAGAS si disponibles
    if ragas_result is not None:
        try:
            scores_df = ragas_result.to_pandas()
            metric_cols = [c for c in scores_df.columns if c in [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall"
            ]]
            for col in metric_cols:
                df[col.replace("_", " ").title()] = scores_df[col].values
        except Exception as e:
            logger.warning(f"Impossible d'ajouter les scores RAGAS au DataFrame: {e}")

    return df


def print_summary(df: pd.DataFrame, ragas_result: Optional[Dict]) -> None:
    """Affiche un résumé dans le terminal."""
    print("\n" + "=" * 70)
    print("RAPPORT D'ÉVALUATION RAG - SportSee NBA Assistant")
    print("=" * 70)
    print(df.to_string(index=False))

    if ragas_result is not None:
        print("\n--- Scores moyens RAGAS ---")
        try:
            scores = ragas_result.to_pandas()
            metric_cols = [c for c in scores.columns if c in [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall"
            ]]
            for col in metric_cols:
                mean_val = scores[col].mean()
                print(f"  {col.replace('_', ' ').title():<30}: {mean_val:.4f}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'affichage des scores moyens: {e}")

    by_cat = df.groupby("Catégorie")["Nb contextes"].mean()
    print("\n--- Contextes récupérés par catégorie ---")
    print(by_cat.to_string())
    print("=" * 70)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation RAGAS du système RAG SportSee NBA"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_avant.json",
        help="Chemin du fichier de résultats JSON (default: results/eval_avant.json)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/eval_avant.csv",
        help="Chemin du fichier CSV des résultats",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Désactiver l'évaluation RAGAS (run RAG only, sans métriques)",
    )
    args = parser.parse_args()

    # Vérification clé API
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY non définie dans .env. Arrêt.")
        sys.exit(1)

    # Créer dossier de sortie
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Charger le vector store
    logger.info("Chargement du vector store...")
    try:
        vs_manager = VectorStoreManager()
        if vs_manager.index is None:
            logger.warning(
                "Index Faiss vide. Exécutez d'abord 'py indexer.py'. "
                "Les contextes seront vides."
            )
    except Exception as e:
        logger.error(f"Impossible de charger le VectorStoreManager: {e}")
        sys.exit(1)

    # Initialiser le client Mistral
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

    # Évaluer chaque question
    logger.info(f"Évaluation de {len(TEST_CASES)} questions métiers...")
    evaluated_samples: List[EvalSample] = []

    for i, sample in enumerate(TEST_CASES, 1):
        logger.info(f"[{i}/{len(TEST_CASES)}] '{sample.question[:60]}...' (cat: {sample.category})")
        answer, contexts = call_rag_pipeline(
            question=sample.question,
            vector_store=vs_manager,
            mistral_client=mistral_client,
        )
        enriched = sample.model_copy(update={"answer": answer, "contexts": contexts})
        evaluated_samples.append(enriched)

    # Évaluation RAGAS
    ragas_result = None
    if not args.no_ragas and RAGAS_AVAILABLE and DATASETS_AVAILABLE:
        logger.info("Construction du dataset RAGAS...")
        ragas_dataset = build_ragas_dataset(evaluated_samples)
        if ragas_dataset:
            logger.info("Lancement de l'évaluation RAGAS...")
            ragas_result = run_ragas_evaluation(ragas_dataset, MISTRAL_API_KEY)
    else:
        if args.no_ragas:
            logger.info("Évaluation RAGAS désactivée (--no-ragas).")
        else:
            logger.warning("RAGAS ou datasets non disponibles. Métriques non calculées.")

    # Rapport
    df = build_report(evaluated_samples, ragas_result)
    print_summary(df, ragas_result)

    # Export CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Résultats CSV exportés → {csv_path}")

    # Export JSON
    export_data = {
        "meta": {
            "total_questions": len(evaluated_samples),
            "categories": {
                cat: sum(1 for s in evaluated_samples if s.category == cat)
                for cat in ["simple", "complex", "noisy"]
            },
            "ragas_available": RAGAS_AVAILABLE and not args.no_ragas,
        },
        "samples": [
            {
                "question": s.question,
                "answer": s.answer,
                "contexts_count": len(s.contexts),
                "ground_truth": s.ground_truth,
                "category": s.category,
            }
            for s in evaluated_samples
        ],
        "ragas_scores": None,
    }

    if ragas_result is not None:
        try:
            scores_df = ragas_result.to_pandas()
            metric_cols = [c for c in scores_df.columns if c in [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall"
            ]]
            export_data["ragas_scores"] = {
                col: {
                    "mean": float(scores_df[col].mean()),
                    "min": float(scores_df[col].min()),
                    "max": float(scores_df[col].max()),
                }
                for col in metric_cols
            }
        except Exception as e:
            logger.warning(f"Erreur export scores JSON: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Résultats JSON exportés → {output_path}")
    logger.info("Évaluation terminée.")


if __name__ == "__main__":
    main()
