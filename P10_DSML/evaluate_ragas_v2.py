# evaluate_ragas_v2.py
"""
Script d'évaluation RAGAS v2 — avec routage intelligent SQL / FAISS.

Différence clé avec evaluate_ragas.py (v1) :
  v1 → toutes les questions passent par le pipeline FAISS (textes)
  v2 → les questions chiffrées/numériques sont routées vers le SQL Tool
       (comme le fait MistralChat.py en production)

Ce routage permet de mesurer les vraies performances du système complet,
pas seulement du composant RAG vectoriel.

Modes disponibles :
  --mode rag   → 100% FAISS (comportement identique à v1)
  --mode sql   → routage automatique SQL/FAISS (défaut, comportement production)
  --mode force-sql → toutes les questions passent par le SQL Tool

Usage :
    py evaluate_ragas_v2.py
    py evaluate_ragas_v2.py --output results/eval_apres_v2.json --csv results/eval_apres_v2.csv
    py evaluate_ragas_v2.py --mode rag --output results/eval_avant_v2.json
    py evaluate_ragas_v2.py --no-ragas
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix conflit OpenMP entre PyTorch et FAISS sur Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from dotenv import load_dotenv

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
    logger.error(f"Erreur d'import utils : {e}")
    sys.exit(1)

try:
    from sql_tool import NBAQueryTool
    SQL_TOOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SQL Tool non disponible : {e}")
    SQL_TOOL_AVAILABLE = False

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
    logger.warning("RAGAS non installé. Exécutez : pip install ragas>=0.2.0 datasets")
    RAGAS_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("'datasets' non installé. Exécutez : pip install datasets")
    DATASETS_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain_mistralai import MistralAIEmbeddings
    LANGCHAIN_MISTRAL_AVAILABLE = True
except ImportError:
    LANGCHAIN_MISTRAL_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    logger.error("mistralai non installé.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Jeu de tests métiers NBA (identique à v1)
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
# Routage — repris de MistralChat.py
# ---------------------------------------------------------------------------

SQL_KEYWORDS = [
    "pourcentage", "%", "statistique", "stats", "meilleur",
    "plus de", "moins de", "combien", "quel joueur a", "compare",
    "top ", "classement", "rang", "total", "moyenne", "saison",
    "points", "rebonds", "passes", "assists", "blocks", "steals",
    "tirs", "3 points", "lancers francs", "victoires", "défaites",
    "bilan", "ratio", "efficacité", "usg", "ts%", "efg",
    # variantes bruitées / abréviées
    "marqueur", "rebondeur", "intercepteur", "passeur", "ekip", "stats???",
    "stl", "ast", "pts", "reb",
]


def _is_numerical_question(question: str) -> bool:
    """Détecte si la question nécessite le SQL Tool (données chiffrées)."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in SQL_KEYWORDS)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def call_rag_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
    k: int = SEARCH_K,
) -> Tuple[str, List[str]]:
    """Pipeline FAISS → Mistral. Retourne (réponse, contextes)."""
    query = RAGQuery(question=question, k=k)
    try:
        results = vector_store.search(query.question, k=query.k)
    except Exception as e:
        logger.warning(f"Recherche vector store échouée pour '{question}': {e}")
        results = []

    contexts = [r["text"] for r in results]
    context_str = "\n\n---\n\n".join(
        f"Source: {r['metadata'].get('source', 'Inconnue')}\nContenu: {r['text']}"
        for r in results
    ) if results else "Aucune information trouvée dans la base de connaissances."

    system_prompt = (
        f"Tu es un analyste NBA expert. Réponds uniquement en te basant sur le contexte fourni.\n"
        f"Si le contexte ne contient pas l'information, dis-le clairement.\n\n"
        f"CONTEXTE:\n{context_str}\n\n"
        f"QUESTION: {question}\n\nRÉPONSE:"
    )
    try:
        response = mistral_client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content if response.choices else "Pas de réponse."
    except Exception as e:
        logger.error(f"Erreur LLM RAG pour '{question}': {e}")
        answer = f"Erreur LLM : {e}"

    return answer, contexts


def call_sql_pipeline(
    question: str,
    sql_tool: NBAQueryTool,
    mistral_client: Mistral,
) -> Tuple[str, List[str]]:
    """
    Pipeline SQL Tool → Mistral. Retourne (réponse, contextes).

    Le SQL Tool interroge SQLite et renvoie un résultat tabulaire brut.
    Ce résultat est ensuite reformulé par Mistral en réponse naturelle.
    Le résultat SQL brut sert de "contexte" pour RAGAS (faithfulness, precision).
    """
    # 1. Exécuter la requête SQL
    sql_result = sql_tool.run(question)

    # 2. Reformuler en français naturel via Mistral
    synthesis_prompt = (
        f"Tu es un analyste NBA expert. "
        f"Voici les données SQL extraites de la base NBA pour répondre à la question :\n\n"
        f"QUESTION : {question}\n\n"
        f"DONNÉES SQL :\n{sql_result}\n\n"
        f"Formule une réponse claire et concise en français, en citant les chiffres clés."
        f" Si les données sont vides ou si la requête a échoué, dis-le clairement."
    )
    try:
        response = mistral_client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content if response.choices else sql_result
    except Exception as e:
        logger.error(f"Erreur LLM synthèse SQL pour '{question}': {e}")
        answer = sql_result  # fallback : retourner le résultat brut

    # Le résultat SQL brut est le "contexte" que RAGAS va évaluer
    contexts = [sql_result]
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

    if LANGCHAIN_MISTRAL_AVAILABLE:
        try:
            # ChatOpenAI sur endpoint OpenAI-compatible Mistral
            # => corrige le TypeError += dict+dict de answer_relevancy avec ChatMistralAI
            langchain_llm = ChatOpenAI(
                model="mistral-small-latest",
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
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
            logger.error(f"Erreur évaluation RAGAS (OpenAI-compat Mistral): {e}")

    # Fallback embeddings locaux
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        emb_wrapper = LangchainEmbeddingsWrapper(emb)
        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy],
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
    sources: List[str],
    ragas_result: Optional[Dict],
) -> pd.DataFrame:
    """Construit un tableau de résultats par question."""
    rows = []
    for s, src in zip(samples, sources):
        row = {
            "Catégorie": s.category,
            "Source": src,
            "Question": s.question[:75] + "..." if len(s.question) > 75 else s.question,
            "Réponse (extrait)": s.answer[:100] + "..." if len(s.answer) > 100 else s.answer,
            "Nb contextes": len(s.contexts),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

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


def print_summary(df: pd.DataFrame, ragas_result: Optional[Dict], mode: str) -> None:
    """Affiche un résumé dans le terminal."""
    print("\n" + "=" * 75)
    print(f"RAPPORT D'ÉVALUATION RAG v2 — SportSee NBA Assistant [mode: {mode}]")
    print("=" * 75)
    print(df.to_string(index=False))

    if ragas_result is not None:
        print("\n--- Scores moyens RAGAS (toutes questions) ---")
        try:
            scores = ragas_result.to_pandas()
            metric_cols = [c for c in scores.columns if c in [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall"
            ]]
            for col in metric_cols:
                mean_val = scores[col].mean()
                print(f"  {col.replace('_', ' ').title():<30}: {mean_val:.4f}")

            # Scores par source
            print("\n--- Scores RAGAS par source de réponse ---")
            if "Source" in df.columns:
                for src in df["Source"].unique():
                    idx = df["Source"] == src
                    print(f"\n  [{src.upper()}]")
                    for col in metric_cols:
                        col_title = col.replace("_", " ").title()
                        if col_title in df.columns:
                            mean_src = df.loc[idx, col_title].mean()
                            print(f"    {col_title:<28}: {mean_src:.4f}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'affichage des scores: {e}")

    by_cat = df.groupby(["Catégorie", "Source"])["Nb contextes"].mean()
    print("\n--- Contextes récupérés par catégorie / source ---")
    print(by_cat.to_string())
    print("=" * 75)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation RAGAS v2 du système complet SportSee NBA (SQL + RAG)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_apres_v2.json",
        help="Chemin du fichier de résultats JSON",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/eval_apres_v2.csv",
        help="Chemin du fichier CSV des résultats",
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "sql", "force-sql"],
        default="sql",
        help=(
            "Stratégie de réponse :\n"
            "  rag       → 100%% FAISS (identique à v1)\n"
            "  sql       → routage automatique SQL/FAISS (défaut, comportement production)\n"
            "  force-sql → toutes les questions passent par le SQL Tool"
        ),
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Désactiver l'évaluation RAGAS (run pipeline only, sans métriques)",
    )
    args = parser.parse_args()

    # Vérification clé API
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY non définie dans .env. Arrêt.")
        sys.exit(1)

    # Créer dossiers de sortie
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    # Charger le vector store (nécessaire en mode rag et sql)
    vs_manager = None
    if args.mode in ("rag", "sql"):
        logger.info("Chargement du vector store...")
        try:
            vs_manager = VectorStoreManager()
            if vs_manager.index is None:
                logger.warning("Index FAISS vide. Exécutez 'py indexer.py'.")
        except Exception as e:
            logger.error(f"Impossible de charger le VectorStoreManager: {e}")
            if args.mode == "rag":
                sys.exit(1)

    # Charger le SQL Tool (nécessaire en mode sql et force-sql)
    sql_tool = None
    if args.mode in ("sql", "force-sql"):
        if SQL_TOOL_AVAILABLE:
            try:
                sql_tool = NBAQueryTool()
                logger.info("SQL Tool chargé.")
            except Exception as e:
                logger.warning(f"SQL Tool non initialisé : {e}")
        else:
            logger.warning("sql_tool.py non disponible — questions chiffrées routées vers FAISS.")

    # Initialiser le client Mistral
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

    # Évaluer chaque question
    logger.info(f"Évaluation de {len(TEST_CASES)} questions [mode: {args.mode}]...")
    evaluated_samples: List[EvalSample] = []
    sources: List[str] = []  # trace la source utilisée pour chaque question

    for i, sample in enumerate(TEST_CASES, 1):
        logger.info(
            f"[{i}/{len(TEST_CASES)}] '{sample.question[:55]}...' (cat: {sample.category})"
        )

        use_sql = (
            args.mode == "force-sql"
            or (args.mode == "sql" and _is_numerical_question(sample.question))
        )

        if use_sql and sql_tool is not None:
            logger.info(f"  → Routage : SQL Tool (SQLite)")
            answer, contexts = call_sql_pipeline(
                question=sample.question,
                sql_tool=sql_tool,
                mistral_client=mistral_client,
            )
            source = "sql"
        else:
            if use_sql and sql_tool is None:
                logger.warning("  → SQL Tool indisponible, fallback FAISS")
            else:
                logger.info(f"  → Routage : FAISS (vector store)")
            answer, contexts = call_rag_pipeline(
                question=sample.question,
                vector_store=vs_manager,
                mistral_client=mistral_client,
            )
            source = "rag"

        enriched = sample.model_copy(update={"answer": answer, "contexts": contexts})
        evaluated_samples.append(enriched)
        sources.append(source)

    # Résumé du routage
    n_sql = sources.count("sql")
    n_rag = sources.count("rag")
    logger.info(f"Routage : {n_sql} question(s) SQL, {n_rag} question(s) FAISS")

    # Évaluation RAGAS
    ragas_result = None
    if not args.no_ragas and RAGAS_AVAILABLE and DATASETS_AVAILABLE:
        logger.info("Construction du dataset RAGAS...")
        ragas_dataset = build_ragas_dataset(evaluated_samples)
        if ragas_dataset:
            logger.info("Lancement de l'évaluation RAGAS...")
            ragas_result = run_ragas_evaluation(ragas_dataset, MISTRAL_API_KEY)
    elif args.no_ragas:
        logger.info("Évaluation RAGAS désactivée (--no-ragas).")
    else:
        logger.warning("RAGAS ou datasets non disponibles.")

    # Rapport
    df = build_report(evaluated_samples, sources, ragas_result)
    print_summary(df, ragas_result, args.mode)

    # Export CSV
    df.to_csv(args.csv, index=False, encoding="utf-8-sig")
    logger.info(f"Résultats CSV exportés → {args.csv}")

    # Export JSON
    export_data = {
        "meta": {
            "version": "v2",
            "mode": args.mode,
            "total_questions": len(evaluated_samples),
            "routing": {"sql": n_sql, "rag": n_rag},
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
                "source": src,
                "contexts_count": len(s.contexts),
                "ground_truth": s.ground_truth,
                "category": s.category,
            }
            for s, src in zip(evaluated_samples, sources)
        ],
        "ragas_scores": None,
    }

    if ragas_result is not None:
        try:
            scores_df = ragas_result.to_pandas()
            metric_cols = [c for c in scores_df.columns if c in [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall"
            ]]
            # Scores globaux
            export_data["ragas_scores"] = {
                col: {
                    "mean": float(scores_df[col].mean()),
                    "min": float(scores_df[col].min()),
                    "max": float(scores_df[col].max()),
                }
                for col in metric_cols
            }
            # Scores par source
            export_data["ragas_scores_by_source"] = {}
            for src in ["sql", "rag"]:
                idx = [i for i, s in enumerate(sources) if s == src]
                if idx:
                    export_data["ragas_scores_by_source"][src] = {
                        col: {
                            "mean": float(scores_df.iloc[idx][col].mean()),
                            "count": len(idx),
                        }
                        for col in metric_cols
                    }
        except Exception as e:
            logger.warning(f"Erreur export scores JSON: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Résultats JSON exportés → {args.output}")
    logger.info("Évaluation v2 terminée.")


if __name__ == "__main__":
    main()
