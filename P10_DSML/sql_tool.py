# sql_tool.py
"""
LangChain SQL Tool pour interroger la base de données NBA (SQLite).

Ce module expose :
  - NBAQueryTool : outil LangChain utilisé par l'agent pour répondre
    aux questions chiffrées (stats, classements, comparaisons).

Fonctionnement :
  1. Le LLM reçoit la question de l'utilisateur.
  2. Il appelle NBAQueryTool.run(question).
  3. L'outil génère dynamiquement une requête SQL via le LLM (few-shot prompting).
  4. La requête est exécutée sur SQLite et le résultat est retourné.
  5. Le LLM synthétise la réponse finale.

Exemples de questions gérées :
  - "Quel joueur a le meilleur % à 3 points ?"
  - "Compare les rebonds de OKC et DEN."
  - "Top 5 des marqueurs de la saison."
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.getenv(
    "NBA_DB_PATH",
    str(Path(__file__).parent / "database" / "nba.db"),
)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

# ---------------------------------------------------------------------------
# Few-shot examples SQL
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
-- Exemple 1 : Meilleur marqueur
-- Question : "Quel joueur a le plus de points cette saison ?"
SELECT player, team, pts
FROM players
ORDER BY pts DESC
LIMIT 1;

-- Exemple 2 : Meilleur % à 3 points (minimum 3 tentatives par match)
-- Question : "Quel joueur a le meilleur % à 3 points parmi ceux qui tirent au moins 3 fois ?"
SELECT player, team, three_pct, three_pa, gp,
       ROUND(three_pa * 1.0 / gp, 2) AS three_pa_per_game
FROM players
WHERE gp > 0 AND three_pa / gp >= 3.0
ORDER BY three_pct DESC
LIMIT 10;

-- Exemple 3 : Comparaison rebonds entre deux équipes
-- Question : "Compare les rebonds de OKC et DEN."
SELECT team,
       ROUND(AVG(oreb), 2) AS avg_oreb,
       ROUND(AVG(dreb), 2) AS avg_dreb,
       ROUND(AVG(reb), 2)  AS avg_reb
FROM players
WHERE team IN ('OKC', 'DEN')
GROUP BY team;

-- Exemple 4 : Bilan victoires/défaites par équipe
-- Question : "Quelle équipe a le meilleur bilan victoires/défaites ?"
SELECT p.team, t.full_name,
       MAX(p.w) AS wins,
       MAX(p.l) AS losses,
       MAX(p.gp) AS games_played,
       ROUND(MAX(p.w) * 100.0 / MAX(p.gp), 1) AS win_pct
FROM players p
LEFT JOIN teams t ON p.team = t.code
GROUP BY p.team, t.full_name
ORDER BY win_pct DESC
LIMIT 5;

-- Exemple 5 : Top 5 des passeurs décisifs
-- Question : "Qui sont les meilleurs passeurs de la saison ?"
SELECT player, team, ast, gp,
       ROUND(ast * 1.0 / gp, 2) AS ast_per_game
FROM players
WHERE gp > 0
ORDER BY ast DESC
LIMIT 5;

-- Exemple 6 : Efficacité offensive (TS%)
-- Question : "Quels joueurs ont le meilleur True Shooting % ?"
SELECT player, team, ts_pct, pts
FROM players
WHERE ts_pct IS NOT NULL AND gp >= 20
ORDER BY ts_pct DESC
LIMIT 10;

-- Exemple 7 : Comparaison domicile/extérieur (via bilan)
-- Question : "Quels joueurs d'équipes domicile fortes ont le meilleur +/- ?"
SELECT player, team, plus_minus, w, l
FROM players
WHERE w > l
ORDER BY plus_minus DESC
LIMIT 10;
"""

# ---------------------------------------------------------------------------
# Prompt de génération SQL
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = """Tu es un expert SQL spécialisé dans les données NBA.
Tu dois générer une requête SQLite valide pour répondre à la question.

SCHÉMA DE LA BASE :
  Table players :
    id, player (TEXT), team (TEXT), age, gp (matchs joués), w (victoires), l (défaites),
    min (minutes/match), pts (points totaux), fgm, fga, fg_pct (% tir),
    three_pm (3pts réussis), three_pa (3pts tentés), three_pct (% 3pts),
    ftm (lancers francs réussis), fta (tentés), ft_pct (% LF),
    oreb (rebonds off.), dreb (rebonds déf.), reb (rebonds totaux),
    ast (passes déc.), tov (pertes balle), stl (interceptions), blk (contres),
    pf (fautes), plus_minus (+/-),
    efg_pct (EFG%), ts_pct (TS%), usg_pct (USG%), pie

  Table teams :
    code (TEXT PK), full_name (TEXT)

EXEMPLES DE REQUÊTES :
{few_shot_examples}

RÈGLES :
- Génère UNIQUEMENT la requête SQL, sans commentaires ni markdown.
- Utilise ROUND() pour les décimaux (2 chiffres).
- Limite à LIMIT 20 sauf si demandé différemment.
- N'utilise que les colonnes existantes dans le schéma.
- Pour les % par match : divise par gp (vérifie gp > 0).
- Pour les comparaisons d'équipes : GROUP BY team.

QUESTION : {question}

REQUÊTE SQL :"""

# ---------------------------------------------------------------------------
# SQL Tool principal
# ---------------------------------------------------------------------------

class NBAQueryTool:
    """
    Outil LangChain-compatible pour répondre aux questions NBA via SQL.

    Interface compatible avec LangChain Tool.run() :
        result = tool.run("Quel joueur a le plus de points ?")
    """

    name: str = "nba_sql_query"
    description: str = (
        "Interroge la base de données NBA SQLite pour répondre à des questions "
        "chiffrées sur les statistiques, classements et comparaisons de joueurs/équipes."
    )

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._engine = None
        self._mistral_client = None
        self._init_db()
        self._init_llm()

    def _init_db(self) -> None:
        """Initialise la connexion SQLite."""
        db_file = Path(self.db_path)
        if not db_file.exists():
            logger.warning(
                f"Base de données '{db_file}' introuvable. "
                "Exécutez d'abord 'py load_excel_to_db.py'."
            )
            return
        try:
            from sqlalchemy import create_engine
            self._engine = create_engine(f"sqlite:///{db_file.as_posix()}", echo=False)
            logger.info(f"SQL Tool connecté à {db_file}.")
        except Exception as e:
            logger.error(f"Erreur connexion SQLite: {e}")

    def _init_llm(self) -> None:
        """Initialise le client Mistral pour la génération SQL."""
        if not MISTRAL_API_KEY:
            logger.warning("MISTRAL_API_KEY manquante — génération SQL indisponible.")
            return
        try:
            from mistralai import Mistral
            self._mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            logger.info("Client Mistral initialisé pour le SQL Tool.")
        except ImportError:
            logger.error("mistralai non installé.")

    def _generate_sql(self, question: str) -> Optional[str]:
        """Génère une requête SQL à partir de la question via le LLM."""
        if self._mistral_client is None:
            return self._fallback_sql(question)

        prompt = SQL_GENERATION_PROMPT.format(
            few_shot_examples=FEW_SHOT_EXAMPLES,
            question=question,
        )
        try:
            response = self._mistral_client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            sql = response.choices[0].message.content.strip()
            # Nettoyer les balises markdown éventuelles
            sql = re.sub(r"```sql\s*", "", sql)
            sql = re.sub(r"```\s*", "", sql)
            sql = sql.strip().rstrip(";") + ";"
            logger.info(f"SQL généré : {sql[:100]}...")
            return sql
        except Exception as e:
            logger.error(f"Erreur génération SQL: {e}")
            return self._fallback_sql(question)

    def _fallback_sql(self, question: str) -> str:
        """Requête fallback générique quand le LLM est indisponible."""
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["point", "marqueur", "scorer"]):
            return "SELECT player, team, pts FROM players ORDER BY pts DESC LIMIT 10;"
        if any(kw in q_lower for kw in ["rebond", "rebound"]):
            return "SELECT player, team, reb, oreb, dreb FROM players ORDER BY reb DESC LIMIT 10;"
        if any(kw in q_lower for kw in ["passe", "assist"]):
            return "SELECT player, team, ast FROM players ORDER BY ast DESC LIMIT 10;"
        if any(kw in q_lower for kw in ["3 point", "trois point", "3pts", "3p%"]):
            return (
                "SELECT player, team, three_pct, three_pa FROM players "
                "WHERE three_pa > 0 ORDER BY three_pct DESC LIMIT 10;"
            )
        if any(kw in q_lower for kw in ["équipe", "team", "bilan", "victoire"]):
            return (
                "SELECT team, MAX(w) AS wins, MAX(l) AS losses "
                "FROM players GROUP BY team ORDER BY wins DESC LIMIT 10;"
            )
        return "SELECT player, team, pts, reb, ast FROM players ORDER BY pts DESC LIMIT 15;"

    def _execute_sql(self, sql: str) -> str:
        """Exécute la requête SQL et retourne le résultat sous forme de texte."""
        if self._engine is None:
            return "Base de données non disponible. Exécutez 'py load_excel_to_db.py'."

        import pandas as pd
        from sqlalchemy import text

        try:
            with self._engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)

            if df.empty:
                return "Aucun résultat trouvé pour cette requête."

            # Formater le résultat
            result_lines = [f"Requête SQL : {sql}", "", "Résultats :"]
            result_lines.append(df.to_string(index=False, max_rows=20))
            result_lines.append(f"\n({len(df)} ligne(s) retournée(s))")
            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Erreur exécution SQL '{sql}': {e}")
            return f"Erreur lors de l'exécution SQL : {e}"

    def run(self, question: str) -> str:
        """
        Interface principale : question → SQL → résultat textuel.
        Compatible avec LangChain Tool.run().
        """
        if self._engine is None:
            return (
                "Base de données NBA non disponible. "
                "Exécutez 'py load_excel_to_db.py' pour l'initialiser."
            )

        logger.info(f"SQL Tool appelé pour : '{question}'")
        sql = self._generate_sql(question)
        if not sql:
            return "Impossible de générer une requête SQL pour cette question."

        return self._execute_sql(sql)

    # Compatibilité LangChain BaseTool
    def __call__(self, question: str) -> str:
        return self.run(question)


# ---------------------------------------------------------------------------
# Point d'entrée (test rapide)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tool = NBAQueryTool()

    test_questions = [
        "Quel joueur a le plus de points cette saison ?",
        "Top 5 des meilleurs rebondeurs.",
        "Quel joueur a le meilleur % à 3 points parmi ceux qui tirent au moins 3 fois par match ?",
        "Compare les statistiques de rebonds entre OKC et DEN.",
        "Quelle équipe a le meilleur bilan victoires/défaites ?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question : {q}")
        print("-" * 60)
        result = tool.run(q)
        print(result)
