# load_excel_to_db.py
"""
Pipeline d'ingestion des données Excel NBA vers SQLite.

Usage:
    py load_excel_to_db.py
    py load_excel_to_db.py --excel DATA/regular+NBA+(2).xlsx --db P10_DSML/database/nba.db

Ce script :
  1. Lit le fichier Excel (feuilles 'Données NBA' et 'Equipe')
  2. Valide chaque ligne via les modèles Pydantic (PlayerStats, TeamInfo)
  3. Crée le schéma relationnel SQLite (tables players, teams)
  4. Insère les données validées

Schéma :
    teams   : code (PK), full_name
    players : id (PK), player, team (FK→teams.code), age, gp, w, l,
              min, pts, fgm, fga, fg_pct, three_pm, three_pa, three_pct,
              ftm, fta, ft_pct, oreb, dreb, reb, ast, tov, stl, blk, pf,
              plus_minus, efg_pct, ts_pct, usg_pct, pie
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError
from sqlalchemy import (
    Column, Float, ForeignKey, Integer, String, Text,
    create_engine, inspect as sa_inspect,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

# Ajouter P10_DSML au path pour les imports locaux
_root = Path(__file__).parent
sys.path.insert(0, str(_root / "P10_DSML"))
sys.path.insert(0, str(_root))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import des modèles Pydantic
# ---------------------------------------------------------------------------
try:
    from P10_DSML.utils.pipeline import PlayerStats, TeamInfo
except ImportError:
    try:
        from utils.pipeline import PlayerStats, TeamInfo
    except ImportError:
        logger.error(
            "Impossible d'importer utils.pipeline. "
            "Lancez ce script depuis la racine du projet PROJET10."
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Modèles SQLAlchemy (ORM)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class TeamModel(Base):
    __tablename__ = "teams"

    code = Column(String(4), primary_key=True)
    full_name = Column(String(100), nullable=False)
    players = relationship("PlayerModel", back_populates="team_ref")


class PlayerModel(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player = Column(Text, nullable=False, index=True)
    team = Column(String(4), ForeignKey("teams.code"), nullable=False, index=True)
    age = Column(Integer)
    gp = Column(Integer)
    w = Column(Integer)
    l = Column(Integer)
    min = Column(Float)
    pts = Column(Float)
    fgm = Column(Float)
    fga = Column(Float)
    fg_pct = Column(Float)
    three_pm = Column(Float)
    three_pa = Column(Float)
    three_pct = Column(Float)
    ftm = Column(Float)
    fta = Column(Float)
    ft_pct = Column(Float)
    oreb = Column(Float)
    dreb = Column(Float)
    reb = Column(Float)
    ast = Column(Float)
    tov = Column(Float)
    stl = Column(Float)
    blk = Column(Float)
    pf = Column(Float)
    plus_minus = Column(Float)
    efg_pct = Column(Float)
    ts_pct = Column(Float)
    usg_pct = Column(Float)
    pie = Column(Float)

    team_ref = relationship("TeamModel", back_populates="players")


# ---------------------------------------------------------------------------
# Lecture et validation des données
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    """Convertit une valeur en float, retourne default si impossible."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def load_teams(excel_path: Path) -> Tuple[List[TeamInfo], int]:
    """Charge et valide les équipes depuis la feuille 'Equipe'."""
    df = pd.read_excel(excel_path, sheet_name="Equipe")
    valid_teams: List[TeamInfo] = []
    errors = 0

    for _, row in df.iterrows():
        try:
            team = TeamInfo(
                code=str(row.iloc[0]).strip(),
                full_name=str(row.iloc[1]).strip(),
            )
            valid_teams.append(team)
        except (ValidationError, Exception) as e:
            logger.warning(f"Ligne équipe invalide : {dict(row)} → {e}")
            errors += 1

    logger.info(f"Équipes : {len(valid_teams)} valides, {errors} erreurs.")
    return valid_teams, errors


def load_players(excel_path: Path) -> Tuple[List[PlayerStats], int]:
    """Charge et valide les joueurs depuis la feuille 'Données NBA'."""
    # La première ligne de données contient les en-têtes réels
    raw = pd.read_excel(excel_path, sheet_name="Données NBA", header=0)
    raw.columns = raw.iloc[0]         # La vraie ligne d'en-tête est à l'index 0
    raw = raw.iloc[1:].reset_index(drop=True)
    raw.columns.name = None

    # Renommer la colonne "3PM" (peut apparaître comme datetime.time 15:00)
    cols = list(raw.columns)
    renamed = {}
    for c in cols:
        cs = str(c)
        if "15:00" in cs or cs == "15:00:00":
            renamed[c] = "3PM"
    if renamed:
        raw = raw.rename(columns=renamed)

    # Supprimer les colonnes NaN (colonnes vides à la fin)
    raw = raw.loc[:, raw.columns.notna()]
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("nan")]

    valid_players: List[PlayerStats] = []
    errors = 0

    for idx, row in raw.iterrows():
        player_name = str(row.get("Player", "")).strip()
        if not player_name or player_name.lower() == "nan":
            continue

        try:
            p = PlayerStats(
                player=player_name,
                team=str(row.get("Team", "UNK")).strip(),
                age=_safe_int(row.get("Age"), 25),
                gp=_safe_int(row.get("GP"), 0),
                w=_safe_int(row.get("W"), 0),
                l=_safe_int(row.get("L"), 0),
                min=_safe_float(row.get("Min"), 0.0),
                pts=_safe_float(row.get("PTS"), 0.0),
                fgm=_safe_float(row.get("FGM"), 0.0),
                fga=_safe_float(row.get("FGA"), 0.0),
                fg_pct=_safe_float(row.get("FG%"), 0.0),
                three_pm=_safe_float(row.get("3PM", row.get("3PM", 0.0)), 0.0),
                three_pa=_safe_float(row.get("3PA"), 0.0),
                three_pct=_safe_float(row.get("3P%"), 0.0),
                ftm=_safe_float(row.get("FTM"), 0.0),
                fta=_safe_float(row.get("FTA"), 0.0),
                ft_pct=_safe_float(row.get("FT%"), 0.0),
                oreb=_safe_float(row.get("OREB"), 0.0),
                dreb=_safe_float(row.get("DREB"), 0.0),
                reb=_safe_float(row.get("REB"), 0.0),
                ast=_safe_float(row.get("AST"), 0.0),
                tov=_safe_float(row.get("TOV"), 0.0),
                stl=_safe_float(row.get("STL"), 0.0),
                blk=_safe_float(row.get("BLK"), 0.0),
                pf=_safe_float(row.get("PF"), 0.0),
                plus_minus=_safe_float(row.get("+/-"), 0.0),
                efg_pct=_safe_float(row.get("EFG%"), 0.0) or None,
                ts_pct=_safe_float(row.get("TS%"), 0.0) or None,
                usg_pct=_safe_float(row.get("USG%"), 0.0) or None,
                pie=_safe_float(row.get("PIE"), 0.0) or None,
            )
            valid_players.append(p)
        except ValidationError as e:
            logger.warning(f"Joueur invalide '{player_name}': {e}")
            errors += 1
        except Exception as e:
            logger.warning(f"Erreur inattendue pour '{player_name}' (ligne {idx}): {e}")
            errors += 1

    logger.info(f"Joueurs : {len(valid_players)} valides, {errors} erreurs.")
    return valid_players, errors


# ---------------------------------------------------------------------------
# Insertion en base
# ---------------------------------------------------------------------------

def create_schema(engine) -> None:
    """Crée les tables si elles n'existent pas encore."""
    Base.metadata.create_all(engine)
    logger.info("Schéma SQLite créé/vérifié.")


def insert_teams(session: Session, teams: List[TeamInfo]) -> None:
    inspector = sa_inspect(session.bind)
    existing = {
        row[0] for row in session.execute(
            session.get_bind().connect().execute(
                __import__("sqlalchemy").text("SELECT code FROM teams")
            )
        )
    } if inspector.has_table("teams") else set()

    new_teams = [t for t in teams if t.code not in existing]
    for t in new_teams:
        session.add(TeamModel(code=t.code, full_name=t.full_name))
    session.commit()
    logger.info(f"{len(new_teams)} équipes insérées.")


def insert_players(session: Session, players: List[PlayerStats]) -> None:
    """Insère les joueurs en ignorant les doublons (player+team)."""
    from sqlalchemy import text, select

    existing_query = session.execute(
        text("SELECT player, team FROM players")
    ).fetchall()
    existing = {(r[0], r[1]) for r in existing_query}

    inserted = 0
    skipped = 0
    for p in players:
        if (p.player, p.team) in existing:
            skipped += 1
            continue
        session.add(PlayerModel(
            player=p.player,
            team=p.team,
            age=p.age,
            gp=p.gp,
            w=p.w,
            l=p.l,
            min=p.min,
            pts=p.pts,
            fgm=p.fgm,
            fga=p.fga,
            fg_pct=p.fg_pct,
            three_pm=p.three_pm,
            three_pa=p.three_pa,
            three_pct=p.three_pct,
            ftm=p.ftm,
            fta=p.fta,
            ft_pct=p.ft_pct,
            oreb=p.oreb,
            dreb=p.dreb,
            reb=p.reb,
            ast=p.ast,
            tov=p.tov,
            stl=p.stl,
            blk=p.blk,
            pf=p.pf,
            plus_minus=p.plus_minus,
            efg_pct=p.efg_pct,
            ts_pct=p.ts_pct,
            usg_pct=p.usg_pct,
            pie=p.pie,
        ))
        existing.add((p.player, p.team))
        inserted += 1

    session.commit()
    logger.info(f"{inserted} joueurs insérés, {skipped} doublons ignorés.")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion des données Excel NBA vers SQLite"
    )
    parser.add_argument(
        "--excel",
        type=str,
        default="DATA/regular+NBA+(2).xlsx",
        help="Chemin vers le fichier Excel NBA",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="P10_DSML/database/nba.db",
        help="Chemin vers la base de données SQLite",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    db_path = Path(args.db)

    if not excel_path.exists():
        logger.error(f"Fichier Excel introuvable : {excel_path}")
        sys.exit(1)

    # Créer le dossier de la base si besoin
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(db_url, echo=False)

    # 1. Schéma
    create_schema(engine)

    # 2. Charger et valider les données
    logger.info(f"Lecture de {excel_path}...")
    teams, team_errors = load_teams(excel_path)
    players, player_errors = load_players(excel_path)

    if not teams and not players:
        logger.error("Aucune donnée valide trouvée. Arrêt.")
        sys.exit(1)

    # 3. Insérer
    with Session(engine) as session:
        # Équipes en premier (contrainte FK)
        for t in teams:
            existing = session.get(TeamModel, t.code)
            if not existing:
                session.add(TeamModel(code=t.code, full_name=t.full_name))
        session.commit()
        logger.info(f"{len(teams)} équipes traitées.")

        # Vérifier les équipes connues pour les joueurs
        known_teams = {t.code for t in teams}
        filtered_players = []
        for p in players:
            if p.team in known_teams:
                filtered_players.append(p)
            else:
                logger.warning(f"Équipe inconnue '{p.team}' pour joueur '{p.player}' — ignoré.")

        # Insérer les joueurs
        from sqlalchemy import text
        existing_pairs = {
            (r[0], r[1])
            for r in session.execute(text("SELECT player, team FROM players")).fetchall()
        }
        inserted = 0
        for p in filtered_players:
            if (p.player, p.team) in existing_pairs:
                continue
            session.add(PlayerModel(
                player=p.player, team=p.team, age=p.age, gp=p.gp, w=p.w, l=p.l,
                min=p.min, pts=p.pts, fgm=p.fgm, fga=p.fga, fg_pct=p.fg_pct,
                three_pm=p.three_pm, three_pa=p.three_pa, three_pct=p.three_pct,
                ftm=p.ftm, fta=p.fta, ft_pct=p.ft_pct,
                oreb=p.oreb, dreb=p.dreb, reb=p.reb,
                ast=p.ast, tov=p.tov, stl=p.stl, blk=p.blk, pf=p.pf,
                plus_minus=p.plus_minus, efg_pct=p.efg_pct,
                ts_pct=p.ts_pct, usg_pct=p.usg_pct, pie=p.pie,
            ))
            existing_pairs.add((p.player, p.team))
            inserted += 1
        session.commit()
        logger.info(f"{inserted} joueurs insérés dans '{db_path}'.")

    # Résumé final
    print("\n" + "=" * 50)
    print("INGESTION TERMINÉE")
    print(f"  Base de données : {db_path}")
    print(f"  Équipes  : {len(teams)} ({team_errors} erreurs)")
    print(f"  Joueurs  : {inserted} insérés ({player_errors} erreurs de validation)")
    print("=" * 50)


if __name__ == "__main__":
    main()
