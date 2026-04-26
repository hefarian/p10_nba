# utils/pipeline.py
"""
Modèles Pydantic pour la validation des flux d'entrée/sortie du pipeline RAG.
Sécurise les données à chaque étape : chunking, embedding, requête, réponse.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Modèles de données métier (NBA)
# ---------------------------------------------------------------------------

class PlayerStats(BaseModel):
    """Statistiques d'un joueur NBA issues du fichier Excel."""

    player: str = Field(..., description="Nom du joueur")
    team: str = Field(..., min_length=2, max_length=4, description="Code équipe (3 lettres)")
    age: int = Field(..., ge=18, le=50, description="Âge du joueur")
    gp: int = Field(..., ge=0, description="Matchs joués (Games Played)")
    w: int = Field(..., ge=0, description="Victoires")
    l: int = Field(..., ge=0, description="Défaites")
    min: float = Field(..., ge=0, description="Minutes par match")
    pts: float = Field(..., ge=0, description="Points totaux")
    fgm: float = Field(..., ge=0, description="Field Goals Made")
    fga: float = Field(..., ge=0, description="Field Goals Attempted")
    fg_pct: float = Field(..., ge=0, le=100, description="Pourcentage Field Goal")
    three_pm: float = Field(..., ge=0, description="3-points Made")
    three_pa: float = Field(..., ge=0, description="3-points Attempted")
    three_pct: float = Field(..., ge=0, le=100, description="Pourcentage 3-points")
    ftm: float = Field(..., ge=0, description="Free Throws Made")
    fta: float = Field(..., ge=0, description="Free Throws Attempted")
    ft_pct: float = Field(..., ge=0, le=100, description="Pourcentage lancers francs")
    oreb: float = Field(..., ge=0, description="Rebonds offensifs")
    dreb: float = Field(..., ge=0, description="Rebonds défensifs")
    reb: float = Field(..., ge=0, description="Rebonds totaux")
    ast: float = Field(..., ge=0, description="Passes décisives")
    tov: float = Field(..., ge=0, description="Pertes de balle")
    stl: float = Field(..., ge=0, description="Interceptions")
    blk: float = Field(..., ge=0, description="Contres")
    pf: float = Field(..., ge=0, description="Fautes personnelles")
    plus_minus: float = Field(..., description="+/- différentiel")
    efg_pct: Optional[float] = Field(None, ge=0, description="EFG%")
    ts_pct: Optional[float] = Field(None, ge=0, description="TS%")
    usg_pct: Optional[float] = Field(None, ge=0, description="USG%")
    pie: Optional[float] = Field(None, description="Player Impact Estimate")

    @field_validator("team")
    @classmethod
    def team_uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("player")
    @classmethod
    def player_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le nom du joueur ne peut pas être vide.")
        return v

    @model_validator(mode="after")
    def wins_losses_consistent(self) -> "PlayerStats":
        if self.w + self.l > self.gp:
            raise ValueError(
                f"Victoires ({self.w}) + Défaites ({self.l}) > Matchs joués ({self.gp})"
            )
        return self


class TeamInfo(BaseModel):
    """Informations d'une équipe NBA."""

    code: str = Field(..., min_length=2, max_length=4)
    full_name: str = Field(..., min_length=3)

    @field_validator("code")
    @classmethod
    def code_uppercase(cls, v: str) -> str:
        return v.strip().upper()


# ---------------------------------------------------------------------------
# Modèles RAG pipeline
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """Chunk de document indexé dans le vector store."""

    chunk_id: str = Field(..., description="Identifiant unique du chunk")
    text: str = Field(..., min_length=1, description="Contenu textuel du chunk")
    source: str = Field(default="unknown", description="Fichier source")
    category: Optional[str] = Field(None, description="Catégorie du document")
    start_index: int = Field(default=-1, description="Position de début dans le document d'origine")

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le chunk ne peut pas être vide ou uniquement des espaces.")
        return v


class RAGQuery(BaseModel):
    """Requête entrante dans le pipeline RAG."""

    question: str = Field(..., min_length=3, description="Question de l'utilisateur")
    k: int = Field(default=5, ge=1, le=20, description="Nombre de chunks à récupérer")
    min_score: float = Field(default=0.0, ge=0.0, description="Score minimum de similarité")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La question ne peut pas être vide.")
        return v


class RetrievedChunk(BaseModel):
    """Chunk récupéré lors d'une recherche dans le vector store."""

    chunk_id: str
    text: str
    source: str
    score: float = Field(..., ge=0.0, description="Score de similarité (0-100)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Réponse produite par le pipeline RAG + LLM."""

    question: str
    answer: str = Field(..., min_length=1)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    model_used: str = Field(default="mistral-small-latest")
    had_context: bool = Field(default=True)
    source_type: str = Field(default="vector_store", description="'vector_store' ou 'sql_tool'")

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La réponse du LLM ne peut pas être vide.")
        return v


# ---------------------------------------------------------------------------
# Modèle pour l'évaluation RAGAS
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    """Un échantillon pour l'évaluation RAGAS."""

    question: str
    answer: str
    contexts: List[str] = Field(default_factory=list)
    ground_truth: Optional[str] = Field(None, description="Réponse de référence (si disponible)")
    category: str = Field(
        default="simple",
        description="Catégorie du test : simple | complex | noisy"
    )

    @field_validator("category")
    @classmethod
    def valid_category(cls, v: str) -> str:
        allowed = {"simple", "complex", "noisy"}
        if v not in allowed:
            raise ValueError(f"Catégorie invalide '{v}'. Valeurs autorisées : {allowed}")
        return v
