# plot_tool.py
"""
PlotTool — Outil LangChain de génération dynamique de graphiques matplotlib.

Prend en entrée des données structurées (JSON) et retourne une image PNG
encodée en base64, affichable directement dans Streamlit via st.image().

Types de graphiques supportés :
  - bar              → comparaisons verticales
  - horizontal_bar   → comparaisons horizontales (nombreux labels)
  - line             → évolutions temporelles
  - pie              → répartitions / proportions
  - scatter          → corrélations

Format d'entrée JSON :
  {
      "chart_type": "bar",
      "data": [{"label": "LeBron James", "value": 27.2}, ...],
      "title": "Points par match — Top 5",
      "x_label": "Joueur",          (optionnel)
      "y_label": "Points / match"   (optionnel)
  }

Usage standalone :
    py plot_tool.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Dict, List, Union

import matplotlib
matplotlib.use("Agg")  # backend non-interactif (pas de fenêtre GUI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

# ── Palette visuelle (couleurs NBA) ───────────────────────────────────────────

_NBA_COLORS = [
    "#1D428A",  # bleu Celtics / NBA
    "#C8102E",  # rouge
    "#FFC72C",  # or Lakers
    "#007A33",  # vert Celtics
    "#552583",  # violet Lakers
    "#CE1141",  # rouge Bulls
    "#00471B",  # vert Bucks
    "#F58426",  # orange Knicks
    "#007AC1",  # bleu OKC
    "#860038",  # bordeaux Cavaliers
]

_FIG_SIZE = (11, 6)
_DPI = 120


# ── NBAPlotTool ───────────────────────────────────────────────────────────────

class NBAPlotTool:
    """
    Outil LangChain-compatible pour générer des graphiques NBA.

    Compatibilité LangChain : expose les attributs ``name`` et ``description``
    et la méthode ``run(input_data) -> str``.
    """

    name: str = "nba_plot_tool"
    description: str = (
        "Génère un graphique matplotlib à partir de données NBA structurées. "
        "Entrée : JSON avec chart_type, data (list of {label, value}), title, "
        "x_label, y_label. Retourne une image PNG encodée en base64."
    )

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Génère le graphique et retourne la chaîne base64.

        Args:
            input_data: dict ou JSON string avec :
                - chart_type (str)  : "bar" | "horizontal_bar" | "line" | "pie" | "scatter"
                - data (list)       : [{"label": str, "value": float}, ...]
                - title (str)       : titre affiché
                - x_label (str)     : label axe X (optionnel)
                - y_label (str)     : label axe Y (optionnel)

        Returns:
            Chaîne base64 PNG.
        """
        params = self._parse_input(input_data)
        chart_type = params.get("chart_type", "bar").lower().replace(" ", "_")
        data: List[Dict] = params.get("data", [])
        title: str = params.get("title", "")
        x_label: str = params.get("x_label", "")
        y_label: str = params.get("y_label", "")

        if not data:
            raise ValueError("Le champ 'data' est vide ou absent.")

        labels = [str(item.get("label", "?")) for item in data]
        values = [float(item.get("value", 0)) for item in data]

        dispatch = {
            "bar":            self._bar_chart,
            "horizontal_bar": self._horizontal_bar_chart,
            "line":           self._line_chart,
            "pie":            self._pie_chart,
            "scatter":        self._scatter_chart,
        }

        if chart_type not in dispatch:
            logger.warning("Type '%s' inconnu — fallback 'bar'.", chart_type)
            chart_type = "bar"

        fig = dispatch[chart_type](labels, values, title, x_label, y_label)
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # Générateurs de graphiques
    # ------------------------------------------------------------------

    def _bar_chart(
        self, labels: List[str], values: List[float],
        title: str, x_label: str, y_label: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        colors = self._pick_colors(len(labels))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9, color="#333333")
        self._style_axes(ax, title, x_label, y_label, rotate_x=True)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        fig.tight_layout()
        return fig

    def _horizontal_bar_chart(
        self, labels: List[str], values: List[float],
        title: str, x_label: str, y_label: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        colors = self._pick_colors(len(labels))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.invert_yaxis()
        self._style_axes(ax, title, y_label or x_label, "", rotate_x=False)
        fig.tight_layout()
        return fig

    def _line_chart(
        self, labels: List[str], values: List[float],
        title: str, x_label: str, y_label: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.plot(
            labels, values,
            marker="o", color=_NBA_COLORS[0], linewidth=2.5,
            markersize=8, markerfacecolor=_NBA_COLORS[1],
        )
        for lbl, val in zip(labels, values):
            ax.annotate(
                f"{val:.1f}", (lbl, val),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color="#333333",
            )
        self._style_axes(ax, title, x_label, y_label, rotate_x=True)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    def _pie_chart(
        self, labels: List[str], values: List[float],
        title: str, x_label: str, y_label: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = self._pick_colors(len(labels))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.82,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        fig.tight_layout()
        return fig

    def _scatter_chart(
        self, labels: List[str], values: List[float],
        title: str, x_label: str, y_label: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        x_vals = list(range(len(values)))
        ax.scatter(x_vals, values, c=_NBA_COLORS[0], s=120, zorder=3, edgecolors="white")
        for i, (lbl, val) in enumerate(zip(labels, values)):
            ax.annotate(
                lbl, (i, val),
                textcoords="offset points", xytext=(5, 5),
                fontsize=9, color="#333333",
            )
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        self._style_axes(ax, title, x_label, y_label, rotate_x=False)
        ax.grid(linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_colors(n: int) -> List[str]:
        """Retourne n couleurs en cyclant sur la palette NBA."""
        return (_NBA_COLORS * ((n // len(_NBA_COLORS)) + 1))[:n]

    @staticmethod
    def _style_axes(
        ax: plt.Axes, title: str, x_label: str, y_label: str, rotate_x: bool
    ) -> None:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        if x_label:
            ax.set_xlabel(x_label, fontsize=11)
        if y_label:
            ax.set_ylabel(y_label, fontsize=11)
        if rotate_x:
            ax.tick_params(axis="x", rotation=28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @staticmethod
    def _parse_input(input_data: Union[str, Dict]) -> Dict[str, Any]:
        if isinstance(input_data, dict):
            return input_data
        try:
            return json.loads(input_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Entrée JSON invalide : {exc}") from exc

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


# ── Helpers Streamlit ─────────────────────────────────────────────────────────

def base64_to_image_bytes(b64_str: str) -> bytes:
    """Convertit une chaîne base64 en bytes utilisable par ``st.image()``."""
    return base64.b64decode(b64_str)


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tool = NBAPlotTool()

    samples = [
        {
            "chart_type": "bar",
            "data": [
                {"label": "Giannis Antetokounmpo", "value": 30.4},
                {"label": "Joel Embiid",             "value": 33.1},
                {"label": "LeBron James",            "value": 27.2},
                {"label": "Stephen Curry",           "value": 29.4},
                {"label": "Nikola Jokic",            "value": 24.5},
            ],
            "title": "Points par match — Top 5",
            "x_label": "Joueur",
            "y_label": "PTS / match",
            "_outfile": "test_bar.png",
        },
        {
            "chart_type": "horizontal_bar",
            "data": [
                {"label": "Boston Celtics",        "value": 64},
                {"label": "Oklahoma City Thunder", "value": 57},
                {"label": "Cleveland Cavaliers",   "value": 64},
                {"label": "Houston Rockets",       "value": 52},
                {"label": "Golden State Warriors", "value": 48},
                {"label": "Denver Nuggets",        "value": 50},
            ],
            "title": "Victoires par équipe (saison régulière)",
            "x_label": "Victoires",
            "_outfile": "test_hbar.png",
        },
        {
            "chart_type": "pie",
            "data": [
                {"label": "Boston Celtics",        "value": 64},
                {"label": "Oklahoma City Thunder", "value": 57},
                {"label": "Cleveland Cavaliers",   "value": 64},
                {"label": "Houston Rockets",       "value": 52},
            ],
            "title": "Victoires — Top 4",
            "_outfile": "test_pie.png",
        },
        {
            "chart_type": "line",
            "data": [
                {"label": "Oct", "value": 24.1},
                {"label": "Nov", "value": 26.3},
                {"label": "Déc", "value": 28.8},
                {"label": "Jan", "value": 30.2},
                {"label": "Fév", "value": 27.5},
                {"label": "Mar", "value": 31.4},
            ],
            "title": "Évolution points/match — Stephen Curry",
            "x_label": "Mois",
            "y_label": "PTS / match",
            "_outfile": "test_line.png",
        },
    ]

    for sample in samples:
        out = sample.pop("_outfile")
        b64 = tool.run(sample)
        with open(out, "wb") as fh:
            fh.write(base64.b64decode(b64))
        print(f"✓ {out}")
