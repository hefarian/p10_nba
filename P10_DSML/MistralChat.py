# MistralChat.py (version RAG + SQL Tool + Logfire)
import streamlit as st
import os
import json
import uuid
import logging
from datetime import datetime
from mistralai import Mistral
from dotenv import load_dotenv

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
    from utils.pipeline import RAGQuery, RAGResponse, RetrievedChunk
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# --- Logfire (observabilité) ---
try:
    import logfire
    # Priorité : token env var, sinon config locale (.logfire/), sinon mode console
    _LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    if _LOGFIRE_TOKEN:
        logfire.configure(token=_LOGFIRE_TOKEN, service_name="sportsee-rag")
    else:
        try:
            # Tente la config locale (logfire auth + logfire projects use)
            logfire.configure(service_name="sportsee-rag")
        except Exception:
            # Pas de credentials locaux (ex: Docker) → mode console uniquement
            logfire.configure(send_to_logfire=False, service_name="sportsee-rag")
    logfire.instrument_pydantic()   # trace les validations Pydantic
    LOGFIRE_ENABLED = True
except ImportError:
    LOGFIRE_ENABLED = False
    logging.getLogger(__name__).warning("Logfire non installé — observabilité désactivée.")


# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    client = Mistral(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# --- Chargement du SQL Tool (mis en cache) ---
@st.cache_resource
def get_sql_tool():
    try:
        from sql_tool import NBAQueryTool
        tool = NBAQueryTool()
        logging.info("SQL Tool NBA chargé.")
        return tool
    except ImportError:
        logging.warning("sql_tool.py non trouvé — outil SQL désactivé.")
        return None
    except Exception as e:
        logging.warning(f"Erreur chargement SQL Tool: {e}")
        return None

sql_tool = get_sql_tool()

# --- Chargement du Plot Tool (mis en cache) ---
@st.cache_resource
def get_plot_tool():
    try:
        from plot_tool import NBAPlotTool
        tool = NBAPlotTool()
        logging.info("Plot Tool NBA chargé.")
        return tool
    except ImportError:
        logging.warning("plot_tool.py non trouvé — visualisations désactivées.")
        return None
    except Exception as e:
        logging.warning(f"Erreur chargement Plot Tool: {e}")
        return None

plot_tool = get_plot_tool()

# --- Gestion des conversations persistantes ---
CONVERSATIONS_DIR = os.path.join(os.path.dirname(__file__), "conversations")
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


def _list_conversations() -> list[dict]:
    """Retourne la liste des conversations sauvegardées, triées par date décroissante."""
    convs = []
    for fname in os.listdir(CONVERSATIONS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(CONVERSATIONS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                convs.append({"id": data["id"], "name": data["name"], "path": path,
                               "created_at": data.get("created_at", "")})
            except Exception:
                pass
    return sorted(convs, key=lambda x: x["created_at"], reverse=True)


def _save_conversation(conv_id: str, name: str, messages: list[dict]):
    """Sauvegarde la conversation courante dans un fichier JSON."""
    path = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "id": conv_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "messages": messages,
        }, f, ensure_ascii=False, indent=2)


def _load_conversation(path: str) -> dict | None:
    """Charge une conversation depuis un fichier JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _delete_conversation(path: str):
    """Supprime un fichier de conversation."""
    try:
        os.remove(path)
    except Exception:
        pass


# Initialisation de l'état de conversation
if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "conv_name" not in st.session_state:
    st.session_state.conv_name = datetime.now().strftime("%d/%m/%Y %H:%M")


SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Fonctions ---

def generer_reponse(prompt_messages: list[dict]) -> str:
    """
    Envoie le prompt (qui inclut maintenant le contexte) à l'API Mistral.
    """
    if not prompt_messages:
         logging.warning("Tentative de génération de réponse avec un prompt vide.")
         return "Je ne peux pas traiter une demande vide."
    try:
        logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")

        if LOGFIRE_ENABLED:
            import logfire
            with logfire.span("mistral_llm_call", model=model, n_messages=len(prompt_messages)):
                response = client.chat.complete(
                    model=model,
                    messages=prompt_messages,
                    temperature=0.1,
                )
        else:
            response = client.chat.complete(
                model=model,
                messages=prompt_messages,
                temperature=0.1,
            )

        if response.choices and len(response.choices) > 0:
            logging.info("Réponse reçue de l'API Mistral.")
            return response.choices[0].message.content
        else:
            logging.warning("L'API n'a pas retourné de choix valide.")
            return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        logging.exception("Erreur API Mistral pendant client.chat.complete")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."


def _is_numerical_question(question: str) -> bool:
    """Détecte si la question porte sur des données chiffrées nécessitant le SQL tool."""
    keywords = [
        "pourcentage", "%", "statistique", "stats", "meilleur",
        "plus de", "moins de", "combien", "quel joueur a", "compare",
        "top ", "classement", "rang", "total", "moyenne", "saison",
        "points", "rebonds", "passes", "assists", "blocks", "steals",
        "tirs", "3 points", "lancers francs", "victoires", "défaites",
        "bilan", "ratio", "efficacité", "usg", "ts%", "efg",
        # Termes supplémentaires pour le routage correct
        "paniers", "panier", "marque", "fgm", "fg%", "shoots",
        "répartition", "détail", "nombre de",
    ]
    q_lower = question.lower()
    # Une demande de visualisation implique toujours des données chiffrées
    if _is_visualization_request(question):
        return True
    return any(kw in q_lower for kw in keywords)


def _is_visualization_request(question: str) -> bool:
    """Détecte si l'utilisateur demande un graphique ou une visualisation."""
    keywords = [
        "graphique", "graph", "visualise", "visualisation", "montre",
        "affiche", "chart", "plot", "courbe", "histogramme",
        "camembert", "barres", "diagramme", "compare visuellement",
        "représente", "évolution", "schéma",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


def _extract_plot_params(question: str, sql_result: str) -> dict | None:
    """
    Demande au LLM de structurer les données SQL en JSON pour le PlotTool.

    Returns:
        dict avec chart_type, data, title, x_label, y_label
        ou None en cas d'échec.
    """
    extraction_prompt = f"""Tu es un assistant qui transforme des résultats SQL NBA en paramètres JSON pour générer un graphique matplotlib.

Résultats SQL disponibles :
{sql_result}

Question de l'utilisateur : {question}

Génère UNIQUEMENT un JSON valide avec cette structure exacte :
{{
    "chart_type": "bar" ou "horizontal_bar" ou "line" ou "pie",
    "data": [{{"label": "NomJoueur", "value": 0.0}}, ...],
    "title": "Titre du graphique",
    "x_label": "Label axe X",
    "y_label": "Label axe Y"
}}

Règles de choix du type :
- "bar"            → comparaison ≤5 éléments
- "horizontal_bar" → comparaison >5 éléments ou labels longs
- "line"           → évolution temporelle
- "pie"            → répartition / proportions
- Inclus au maximum 15 éléments dans "data".
- N'inclus que le JSON, sans texte autour.

JSON :"""

    try:
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # Extraire le JSON même s'il y a du texte autour
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            import json as _json
            return _json.loads(match.group())
    except Exception as exc:
        logging.warning("Extraction paramètres graphique échouée : %s", exc)
    return None


# --- Interface Utilisateur Streamlit ---

# ── Sidebar : gestion des conversations ───────────────────────────────────────
with st.sidebar:
    st.header("💬 Conversations")

    # Nouvelle conversation
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        st.session_state.conv_id = str(uuid.uuid4())
        st.session_state.conv_name = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]
        st.rerun()

    st.divider()

    # Renommer la conversation courante
    new_name = st.text_input("Nom de la conversation", value=st.session_state.conv_name, key="conv_name_input")
    if new_name != st.session_state.conv_name:
        st.session_state.conv_name = new_name

    st.divider()

    # Liste des conversations sauvegardées
    saved = _list_conversations()
    if saved:
        st.subheader("📂 Historique")
        for conv in saved:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(conv["name"], key=f"load_{conv['id']}", use_container_width=True,
                             type="primary" if conv["id"] == st.session_state.conv_id else "secondary"):
                    data = _load_conversation(conv["path"])
                    if data:
                        st.session_state.conv_id = data["id"]
                        st.session_state.conv_name = data["name"]
                        st.session_state.messages = data["messages"]
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{conv['id']}", help="Supprimer"):
                    _delete_conversation(conv["path"])
                    if conv["id"] == st.session_state.conv_id:
                        st.session_state.conv_id = str(uuid.uuid4())
                        st.session_state.conv_name = datetime.now().strftime("%d/%m/%Y %H:%M")
                        st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions."}]
                    st.rerun()
    else:
        st.caption("Aucune conversation sauvegardée.")

# ─────────────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Span Logfire global pour cette interaction
    _logfire_span_ctx = None
    if LOGFIRE_ENABLED:
        import logfire
        _logfire_span_ctx = logfire.span("rag_interaction", question=prompt)
        _logfire_span_ctx.__enter__()

    # === Début de la logique RAG + SQL ===

    source_type = "vector_store"
    context_str = ""
    search_results = []

    # 2. Routing : questions chiffrées → SQL Tool, sinon → Vector Store
    if sql_tool is not None and _is_numerical_question(prompt):
        try:
            logging.info(f"Routing SQL pour: '{prompt}'")
            if LOGFIRE_ENABLED:
                with logfire.span("sql_tool_call", question=prompt):
                    sql_context = sql_tool.run(prompt)
            else:
                sql_context = sql_tool.run(prompt)
            context_str = f"[Données SQL]\n{sql_context}"
            source_type = "sql_tool"
            logging.info("Contexte SQL récupéré.")
        except Exception as e:
            logging.warning(f"SQL Tool échoué, fallback vector store: {e}")

    # 3. Si pas de contexte SQL, utiliser le Vector Store
    if not context_str:
        if vector_store_manager is None:
            st.error("Le service de recherche de connaissances n'est pas disponible.")
            logging.error("VectorStoreManager non disponible pour la recherche.")
            if _logfire_span_ctx:
                _logfire_span_ctx.__exit__(None, None, None)
            st.stop()
        try:
            logging.info(f"Recherche vector store pour: '{prompt}' (k={SEARCH_K})")
            if LOGFIRE_ENABLED:
                with logfire.span("vector_store_search", question=prompt, k=SEARCH_K):
                    search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            else:
                search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            logging.info(f"{len(search_results)} chunks trouvés.")
        except Exception as e:
            st.error(f"Erreur lors de la recherche d'informations pertinentes: {e}")
            logging.exception(f"Erreur vector_store.search pour: {prompt}")
            search_results = []

        context_str = "\n\n---\n\n".join([
            f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in search_results
        ])
        if not context_str:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances."
            logging.warning(f"Aucun contexte trouvé pour: {prompt}")

    # 4. Construire le prompt final
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

    messages_for_api = [
        {"role": "user", "content": final_prompt_for_llm}
    ]

    # === Fin de la logique RAG + SQL ===

    # 5. Afficher indicateur + Générer la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...")

        response_content = generer_reponse(messages_for_api)
        message_placeholder.write(response_content)

        # Indicateur de source
        if source_type == "sql_tool":
            st.caption("📊 Réponse basée sur les données SQL")
        elif search_results:
            st.caption(f"📚 Réponse basée sur {len(search_results)} document(s) indexé(s)")

        # ── Visualisation dynamique ────────────────────────────────────────
        if plot_tool is not None and _is_visualization_request(prompt) and source_type == "sql_tool":
            try:
                plot_params = _extract_plot_params(prompt, context_str)
                if plot_params:
                    from plot_tool import base64_to_image_bytes
                    b64_img = plot_tool.run(plot_params)
                    st.image(
                        base64_to_image_bytes(b64_img),
                        caption=plot_params.get("title", "Graphique NBA"),
                        use_container_width=True,
                    )
                    logging.info("Graphique généré : %s", plot_params.get("title", ""))
                else:
                    logging.warning("Impossible d'extraire les paramètres du graphique.")
            except Exception as _plot_err:
                logging.warning("Erreur génération graphique : %s", _plot_err)

    # 6. Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response_content})

    # 7. Sauvegarder automatiquement la conversation
    _save_conversation(st.session_state.conv_id, st.session_state.conv_name, st.session_state.messages)

    # Fermer le span Logfire
    if _logfire_span_ctx:
        _logfire_span_ctx.__exit__(None, None, None)

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")