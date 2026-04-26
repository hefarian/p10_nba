# SportSee NBA Analyst AI — Système RAG + SQL

Assistant intelligent d'analyse de performance NBA combinant **RAG** (Retrieval-Augmented Generation), un **SQL Tool** pour les données chiffrées, et une observabilité complète via **Pydantic Logfire**.

## Architecture

```
P10_DSML/
├── MistralChat.py          # Application Streamlit (RAG + SQL routing + Logfire)
├── indexer.py              # Script d'indexation FAISS
├── evaluate_ragas.py       # Évaluation automatisée RAGAS
├── sql_tool.py             # LangChain SQL Tool (génération dynamique SQL)
├── inputs/                 # Documents sources (PDF, TXT, DOCX…)
├── vector_db/              # Index FAISS + chunks sérialisés
├── database/               # Base SQLite (nba.db)
├── results/                # Résultats d'évaluation (CSV + JSON)
└── utils/
    ├── config.py           # Variables de configuration
    ├── data_loader.py      # Chargement et OCR des documents
    ├── vector_store.py     # Gestion de l'index vectoriel FAISS
    └── pipeline.py         # Modèles Pydantic (validation entrées/sorties)

load_excel_to_db.py         # Ingestion Excel → SQLite (à la racine du projet)
```

## Flux de données

```
Question utilisateur
        │
        ▼
 _is_numerical_question() ?
        │
   Oui ─┤─── SQL Tool ──► SQLite NBA ──► résultat chiffré
        │
   Non ─┤─── Vector Store FAISS ──► chunks PDF/TXT ──► contexte
        │
        ▼
   LLM Mistral (mistral-small-latest)
        │
        ▼
   Réponse synthétisée + indicateur de source
```

## Prérequis

- Python 3.10+
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai/))
- (Optionnel) Token Logfire pour l'observabilité cloud

## Installation

```bash
# Depuis la racine PROJET10/
git clone <url-du-repo>
cd PROJET10

# Environnement virtuel
py -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Dépendances
pip install -r P10_DSML/requirements.txt
```

## Configuration

Créez un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=votre_clé_api_mistral

# Optionnel — Logfire (https://logfire.pydantic.dev/)
LOGFIRE_TOKEN=votre_token_logfire

# Optionnel — chemin base SQLite (défaut : P10_DSML/database/nba.db)
NBA_DB_PATH=P10_DSML/database/nba.db
```

## Utilisation

### 1. Ingérer les données Excel → SQLite

```bash
py load_excel_to_db.py
# Ou avec chemins personnalisés :
py load_excel_to_db.py --excel DATA/regular+NBA+(2).xlsx --db P10_DSML/database/nba.db
```

### 2. Indexer les documents (PDF, TXT…)

Placez vos fichiers dans `P10_DSML/inputs/`, puis :

```bash
cd P10_DSML
py indexer.py
```

### 3. Lancer l'assistant

```bash
cd P10_DSML
streamlit run MistralChat.py
```

### 4. Évaluer les performances (RAGAS)

```bash
cd P10_DSML

# Évaluation complète (avant SQL tool)
py evaluate_ragas.py --output results/eval_avant.json --csv results/eval_avant.csv

# Sans métriques RAGAS (RAG seul, plus rapide)
py evaluate_ragas.py --no-ragas

# Évaluation après intégration SQL
py evaluate_ragas.py --output results/eval_apres.json --csv results/eval_apres.csv
```

### 5. Tester le SQL Tool en standalone

```bash
cd P10_DSML
py sql_tool.py
```

## Métriques RAGAS

| Métrique | Description |
|---|---|
| `faithfulness` | La réponse est-elle fidèle aux contextes récupérés ? |
| `answer_relevancy` | La réponse répond-elle à la question ? |
| `context_precision` | Les contextes récupérés sont-ils pertinents ? |
| `context_recall` | Le contexte couvre-t-il la réponse de référence ? |

Les résultats sont exportés en CSV et JSON dans `results/`.

## Validation Pydantic

Toutes les données transitent par des modèles Pydantic (`utils/pipeline.py`) :

- `PlayerStats` — validation des stats joueurs (cohérence W+L≤GP, ranges de %)
- `RAGQuery` — validation de la requête entrante
- `RAGResponse` — validation de la réponse LLM
- `EvalSample` — validation des cas de test RAGAS

## Observabilité Logfire

Si `LOGFIRE_TOKEN` est défini, chaque interaction est tracée :
- span `rag_interaction` (global)
- span `vector_store_search` (retrieval)
- span `sql_tool_call` (SQL routing)
- span `mistral_llm_call` (génération)

Sans token, Logfire fonctionne en mode local (logs uniquement).

## Formats de documents supportés

| Format | Méthode |
|---|---|
| PDF (texte) | PyPDF2 |
| PDF (image/scan) | PyMuPDF + EasyOCR |
| TXT | lecture directe |
| DOCX | python-docx |
| CSV | pandas |
| JSON | json stdlib |


L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `utils/query_classifier.py`

Détermine si une requête nécessite une recherche RAG :
- Analyse des mots-clés
- Classification avec le modèle Mistral
- Détection des questions spécifiques vs générales

### `utils/database.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation

