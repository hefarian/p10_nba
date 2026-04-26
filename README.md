# SportSee NBA Analyst AI — Système RAG + SQL

Assistant intelligent d'analyse de performance NBA combinant **RAG** (Retrieval-Augmented Generation), un **SQL Tool** pour les données chiffrées, et une observabilité complète via **Pydantic Logfire**.

## Fonctionnalités

- 🔍 **Recherche sémantique** (FAISS + embeddings Mistral) sur documents PDF/TXT
- 📊 **SQL Tool** (LangChain) pour les questions chiffrées → SQLite NBA (568 joueurs)
- 📈 **PlotTool** — génération dynamique de graphiques matplotlib en base64
- 🛡️ **Validation Pydantic v2** à chaque étape du pipeline
- 🔭 **Observabilité Logfire** — spans RAG, SQL, LLM tracés
- 🐳 **Docker** — démarrage en une commande, Tesseract OCR inclus
- 📋 **Évaluation RAGAS** — faithfulness, answer_relevancy, context_precision

## Prérequis

- Python 3.10+ **ou** Docker + Docker Compose
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai/))

## Structure du projet

```
PROJET10/
├── .gitignore
├── .dockerignore
├── Dockerfile                      # Image python:3.12-slim + Tesseract
├── docker-compose.yml              # Orchestration sportsee-rag + tika
├── entrypoint.sh                   # Démarrage séquentiel (SQLite → FAISS → Streamlit)
├── load_excel_to_db.py             # Ingestion Excel NBA → SQLite
│
├── DATA/
│   ├── regular+NBA+(2).xlsx        # Données saison NBA (569 joueurs)
│   ├── Reddit+1+(1).pdf            # Corpus documentaire (4 PDFs Reddit NBA)
│   ├── Reddit+2+(1).pdf
│   ├── Reddit+3+(1).pdf
│   └── Reddit+4+(1).pdf
│
└── P10_DSML/
    ├── MistralChat.py              # Application Streamlit — routage SQL/FAISS + Logfire
    ├── indexer.py                  # Indexation FAISS des documents
    ├── sql_tool.py                 # SQL Tool LangChain (génération dynamique SQL)
    ├── plot_tool.py                # PlotTool — graphiques matplotlib base64
    ├── evaluate_ragas.py           # Évaluation RAGAS (FAISS seul)
    ├── evaluate_ragas_v2.py        # Évaluation RAGAS (routing SQL/FAISS — production)
    ├── ocr_compare.py              # Benchmark OCR : pytesseract / Tika / Nanonets
    ├── requirements.txt
    ├── analyse_ocr.md              # Rapport comparatif OCR
    │
    ├── inputs/                     # Documents sources indexés
    │   ├── Reddit 1.pdf
    │   ├── Reddit 2.pdf
    │   ├── Reddit 3.pdf
    │   ├── Reddit 4.pdf
    │   └── regular NBA.xlsx
    │
    ├── database/
    │   └── nba.db                  # SQLite — tables players (568) + teams (30)
    │
    ├── vector_db/
    │   ├── faiss_index.idx         # Index vectoriel FAISS
    │   └── document_chunks.pkl     # Chunks sérialisés
    │
    ├── results/
    │   ├── eval_avant.json / .csv          # RAGAS — FAISS seul, code v0
    │   ├── eval_apres.json / .csv          # RAGAS — FAISS seul, code v1
    │   ├── eval_apres_v2.json / .csv       # RAGAS — routing SQL/FAISS (production)
    │   ├── eval_avant_quick.json / .csv    # Mode --no-ragas rapide
    │   └── ocr_comparison.json / .csv      # Benchmark OCR
    │
    └── utils/
        ├── config.py               # Variables de configuration centralisées
        ├── data_loader.py          # Chargement PDF/TXT/DOCX + OCR pytesseract
        ├── vector_store.py         # Gestion index FAISS + embeddings Mistral
        └── pipeline.py             # Modèles Pydantic v2 (PlayerStats, RAGQuery…)
```

## Flux de données

```
Question utilisateur
        │
        ▼
 _is_numerical_question() ?
        │
   Oui ─┤─── SQL Tool ──► SQLite NBA ──► résultat chiffré ──► PlotTool (si graphique demandé)
        │
   Non ─┤─── Vector Store FAISS ──► chunks PDF/TXT ──► contexte
        │
        ▼
   LLM Mistral (mistral-small-latest)
        │
        ▼
   Réponse + badge source (📊 SQL / 📚 Docs) + Logfire trace
```

## Installation

### Option A — Docker (recommandé)

```bash
git clone https://github.com/hefarian/p10_nba.git
cd p10_nba

# Créer le fichier de config
echo "MISTRAL_API_KEY=votre_clé" > P10_DSML/.env

# Premier lancement (build + ingestion + indexation + Streamlit)
docker compose up --build
```

L'application est accessible sur **http://localhost:8508**.

### Option B — Python local

```bash
git clone https://github.com/hefarian/p10_nba.git
cd p10_nba

py -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r P10_DSML/requirements.txt

# Installer Tesseract OCR (Windows) : https://github.com/UB-Mannheim/tesseract/wiki
# Puis ajouter au PATH

echo "MISTRAL_API_KEY=votre_clé" > P10_DSML/.env
```

## Configuration

Fichier `P10_DSML/.env` :

```env
# Obligatoire
MISTRAL_API_KEY=votre_clé_mistral

# Optionnel
MODEL_ID=mistral-small-latest
EMBEDDING_MODEL=mistral-embed
NBA_DB_PATH=P10_DSML/database/nba.db
LOGFIRE_TOKEN=votre_token_logfire
NANONETS_API_KEY=votre_clé_nanonets
```

## Utilisation

### 1. Ingérer les données NBA → SQLite

```bash
py load_excel_to_db.py --excel DATA/regular+NBA+(2).xlsx --db P10_DSML/database/nba.db
```

### 2. Indexer les documents FAISS

```bash
cd P10_DSML
py indexer.py
```

### 3. Lancer l'assistant Streamlit

```bash
cd P10_DSML
streamlit run MistralChat.py
# → http://localhost:8501
```

### 4. Évaluation RAGAS

```bash
cd P10_DSML

# Évaluation FAISS seul (avant SQL Tool)
py evaluate_ragas.py --output results/eval_avant.json --csv results/eval_avant.csv

# Mode rapide sans métriques
py evaluate_ragas.py --no-ragas

# Évaluation production (routing SQL/FAISS)
py evaluate_ragas_v2.py --output results/eval_apres_v2.json --csv results/eval_apres_v2.csv
```

### 5. Benchmark OCR

```bash
cd P10_DSML
py ocr_compare.py --output results/ocr_comparison.csv
```

## Résultats RAGAS mesurés

| Évaluation | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| eval_avant (FAISS, v0) | 0.569 | N/A | 0.259 |
| eval_apres (FAISS, v1) | 0.553 | N/A | 0.370 |
| **eval_apres_v2 (SQL/FAISS)** | **0.879** | **0.810** | **0.444** |

## Benchmark OCR mesuré (84 pages, 4 PDFs Reddit)

| Moteur | Mots/page | Chars/page | Temps/page | Coût |
|---|---|---|---|---|
| EasyOCR (référence) | 293 | 1 746 | 8,6 s | Gratuit |
| **pytesseract (retenu)** | **289** | **1 652** | **0,92 s** | Gratuit |
| Apache Tika | 289 | 1 682 | 0,81 s | Gratuit |
| Nanonets | 304 | 1 869 | 7,31 s | 0,01 $/page |

## Commandes Docker utiles

```bash
docker compose up --build      # Premier lancement
docker compose up -d           # Relancer sans rebuild
docker compose logs -f         # Logs en direct
docker compose down            # Arrêter
```

