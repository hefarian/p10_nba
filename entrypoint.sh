#!/bin/bash
set -e

echo "=============================================="
echo " SportSee NBA Analyst AI — Démarrage"
echo "=============================================="

cd /app

# ── Étape 1 : Ingestion Excel → SQLite ────────────────────────────────────────
echo ""
echo "[1/3] Ingestion des données Excel → SQLite..."
python load_excel_to_db.py \
    --excel "DATA/regular+NBA+(2).xlsx" \
    --db P10_DSML/database/nba.db

# ── Étape 2 : Indexation FAISS ────────────────────────────────────────────────
echo ""
cd /app/P10_DSML
if [ -f "vector_db/faiss_index.idx" ] && [ -f "vector_db/document_chunks.pkl" ]; then
    echo "[2/3] Index FAISS déjà présent — indexation ignorée."
else
    echo "[2/3] Indexation des documents dans FAISS..."
    python indexer.py
fi

# ── Étape 3 : Lancement Streamlit ─────────────────────────────────────────────
echo ""
echo "[3/3] Démarrage de l'assistant Streamlit sur le port 8508..."
exec streamlit run MistralChat.py \
    --server.port 8508 \
    --server.address 0.0.0.0 \
    --server.headless true
