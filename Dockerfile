# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Dépendances système (PyMuPDF, EasyOCR, OpenCV) ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dépendances Python (layer mis en cache si requirements.txt inchangé) ─────
COPY P10_DSML/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "mistralai>=1.5.0,<2.0.0" && \
    pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
COPY . .

# ── Configuration Streamlit (port 8508, pas de navigateur) ───────────────────
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nport = 8508\naddress = "0.0.0.0"\nheadless = true\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

# ── Normaliser les fins de ligne (CRLF → LF) et rendre exécutable ────────────
RUN sed -i 's/\r//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8508

ENTRYPOINT ["/app/entrypoint.sh"]
