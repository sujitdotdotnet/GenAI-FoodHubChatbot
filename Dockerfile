# Hugging Face Space (Docker SDK) - Streamlit app with ChromaDB policy RAG
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Hugging Face runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create the app folder as the non-root user so ChromaDB can write its index there
RUN mkdir -p $HOME/app
WORKDIR $HOME/app

# Install dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Application files
COPY --chown=user app.py policy_rag.py build_index.py customer_orders.db banner.jpg ./
COPY --chown=user policies ./policies
COPY --chown=user .streamlit ./.streamlit

# RAG indexing at build time: chunk the policies, download the embedding model (cached in
# ~/.cache/chroma) and build the ChromaDB index, so the first start of the Space is fast.
# If this step fails (e.g. no network at build time) the app still works: it indexes at start-up
# or falls back to TF-IDF.
RUN python build_index.py || echo "WARNING: index pre-build skipped; the app will index on start-up"

# Must match `app_port` in README.md
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
