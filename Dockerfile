# CAL — Ambiente de desenvolvimento Python
# Baseado em Python 3.11 slim; todas as dependências instaladas em build time.
# O código é montado como volume — não é copiado para a imagem.

FROM python:3.11-slim

# ─── System dependencies ───────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
        git \
        vim \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ─── Python environment ────────────────────────────────────
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copiar requirements antes do código para maximizar cache de layers.
# Fases comentadas são instaladas todas — rebuild rápido quando se descomenta.
COPY cal/requirements.txt /tmp/requirements.txt

# Instalar dependências base (Fase 1) + dependências das fases seguintes
# para que o ambiente fique pronto sem rebuild.
RUN pip install \
        psycopg2-binary==2.9.9 \
        pandas==2.2.2 \
        requests==2.32.3 \
        numpy==1.26.4 \
        python-dotenv==1.0.1 \
        rich==13.7.1 \
        # Fase 2+ — modelos
        scikit-learn==1.4.2 \
        statsmodels==0.14.2 \
        scipy==1.13.0 \
        # Fase 5 — API & dashboard
        fastapi==0.111.0 \
        "uvicorn[standard]==0.30.0" \
        streamlit==1.35.0 \
        plotly==5.22.0 \
        # Exploração
        jupyterlab==4.2.0 \
        ipykernel==6.29.4 \
        # Fase 6 — relatórios PDF
        matplotlib==3.9.0 \
        reportlab==4.2.0

# ─── Criar directório de trabalho e notebooks ──────────────
RUN mkdir -p /app/cal /app/notebooks

# ─── Portas documentadas (exposição real via docker-compose) ─
EXPOSE 8000 8501 8888

# Comando por defeito: container fica vivo para exec interactivo.
# Para serviços específicos, sobrepor no docker-compose.yml.
CMD ["tail", "-f", "/dev/null"]
