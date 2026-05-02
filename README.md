# CAL — Criticar a Arbitragem Legalmente

> Plataforma de análise estatística de arbitragem na Primeira Liga portuguesa (2017/18 – presente)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://www.docker.com/)
[![Live](https://img.shields.io/badge/Live-cal.terradigital.net-green)](https://cal.terradigital.net)

---

## Sobre o projecto

O CAL é uma plataforma de análise estatística que aplica métodos quantitativos rigorosos para detectar padrões de bias na arbitragem do futebol português. O objectivo não é acusar árbitros individualmente, mas fornecer uma ferramenta transparente e baseada em dados que permita uma discussão informada sobre a qualidade da arbitragem.

### Metodologia

- **Teste binomial** com correcção de Benjamini-Hochberg (FDR) para múltiplas comparações
- **PressureBiasIndex (PBI)** — índice proprietário de bias por pressão do público
- **Modelos de machine learning** (scikit-learn) para detecção de anomalias estatísticas
- Dados históricos de 9 épocas da Primeira Liga (2017/18 a 2025/26)

---

## Funcionalidades

| Vista | Descrição |
|-------|-----------|
| **Árbitro específico** | Perfil detalhado de cada árbitro com métricas de bias |
| **Desvios extremos** | Jogos com decisões estatisticamente anómalas |
| **Heatmap geral** | Visão panorâmica de todos os árbitros por época |
| **Por equipa** | Análise de como cada equipa é tratada pelos árbitros |
| **Perfil de equipa** | Estatísticas detalhadas por clube |
| **Alertas & Comparação** | Alertas automáticos e comparação entre árbitros |

---

## Stack técnica

```
Frontend    → Streamlit + Plotly
Backend     → Python 3.11
Base dados  → PostgreSQL 15 + psycopg2
ML          → scikit-learn, statsmodels, scipy
Deploy      → Docker Compose + Nginx + Let's Encrypt
Servidor    → Hetzner Cloud CPX22 (Ubuntu 24.04)
```

---

## Estrutura do projecto

```
cal/
├── cal/
│   ├── analysis/           ← Módulos de análise estatística
│   │   ├── alerts.py
│   │   ├── bias_history.py
│   │   ├── cards_by_team.py
│   │   └── home_bias.py
│   ├── dashboard/
│   │   └── app.py          ← Aplicação Streamlit principal
│   ├── features/
│   │   └── engineering.py  ← Feature engineering para ML
│   ├── ingest/
│   │   ├── footballdata.py ← Ingestão de dados (football-data.co.uk)
│   │   └── sofascore.py    ← Ingestão de árbitros (Sofascore)
│   ├── models/
│   │   └── train.py        ← Treino de modelos ML
│   ├── bias_engine.py      ← Motor central de detecção de bias
│   ├── db.py               ← Gestão de ligações à base de dados
│   ├── schema.sql          ← Schema PostgreSQL
│   └── requirements.txt
├── cal_splash.html         ← Splash screen (ficha técnica do projecto)
├── cal_architecture.html   ← Documentação completa da arquitectura
├── docker-compose.yml
├── Dockerfile
├── .env.example            ← Template de configuração
├── run_ingest.py           ← Script de ingestão de dados
├── run_referees.py         ← Script de ingestão de árbitros
├── run_features.py         ← Script de feature engineering
├── run_models.py           ← Script de treino de modelos
├── run_bias.py             ← Script de cálculo de bias
├── start.sh / stop.sh      ← Scripts de arranque (Linux/macOS)
└── start.bat / stop.bat    ← Scripts de arranque (Windows)
```

---

## Instalação e execução local

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e a correr
- Nenhuma outra dependência necessária

### Passos

```bash
# 1. Clonar o repositório
git clone https://github.com/ppestana/cal.git
cd cal

# 2. Configurar variáveis de ambiente
cp .env.example .env
nano .env   # definir DB_PASSWORD

# 3. Arrancar
./start.sh          # Linux/macOS
start.bat           # Windows

# 4. Ingestão de dados
docker compose exec app python run_ingest.py
docker compose exec app python run_referees.py
docker compose exec app python run_features.py
docker compose exec app python run_models.py
docker compose exec app python run_bias.py
```

A aplicação fica disponível em **http://localhost:8501**

### Ficheiro .env

```bash
# Base de dados PostgreSQL
DB_NAME=cal
DB_USER=cal_user
DB_PASSWORD=<password_à_tua_escolha>
DB_PORT=5432
```

---

## Fontes de dados

| Fonte | Dados | Cobertura |
|-------|-------|-----------|
| [football-data.co.uk](https://www.football-data.co.uk/) | Resultados, golos, cartões | 2017/18 – presente |
| [Sofascore](https://www.sofascore.com/) | Árbitros por jogo | 2017/18 – presente |

---

## Demonstração

A aplicação está disponível em produção em **[cal.terradigital.net/app](https://cal.terradigital.net/app)**

> A primeira visita a `cal.terradigital.net` apresenta um splash screen com a ficha técnica do projecto antes de entrar na aplicação.

---

## Declaração de utilização de IA

Em conformidade com o EU AI Act (Regulamento UE 2024/1689).  
Claude (Anthropic, 2025–2026) foi utilizado como assistente de programação e documentação. Todo o código foi revisto, testado e validado pelo autor. Nenhum dado foi gerado ou modificado por IA.

---

## Autor

**Pedro Pestana** — GIS & geospatial systems specialist, full-stack developer  
[pedropestana.com](https://www.pedropestana.com) · [TerraDigital](https://www.terradigital.net) · [LinkedIn](https://www.linkedin.com/in/pmpestana)

---

## Licença

MIT License — ver [LICENSE](LICENSE)  
Os dados utilizados são de fontes públicas. Uso comercial dos dados não autorizado.
