# CAL — Setup do Ambiente Docker

## Conceito

Todos os ficheiros do projecto — código, base de dados, configuração —
residem no **disco externo**. Ao ligar o disco a qualquer computador com
Docker instalado, o ambiente arranca imediatamente com:

```
./start.sh        # Linux / macOS
start.bat         # Windows
```

---

## Pré-requisito único: Docker Desktop

| Sistema    | Download                                      |
|------------|-----------------------------------------------|
| Windows    | https://docs.docker.com/desktop/install/windows-install/ |
| macOS      | https://docs.docker.com/desktop/install/mac-install/     |
| Linux      | https://docs.docker.com/engine/install/       |

Não é necessário instalar Python, PostgreSQL, nem qualquer outra dependência
no computador host — tudo corre dentro dos containers.

---

## Estrutura de ficheiros no disco externo

```
CAL/
├── cal/                ← Código do projecto (git)
│   ├── schema.sql
│   ├── db.py
│   ├── ingest/
│   │   └── footballdata.py
│   └── requirements.txt
│
├── pgdata/             ← Base de dados PostgreSQL (auto-criado, nunca editar)
├── notebooks/          ← Jupyter notebooks de exploração (auto-criado)
│
├── docker-compose.yml
├── Dockerfile
├── .env                ← Credenciais (criar a partir de .env.example)
├── .env.example
├── .gitignore
├── start.sh  / start.bat
├── stop.sh   / stop.bat
├── run_ingest.py
└── README_DOCKER.md    ← este ficheiro
```

---

## Primeira execução

### 1. Criar o ficheiro .env

```bash
cp .env.example .env
nano .env           # definir DB_PASSWORD
```

**Alterar apenas `DB_PASSWORD`** — os restantes valores são opcionais.

### 2. Arrancar

```bash
chmod +x start.sh stop.sh    # só necessário uma vez no Linux/macOS
./start.sh
```

Na primeira execução:
- A imagem Python é construída (~3–5 minutos, apenas uma vez por computador)
- O PostgreSQL inicializa o schema automaticamente a partir de `cal/schema.sql`

### 3. Entrar no shell de desenvolvimento

```bash
docker compose exec dev bash
```

Dentro do container, o directório `/app` é o código do projecto em tempo real —
qualquer edição no disco externo reflecte imediatamente.

---

## Comandos frequentes

```bash
# Arrancar o ambiente
./start.sh

# Shell de desenvolvimento
docker compose exec dev bash

# Correr a ingestão de dados
docker compose exec dev python run_ingest.py "2024/25"
docker compose exec dev python run_ingest.py          # todas as temporadas

# Ver logs da base de dados
docker compose logs db -f

# Ver logs do container dev
docker compose logs dev -f

# Parar (dados preservados)
./stop.sh

# Parar e remover containers (dados preservados em pgdata/)
docker compose down

# Rebuildar a imagem (após alterações ao Dockerfile ou requirements)
./start.sh --build

# Acesso directo ao PostgreSQL
docker compose exec db psql -U cal_user -d cal

# pgAdmin (interface web para a DB) — porta 5050
docker compose --profile admin up -d
# → abrir http://localhost:5050
# → adicionar servidor: host=db, port=5432, user=cal_user
```

---

## Serviços e portas

| Serviço    | Porta | Quando disponível      |
|------------|-------|------------------------|
| PostgreSQL | 5432  | Sempre (Fase 1+)       |
| FastAPI    | 8000  | Fase 5                 |
| Streamlit  | 8501  | Fase 5                 |
| Jupyter    | 8888  | Sempre (exploração)    |
| pgAdmin    | 5050  | `--profile admin`      |

---

## Mudar de computador

1. **Ligar o disco externo** ao novo computador
2. **Confirmar que Docker Desktop está instalado** e a correr
3. Executar `./start.sh` (ou `start.bat` no Windows)
4. Na primeira vez nesse computador, a imagem é construída (~3–5 min)
5. Os dados da base de dados estão em `pgdata/` — já lá estão, nada a fazer

O schema **não** é re-executado se `pgdata/` já existir. É seguro mudar
de computador sem perder dados.

---

## Iniciar Jupyter Lab

```bash
docker compose exec dev jupyter lab \
    --ip=0.0.0.0 --port=8888 \
    --no-browser --NotebookApp.token=''
```

Abrir em: http://localhost:8888

---

## Resolução de problemas

**"Docker daemon não está a correr"**
→ Iniciar Docker Desktop e aguardar o ícone ficar verde.

**"Port 5432 is already in use"**
→ Outro PostgreSQL está a correr no host. Alterar `DB_PORT=5433` no `.env`.

**Dados parecem estar em falta após mudar de computador**
→ Confirmar que o disco externo está montado e que se está a correr
  `./start.sh` a partir da raiz do disco (`CAL/`).

**Rebuildar a imagem após alterar requirements.txt**
```bash
./start.sh --build
```
