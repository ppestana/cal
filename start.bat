@echo off
:: CAL — Arranque do ambiente de desenvolvimento (Windows)
:: Requer Docker Desktop instalado e a correr.

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  CAL ^— Criticar a Arbitragem Legalmente
echo  ----------------------------------------
echo.

:: Verificar Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo  [ERRO] Docker nao encontrado ou nao esta a correr.
    echo         Iniciar Docker Desktop e tentar novamente.
    pause
    exit /b 1
)

:: Verificar .env
if not exist ".env" (
    if exist ".env.example" (
        echo  [!] .env nao encontrado. A criar a partir de .env.example ...
        copy ".env.example" ".env" >nul
        echo  [!] EDITAR .env e definir DB_PASSWORD antes de continuar.
        echo      notepad .env
        pause
        exit /b 1
    ) else (
        echo  [ERRO] .env e .env.example em falta.
        pause
        exit /b 1
    )
)

:: Criar pgdata se nao existir
if not exist "pgdata" (
    echo  [i] Primeira execucao detectada. A criar pgdata\ ...
    mkdir pgdata
)

:: Criar notebooks se nao existir
if not exist "notebooks" mkdir notebooks

:: Build se imagem nao existir
docker image inspect cal_dev:latest >nul 2>&1
if errorlevel 1 (
    echo  [i] A construir imagem cal_dev (primeira vez - pode demorar) ...
    docker compose up -d --build
) else (
    echo  [i] A iniciar containers ...
    docker compose up -d
)

if errorlevel 1 (
    echo  [ERRO] Falha ao iniciar containers.
    pause
    exit /b 1
)

:: Aguardar DB
echo  [i] A aguardar PostgreSQL ...
set /a COUNT=0
:wait_loop
docker compose exec -T db pg_isready -U cal_user -d cal >nul 2>&1
if not errorlevel 1 goto db_ready
set /a COUNT+=1
if %COUNT% geq 30 (
    echo  [ERRO] PostgreSQL nao respondeu.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_loop

:db_ready
echo  [OK] PostgreSQL pronto.
echo.
echo  ----------------------------------------
echo   PostgreSQL  ^> localhost:5432
echo   FastAPI     ^> http://localhost:8000
echo   Streamlit   ^> http://localhost:8501
echo   Jupyter     ^> http://localhost:8888
echo.
echo   Shell de desenvolvimento:
echo   docker compose exec dev bash
echo.
echo   Correr ingestao:
echo   docker compose exec dev python run_ingest.py "2024/25"
echo.
echo   Parar o ambiente:
echo   stop.bat
echo  ----------------------------------------
echo.
pause
