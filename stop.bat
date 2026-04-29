@echo off
cd /d "%~dp0"
echo Parar containers CAL ...
docker compose down
echo Containers parados. Dados preservados em pgdata\
echo Para voltar: start.bat
pause
