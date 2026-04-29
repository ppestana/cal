#!/usr/bin/env python
"""
run_pdf_report.py — Gerar relatório PDF por árbitro/época (Fase 6)

Uso:
    python run_pdf_report.py "Artur Soares Dias" "2023/24"
    python run_pdf_report.py "Luís Godinho" "2022/23"
    python run_pdf_report.py "Jorge de Sousa"        # todas as épocas

O PDF é guardado em /app/reports/ dentro do container.
Para aceder ao ficheiro no Mac:
    docker compose cp dev:/app/reports/<ficheiro>.pdf ~/Downloads/
"""
import sys, logging, dotenv
from rich.logging import RichHandler
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[RichHandler(show_path=False)])

from cal.reports.pdf_report import generate_report

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    print("Uso: python run_pdf_report.py 'Nome Árbitro' ['Época']")
    print("Ex:  python run_pdf_report.py 'Artur Soares Dias' '2023/24'")
    sys.exit(1)

referee = args[0]
season  = args[1] if len(args) > 1 else None

try:
    path = generate_report(referee, season)
    print(f"✓ PDF gerado: {path}")
    print(f"\nPara transferir para o Mac:")
    import os
    filename = os.path.basename(path)
    print(f"  docker compose cp dev:/app/reports/{filename} ~/Downloads/")
except ValueError as e:
    print(f"✗ Erro: {e}")
    sys.exit(1)
