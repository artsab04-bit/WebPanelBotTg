# Hyper Collision Support Panel

## Запуск backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Запуск frontend

```bash
cd frontend
npm install
npm run dev
```

## Что реализовано

- FastAPI backend с SQLite (WAL), JWT auth (Telegram login hash verify), CRUD API.
- WebSocket `/ws` для realtime-событий.
- Импорт архивных тикетов из `ticket_*.txt`.
- Дашборд и экспорт аналитики CSV/XLSX.
- SPA-заготовка React/Vite на русском интерфейсе с 3-колоночным layout.
