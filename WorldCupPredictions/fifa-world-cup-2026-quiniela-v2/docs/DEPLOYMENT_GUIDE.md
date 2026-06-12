# Deployment Guide

## Local

```bash
git clone <repo-url>
cd fifa-world-cup-2026-quiniela-v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/.env.example .env
make api
```

## Docker

```bash
make docker-build
make docker-up
```

The compose file mounts `data/`, `models/`, `outputs/` and `logs/` from the host so that artifacts persist between container restarts.

## Server

Recommended: a single VM (2 vCPU, 4 GB RAM, 20 GB disk) running:

- Python 3.11 + virtualenv.
- The project deployed under `/opt/wc2026`.
- `systemd` unit running `uvicorn src.api.main:app`.
- Cron schedule for daily `scripts/update_after_matchday.py`.

Example `systemd` unit:

```
[Unit]
Description=World Cup 2026 Quiniela API
After=network.target

[Service]
WorkingDirectory=/opt/wc2026
Environment="PATH=/opt/wc2026/.venv/bin"
ExecStart=/opt/wc2026/.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Environment variables

| Variable                  | Default                       | Description                                 |
| ------------------------- | ----------------------------- | ------------------------------------------- |
| `ENV`                     | `development`                 | Deployment environment label                |
| `LOG_LEVEL`               | `INFO`                        | Logger level                                |
| `FOOTBALL_DATA_API_KEY`   | empty                         | football-data.org API key                   |
| `SIMULATION_RUNS`         | `10000`                       | Monte-Carlo runs                            |
| `API_HOST`                | `0.0.0.0`                     | Bind host                                   |
| `API_PORT`                | `8000`                        | Bind port                                   |
| `CONFIG_PATH`             | `config/config.yaml`          | Main config path                            |
| `FEATURES_CONFIG_PATH`    | `config/features.yaml`        | Features config path                        |
| `MODEL_PARAMS_PATH`       | `config/model_params.yaml`    | Model hyperparameter config                 |
| `STRATEGY_CONFIG_PATH`    | `config/strategy.yaml`        | Strategy config                             |

## Production recommendations

- Run behind a reverse proxy (nginx / Caddy) with TLS.
- Set `API_RELOAD=false` in production.
- Pre-warm artifacts before exposing traffic: bootstrap data, (optionally) `build_squad_values.py`, build ratings, run pipeline, train models, generate picks/scorelines and simulate. On Windows, `run_all.bat` does the whole pre-warm end-to-end.
- Schedule the daily update via cron (`update_after_matchday.py`) or `update_matchday.bat` on Windows.
- Monitor `logs/errors/` and the `/health` endpoint.
