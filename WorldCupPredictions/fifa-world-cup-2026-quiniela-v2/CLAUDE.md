# CLAUDE.md

Guía de contexto para sesiones de Claude Code que trabajen sobre este repositorio. Léela antes de proponer cambios o ejecutar comandos no triviales.

---

## 1. Resumen del proyecto

**Nombre:** FIFA World Cup 2026 Quiniela Predictor V2

**Qué hace:** Sistema end-to-end para predecir partidos del Mundial 2026, simular el torneo completo vía Monte-Carlo, y generar hojas de quiniela en 4 perfiles de riesgo (`safe`, `balanced`, `aggressive`, `contrarian`). Se actualiza dinámicamente jornada a jornada durante el Mundial.

**Para quién:** Un único usuario (DSarceno, dsarceno68@gmail.com) que va a jugar quinielas durante el Mundial 2026.

**Objetivo principal:** **Maximizar el valor esperado en quinielas**, no la accuracy académica. Por eso la capa de probabilidad y la capa de selección de picks están desacopladas — el mismo forecast alimenta los 4 perfiles.

---

## 2. Stack técnico

- **Lenguaje:** Python 3.10 / 3.11 (probado con 3.11 en venv `worldCup`).
- **Datos:** `pandas==2.1.4`, `numpy==1.24.4`, `scipy==1.11.4`.
- **ML:** `scikit-learn==1.4.2`, `xgboost==2.0.3`, `statsmodels==0.14.2`.
- **API:** `fastapi==0.110.0`, `uvicorn==0.29.0`, `pydantic==2.6.4`.
- **HTTP:** `httpx==0.27.0` (fuentes externas), `requests==2.31.0`.
- **Persistencia:** `joblib==1.3.2` para pickles de modelos.
- **Config:** `python-dotenv==1.0.1`, `pyyaml==6.0.1`.
- **Test:** `pytest==8.1.1`, `pytest-cov==4.1.0`.
- **Format:** `black==24.3.0` (line-length 100), `isort==5.13.2` (perfil `black`), `pylint==3.1.0`.
- **Plotting:** `matplotlib==3.8.3` con backend `Agg`.
- **Reportes:** LaTeX (pdflatex). Si no está instalado, el pipeline lo salta sin error.

**Servicios externos:**
- `football-data.org` — fixtures y resultados oficiales del Mundial. Free tier requiere API key en `.env` (`FOOTBALL_DATA_API_KEY`).
- Kaggle dataset `martj42/international-football-results-from-1872-to-2017` — histórico (~46k partidos hasta ~2024). El usuario descarga `results.csv` manualmente.
- StatsBomb Open Data — opcional; si falta, se salta sin error.

---

## 3. Estructura del proyecto

```
fifa-world-cup-2026-quiniela-v2/
├── src/                     # 60 módulos. Layered architecture, deps unidireccionales.
│   ├── utils/               # logging, config, io, dates, metrics, plotting, constants.
│   ├── data/                # ingesta y unificación. Clients: football_data, kaggle, statsbomb, fifa_rankings.
│   ├── ratings/             # Elo, PI, Form, ensemble z-scored.
│   ├── features/            # team/match/market/fatigue/tournament + build_features (entrypoint).
│   ├── models/              # base, multinomial, xgboost, poisson (Dixon-Coles), calibration, factory.
│   ├── ensemble/            # blender (pesos config) + pick_optimizer (perfiles de riesgo).
│   ├── simulation/          # group_stage + VectorizedGroupStageEngine, knockout, bracket, tournament_simulator.
│   ├── prediction/          # predictor, score_predictor, quiniela_strategy, daily_update, feature_builder.
│   ├── training/            # trainer (fit completo), evaluator, backtester, cross_validation.
│   └── api/                 # FastAPI: /health /predict/* /simulate/* /strategy/* /update/* /diagnostics/*.
├── scripts/                 # 10 CLI entrypoints (cada uno con setup_logging + parse_args).
├── tests/unit/              # pytest. Fixtures sintéticas en conftest.py.
├── config/                  # config.yaml, features.yaml, model_params.yaml, strategy.yaml, .env.example.
├── data/
│   ├── raw/                 # snapshots inmutables: football_data/, kaggle/, statsbomb/, manual/, tournament_updates/, fifa_rankings/.
│   ├── interim/             # matches_unified.csv (tabla canónica, única fuente de verdad).
│   └── processed/           # feature_matrix.csv.
├── models/                  # *.pkl: rating_ensemble, multinomial, xgboost, poisson, calibrator, feature_columns.
├── outputs/                 # predictions/, picks/, simulations/, diagnostics/.
├── logs/                    # rotating: pipeline/, training/, prediction/, updates/, errors/.
├── reports/                 # academic/ (9 secciones + main.tex), dashboard/ (3 secciones + main.tex).
├── docs/                    # 11 .md de documentación + diagramas Mermaid.
├── docker/                  # Dockerfile + docker-compose.yml.
├── notebooks/               # 4 notebooks de exploración/diagnóstico.
├── run_all.bat              # Pipeline completo end-to-end para Windows.
├── Makefile                 # Targets equivalentes para Unix-likes.
├── pyproject.toml           # black/isort/pylint/pytest config.
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
└── REPOSITORY_COMPLETENESS_REPORT.md
```

**Dependencia de capas (estricta, unidireccional):**
```
utils → data → ratings → features → models → ensemble → simulation/prediction/training → api
```
Cualquier import que rompa esta dirección es un *code smell* — discútelo antes de mergear.

---

## 4. Flujo de desarrollo

### Instalación

```powershell
cd fifa-world-cup-2026-quiniela-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# o: source .venv/bin/activate    # Unix
pip install -r requirements.txt
cp config/.env.example .env       # editar FOOTBALL_DATA_API_KEY si la tienes
```

### Pipeline completo end-to-end

```powershell
run_all.bat                       # Windows, hace todo + reportes LaTeX
make pipeline && make train && make predict && make picks && make simulate    # Unix-likes
```

### Comandos individuales (orden esperado)

```powershell
python scripts/import_kaggle_history.py --csv data/raw/kaggle/results.csv --start-year 2014 --replace
python scripts/bootstrap_historical_data.py --competitions WC --start-year 2026 --end-year 2026 --no-kaggle
python scripts/build_ratings.py
python scripts/run_pipeline.py            # idempotente: NO recolecta, solo features
python scripts/train_models.py
python scripts/predict_group_stage.py
python scripts/export_quiniela_sheet.py
python scripts/simulate_tournament.py --n-runs 2000
```

### Tests

```powershell
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Lint / Format

```powershell
black src tests scripts
isort src tests scripts
pylint src tests
```

### API local

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger: http://localhost:8000/docs
```

### Workflow diario durante el Mundial

```powershell
python scripts/update_after_matchday.py --date 2026-06-15
python scripts/build_ratings.py
python scripts/run_pipeline.py
python scripts/train_models.py
python scripts/predict_group_stage.py
python scripts/export_quiniela_sheet.py
python scripts/simulate_tournament.py --n-runs 2000
```

---

## 5. Convenciones de código

- **Type hints obligatorios** en 100% de funciones y métodos públicos (`from __future__ import annotations` está activo).
- **Docstrings Google-style** en todas las clases y funciones públicas (`Args:`, `Returns:`, `Raises:`).
- **Logging vía `src.utils.logging_config.get_logger(__name__)`** — nunca `print()` en código de producción. Los scripts hacen `setup_logging(log_file="logs/<area>/<name>.log")` al inicio de `main()`.
- **Manejo de excepciones específico**: nunca `except Exception` mudo; siempre `except (httpx.HTTPError, ValueError) as exc: logger.error(...)`.
- **Imports ordenados**: stdlib → third-party → local (`from src.*`). `isort` configurado con perfil black.
- **Black + line-length 100**.
- **Nombres**: módulos `snake_case`, clases `PascalCase`, funciones/vars `snake_case`, constantes `UPPER_SNAKE`.
- **Dataclasses** para containers ligeros (`@dataclass` o `@dataclass(frozen=True)` cuando aplica).
- **Patrón Strategy/Factory** en `src/models/model_factory.py` y `src/ensemble/pick_optimizer.py`.
- **Configuración por capas**: `config.yaml` (general) → `features.yaml` / `model_params.yaml` / `strategy.yaml` (específicas). Acceso vía `Config.get("dotted.path", default)`.
- **Variables de entorno** en `.env` (gitignored), template en `config/.env.example`.

---

## 6. Reglas importantes

### SIEMPRE

- **Verificar paths antes de leer/escribir** con `ensure_dir()` (en `src/utils/io.py`).
- **Filtrar partidos "por jugar" con `outcome.isna()`**, NO con `score_a.isna()`. `score_a` puede haber sido tocado por `_fillna_numeric` aunque ahora esté protegido.
- **Usar el cliente de logging del proyecto** (`get_logger(__name__)`), no `logging.getLogger()` directo.
- **Persistir snapshots crudos en `data/raw/<source>/`** con timestamp UTC antes de cualquier transformación.
- **Tratar `data/interim/matches_unified.csv` como única fuente de verdad** para partidos. Todo lo demás se deriva.
- **Mergear con la tabla canónica existente al ingerir nuevos datos** (`merge_with_existing=True` por default en `ResultsCollector`). El dedup es por `(date, team_a, team_b)`, `keep="last"`.
- **Usar el `VectorizedGroupStageEngine` y `cache_predictions=True` en el simulador** para corridas Monte-Carlo. El scalar `simulate_group_stage` queda solo para tests/API.
- **Type hints + docstrings** en TODO código nuevo.

### NUNCA

- **NUNCA usar `lambda` como `default_factory` de `defaultdict`** si el objeto se va a serializar con `joblib`/`pickle`. Rompe el dump. Usa `__getstate__` / `__setstate__` para convertir a dict plano al picklear (ver `src/ratings/elo.py`, `pi_rating.py`, `form_rating.py`).
- **NUNCA hacer `out["col"] = raw["col"]` después de un `dropna` sin `reset_index(drop=True)`** sobre `raw`. Causa alineación de índice silenciosa: la mayoría de filas quedan NaN. Patrón correcto: `raw = raw.dropna(...).reset_index(drop=True)` ANTES de construir `out`.
- **NUNCA tratar `score_a` / `score_b` / `match_id` como features**. Son metadata. `_fillna_numeric` en `src/features/build_features.py` los excluye explícitamente (`_METADATA_NUMERIC_COLUMNS`).
- **NUNCA hacer que `run_pipeline.py` recolecte datos por default**. Es idempotente: solo lee canónico + construye features. La recolección se hace en `bootstrap_historical_data.py` o `update_after_matchday.py`.
- **NUNCA llamar a `predict_fn(a, b)` dentro del hot loop del simulador**. Pasa por la cache (`TournamentSimulator(cache_predictions=True)`). 200x-1500x diferencia.
- **NUNCA romper la dirección de dependencias entre capas** (ver §3).
- **NUNCA commitear `.env`**, `data/raw/`, `data/interim/`, `data/processed/`, `models/*.pkl`, `outputs/`, `logs/`. Están en `.gitignore`.
- **NUNCA usar `competition="group"` o `stage="group"` como filtro de "fase de grupos del Mundial"**. El Kaggle client etiqueta histórico con `stage="historical"` precisamente para evitar esa contaminación.
- **NUNCA cambiar `CANONICAL_COLUMNS` sin actualizar también** `src/data/data_loader.py` (default columns), `src/data/kaggle_results_client.py`, `_normalize_football`, `_normalize_statsbomb` y los scripts que ingieren.

---

## 7. Áreas delicadas

| Archivo / módulo | Por qué hay que tener cuidado |
|---|---|
| `src/ratings/{elo,pi_rating,form_rating}.py` | Usan `defaultdict(lambda: ...)`. Implementan `__getstate__/__setstate__` para que joblib funcione. Si tocas el constructor o el estado interno, REVISA el pickling con un test. |
| `src/data/kaggle_results_client.py` | Bug histórico de alineación de índice: el `raw.reset_index(drop=True)` antes de construir `out` es **crítico**. No lo quites. |
| `src/features/build_features.py` (`_fillna_numeric`) | Excluye `score_a`/`score_b`/`match_id` del fill. Si agregas otra columna metadata numérica, sumala a `_METADATA_NUMERIC_COLUMNS`. |
| `src/simulation/tournament_simulator.py` | El cache se construye en `__init__`. Costoso (45s para 48 equipos). Reusa el simulador si vas a correr varios `n_runs` distintos. |
| `src/simulation/group_stage.py::VectorizedGroupStageEngine` | El sort por grupos usa `np.lexsort` con orden de keys `(noise, gf, gd, points)` y `[::-1]`. Cambiar el orden rompe los tiebreaks FIFA. |
| `src/simulation/knockout.py` | Tiene manejo de "bye" para rondas con número impar de equipos. Si cambias el flujo, mantén ese fallback o el simulador truena en rondas mal calibradas. |
| `src/prediction/predictor.py::predict_proba` | Salta multinomial/XGBoost cuando faltan feature columns. Esto permite que la API y el simulador funcionen con solo `(team_a, team_b)`. No conviertas esto en error duro. |
| `scripts/run_pipeline.py` | Es idempotente — no recolecta a menos que pases `--collect`. Esto es intencional para no sobreescribir `matches_unified.csv` accidentalmente. |
| `scripts/simulate_tournament.py::_select_fixtures` | Filtra estrictamente por `competition=WC ∧ year=2026 ∧ outcome.isna() ∧ (stage~"GROUP" ∨ group≠"")`. Cualquier cambio aquí puede meter partidos históricos en la simulación. |
| `src/api/main.py` | Carga modelos de disco vía `BaseOutcomeModel.load(...)` con `try/except FileNotFoundError`. Si los pkl no existen, los endpoints siguen respondiendo (degradado a ratings + Poisson). No hagas esto fallar duro. |
| Tabla canónica `matches_unified.csv` | Única fuente de verdad. Si la corrompes (formato, columnas faltantes, fechas no parseables) **todo el pipeline downstream falla**. Valida con `validate_match_dataframe` antes de persistir. |

---

## 8. Testing y validación

### Tests unitarios

```powershell
pytest tests/ -v
```

Cubre: ratings, features, models, simulation, strategy. Fixtures sintéticas en `tests/conftest.py` (no necesitan datos reales).

### Smoke check completo

```powershell
run_all.bat
```

Si termina con `PIPELINE COMPLETADO EXITOSAMENTE`, el end-to-end funciona.

### Validaciones rápidas post-cambio

```powershell
REM Tabla canónica
python -c "import pandas as pd; df = pd.read_csv('data/interim/matches_unified.csv'); print('Rows:', len(df), 'Cols:', list(df.columns))"

REM Feature matrix
python -c "import pandas as pd; df = pd.read_csv('data/processed/feature_matrix.csv'); print('Rows:', len(df), 'outcome NaN:', df['outcome'].isna().sum())"

REM Sanity de outputs simulator
python -c "import pandas as pd; df = pd.read_csv('outputs/simulations/championship_probabilities.csv'); print(df.head(10)); print('Suma:', df['championship_prob'].sum())"
```

### Validaciones de invariantes

- Probabilidades `[p_home, p_draw, p_away]` deben sumar 1.0 (tolerancia 1e-6).
- `championship_probs["championship_prob"].sum()` ≈ 1.0.
- Standings: cada grupo tiene `len(teams) == 4` y todas las filas con `matches == 3`.
- Picks: cada quiniela tiene N filas (una por fixture), valores en `{"H","D","A"}`.

### Si tocaste el simulador

Asegúrate de que `pytest tests/unit/test_simulation.py -v` pase. Hace un dry-run con grupos sintéticos y probas uniformes.

### Si tocaste un modelo

Asegúrate de que `pytest tests/unit/test_models.py -v` pase. Verifica que `predict_proba` sume 1 row-wise.

---

## 9. Contexto para futuras actualizaciones

### Agregar un nuevo modelo de outcome

1. Crea `src/models/<tu_modelo>.py` heredando de `BaseOutcomeModel`.
2. Implementa `fit(X, y)` y `predict_proba(X)`.
3. Registra en `src/models/model_factory.py::build_model()`.
4. Agrega hyperparams en `config/model_params.yaml`.
5. Engancha en `src/training/trainer.py::Trainer.fit()`.
6. Engancha en `src/prediction/predictor.py::predict_proba()` (con la verificación de feature columns).
7. Agrega un peso en `src/ensemble/blender.py::BlendWeights`.
8. Escribe un test en `tests/unit/test_models.py`.

### Agregar una nueva feature

1. Si es **team-level**: edita `src/features/team_features.py`.
2. Si es **match-context**: edita `src/features/match_features.py`.
3. Si es **derivada de la diff entre equipos**: edita `src/features/build_features.py::_add_strategy_features` o agrega un nuevo paso ahí.
4. Agrega el nombre a `NUMERIC_FEATURE_COLUMNS` en `src/features/build_features.py` para que `select_feature_columns` la detecte.
5. Documenta en `docs/DATA_DICTIONARY.md`.

### Agregar una nueva fuente de datos

1. Crea `src/data/<tu_cliente>.py` con `is_available()` y `load(...)`.
2. Registra en `src/data/sources.py::SOURCES`.
3. Acepta como parámetro opcional en `ResultsCollector.__init__`.
4. Agrega su `_normalize_<source>` privado que devuelve `CANONICAL_COLUMNS`.
5. Llámalo desde `collect_all()`.
6. Documenta en `docs/USER_GUIDE.md` y `config/config.yaml`.

### Agregar un perfil de riesgo de quiniela

1. Edita `src/ensemble/pick_optimizer.py::DEFAULT_PROFILES`.
2. Agrega la rama lógica en `PickOptimizer.pick()`.
3. Suma el nombre a `src/utils/constants.py::RISK_PROFILES`.
4. Agrega config en `config/strategy.yaml`.
5. `export_quiniela_sheet.py` lo recoge automáticamente porque itera `RISK_PROFILES`.

### Fixing bugs

1. **Reproduce primero**: corre el comando que falla, captura el traceback completo.
2. **Lee `logs/`** del área afectada.
3. **Valida la tabla canónica** y la feature matrix antes de tocar lógica.
4. **Si el bug es en `_fillna_numeric` / alineación de pandas / pickling**: revisa §7 antes de proponer un fix nuevo, probablemente ya hubo un fix histórico.

### Refactors

- Mantén el contrato de los `__init__.py` (exports públicos). Si rompes un export, el código que importa desde el paquete truena silenciosamente.
- Si reorganizas un módulo, verifica imports en `src/api/main.py`, `scripts/*.py` y `tests/unit/*.py`.
- Verifica que `run_all.bat` siga corriendo end-to-end.

### Por confirmar

- Si el usuario tiene plan de pago de football-data.org (free tier limita el histórico del Mundial).
- Si Kaggle `results.csv` se está versionando en algún lado o cada vez se descarga manual (actualmente: manual).
- Si los reportes LaTeX se compilan en CI o solo localmente con `run_all.bat`.

---

## Atajos útiles

- **Logs en vivo**: `Get-Content logs/<area>/<name>.log -Wait` (PowerShell) o `tail -F logs/...` (Unix).
- **Limpiar todo y arrancar de cero**:
  ```powershell
  Remove-Item -Recurse data\interim\*, data\processed\*, models\*.pkl, outputs\*
  run_all.bat
  ```
- **Debug rápido del cache del simulador**: usa `--n-runs 50 --no-cache` para iterar lógica sin esperar el warmup.
- **Ver qué features importan más**: `outputs/diagnostics/feature_importance.csv` (top 20 features del XGBoost).
