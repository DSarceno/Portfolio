# CLAUDE.md

Guía de contexto para sesiones de Claude Code que trabajen sobre este repositorio. Léela antes de proponer cambios o ejecutar comandos no triviales.

---

## Content Generation Context

When generating content from this project:

- Prioritize mathematical rigor over marketing language.
- Explain the theory behind every major algorithm.
- Highlight engineering trade-offs.
- Connect implementation decisions with scientific principles.
- Write as a Data Scientist with a Physics background.
- Avoid generic AI content.
- Emphasize reproducibility and experimentation.
- Include practical MLOps lessons whenever applicable.
- Focus on what was learned, not only what was built.

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
│   ├── ratings/             # Elo, PI, Form, ensemble z-scored (decaimiento temporal opcional).
│   ├── features/            # team/match/market/fatigue/tournament + build_features (entrypoint).
│   ├── models/              # base, multinomial, xgboost, poisson (Skellam para H/D/A + Dixon-Coles para scorelines), calibration, factory.
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
├── notebooks/               # 4 notebooks de análisis (portafolio); estilo compartido en nb_style.py.
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
python scripts/predict_scorelines.py
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
python scripts/predict_scorelines.py
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
- **NUNCA confundir la orientación de la matriz Poisson en `outcome_probabilities`**. `matrix[i, j]` = `P(team_a marca i, team_b marca j)`. Por lo tanto **team_a gana cuando `i > j`** (`np.tril(matrix, k=-1)`); **team_b gana cuando `j > i`** (`np.triu(matrix, k=1)`). El camino default ahora es Skellam (`scipy.stats.skellam`) — más limpio. Si tocas el camino de la matriz, mantén la orientación. Bug histórico: estaban invertidos y arrastraban toda la simulación a equipos equivocados.
- **NUNCA usar `setdefault` en `_record_round`** de `src/simulation/knockout.py`. Tiene que ser asignación directa (`rounds_reached[t] = label`) porque los equipos avanzan y la etiqueta se sobreescribe ronda a ronda. Bug histórico: `setdefault` dejaba a TODOS los equipos en `"r32"` para siempre.

---

## 7. Áreas delicadas

| Archivo / módulo | Por qué hay que tener cuidado |
|---|---|
| `src/ratings/{elo,pi_rating,form_rating}.py` | Usan `defaultdict(lambda: ...)`. Implementan `__getstate__/__setstate__` para que joblib funcione. Si tocas el constructor o el estado interno, REVISA el pickling con un test. |
| `src/data/kaggle_results_client.py` | Bug histórico de alineación de índice: el `raw.reset_index(drop=True)` antes de construir `out` es **crítico**. No lo quites. |
| `src/features/build_features.py` (`_fillna_numeric`) | Excluye `score_a`/`score_b`/`match_id` del fill. Si agregas otra columna metadata numérica, sumala a `_METADATA_NUMERIC_COLUMNS`. |
| `src/simulation/tournament_simulator.py` | El cache se construye en `__init__`. Costoso (45s para 48 equipos). Reusa el simulador si vas a correr varios `n_runs` distintos. |
| `src/simulation/group_stage.py::VectorizedGroupStageEngine` | El sort por grupos usa `np.lexsort` con orden de keys `(noise, gf, gd, points)` y `[::-1]`. Cambiar el orden rompe los tiebreaks FIFA. |
| `src/simulation/knockout.py` | Tiene manejo de "bye" para rondas con número impar de equipos. Si cambias el flujo, mantén ese fallback o el simulador truena en rondas mal calibradas. **`_record_round` usa asignación directa (NO `setdefault`)** — los equipos sobrescriben su última ronda alcanzada al avanzar. |
| `src/models/poisson_model.py::outcome_probabilities` | Orientación crítica de la matriz scoreline (ver §6 NUNCA). El default es Skellam; el camino DC se conserva para diagnóstico. El `fit()` aplica peso `exp(-xi * days_since)` cuando `time_decay_xi > 0` y existe columna `date` en el input. Si llamas `fit()` sin `date`, el decay queda desactivado silenciosamente. |
| `src/training/trainer.py::_lasso_select_features` | Antes de entrenar XGB/multinomial corre LASSO multinomial (L1) sobre el pool de features y poda las que no aportan. Si LASSO falla o deja menos de `min_features=5`, hace fallback. Logguea siempre los features descartados — esa es la diagnostica clave. |
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

## 10. Upgrade de modelado (junio 2026)

Este lote de cambios viene de un análisis comparativo contra la literatura académica (Dixon-Coles 1997, Karlis-Ntzoufras 2009, Groll-Schauberger-Tutz 2015, Groll-Ley 2019). Se aplicó después de detectar resultados anti-realistas (Paraguay/Panama/Iraq con probabilidad de campeonato superior a Spain/Brazil) trazados a **dos bugs** + **señal comprimida** en las potencias UEFA/CONMEBOL.

### Bugs corregidos

1. **`outcome_probabilities` con tri-up/tri-low invertidos** → Poisson votaba al rival equivocado con peso efectivo ~50% en la simulación. Ver §6 (NUNCA) y §7 entrada de `poisson_model.py`.
2. **`_record_round` usando `setdefault`** → `round_reached_probabilities.csv` solo tenía `r32`. Ver §6 (NUNCA) y §7 entrada de `knockout.py`.

### Mejoras de modelado introducidas

| Cambio | Archivo | Mecánica | Default |
|---|---|---|---|
| **K factors recalibrados** | `src/ratings/elo.py::EloConfig` | `k_friendly: 18 → 8`, `k_world_cup: 60 → 80`. Amistosos pesan menos; el Mundial pesa más. | activado |
| **Decaimiento temporal** | `src/ratings/{elo,pi_rating}.py`, `src/models/poisson_model.py` | Peso `exp(-xi * days_to_latest)` aplicado al delta Elo / error PI / promedios Poisson. Replica Dixon-Coles 1997. | `xi_elo=0.0015`, `xi_pi=0.0015`, `xi_poisson=0.0020` (configurable en `config.yaml::ratings.time_decay`) |
| **Skellam para H/D/A** | `src/models/poisson_model.py::outcome_probabilities` | Reemplaza la integración del grid de scorelines + corrección τ. Calibra mejor empates (Karlis & Ntzoufras 2009). El grid Dixon-Coles se mantiene para *exact scoreline* en quinielas. | `use_skellam: true` en `config/model_params.yaml::models.poisson.hyperparameters` |
| **LASSO para selección de features** | `src/training/trainer.py::_lasso_select_features` | L1-multinomial sobre el pool de ~25 features engineered; descarta las que no aportan señal independiente (Groll-Schauberger-Tutz 2015). Floor en 5 features para evitar sobrelimpieza. | `lasso_enabled: true`, `C=0.1` en `config.yaml::training.lasso` |

### Configuración nueva

```yaml
# config/config.yaml
training:
  lasso:
    enabled: true
    C: 0.1                  # baja C ⇒ más poda; sube ⇒ conserva más

ratings:
  time_decay:
    enabled: true
    xi_elo: 0.0015          # half-life ~462 días
    xi_pi: 0.0015
    xi_poisson: 0.0020      # half-life ~347 días

# config/model_params.yaml
models:
  poisson:
    hyperparameters:
      use_skellam: true
      time_decay_xi: 0.0020
```

### Verificaciones rápidas post-cambio

```powershell
REM Confirmar que LASSO podó features
findstr /C:"LASSO kept" logs\training\train_models.log

REM Confirmar que el decay está activo
findstr /C:"xi=" logs\pipeline\build_ratings.log

REM Confirmar Skellam
findstr /C:"skellam=True" logs\training\train_models.log

REM Sanity: round_reached con todas las etapas (no solo r32)
python -c "import pandas as pd; df = pd.read_csv('outputs/simulations/round_reached_probabilities.csv'); print(df['stage'].unique())"
```

### Cómo revertir cambios individualmente

- LASSO: `training.lasso.enabled: false` (training se hace sobre todo el pool).
- Decay temporal: `ratings.time_decay.enabled: false` (xi=0 para todos los ratings).
- Skellam: `models.poisson.hyperparameters.use_skellam: false` (vuelve al grid + τ).
- K factors: editar `EloConfig` (no expuesto en YAML — se cambia en código).

### Compatibilidad

- **No agrega dependencias**. `scipy.stats.skellam` ya estaba en `scipy==1.11.4`.
- **No cambia CLI** de ningún script. `run_all.bat` funciona sin tocar nada.
- **No cambia firma pickle**: los pkls anteriores se invalidan al regenerar el pipeline, pero la estructura es backward-compatible vía defaults en `__init__`.

---

## 10.bis Segundo lote de modelado (junio 2026)

Iteración para "mejorar la predicción" tuneando config. Hallazgo central: **no se puede
tunear a ojo; hay que medir con el backtest**. Se construyó el árbitro (A.8) y se usó para
aceptar/rechazar cada cambio.

### Bug corregido: `ensemble.weights` era config muerta

`model_params.yaml::ensemble.weights` **no se leía en ningún lado**. `MatchPredictor` se
construía sin `blender`, cayendo siempre en los defaults hardcodeados de
`BlendWeights` (0.25/0.20/0.30/0.25), tanto en `predict_group_stage.py`, `simulate_tournament.py`
como en la API. Cambiar el YAML no tenía efecto.

- **Fix:** `BlendWeights.from_model_params()` / `.from_config()` en `src/ensemble/blender.py`
  (recuerda: `Config.get()` solo lee `config.yaml`/`self.main`; los pesos viven en
  `model_params` → hay que acceder vía `config.model_params`). Cableado en los **4 call sites**:
  `predict_group_stage.py`, `simulate_tournament.py`, `api/main.py` y `export_quiniela_sheet.py`
  (este último se cableó en junio 2026 al integrar A.2; antes seguía con pesos default).
  Test: `tests/unit/test_ensemble.py`.

### Cambios de config evaluados con el backtest (2018+2022)

| Cambio | Veredicto | Estado |
|---|---|---|
| `ensemble.weights` ratings 0.25→0.35, poisson 0.25→0.15 | **neutro** en log-loss (mejora la lista de campeones, inocuo en métrica) | aplicado |
| `ratings.shrinkage.k_poisson` 20→35 | neutro en log-loss | aplicado |
| **A.4 mixture prior** (élite/regular) | **net-negativo** (log-loss +2.3%, acc −2.4pts) | implementado pero **`enabled: false`** |
| **A.8 backtester cross-tournament** | el árbitro que reveló lo anterior | implementado |

**Moraleja para futuras sesiones:** la lista de `championship_probabilities.csv` es engañosa
(una potencia que sobre-rinde como Morocco sale #1 por mérito de resultados; Brazil sale bajo
por su mal tramo reciente). Eso **no es un bug** sino el límite de un rating basado solo en
resultados. La palanca real de mejora ya no es tunear knobs (el backtest los muestra ~óptimos)
sino **añadir señal ortogonal de talento → A.2 (valor de plantilla / Transfermarkt)**.

---

## Anexo A — Roadmap de mejoras pendientes

Cambios identificados en el análisis de literatura que **NO se aplicaron todavía**, en orden de impacto esperado / esfuerzo. Útil para próximas iteraciones.

### A.1 Consenso de casas de apuestas como prior — **🔥🔥🔥 impacto / 🟡 esfuerzo medio**

La conclusión más unánime de la literatura (Leitner-Zeileis-Hornik 2010, Groll 2026, todos los rankings de competiciones de forecasting): **agregar odds de ≥10 casas de apuestas, despojarlas del overround, promediar en escala logit** produce un benchmark que casi nadie supera por más de 1-2% de log-loss.

**Cómo entrarle:**
1. Scraper o API que tome 1X2 odds de Pinnacle / Bet365 / Bwin / etc. para cada fixture WC 2026.
2. `src/data/bookmaker_client.py` con `fetch_odds(fixture_id) -> dict[bookie, [home, draw, away]]`.
3. Para cada fixture: `1/odds_i`, normalizar (quita overround), `logit`, promedio simple, `softmax` → `(p_home, p_draw, p_away)`.
4. Agregar al feature matrix como `bookmaker_p_home / p_draw / p_away` (3 columnas).
5. Agregar como rama extra en `src/ensemble/blender.py::BlendWeights`.

**Cita:** Zeileis 2018 (https://www.zeileis.org/news/fifa2018/), Groll 2026 R-bloggers.

### A.2 Valor de plantilla — **✅ IMPLEMENTADO (junio 2026)**

> **Veredicto del backtest: MEJORA. Activado por default.** A/B sobre 2018+2022:
> log-loss 0.9994 → **0.9905**, Brier 0.1976 → 0.1955, RPS 0.2117 → 0.2082 (accuracy
> igual). Ambas features sobreviven al LASSO (importancia ~0.05/0.04). Spec:
> `docs/superpowers/specs/2026-06-11-a2-squad-value-design.md`.

Covariable #2 tras el Elo (Groll et al.). Captura **talento actual**, que el Elo (resultados) no ve. Implementación:

- **Fuente:** Kaggle `davidcariboo/player-scores` en `data/raw/kaggle/players-scores/`. **Proxy por nacionalidad** (`country_of_citizenship` + `player_valuations` as-of fecha), **mismo método para los 3 snapshots** (2018/2022/2026) → escala consistente train↔predicción (clave para que el modelo transfiera). NO se usa `national_teams.csv` (rompería esa consistencia).
- **`scripts/build_squad_values.py`** → `data/raw/squad_values/squad_values.csv` (snapshots `team, as_of_date, *_total_meur, *_top11_meur, source`). Idempotente; re-correr solo si cambia el dump Kaggle o el override manual. Override opcional: `squad_values_2026_manual.csv`.
- **`src/data/squad_value_client.py`** (`SquadValueClient`, `build_citizenship_snapshot`, `SQUAD_VALUE_NAME_MAP`). Registrado en `sources.py`.
- **`src/features/squad_value_features.py`** — paso a **nivel de partido** (no team-static, porque depende de la fecha): as-of join por fecha del match → `squad_value_total_diff`, `squad_value_top11_diff` (`log1p` home − away); faltantes → mediana de confederación. Enganchado en `build_match_feature_matrix(squad_values=...)`, `run_pipeline.py`, `feature_builder.py`, `backtester.py`. Config: `data.sources.squad_value`.
- **Simulador con modelo completo:** antes el Monte-Carlo corría solo con ratings+Poisson (`predict_single(a,b)` saltaba XGB/multinomial por features ausentes). Ahora `feature_builder.build_pairwise_feature_matrix(teams)` construye fixtures sintéticos neutrales para los `n*(n-1)` pares, los pasa por **el mismo** `build_match_feature_matrix` (squad value as-of + diffs de fuerza + contexto neutro) y los pre-scorea con el **modelo completo blended**; `simulate_tournament.py` hace lookup en el hot loop. Efecto: el campeón ahora refleja talento (Brazil #15→#5, Morocco #2→#10, Spain #1 a 12%). Si tocas el simulador, mantén este pre-scoring — sin él vuelve al modelo degradado.

**Cita:** Groll-Schauberger-Tutz 2015, Groll-Ley 2019.

### A.3 Plus-minus / PageRank rating — **🔥🔥 impacto / 🟡 esfuerzo medio**

El modelo de producción Groll 2026 lo usa; Hubáček 2019 (ganador del Soccer Prediction Challenge) usa PageRank sobre grafo de partidos. **Uncorrelated con Elo** — agrega información ortogonal.

**PageRank (más simple que plus-minus, no requiere data de jugadores):**
1. Construir grafo `team_a → team_b` con peso = goal_margin (o W=3/D=1/L=0).
2. `scipy.sparse` + `networkx.pagerank()` → vector de rankings.
3. Snapshot en `outputs/diagnostics/pagerank.csv`.
4. Sumar al ensemble en `RatingEnsemble` con peso 0.10-0.15.

**Cita:** Hubáček-Šourek-Železný 2019 (Springer ML 108).

### A.4 Mixture prior en shrinkage — **✅ IMPLEMENTADO pero DESACTIVADO (junio 2026)**

> **Veredicto del backtest: NET-NEGATIVO. `mixture_prior.enabled: false` por default.**
> Implementado en `src/ratings/shrinkage.py::RatingShrinker` (sub-priors `elite`/`regular`
> por confederación + `classify_elite` con compuerta `min_matches_for_elite` anti-minnow).
> El A/B con `run_tournament_backtest` (2018+2022) mostró que con los priors élite/regular
> elegidos a mano (1740/1560…) **empeora** log-loss +2.3% y accuracy −2.4pts. Hizo que la
> lista de campeones se *viera* mejor (Spain #1) pero degradó la predicción por partido —
> que es la métrica relevante para quinielas. **No reactivar sin re-tunear los priors con un
> optimizador y re-correr el backtest.** Config: `ratings.shrinkage.mixture_prior.*`.

El shrinkage Gaussiano jala todos los equipos hacia la media de su confederación. Baio-Blangiardo 2010 proponen un **mixture prior** (élite vs resto). La mecánica quedó cableada y testeada (`tests/unit/test_ratings.py`), solo desactivada. Para re-tunear: ajustar los dicts `CONFEDERATION_*_PRIOR_ELITE/REGULAR` y validar contra el backtest antes de prender el flag.

**Cita:** Baio-Blangiardo 2010 (https://discovery.ucl.ac.uk/16040/).

### A.5 Habilitar features StatsBomb — **🔥 impacto / 🔴 esfuerzo alto**

Cliente y data ya están integrados pero **no producen features**. xG agregado por equipo (`xg_for_per90`, `xg_against_per90`) es señal independiente. Aporta marginal en selecciones (poca muestra ~10 partidos/año), pero útil.

**Cómo entrarle:**
1. `src/features/team_features.py` consume `data/raw/statsbomb/*.json` agregado por equipo.
2. Solo selecciones top que aparecen en StatsBomb open data (no todas).
3. Probable bajo aporte tras LASSO; vale validar.

### A.6 Calibrador sobre held-out tournament — **🔥 impacto / 🟢 esfuerzo bajo**

El calibrador isotónico se ajusta sobre `val_features` del temporal_split. Si `validation_year=2025` y XGBoost entrena hasta 2024, no hay leak. Pero conviene validar explícitamente que el calibrador NUNCA ve datos del set de entrenamiento del clasificador base.

**Cómo entrarle:**
1. En `src/training/trainer.py::fit`, asertar `set(train_features.index) ∩ set(val_features.index) == ∅`.
2. Agregar test unitario que reproduzca el split y verifique la disjuntez.

### A.7 Benchmark contra casas de apuestas — **🔥 valor diagnóstico / 🟡 esfuerzo medio**

Reporta `log_loss(modelo)` vs `log_loss(bookmaker_consensus)` sobre el set de validación. La literatura es unánime: **si el modelo no supera al consenso en log-loss, el consenso ES tu modelo**.

**Cómo entrarle:**
1. Dependencia de A.1 (necesitas odds).
2. `src/training/evaluator.py::compare_against_bookmaker(model_proba, bookie_proba, y)`.
3. Salida en `outputs/diagnostics/benchmark_vs_bookmaker.csv` con log-loss por torneo y delta.

### A.8 Backtest cross-tournament — **✅ IMPLEMENTADO (junio 2026)**

`src/training/backtester.py::run_tournament_backtest` + `scripts/backtest_tournaments.py`.
Entrena el **stack completo** (RatingEnsemble + shrinkage + multinomial + XGBoost + Poisson,
blended con los pesos de config) sobre `año < Y` y evalúa en el torneo `== Y`, **sin leakage**:
los ratings y las features de test se derivan solo de datos pre-Y (el composite se ajusta con
`train` y se pasa a `build_match_feature_matrix`). Métricas: log-loss, Brier, **RPS ordinal**,
accuracy, macro-F1, ECE.

```powershell
python scripts/backtest_tournaments.py --years 2018 2022
# -> outputs/diagnostics/backtest_WC_2018_2022.csv
```

**Este es el árbitro.** Cualquier cambio de modelado (pesos, K, mixture, features nuevas) se
valida aquí ANTES de creerle a la lista de campeones. Baseline actual (config de producción):
log-loss ~0.999, accuracy ~0.55 sobre 2018+2022 (uniforme = 1.0986). Para A/B rápido, llamar
`run_tournament_backtest(matches, [2018,2022], shrinker_factory=..., blend_weights=...)`
directo con distintas configs (ver cómo se hizo el A/B de A.4).

### Prioridad sugerida si se retoma

Si retomas el modelo: **A.2 (Transfermarkt via Kaggle player-scores)** primero — es la fuente que más cita la literatura, está en un Kaggle que ya conoces. Luego **A.8 (backtest)** para medir el impacto cuantitativamente. **A.1 (bookmaker consensus)** es el santo grial pero requiere infraestructura externa nueva.

**No retomes A.5 (StatsBomb)** salvo que quieras juguete; el ROI para selecciones es muy bajo.

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
