# A.2 — Valor de plantilla (squad value) · Diseño

**Fecha:** 2026-06-11
**Estado:** aprobado en brainstorming, pendiente de review del spec.
**Objetivo:** añadir la covariable de **valor de plantilla** (la #2 más predictiva tras el Elo
en la literatura, Groll et al. 2015/2019), que captura **talento actual** — algo que un rating
basado en resultados (Elo/PI) no ve cuando una potencia rota plantilla o atraviesa mala forma
(p.ej. Brazil infravalorado, Morocco sobrevalorado en la corrida actual).

El cambio se acepta **solo si mejora** log-loss/Brier/RPS en el backtest 2018+2022 (mismo
árbitro que descartó A.4). La estética de la lista de campeones no decide nada.

---

## 1. Decisiones ya tomadas (brainstorming)

- **Fuente: híbrida.** Kaggle `davidcariboo/player-scores` para el histórico (backtest) +
  valores actuales para 2026 (producción).
- **Features: total + top-11.** `squad_value_total` y `squad_value_top11` (suma de los 11
  jugadores más valiosos). El top-11 es más robusto al ruido del proxy por nacionalidad.
- **Mecanismo: snapshots con fecha + as-of join** (sin leakage, mismo camino para backtest y
  producción).

---

## 2. Datos disponibles (verificado en `data/raw/kaggle/players-scores/`)

| Archivo | Uso |
|---|---|
| `player_valuations.csv` | **Serie temporal** de valor por jugador (`player_id, date, market_value_in_eur`). Base del histórico. |
| `players.csv` | `country_of_citizenship`, `current_national_team_id`, `market_value_in_eur`, `date_of_birth`. Mapeo jugador→selección/nacionalidad. |
| `national_teams.csv` | Snapshot **actual** por selección: `country_name, confederation, total_market_value, average_age, squad_size`. Base del 2026 auto-derivado. |

**Mejora sobre el plan original:** el 2026 se puede **auto-derivar** de `national_teams.csv`
(total) + `players.csv` agrupado por `current_national_team_id` (top-11). El CSV manual deja de
ser obligatorio y pasa a ser una **capa de override** opcional (corregir selecciones donde el
dump esté desactualizado). Esto reduce el trabajo manual del usuario casi a cero.

---

## 3. Esquema canónico de snapshots

`data/raw/squad_values/squad_values.csv` (lo genera el script; **no editar a mano**):

```
team, as_of_date, squad_value_total_meur, squad_value_top11_meur, source
```

- `team`: nombre **exacto** como en `matches_unified.csv`.
- `as_of_date`: validez del snapshot (`YYYY-MM-DD`).
- valores en **millones de EUR**.
- `source`: `kaggle_citizenship` (histórico proxy) | `kaggle_national_team` (2026 derivado) |
  `manual` (override del usuario).

**Override del usuario:** `data/raw/squad_values/squad_values_2026_manual.csv` (la plantilla ya
generada, 48 filas). Si una fila tiene valores, **gana** sobre el derivado para esa
`(team, as_of_date)`.

---

## 4. Componente nuevo: `scripts/build_squad_values.py`

Genera `squad_values.csv` mergeando tres orígenes. CLI:
`python scripts/build_squad_values.py --historical-dates 2018-06-01 2022-06-01 --current-date 2026-06-01`

Pasos:
1. **Histórico (cada fecha `D` en `--historical-dates`)** — proxy por nacionalidad:
   - De `player_valuations`, tomar por jugador el valor más reciente con `date <= D`.
   - Unir con `players` por `player_id` → `country_of_citizenship`.
   - Mapear citizenship → nombre canónico (`SQUAD_VALUE_NAME_MAP`).
   - Por selección: `total` = suma de todos los citizens; `top11` = suma de los 11 mayores.
   - Emitir filas con `source=kaggle_citizenship`.
2. **Actual (`--current-date`)** — convocatoria real (snapshot actual):
   - `total` desde `national_teams.csv::total_market_value`.
   - `top11` desde `players` filtrado por `current_national_team_id`, top-11 por
     `market_value_in_eur`.
   - `source=kaggle_national_team`.
3. **Override manual:** leer `squad_values_2026_manual.csv`; filas con valores no vacíos pisan
   las derivadas en `(team, as_of_date)` con `keep="last"`.
4. Escribir `squad_values.csv` ordenado por `(team, as_of_date)`. Loguear cobertura (cuántas de
   las 48 selecciones quedaron con dato por snapshot).

**Mapeo de nombres** (`SQUAD_VALUE_NAME_MAP`, dict en `src/data/squad_value_client.py`): cubre
las diferencias conocidas (`USA`→`United States`, `Korea, South`→`South Korea`,
`Cote d'Ivoire`→`Ivory Coast`, `Czech Republic`→`Czechia`, `Türkiye`→`Turkey`,
`DR Congo`→`Congo DR`, `Cape Verde`→`Cape Verde Islands`, `Curacao`→`Curaçao`, etc.). Cualquier
selección de las 48 sin match se loguea como warning.

---

## 5. Componente nuevo: `src/data/squad_value_client.py`

`SquadValueClient` con:
- `is_available() -> bool` — existe `squad_values.csv`.
- `load() -> pd.DataFrame` — canónico validado (columnas, fechas parseables, valores ≥ 0).
- `as_of(team, date) -> (total, top11)` — snapshot con `as_of_date` máxima ≤ `date`; `NaN` si
  no hay.

Registrado en `src/data/sources.py::SOURCES["squad_value"]`.

---

## 6. Integración en features (paso a nivel de partido, date-aware)

El valor de plantilla **depende de la fecha del partido**, así que NO va en
`compute_team_features` (estático por equipo). Se añade un paso nuevo:

- **`src/features/squad_value_features.py::compute_squad_value_features(match_features, squad_values)`**
  - Entrada: `match_features` (ya tiene `date, team_a, team_b`) + tabla de snapshots.
  - Para cada partido: as-of join por equipo y fecha → `total_a, top11_a, total_b, top11_b`.
  - Transformar con `log1p` (valores de cola pesada).
  - Emitir `squad_value_total_diff = log1p(total_a) - log1p(total_b)` y
    `squad_value_top11_diff` (idem). También conserva `*_a`/`*_b` logueados por diagnóstico.
  - Faltantes → mediana de la confederación a esa fecha (no 0, para no inventar señal); si la
    confederación tampoco tiene dato, mediana global del snapshot.
- **`build_features.py::build_match_feature_matrix`**: nuevo parámetro opcional
  `squad_values: Optional[pd.DataFrame]`. Si se pasa, llamar el paso nuevo después de
  `compute_match_features` (que ya expone `date`). Sin él, el pipeline funciona igual que hoy
  (degradación limpia).
- **`NUMERIC_FEATURE_COLUMNS`** += `squad_value_total_diff`, `squad_value_top11_diff`.
- **`src/prediction/feature_builder.py::build_inference_feature_matrix`**: cargar
  `SquadValueClient().load()` (si disponible) y pasarlo.
- **`src/training/backtester.py::run_tournament_backtest`**: cargar y pasar el snapshot, para
  que el A/B mida estas features sin leakage (as-of por fecha lo garantiza).

---

## 7. Config y registro

`config/config.yaml`:
```yaml
data:
  sources:
    squad_value:
      enabled: true
      historical_dates: ["2018-06-01", "2022-06-01"]
      current_date: "2026-06-01"
```
`features.yaml`: documentar las 2 features nuevas. Si `enabled: false`, el builder no pasa el
snapshot y las features no se generan (LASSO simplemente no las ve).

---

## 8. Validación (el árbitro)

1. Regenerar `squad_values.csv`.
2. A/B con `run_tournament_backtest` (2018+2022): **con** vs **sin** `squad_values`, comparando
   log-loss/Brier/RPS/accuracy.
3. Inspeccionar `feature_importance.csv` y el log de LASSO (¿sobreviven los diffs?).
4. **Criterio de aceptación:** se quedan activadas solo si mejoran (o al menos no empeoran) el
   log-loss agregado. Si empeoran, se documentan como A.4 (implementadas, `enabled: false`).
5. Sanidad cualitativa secundaria: re-correr el simulador y ver si Brazil/Germany suben y
   Morocco baja hacia valores más realistas — pero esto **no** decide, solo ilustra.

---

## 9. Tests (`tests/unit/`)

- `test_squad_value_client.py`: `load()` valida esquema; `as_of()` toma el snapshot correcto
  (último ≤ fecha; `NaN` antes del primero).
- `test_squad_value_features.py`: as-of join correcto sobre fixture sintética con 2 snapshots;
  `log1p` y diff; fillna por mediana de confederación.
- agregación Kaggle: test del top-11 y del proxy por citizenship sobre un mini-DataFrame.
- El A/B del backtest es la validación de integración (no unit test).

---

## 10. Documentación y `run_all.bat` a actualizar (requisito del usuario)

- `docs/DATA_DICTIONARY.md`: las 2 features nuevas + el esquema de snapshots.
- `docs/USER_GUIDE.md`: nueva fuente + cómo correr `build_squad_values.py` + cómo usar el
  override manual.
- `config/config.yaml` / `config/features.yaml`: ya cubierto arriba.
- `CLAUDE.md`: §3 (estructura: nuevos módulos/carpeta), Anexo A.2 → **implementado**, §10.bis o
  nueva entrada con el resultado del backtest; nota de pipeline.
- **`run_all.bat`**: insertar `python scripts/build_squad_values.py ...` **antes** de
  `run_pipeline.py` (las features lo necesitan). `Makefile`: target equivalente
  `squad-values`.

> El paso de `build_squad_values.py` es **idempotente** y solo se re-corre cuando cambia el
> dump de Kaggle o el override manual; no descarga nada por sí mismo.

---

## 11. Qué provee el usuario

1. ✅ Kaggle `players-scores/` ya está en `data/raw/kaggle/players-scores/`.
2. **(Opcional)** Llenar/corregir `squad_values_2026_manual.csv` solo si quiere overridear los
   valores 2026 auto-derivados de `national_teams.csv`. Ya **no es obligatorio**.

---

## 12. Riesgos / caveats

- **Proxy por nacionalidad (histórico):** incluye ciudadanos no convocados y omite
  nacionalizados. El `top11` mitiga (solo cuenta a los mejores). Documentado; el backtest dirá
  si aun así aporta.
- **`national_teams.csv` es snapshot actual**, no datado: se asume válido para `2026-06-01`. Si
  está desactualizado, el override manual corrige.
- **Cobertura:** selecciones pequeñas (Curaçao, Cape Verde) pueden tener pocos jugadores
  valuados → fillna por confederación. Se loguea cobertura.
- **Mapeo de nombres:** fuente de bugs silenciosos. Warning explícito por cada una de las 48
  que no mapee.

---

## 13. Fuera de alcance (YAGNI)

- mean_age, n_legionnaires, n_ucl_players (descartadas en brainstorming).
- Reconstrucción de convocatorias históricas reales vía `game_lineups.csv` (complejo; el proxy
  + top11 es suficiente para una primera iteración medible).
- Scraping en vivo de Transfermarkt.
