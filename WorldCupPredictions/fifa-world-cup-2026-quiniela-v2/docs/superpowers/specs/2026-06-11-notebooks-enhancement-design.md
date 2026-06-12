# Mejora de notebooks (análisis profundos + presentables) · Diseño

**Fecha:** 2026-06-11
**Estado:** aprobado en brainstorming, pendiente de review del spec.
**Objetivo:** convertir los 4 notebooks de exploración básica en piezas **de portafolio**:
estadísticamente profundas, visualmente pulidas y útiles para decidir quinielas. Deben poder
**presentarse** (renderizadas) y aprovechar lo nuevo del proyecto (valor de plantilla A.2,
marcadores, backtest A.8, calibración).

---

## 1. Decisiones tomadas (brainstorming)

- **Visual:** `matplotlib + seaborn` (estático, renderiza en GitHub/PDF/nbviewer). Se agrega
  `seaborn` a `requirements.txt`.
- **Estructura:** mejorar los **4 notebooks en sitio** (arco: datos → fuerza → modelos →
  estrategia). No consolidar, no agregar un 5º.
- **Limpieza:** eliminar las celdas scratch/rotas del notebook 04 (incluida la del
  `AttributeError` de `poisson.attack` y la de `r32` con datos viejos).
- **Cada notebook** lleva: intro narrativa (propósito, datos, qué mirar) y un bloque
  **"Hallazgos clave"** al cierre.

---

## 2. Infraestructura compartida

### `notebooks/nb_style.py`
Módulo importado al inicio de cada notebook (cwd de notebooks = `notebooks/`, import directo).
Provee:
- `apply_theme()` — `seaborn.set_theme(style="whitegrid", context="notebook")`, paleta y
  `matplotlib` rcParams (fuente, tamaños, dpi, colores de ejes). Tema único y consistente.
- `PALETTE` y `CONFEDERATION_COLORS` (UEFA, CONMEBOL, CONCACAF, AFC, CAF, OFC) — color por
  confederación reusable en rankings/scatters.
- Helpers: `kpi_header(items)` (fila de tarjetas KPI con matplotlib), `annotate_barh(ax, ...)`
  (etiquetas de valor en barras), `finding(text)` (caja markdown de hallazgo). Mínimos y
  testeables a ojo.
- `team_confederation(team)` reutiliza `src.ratings.shrinkage.TEAM_CONFEDERATION` (no duplicar).

### Robustez
Cada celda que dependa de outputs opcionales (`squad_values.csv`, `scoreline_predictions.csv`,
`outputs/simulations/*`) verifica existencia y, si falta, imprime un aviso claro y sigue (no
rompe la ejecución del notebook).

### Ejecución
Al terminar, ejecutar los 4 notebooks (`jupyter nbconvert --to notebook --execute --inplace`)
para dejarlos con gráficos renderizados. Requiere pipeline ya corrido (datos + modelos +
outputs presentes; ya lo están).

---

## 3. Pequeña adición de código reusable (justificada)

Para el diagrama de **calibración out-of-sample** y la **visualización del backtest** sin
duplicar la lógica leakage-safe en el notebook:

- `src/training/backtester.py::collect_holdout_predictions(matches, year, ...) -> (proba, y_true)`
  Entrena el stack completo sobre `< year` y devuelve las probabilidades blended y las
  etiquetas reales del torneo `== year` (un solo fold). Reusa la misma maquinaria que
  `run_tournament_backtest` (ratings+shrinkage+modelos+blend, as-of features). El notebook 03 la
  usa para (a) la reliability curve y (b) métricas vs baseline. Test unitario mínimo: forma y
  rango de la salida sobre datos sintéticos o un año real.

Es la **única** adición de código fuera de `notebooks/`. Todo lo demás vive en los notebooks.

---

## 4. Contenido por notebook

> Convención común: intro markdown → secciones con narrativa breve → "Hallazgos clave".
> Lo marcado **(nuevo)** no existe hoy.

### 01 — Datos y panorama del torneo
1. **KPI header (nuevo):** n partidos, rango de fechas, n equipos, n competencias, n fixtures WC2026.
2. Partidos en el tiempo: línea/área estilizada con años de Mundial anotados.
3. Composición por competencia: barh con % y conteo.
4. **Análisis de goles (nuevo):** distribución de goles por partido; goles local vs visitante;
   **ventaja de local y su tendencia** por año (¿declina?); tasa de empate por año.
5. **Grupos de la muerte (nuevo):** por grupo WC2026, dificultad = fuerza compuesta agregada
   (media y suma top-2, ya que clasifican 2). Ranking de grupos + tabla de integrantes.
6. Fiabilidad de ratings: partidos históricos por equipo WC2026 (cobertura → ratings inestables).
7. Hallazgos clave.

### 02 — Modelos de fuerza de equipos
1. Ajuste de Elo/PI/Form (resumen).
2. Distribuciones con **KDE** (seaborn) de los tres ratings.
3. Correlación entre ratings: heatmap seaborn anotado.
4. **Valor de plantilla vs Elo (nuevo):** scatter de los 48 (squad value 2026 top-11 vs Elo),
   recta de regresión, equipos etiquetados, resaltar outliers (Brazil sub-valorado por Elo,
   Morocco sobre-valorado). **La historia central del proyecto, visualizada.**
5. **Fuerza por confederación (nuevo):** boxplots de composite por confederación.
6. Ranking de los 48 (barh coloreado por `CONFEDERATION_COLORS`).
7. **Mayores desacuerdos entre sistemas (nuevo):** tabla de equipos con mayor |z(Elo) − z(PI)|
   o ranking dispar (señal de incertidumbre).
8. Hallazgos clave.

### 03 — Modelos de pronóstico y calibración
1. Carga de modelos (resumen).
2. Feature importance **agrupada por categoría** (fuerza / contexto / squad value), barras
   coloreadas por grupo.
3. **Acuerdo entre modelos (nuevo):** sobre todas las fixtures, correlación/scatter de las
   probabilidades de XGB vs Multinomial vs Poisson vs ratings (¿son ortogonales o redundantes?).
4. **Diagrama de calibración / reliability (nuevo):** vía `collect_holdout_predictions` (p.ej.
   held-out 2022), binear prob predicha vs frecuencia observada, recta diagonal ideal, reportar
   ECE y Brier. **Rigor estadístico clave.**
5. **Backtest visual (nuevo):** barras de log-loss/Brier/RPS del modelo vs baseline uniforme
   (`ln 3 ≈ 1.0986`) y el A/B con/sin valor de plantilla (resultado ya conocido: 0.9994→0.9905).
6. Scoreline grid Poisson (heatmap pulido) + top-k marcadores.
7. Deep-dive de un partido: comparación por modelo (usando **pesos de config**, no default).
8. Hallazgos clave.

### 04 — Simulación y estrategia de quiniela
0. **Borrar celdas scratch** (extras marcadas).
1. Favoritos al título: barh coloreado por confederación (modelo completo, ya refleja talento).
2. Probabilidad de llegar a cada ronda: heatmap pulido (verificar que `round_reached` traiga
   todas las etapas, no solo `r32`).
3. Clasificación a octavos: top/bottom estilizado.
4. **Valor Esperado / ROI de los 4 perfiles (nuevo):** bajo las reglas de `strategy.yaml`
   (`correct_1x2`, `upset_bonus`, …), EV por pick = Σ P(outcome)·puntos(pick,outcome); total y
   **vista riesgo-retorno** (EV vs desviación de puntos) de los 4 perfiles. Sirve directo a
   "maximizar valor esperado". (1X2 como núcleo; bonus de marcador exacto como extensión si es
   barato.)
5. Divergencia entre perfiles: partidos donde difieren (mayor valor estratégico), pulido.
6. **Distribución de marcadores más probables (nuevo):** del `scoreline_predictions.csv` —
   frecuencia de cada marcador modal, cuántos partidos modan a 1-1/0-0, etc.
7. Finales más frecuentes (campeón/subcampeón) pulido.
8. Hallazgos clave.

---

## 5. Métodos estadísticos (notas de implementación)

- **Calibración:** held-out temporal real (train `< year`, test `== year`) vía
  `collect_holdout_predictions` → sin leakage. Bins de ~10; ECE = Σ (n_b/N)·|conf_b − acc_b|.
- **EV de perfiles:** por pick, `EV = p_pick·correct_1x2 (+ upset_bonus·p_pick si el pick es
  underdog)`. Riesgo = desviación estándar de la variable de puntos por pick. Comparar los 4.
- **Dificultad de grupo:** `mean(composite)` y `sum(top2 composite)` por grupo; ranking.
- **Squad value vs Elo:** ambos z-scoreados; recta OLS; residual grande = "infra/ sobre-valorado".
- **Desacuerdo entre modelos:** correlación de Pearson entre vectores de prob por outcome sobre
  el set de fixtures.

---

## 6. Fuera de alcance (YAGNI)

- SHAP / explicabilidad avanzada (feature importance de XGB basta para v1).
- Interactividad (plotly) — descartado en brainstorming.
- Reescribir los reportes LaTeX de `reports/` (otra entrega).
- Nuevos modelos o cambios de modelado (esto es solo análisis/visualización).

---

## 7. Validación / criterios de calidad

- Los 4 notebooks **ejecutan de principio a fin sin error** (`nbconvert --execute`).
- Estilo consistente vía `nb_style` (un solo tema).
- Cada notebook tiene intro + "Hallazgos clave".
- Cada gráfico: título, ejes etiquetados, fuente/unidades claras, anotaciones donde aporten.
- Degradación limpia si falta un output opcional.
- `pytest tests/unit/ -q` sigue verde (por la adición en `backtester.py` + su test).
- `requirements.txt` incluye `seaborn`.

---

## 8. Archivos tocados

- **Nuevos:** `notebooks/nb_style.py`, test para `collect_holdout_predictions`.
- **Modificados:** los 4 `.ipynb`, `src/training/backtester.py`, `requirements.txt`.
- **Docs:** breve nota en `CLAUDE.md` (§3 estructura: `nb_style.py`) si aplica.
