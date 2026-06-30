# Marco teórico y fundamentación técnica

**Proyecto:** FIFA World Cup 2026 Quiniela Predictor V2
**Versión del documento:** 1.0 (junio 2026)
**Autor del sistema:** DSarceno
**Naturaleza del documento:** Marco teórico formal con fines de revisión académica.

---

## Índice

1. [Introducción](#1-introducción)
2. [Fundamentos teóricos](#2-fundamentos-teóricos)
3. [Técnicas utilizadas](#3-técnicas-utilizadas)
4. [Métodos y procesos](#4-métodos-y-procesos)
5. [Arquitectura del proyecto](#5-arquitectura-del-proyecto)
6. [Herramientas, frameworks y librerías](#6-herramientas-frameworks-y-librerías)
7. [Algoritmos y lógica relevante](#7-algoritmos-y-lógica-relevante)
8. [Bibliografía y referencias teóricas](#8-bibliografía-y-referencias-teóricas)
9. [Relación entre implementación y teoría](#9-relación-entre-implementación-y-teoría)
10. [Limitaciones, supuestos y oportunidades de mejora](#10-limitaciones-supuestos-y-oportunidades-de-mejora)
11. [Conclusión](#11-conclusión)

---

## 1. Introducción

### 1.1. Propósito del proyecto

El sistema **FIFA World Cup 2026 Quiniela Predictor V2** es una plataforma de pronóstico cuantitativo orientada a la Copa Mundial de la FIFA 2026 (48 selecciones, 12 grupos, 104 partidos). Su objetivo terminal es **maximizar el valor esperado** del jugador de quinielas mediante la generación, jornada a jornada, de hojas de pronóstico estratificadas en cuatro perfiles de riesgo (`safe`, `balanced`, `aggressive`, `contrarian`).

A diferencia de un sistema puramente académico —cuya función objetivo es minimizar log-loss o Brier score sobre un conjunto de validación—, este sistema desacopla deliberadamente dos capas: (i) una capa de **estimación probabilística calibrada** y (ii) una capa de **selección de pronósticos** condicionada al perfil de riesgo del jugador. Esta separación es coherente con la literatura de *forecasting* deportivo aplicado al mercado de apuestas (cf. Dixon & Coles, 1997; Constantinou & Fenton, 2013).

### 1.2. Problema que resuelve

El problema central es la **predicción de outcomes y marcadores** en un torneo internacional de alta incertidumbre y baja densidad de partidos por equipo. Esta característica distingue de manera fundamental al fútbol de selecciones de los campeonatos de clubes:

- Cada selección disputa entre 8 y 15 partidos oficiales por año.
- La composición de cada equipo varía drásticamente entre convocatorias.
- La heterogeneidad entre confederaciones (UEFA, CONMEBOL, AFC, CAF, CONCACAF, OFC) genera asimetrías en la "fuerza de calendario" (*strength of schedule*).
- El conjunto disponible de partidos oficiales recientes para entrenamiento es del orden de 10³ observaciones, no 10⁵ como en ligas de clubes.

Esta baja densidad invalida la mayor parte de los enfoques de aprendizaje profundo que requieren grandes muestras (cf. Hubáček, Šourek & Železný, 2019) y favorece a modelos estructurales: Poisson para goles, sistemas de rating Bradley-Terry/Elo para fuerza de equipo, y *gradient boosting* sobre conjuntos reducidos de covariables.

### 1.3. Contexto técnico y académico

El sistema se inscribe en la tradición de *modelos generativos para resultados de fútbol* iniciada por Maher (1982) y consolidada por Dixon & Coles (1997). Combina tres ingredientes ampliamente validados:

1. **Sistemas de rating dinámicos**: Elo (Elo, 1978), PI Rating (Constantinou & Fenton, 2013) y rolling form.
2. **Modelos generativos de marcador**: Poisson independiente con corrección de baja anotación (Dixon & Coles, 1997) y distribución Skellam para la diferencia de goles (Karlis & Ntzoufras, 2009).
3. **Clasificadores discriminativos sobre features ingenieriles**: regresión logística multinomial y *gradient boosting* (XGBoost), siguiendo el patrón de modelo híbrido validado por Groll, Ley, Schauberger & Van Eetvelde (2019) para la Copa Mundial 2018.

La arquitectura del sistema, además, replica el patrón estándar de la literatura aplicada: (a) calibración isotónica posterior al clasificador base (Zadrozny & Elkan, 2002), (b) ensamble probabilístico ponderado, y (c) simulación Monte-Carlo del torneo completo para derivar probabilidades de eventos de orden superior (clasificación a octavos, llegar a semifinales, ganar el torneo).

---

## 2. Fundamentos teóricos

### 2.1. Distribución de Poisson para el conteo de goles

**Definición.** Una variable aleatoria $X$ sigue una distribución de Poisson de parámetro $\lambda > 0$ si $P(X = k) = e^{-\lambda} \lambda^k / k!$ para $k \in \mathbb{N}_0$.

**Aplicación.** Maher (1982) propuso modelar el número de goles que un equipo $a$ anota contra un equipo $b$ como una variable Poisson independiente con parámetro:

$$\lambda_{ab} = \mu \cdot \alpha_a \cdot \beta_b \cdot h$$

donde $\alpha_a$ es el "ataque" de $a$, $\beta_b$ es la "defensa" de $b$, $\mu$ es la tasa global de goles y $h$ es la ventaja del local. La hipótesis de independencia entre $G_a$ y $G_b$ es estadísticamente conveniente pero empíricamente cuestionable; Dixon & Coles (1997) introdujeron una corrección $\tau(\lambda_a, \lambda_b, \rho)$ aplicada a los marcadores bajos $\{0\text{-}0, 1\text{-}0, 0\text{-}1, 1\text{-}1\}$ para mejorar la calibración.

**Implementación.** El modelo Poisson se implementa en [`src/models/poisson_model.py`](src/models/poisson_model.py), clase `PoissonScoreModel`. Los parámetros $\alpha_a$ y $\beta_b$ se estiman mediante promedios marginales ponderados; el factor $\mu$ se deriva del promedio global ponderado; $h$ se obtiene del diferencial entre goles de local y visitante.

### 2.2. Distribución Skellam para la diferencia de goles

**Definición.** Si $G_a \sim \text{Poisson}(\lambda_a)$ y $G_b \sim \text{Poisson}(\lambda_b)$ son independientes, la variable $D = G_a - G_b$ sigue una distribución Skellam:

$$P(D = k) = e^{-(\lambda_a + \lambda_b)} \left(\frac{\lambda_a}{\lambda_b}\right)^{k/2} I_{|k|}(2\sqrt{\lambda_a \lambda_b})$$

donde $I_{|k|}$ es la función de Bessel modificada de primera especie.

**Ventaja.** El soporte de $D$ es $\mathbb{Z}$ y las probabilidades de outcome se derivan directamente:

- $P(\text{H}) = P(D > 0) = 1 - F_D(0)$
- $P(\text{D}) = P(D = 0) = p_D(0)$
- $P(\text{A}) = P(D < 0) = F_D(-1)$

Esto elimina la necesidad de integrar sobre una matriz de marcadores y produce **mejor calibración de empates** que la corrección $\tau$ de Dixon-Coles (cf. Karlis & Ntzoufras, 2009).

**Implementación.** Camino por defecto en [`src/models/poisson_model.py`](src/models/poisson_model.py), método `outcome_probabilities`, controlado por la bandera `use_skellam`. Se utiliza `scipy.stats.skellam` para el cómputo numérico.

### 2.3. Sistema Elo

**Definición.** El rating Elo (Elo, 1978) asigna a cada agente $i$ una valoración escalar $r_i \in \mathbb{R}$. La probabilidad esperada de victoria de $a$ sobre $b$ es:

$$E_a = \frac{1}{1 + 10^{-(r_a - r_b)/400}}$$

Tras observar un resultado $S_a \in \{0, 0.5, 1\}$, el rating se actualiza como:

$$r_a \leftarrow r_a + K \cdot (S_a - E_a)$$

donde $K > 0$ es la **constante de aprendizaje**.

**Extensiones aplicadas.** El sistema implementa tres extensiones estándar:

1. **K dependiente de competición** (cf. World Football Elo Ratings, FiveThirtyEight SPI): $K \in \{80, 40, 30, 24, 8\}$ para Mundial, Continentales, Eliminatorias, Otros y Amistosos respectivamente. La calibración eleva el peso del Mundial y reduce el de amistosos para evitar el ruido inducido por convocatorias rotativas.
2. **Multiplicador de margen de goles** (estilo FiveThirtyEight): $m(g) = 1$ si $|g|=1$, $1.5$ si $|g|=2$, $(11+|g|)/8$ si $|g| \geq 3$. Modula la magnitud del cambio según la diferencia de marcador.
3. **Decaimiento temporal exponencial** (Dixon & Coles, 1997): $\Delta r_a \mapsto w_t \cdot \Delta r_a$ con $w_t = \exp(-\xi \cdot \Delta_{\text{días}})$, $\xi \approx 0.0015$. Half-life aproximada de 15 meses.

**Implementación.** [`src/ratings/elo.py`](src/ratings/elo.py), clase `EloRating`, parametrizada por `EloConfig`.

### 2.4. PI Rating

**Definición.** Constantinou & Fenton (2013) proponen un sistema de rating dinámico que mantiene **dos ratings por equipo** —uno para condición de local y otro para visitante— con dos tasas de aprendizaje: una intra-condición ($\lambda$) y otra cross-condición ($\gamma$).

Dada la diferencia esperada de goles $\hat{\delta} = (10^{|r_h - r_a|/c} - 1) \cdot \text{sign}(r_h - r_a)$ y el error observado $e = \delta_{\text{obs}} - \hat{\delta}$, las actualizaciones son:

$$
\begin{aligned}
r_h(a) &\leftarrow r_h(a) + \lambda \cdot e \\
r_a(a) &\leftarrow r_a(a) + \gamma \lambda \cdot e \\
r_a(b) &\leftarrow r_a(b) - \lambda \cdot e \\
r_h(b) &\leftarrow r_h(b) - \gamma \lambda \cdot e
\end{aligned}
$$

con $\lambda = 0.054$, $\gamma = 0.79$, $c = 3.0$ (valores reportados en el paper original).

**Implementación.** [`src/ratings/pi_rating.py`](src/ratings/pi_rating.py), clase `PIRating`. El proyecto añade decaimiento temporal análogo al de Elo (parámetro `time_decay_xi`).

### 2.5. Rolling form rating

**Definición.** El form rating es un indicador de momentum reciente. Mantiene una ventana móvil de los últimos $n=5$ partidos por equipo y combina puntos por resultado con diferencia de goles:

$$\text{form}(a) = \sum_{t=T-n+1}^{T} P_t(a) + \alpha \sum_{t=T-n+1}^{T} \delta_t(a)$$

con $P_t \in \{0, 1, 3\}$ y $\alpha = 0.25$.

**Implementación.** [`src/ratings/form_rating.py`](src/ratings/form_rating.py), clase `FormRating`. Utiliza `collections.deque(maxlen=5)` para la ventana móvil.

### 2.6. Calibración de probabilidades

Los clasificadores probabilísticos producen, en general, scores que no son probabilidades bien calibradas: un modelo puede estar sistemáticamente sobreconfiado o sub-confiado.

**Regresión isotónica** (Zadrozny & Elkan, 2002). Dado un conjunto de pares $(\hat{p}_i, y_i)$, encuentra una función monótona no decreciente $\phi: [0,1] \to [0,1]$ que minimiza $\sum_i (\phi(\hat{p}_i) - y_i)^2$. No paramétrica, sin supuestos sobre la forma de la deformación.

**Calibración de Platt** (Platt, 1999). Ajusta una sigmoide $\phi(\hat{p}) = 1 / (1 + e^{A \hat{p} + B})$ por máxima verosimilitud. Paramétrica, suaviza, robusta con poca data.

**Implementación.** [`src/models/calibration.py`](src/models/calibration.py), clase `ProbabilityCalibrator` con estrategia `CalibrationStrategy.ISOTONIC` (default) o `SIGMOID`. La calibración se realiza **clase-a-clase** (one-vs-rest) sobre el conjunto de validación temporal.

### 2.7. Regularización L1 (LASSO)

**Definición.** Tibshirani (1996) introdujo el LASSO (*Least Absolute Shrinkage and Selection Operator*) como una regularización L1 sobre los coeficientes de regresión:

$$\hat{\beta} = \arg\min_\beta \left[ \mathcal{L}(\beta) + \alpha \sum_j |\beta_j| \right]$$

A diferencia de L2, la penalización L1 induce **esparsidad**: algunos $\hat{\beta}_j$ se vuelven exactamente cero, realizando *feature selection* automática.

**Aplicación.** Groll, Schauberger & Tutz (2015) demostraron que para forecasting de torneos internacionales, una selección LASSO sobre el pool de covariables ingenieriles deja típicamente menos de 10 features no nulas, simplificando los modelos downstream sin pérdida de capacidad predictiva.

**Implementación.** [`src/training/trainer.py`](src/training/trainer.py), función `_lasso_select_features`, mediante `sklearn.linear_model.LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial")` con `C=0.1`. Las features con $|\hat{\beta}| < 10^{-8}$ en todas las clases son descartadas.

### 2.8. Shrinkage bayesiano hacia priors confederacionales

**Definición.** En presencia de muestras pequeñas, la estimación de máxima verosimilitud tiene alta varianza. James & Stein (1961) probaron que es admisible "encoger" estas estimaciones hacia un prior, reduciendo el riesgo bajo error cuadrático medio.

**Forma aplicada.** Para un equipo con $n$ partidos observados y rating empírico $\hat{r}$, el rating *encogido* es:

$$r^* = \frac{n}{n + K} \hat{r} + \frac{K}{n + K} r_0$$

donde $r_0$ es el prior asignado por confederación y $K$ controla la fuerza del encogimiento.

**Justificación contextual.** Selecciones con poca historia frente a rivales fuertes (e.g., naciones de OFC o pequeñas naciones de AFC) tienden a tener ratings empíricos infladamente alterados por una *strength of schedule* débil. El shrinkage hacia el promedio confederacional mitiga esta distorsión.

**Implementación.** [`src/ratings/shrinkage.py`](src/ratings/shrinkage.py), clase `RatingShrinker` (defaults $K_{\text{elo}}=K_{\text{pi}}=30$, $K_{\text{poisson}}=35$). Priors de Elo: UEFA/CONMEBOL=1620, CONCACAF=1430, AFC=1400, CAF=1420, OFC=1200.

**Mixture prior (A.4) — implementado pero desactivado.** La mixtura élite/regular por confederación que sugieren Baio & Blangiardo (2010) está implementada (`classify_elite` con compuerta `min_matches_for_elite` para que un equipo de pocos partidos nunca sea élite). Sin embargo, el backtest cross-tournament (§7.12) mostró que con los priors élite/regular calibrados a mano **empeora** el log-loss held-out (+2.3%), por lo que `ratings.shrinkage.mixture_prior.enabled = false` por defecto. Es la ilustración del principio metodológico del proyecto: la realismo visual de la lista de campeones no decide; el backtest sí.

### 2.9. Simulación Monte-Carlo

**Definición.** La simulación de Monte-Carlo (Metropolis & Ulam, 1949) estima la distribución de un evento complejo mediante el muestreo repetido de los procesos estocásticos subyacentes.

**Aplicación.** Para estimar $P(\text{equipo } X \text{ es campeón})$ no existe forma cerrada porque el bracket depende endógenamente de los resultados de la fase de grupos. La aproximación es:

$$\hat{P}(X = \text{campeón}) = \frac{1}{N} \sum_{r=1}^N \mathbb{1}[\text{X gana torneo en simulación } r]$$

con $N = 2000$ por defecto. La precisión del estimador escala como $O(N^{-1/2})$.

**Implementación.** [`src/simulation/tournament_simulator.py`](src/simulation/tournament_simulator.py), clase `TournamentSimulator`. Incluye dos optimizaciones críticas: (i) cache de predicciones para todos los pares ordenados de equipos antes del bucle principal, y (ii) motor vectorizado `VectorizedGroupStageEngine` que ejecuta la fase de grupos con operaciones sobre arrays NumPy. Cada par se pre-scorea con el **modelo blended completo** (no solo ratings+Poisson): `feature_builder.build_pairwise_feature_matrix` arma fixtures sintéticos neutros para los $n(n-1)$ pares y los pasa por el mismo pipeline de features (squad value as-of + diffs de fuerza + contexto neutro), de modo que la distribución de campeón refleja talento (§7.9).

### 2.10. Valor de plantilla como proxy de talento (A.2)

**Definición.** El Elo y el PI son ratings *basados en resultados*: premian lo que ya ocurrió y reaccionan tarde. La literatura de pronóstico de selecciones (Groll, Schauberger & Tutz 2015; Groll & Ley 2019) muestra que el **valor de mercado de la plantilla** es la covariable más predictiva después del Elo, porque captura el **talento actual** — exactamente lo que un rating de resultados no ve cuando una potencia rota plantilla o atraviesa mala forma.

**Forma aplicada.** Por selección se agrega el valor de mercado de los jugadores (proxy por `country_of_citizenship`, dataset Kaggle `player-scores` derivado de Transfermarkt) a la fecha del partido (as-of), y se usan dos transformaciones: el total y la suma del **top-11** (el XI más valioso, más robusto al ruido del proxy). Las features de partido son diferencias log: $\Delta = \log(1+v_a) - \log(1+v_b)$.

**Consistencia de escala.** El mismo método de agregación se usa para todos los snapshots (2018/2022/2026), de modo que la feature está en la misma escala en entrenamiento y predicción — condición necesaria para que el modelo transfiera.

**Validación.** Aceptado por el backtest cross-tournament (log-loss held-out 0.9994 → 0.9905; §7.12). Implementación: [`src/data/squad_value_client.py`](src/data/squad_value_client.py), [`src/features/squad_value_features.py`](src/features/squad_value_features.py).

---

## 3. Técnicas utilizadas

Esta sección detalla las técnicas implementadas, los módulos correspondientes y el problema específico que resuelve cada una.

### 3.1. Ingesta unificada con cliente-fuente y normalización canónica

**Técnica.** Patrón Registry de fuentes externas (`src/data/sources.py`) más clientes con interfaz uniforme (`load() -> pd.DataFrame`) que retornan filas conformes a `CANONICAL_COLUMNS` (`date`, `team_a`, `team_b`, `score_a`, `score_b`, `competition`, `stage`, `group`, `neutral_venue`).

**Módulos.** [`src/data/football_data_client.py`](src/data/football_data_client.py), [`src/data/kaggle_results_client.py`](src/data/kaggle_results_client.py), [`src/data/statsbomb_open_client.py`](src/data/statsbomb_open_client.py), [`src/data/results_collector.py`](src/data/results_collector.py).

**Problema que resuelve.** Heterogeneidad de fuentes (REST API, CSV plano, JSON anidado). Estandariza los esquemas en un único `data/interim/matches_unified.csv` que actúa como **única fuente de verdad** para todo el pipeline downstream.

### 3.2. Deduplicación temporal por triple clave

**Técnica.** Al fusionar nuevos partidos con la tabla canónica, se realiza una deduplicación por la clave compuesta `(date, team_a, team_b)` conservando la observación más reciente (`keep="last"`).

**Módulo.** [`src/data/results_collector.py`](src/data/results_collector.py), método `merge_with_existing`.

**Problema que resuelve.** Las fuentes externas pueden contener registros parcialmente actualizados (e.g., partidos con marcador NaN que luego se completan). El criterio "última versión gana" garantiza que los resultados sobreescriban a los placeholders.

### 3.3. Ratings ortogonales en ensamble z-scored

**Técnica.** Tres sistemas de rating independientes (Elo, PI, Form) se combinan en un índice compuesto mediante normalización z y promedio ponderado:

$$\text{composite}(a) = \frac{w_e z_e(a) + w_p z_p(a) + w_f z_f(a)}{w_e + w_p + w_f}$$

con pesos `EnsembleWeights(elo=0.55, pi=0.30, form=0.15)`.

**Módulo.** [`src/ratings/rating_ensemble.py`](src/ratings/rating_ensemble.py), clase `RatingEnsemble`.

**Problema que resuelve.** Cada rating captura una dimensión distinta: Elo, fuerza a largo plazo y simétrica; PI, asimetría local-visitante; Form, momentum reciente. La z-normalización los pone en una escala común; el promedio reduce la varianza del estimador compuesto frente a usar uno solo.

### 3.4. Ingeniería de features estratificadas

**Técnica.** Cinco subsistemas de features:

- **Team features** ([`src/features/team_features.py`](src/features/team_features.py)): ratings, ranking FIFA, ataque/defensa, reputación histórica.
- **Match features** ([`src/features/match_features.py`](src/features/match_features.py)): diferenciales entre equipos, etapa del torneo, anfitrión.
- **Market features** ([`src/features/market_features.py`](src/features/market_features.py)): proxies de sesgo público y sobrerreacción FIFA.
- **Fatigue features** ([`src/features/fatigue_features.py`](src/features/fatigue_features.py)): días de descanso, carga de viaje por matriz continental, índice de fatiga.
- **Tournament state features** ([`src/features/tournament_features.py`](src/features/tournament_features.py)): puntos acumulados y goal-diff durante el Mundial.

Adicionalmente, `_add_strategy_features` ([`src/features/build_features.py`](src/features/build_features.py)) deriva features estratégicas (`volatility_index`, `upset_window_score`, `draw_trap_score`, `favorite_fragility_score`, etc.) a partir de los diferenciales de rating, destinadas a alimentar la capa de selección de picks.

**Problema que resuelve.** Convierte la tabla canónica de partidos en una matriz de covariables apta para clasificadores discriminativos. La estratificación facilita el mantenimiento y permite que cada subsistema sea probado y modificado independientemente.

### 3.5. Selección de features por L1 (LASSO multinomial)

**Técnica.** Antes de entrenar XGBoost y la regresión multinomial, se ejecuta una regresión logística multinomial con regularización L1 sobre el pool completo de features. Las features con coeficiente nulo en todas las clases son descartadas; las restantes alimentan los modelos principales.

**Módulo.** [`src/training/trainer.py`](src/training/trainer.py), función `_lasso_select_features`.

**Problema que resuelve.** El pool incluye ~25 features ingenieriles, varias de ellas correlacionadas o redundantes (e.g., `volatility_index` y `match_entropy` ambas derivadas de `tanh(elo_diff/250)`). LASSO realiza selección automática siguiendo el patrón validado por Groll, Schauberger & Tutz (2015), evitando sobreajuste en clasificadores no parámetricos.

### 3.6. Modelo Poisson con decaimiento temporal y opción Skellam

**Técnica.** Estimación ponderada de parámetros $\alpha_a, \beta_b, \mu, h$:

$$\hat{\mu} = \frac{\sum_t w_t \cdot (g_a^t + g_b^t)/2}{\sum_t w_t}, \quad w_t = \exp(-\xi \cdot \Delta_{\text{días}})$$

y análogamente para attack y defense. La derivación de outcome puede usar el camino Skellam (default) o la integración de la matriz de marcadores con corrección Dixon-Coles.

**Módulo.** [`src/models/poisson_model.py`](src/models/poisson_model.py), clase `PoissonScoreModel`.

**Problema que resuelve.** El Poisson clásico asume estacionariedad. El decaimiento exponencial introduce no-estacionariedad suave: los resultados recientes dominan los lejanos. Skellam, por su parte, mejora la calibración de empates frente a la corrección $\tau$ ad-hoc de Dixon-Coles.

### 3.7. Ensamble ponderado de probabilidades

**Técnica.** Sea $\mathbf{p}^{(k)} \in \Delta^2$ la salida del modelo $k \in \{\text{ratings}, \text{multinomial}, \text{xgboost}, \text{poisson}\}$. La probabilidad ensamblada es:

$$\mathbf{p} = \text{normalize}\left( \sum_k w_k \mathbf{p}^{(k)} \right)$$

con renormalización condicional a los modelos disponibles (skip cuando faltan features para multinomial/XGBoost).

**Módulo.** [`src/ensemble/blender.py`](src/ensemble/blender.py), clase `ProbabilityBlender`.

**Problema que resuelve.** Promediación robusta de fuentes heterogéneas. Reduce la varianza del estimador (cf. principio del *ensemble averaging* en Dietterich, 2000).

### 3.8. Calibración posterior

**Técnica.** Tras el blending, las probabilidades pasan por un calibrador ajustado sobre el conjunto de validación temporal. Estrategias disponibles: isotonic (no paramétrica) y sigmoid/Platt (paramétrica).

**Módulo.** [`src/models/calibration.py`](src/models/calibration.py), clase `ProbabilityCalibrator`.

**Problema que resuelve.** Reduce el sesgo sistemático de los clasificadores discriminativos (XGBoost, en particular, tiende a estar mal calibrado por defecto). Mejora el log-loss y el Brier score del estimador final sin alterar el ranking de outcomes.

### 3.9. Simulación Monte-Carlo vectorizada

**Técnica.** Para cada simulación: (i) generar resultados de la fase de grupos por muestreo de la categorical $(p_H, p_D, p_A)$ y luego goles condicionales por muestreo Poisson, (ii) ordenar standings con tiebreakers FIFA usando `np.lexsort`, (iii) seleccionar los 8 mejores terceros lugares, (iv) construir el bracket de R32, (v) simular las cinco rondas eliminatorias propagando ganadores.

**Módulo.** [`src/simulation/tournament_simulator.py`](src/simulation/tournament_simulator.py), [`src/simulation/group_stage.py`](src/simulation/group_stage.py), [`src/simulation/knockout.py`](src/simulation/knockout.py), [`src/simulation/bracket_generator.py`](src/simulation/bracket_generator.py).

**Problema que resuelve.** Estima probabilidades de eventos compuestos (clasificación, llegar a determinada ronda, campeonato) cuya forma cerrada es intratable por el acoplamiento bracket–resultados.

### 3.10. Cache de predicciones para reducción factorial

**Técnica.** Antes del bucle principal de simulaciones, se pre-computa la matriz $P[a, b] \in \mathbb{R}^3$ para todo par ordenado $(a, b)$ con $a \neq b$. Dentro del bucle, cada predicción se reduce a una búsqueda $O(1)$ en diccionario.

**Módulo.** [`src/simulation/tournament_simulator.py`](src/simulation/tournament_simulator.py), método `_build_cached_predict_fn`.

**Problema que resuelve.** Sin cache, una corrida de $N = 2000$ simulaciones requiere $\sim 8N$ llamadas a `predict_fn`. Con cache, el costo total es $O(n^2 + N \cdot \text{lookup})$ donde $n = 48$. En la práctica, esto representa un *speedup* observado de 200x a 1500x según el costo de `predict_fn`.

### 3.11. Selección de picks por perfil de riesgo

**Técnica.** Cuatro perfiles aplican lógica diferenciada al vector de probabilidades calibrado:

| Perfil | favorite_threshold | upset_tolerance | draw_bias | Política |
|---|---|---|---|---|
| safe | 0.58 | 0.05 | 0.90 | argmax estricto, prioriza favoritos |
| balanced | 0.50 | 0.10 | 1.00 | fallback a empate si favorito no supera 0.50 |
| aggressive | 0.43 | 0.18 | 1.10 | acepta sorpresas si `upset_window_score ≥ 0.5` |
| contrarian | 0.38 | 0.25 | 1.15 | penaliza reputación; busca diferenciación |

**Módulo.** [`src/ensemble/pick_optimizer.py`](src/ensemble/pick_optimizer.py), clase `PickOptimizer`; [`src/prediction/quiniela_strategy.py`](src/prediction/quiniela_strategy.py).

**Problema que resuelve.** En un *pool* de quiniela, el valor esperado del jugador no depende solo de la accuracy individual sino de la **diferenciación** frente al resto de participantes. Los perfiles `aggressive` y `contrarian` están diseñados para sacrificar accuracy por diferenciación, siguiendo principios de teoría de juegos aplicada a apuestas mutuas (cf. literatura de *pari-mutuel betting*).

### 3.12. API REST con FastAPI

**Técnica.** Endpoints sincrónicos sobre los servicios principales: salud, equipos, partidos, predicciones, simulación, estrategia, actualización, diagnósticos.

**Módulo.** [`src/api/main.py`](src/api/main.py).

**Problema que resuelve.** Exposición programática del sistema. Permite que un cliente externo (dashboard, notebook, integración con bots) consuma predicciones sin invocar los scripts CLI.

---

## 4. Métodos y procesos

### 4.1. Flujo general del sistema

El sistema opera como un pipeline ETL/ML lineal con nueve etapas:

```
[1] Ingesta histórica (Kaggle, FIFA rankings, manual)
        ↓
[2] Ingesta de fixtures actuales (football-data.org)
        ↓
[3] Unificación canónica → data/interim/matches_unified.csv
        ↓
[4] Cómputo de ratings (Elo, PI, Form) → models/rating_ensemble.pkl
        ↓
[5] Construcción de feature matrix → data/processed/feature_matrix.csv
        ↓
[6] Entrenamiento (LASSO → multinomial, XGBoost, Poisson, calibrator)
        ↓
[7] Predicción match-by-match → outputs/predictions/
        ↓
[8] Generación de picks por perfil → outputs/picks/
        ↓
[9] Simulación Monte-Carlo del torneo → outputs/simulations/
```

La orquestación está implementada en [`run_all.bat`](run_all.bat) (Windows) y [`Makefile`](Makefile) (Unix-likes). Cada etapa es idempotente y persistible.

### 4.2. Procesamiento de datos

**Etapa de ingesta.** Cada cliente externo persiste un snapshot crudo con timestamp UTC en `data/raw/<source>/`. El `ResultsCollector` ([`src/data/results_collector.py`](src/data/results_collector.py)) orquesta la consulta, normalización a `CANONICAL_COLUMNS` y fusión con la tabla canónica existente.

**Etapa de unificación.** Después de cada ingesta, la tabla canónica se valida con `validate_match_dataframe` (esquema, dtypes, fechas parseables) antes de persistir. Cualquier corrupción aborta el pipeline.

**Etapa de features.** `build_match_feature_matrix` ([`src/features/build_features.py`](src/features/build_features.py)) ensambla los cinco subsistemas de features sobre la tabla canónica más el composite de ratings, las rankings FIFA y, opcionalmente, el estado del torneo.

**Fill de NaN.** `_fillna_numeric` rellena con 0.0 todas las columnas numéricas **excepto las marcadas como metadata** (`score_a`, `score_b`, `match_id`): estos campos deben permanecer NaN para los partidos por jugar, dado que el simulador los usa como señal de "fixture pendiente".

### 4.3. Lógica de entrenamiento

El entrenador ([`src/training/trainer.py`](src/training/trainer.py), clase `Trainer`) ejecuta en orden:

1. **Selección de features** vía LASSO (si `lasso_select=True`).
2. **Ajuste del clasificador multinomial** (logística con StandardScaler).
3. **Ajuste de XGBoost** sobre el mismo subset de features.
4. **Ajuste del Poisson** sobre `(team_a, team_b, score_a, score_b, date)`, con decaimiento temporal opcional.
5. **Shrinkage Poisson** (si `shrinker` fue provisto): pull de los parámetros attack/defense hacia priors confederacionales.
6. **Calibrador** sobre el conjunto de validación: ajuste isotónico/sigmoid por clase de las probabilidades de XGBoost.

Los artefactos se persisten como `models/<name>.pkl` mediante `save_pickle` (wrapper sobre `joblib.dump`).

### 4.4. Lógica de predicción

`MatchPredictor.predict_proba` ([`src/prediction/predictor.py`](src/prediction/predictor.py)) opera en cinco pasos:

1. Computar $\mathbf{p}^{\text{ratings}}$ vía `EloRating.predict` si Elo está disponible.
2. Computar $\mathbf{p}^{\text{multinomial}}$ y $\mathbf{p}^{\text{xgb}}$ **solo si** todas las feature columns requeridas están presentes en el input. En su ausencia (caso típico del simulador, que pasa solo `team_a`/`team_b`), los modelos se omiten silenciosamente.
3. Computar $\mathbf{p}^{\text{poisson}}$ vía `PoissonScoreModel.predict_proba`.
4. Combinar con `ProbabilityBlender`, renormalizando por la suma de pesos activos.
5. Aplicar calibrador si está disponible.

Este patrón de **degradación elegante** (skip de modelos cuando faltan features) es esencial para que la API y el simulador funcionen con inputs mínimos.

### 4.5. Lógica de simulación

`TournamentSimulator.run` ([`src/simulation/tournament_simulator.py`](src/simulation/tournament_simulator.py)) ejecuta:

```python
for run_idx in range(n_runs):
    rng = np.random.default_rng(seed + run_idx)
    gs = group_stage_engine.simulate(rng)
    qualifications.extend(gs.qualified_top_two + gs.best_thirds)
    bracket = build_round_of_32_bracket(gs.standings, gs.best_thirds)
    ko = simulate_knockout(bracket, predict_fn, rng=rng)
    championships[ko.champion] += 1
```

Tras $N$ corridas, se agregan estadísticos: probabilidad de clasificación al R32, probabilidad de alcanzar cada ronda, probabilidad de campeonato.

### 4.6. Manejo de errores y validaciones

El sistema sigue tres principios:

1. **Excepciones específicas**: nunca `except Exception` mudo; siempre tipos concretos como `httpx.HTTPError`, `ValueError`, `FileNotFoundError`, con `logger.error/warning` y degradación cuando es posible.
2. **Degradación silenciosa en endpoints**: la API carga modelos con `try/except FileNotFoundError`; en su ausencia continúa con ratings + Poisson.
3. **Validación pre-persistencia**: la tabla canónica se valida con `validate_match_dataframe` antes de sobreescribir el CSV de interim.

### 4.7. Comunicación entre módulos

La comunicación intra-pipeline se realiza mediante **archivos serializados** (CSV para datos tabulares, pickle/joblib para modelos), no por *in-memory message passing*. Esto:

- Hace cada etapa independientemente re-ejecutable.
- Facilita la auditoría (cada output queda persistido).
- Permite distribuir etapas en máquinas distintas si se requiriera escalar.

La API expone los mismos servicios pero opera sobre los artefactos persistidos, no recomputa de cero.

### 4.8. Autenticación

**No hay capa de autenticación implementada en la API** (la dependencia `python-jose` o `passlib` no aparece en `requirements.txt`). El sistema asume **entorno confiable de uso personal**. Para despliegues con exposición pública se requeriría agregar middleware de autenticación (recomendado: OAuth2 o API keys).

> *Inferencia:* La ausencia de auth es consistente con el objetivo declarado en el README ("uso individual del autor"), no un descuido.

### 4.9. Workflow diario durante el torneo

El script [`scripts/update_after_matchday.py`](scripts/update_after_matchday.py) ejecuta:

1. `TournamentUpdater.append_results(date)`: descarga los resultados oficiales de la jornada vía football-data.org.
2. Reentrena ratings (Elo, PI, Form) incorporando los nuevos partidos.
3. Recomputa la matriz de features.
4. Reentrena los modelos (multinomial, XGBoost, Poisson, calibrador).
5. Regenera predicciones para los partidos restantes.
6. Re-exporta las cuatro hojas de quiniela.

Frecuencia esperada: una ejecución diaria durante los 31 días del torneo.

---

## 5. Arquitectura del proyecto

### 5.1. Vista general en capas

El sistema sigue una **arquitectura en capas con dependencias estrictamente unidireccionales**:

```
utils → data → ratings → features → models → ensemble → simulation/prediction/training → api
```

Cualquier importación que viole esta dirección se considera un *code smell*. Esto facilita el razonamiento sobre el grafo de dependencias y permite que cada capa sea verificable independientemente.

### 5.2. Mapa de módulos

| Capa | Subdirectorio | Responsabilidad |
|---|---|---|
| Utilitarios | `src/utils/` | logging, config, IO, fechas, métricas, constantes, plotting |
| Datos | `src/data/` | clientes externos, registry de fuentes, recolector, *data loader* |
| Ratings | `src/ratings/` | Elo, PI, Form, ensemble, shrinkage confederacional |
| Features | `src/features/` | team, match, market, fatigue, tournament + entrypoint `build_features` |
| Modelos | `src/models/` | base abstracta, multinomial, XGBoost, Poisson, calibración, factory |
| Ensemble | `src/ensemble/` | blender de probabilidades, optimizador de picks |
| Simulación | `src/simulation/` | fase de grupos (scalar + vectorizada), knockout, bracket, simulador completo |
| Predicción | `src/prediction/` | predictor, score_predictor, quiniela_strategy, daily_update, feature_builder |
| Entrenamiento | `src/training/` | trainer, evaluator, backtester, cross_validation |
| API | `src/api/` | endpoints FastAPI sobre los servicios anteriores |

### 5.3. Patrones de diseño identificados

**Strategy** (explícito). [`src/models/calibration.py`](src/models/calibration.py) implementa la enumeración `CalibrationStrategy` y selecciona dinámicamente entre regresión isotónica y Platt en `ProbabilityCalibrator`. [`src/ensemble/pick_optimizer.py`](src/ensemble/pick_optimizer.py) aplica el mismo patrón con los cuatro perfiles de riesgo.

**Factory** (explícito). [`src/models/model_factory.py`](src/models/model_factory.py), función `build_model(name, ...)`, construye instancias concretas (`MultinomialOutcomeModel`, `XGBoostOutcomeModel`, `PoissonScoreModel`) sin que el cliente conozca las clases.

**Repository** (inferencia). Los clientes de datos en [`src/data/`](src/data/) actúan como repositorios sobre fuentes heterogéneas, exponiendo una API uniforme `load() -> pd.DataFrame` y aislando al resto del sistema de los detalles de cada fuente.

**Template Method** (inferencia). [`src/models/base_model.py`](src/models/base_model.py) define la clase abstracta `BaseOutcomeModel` con el contrato `fit/predict_proba`, dejando la lógica concreta a las subclases.

**Pipeline** (inferencia). El orden de transformaciones en `Trainer.fit` y en `MatchPredictor.predict_proba` siguen un pipeline conceptual, aunque no se utiliza explícitamente `sklearn.pipeline.Pipeline`.

**Observer/Hook** (no aplica). El sistema no usa observers porque el flujo es batch, no event-driven.

### 5.4. Separación de responsabilidades

El proyecto observa el principio de **Single Responsibility** a nivel de módulo:

- Los clientes de datos no transforman features.
- Los ratings no producen probabilidades de outcome (eso es trabajo de modelos).
- Los modelos no eligen picks (eso es trabajo del `PickOptimizer`).
- El simulador no calibra (recibe `predict_fn` ya calibrada).
- La API no recomputa (lee artefactos persistidos).

Esto facilita pruebas unitarias y permite reemplazar piezas individualmente.

### 5.5. Dependencias internas

El grafo de dependencias internas es un DAG. Los ejemplos representativos:

- `src/training/trainer.py` depende de `src/features/build_features.py`, `src/models/*` y `src/ratings/shrinkage.py`.
- `src/simulation/tournament_simulator.py` depende de `src/simulation/group_stage.py`, `simulation/knockout.py`, `simulation/bracket_generator.py`.
- `src/prediction/predictor.py` depende de `src/models/*`, `src/ratings/elo.py`, `src/ensemble/blender.py`.

### 5.6. Dependencias externas

Las dependencias se enumeran en [`requirements.txt`](requirements.txt) y se categorizan en §6.

### 5.7. Configuración por capas

El sistema separa configuración estática (en YAMLs) de configuración sensible (en variables de entorno):

- [`config/config.yaml`](config/config.yaml): parámetros generales (datos, torneo, training, ratings, simulación, evaluación, logging, paths).
- [`config/model_params.yaml`](config/model_params.yaml): hiperparámetros por modelo.
- [`config/features.yaml`](config/features.yaml): definiciones de features.
- [`config/strategy.yaml`](config/strategy.yaml): parámetros de los perfiles de quiniela.
- `.env`: API keys y secretos (gitignored).

El acceso es uniforme a través de `Config.get("dotted.path", default)`.

---

## 6. Herramientas, frameworks y librerías

### 6.1. Lenguaje y runtime

**Python 3.10 / 3.11.** El proyecto activa `from __future__ import annotations` en todos los módulos para usar la sintaxis PEP 604 (`int | None`) sin requerir Python 3.10 en runtime de evaluación. Las type hints son obligatorias en todas las firmas públicas.

### 6.2. Datos y cálculo numérico

| Librería | Versión | Rol |
|---|---|---|
| `numpy` | 1.24.4 | Arrays, álgebra lineal, RNG `np.random.default_rng`, operaciones vectorizadas (`np.lexsort`, `np.add.at`) en el simulador |
| `pandas` | 2.1.4 | DataFrame canónico de partidos, feature matrix, merges, groupby ponderado |
| `scipy` | 1.11.4 | `scipy.stats.poisson` (cómputo de PMF), `scipy.stats.skellam` (Skellam distribution para H/D/A), `scipy.special` (funciones Bessel implícitas) |

### 6.3. Aprendizaje automático

| Librería | Versión | Rol |
|---|---|---|
| `scikit-learn` | 1.4.2 | `LogisticRegression` (multinomial L1 para LASSO; LBFGS para multinomial base), `StandardScaler`, `IsotonicRegression`, métricas |
| `xgboost` | 2.0.3 | Clasificador `XGBClassifier` con `multi:softprob` |
| `statsmodels` | 0.14.2 | *Por confirmar*: declarado pero su uso concreto no se evidencia en los módulos auditados. Posiblemente utilitario en backtester o evaluator. |

### 6.4. API y servicio web

| Librería | Versión | Rol |
|---|---|---|
| `fastapi` | 0.110.0 | Framework REST asincrónico con generación automática de OpenAPI/Swagger |
| `uvicorn` | 0.29.0 | ASGI server para servir FastAPI en desarrollo |
| `pydantic` | 2.6.4 | Modelos de validación de request/response |

**Ventajas técnicas.** FastAPI ofrece (i) generación automática de documentación, (ii) validación declarativa con type hints, (iii) rendimiento competitivo gracias a Starlette/uvicorn. La elección es coherente con un sistema de tamaño medio que requiere una API limpia sin la complejidad de Django.

### 6.5. HTTP y persistencia

| Librería | Versión | Rol |
|---|---|---|
| `httpx` | 0.27.0 | Cliente HTTP async (fuentes externas, e.g., football-data.org) |
| `requests` | 2.31.0 | Cliente HTTP sincrónico (compatibilidad/fallback) |
| `joblib` | 1.3.2 | Serialización de modelos (`save_pickle`, `load_pickle` envuelven `joblib.dump/load`) |
| `tqdm` | 4.66.2 | Barras de progreso en el simulador y warmup de cache |

### 6.6. Configuración

| Librería | Versión | Rol |
|---|---|---|
| `python-dotenv` | 1.0.1 | Carga de `.env` con secretos |
| `pyyaml` | 6.0.1 | Lectura de los archivos YAML de configuración |

### 6.7. Calidad de código y pruebas

| Librería | Versión | Rol |
|---|---|---|
| `pytest` | 8.1.1 | Framework de pruebas unitarias |
| `pytest-cov` | 4.1.0 | Cobertura de pruebas |
| `black` | 24.3.0 | Formateo con `line-length=100` |
| `isort` | 5.13.2 | Ordenamiento de imports, perfil `black` |
| `pylint` | 3.1.0 | Análisis estático |

### 6.8. Visualización y reportes

| Librería | Versión | Rol |
|---|---|---|
| `matplotlib` | 3.8.3 | Gráficos en notebooks y reportes (backend `Agg` para uso headless) |
| `jupyter` / `notebook` | 1.0.0 / 7.1.2 | Notebooks de exploración y diagnóstico |
| LaTeX (`pdflatex`) | externo | Compilación de reportes académicos y dashboard. Si no está instalado, el pipeline lo salta sin error. |

### 6.9. Servicios externos

| Servicio | Naturaleza | Uso |
|---|---|---|
| football-data.org | REST API (free tier limitada; key en `.env`) | Fixtures, resultados, standings del Mundial |
| Kaggle dataset `martj42/international-football-results-from-1872-to-2017` | CSV de descarga manual | Histórico extendido (~46k partidos) |
| StatsBomb Open Data | JSON open data | Disponible pero no consumido por features (ver §10) |
| FIFA Rankings | snapshots CSV locales | Puntos y posiciones oficiales |

### 6.10. Consideraciones

- El proyecto **no utiliza Docker en producción** automáticamente pero incluye [`docker/Dockerfile`](docker/Dockerfile) y [`docker/docker-compose.yml`](docker/docker-compose.yml) (por confirmar el grado de uso real, no fueron auditados en detalle).
- No usa frameworks de orquestación (Airflow, Prefect). El pipeline es un script secuencial en `.bat`/`Makefile`. Suficiente para un sistema de un solo usuario; insuficiente para producción multi-tenant.
- No usa base de datos relacional. Todo se persiste en CSV/pickle. Coherente con el tamaño de los datos (~50MB) y el patrón batch.

---

## 7. Algoritmos y lógica relevante

### 7.1. Actualización Elo con K competition-aware y goal-margin multiplier

Pseudocódigo:

```
ENTRADA: r_a, r_b, score_a, score_b, competition, neutral, peso w
expected_a ← 1 / (1 + 10^(-(r_a + h - r_b)/400))         // h = 0 si neutral
S_a ← {1 si gana, 0.5 si empata, 0 si pierde}
K ← K_default si competition desconocida, sino mapa
m ← 1 si |Δgoles|=1, 1.5 si =2, (11+|Δgoles|)/8 si ≥3
Δ ← w · K · m · (S_a - expected_a)
r_a ← r_a + Δ
r_b ← r_b - Δ
```

[`src/ratings/elo.py:108-141`](src/ratings/elo.py#L108-L141).

### 7.2. Skellam para H/D/A

Dados $\lambda_a, \lambda_b$ obtenidos de `expected_goals`:

```
p_draw ← skellam.pmf(0, λ_a, λ_b)
p_home ← 1 - skellam.cdf(0, λ_a, λ_b)
p_away ← skellam.cdf(-1, λ_a, λ_b)
return normalize([p_home, p_draw, p_away])
```

[`src/models/poisson_model.py`](src/models/poisson_model.py), método `outcome_probabilities`.

### 7.3. LASSO multinomial para selección de features

```
X_scaled ← StandardScaler().fit_transform(X[candidate_columns])
lasso ← LogisticRegression(penalty='l1', solver='saga', C=0.1, multi_class='multinomial')
lasso.fit(X_scaled, y)
coef_mag ← |lasso.coef_|.max(axis=0)         // máx sobre clases
kept ← {c : coef_mag[c] > 1e-8}
si |kept| < min_features:
    kept ← top-min_features por coef_mag
return sorted(kept)
```

[`src/training/trainer.py`](src/training/trainer.py), función `_lasso_select_features`.

### 7.4. Decaimiento temporal exponencial

Aplicado a Elo, PI y Poisson:

```
ref_date ← max(matches.date)
w_t ← exp(-ξ · (ref_date - date_t).days)
```

con $\xi_{\text{elo}} = \xi_{\text{pi}} = 0.0015$ (half-life ~462 días) y $\xi_{\text{poisson}} = 0.0020$ (half-life ~347 días). Los pesos modulan el delta del rating (Elo/PI) o el promedio ponderado de la estimación de parámetros (Poisson).

### 7.5. Tiebreakers FIFA por `np.lexsort`

En el motor vectorizado [`src/simulation/group_stage.py`](src/simulation/group_stage.py), el orden final de un grupo se obtiene mediante un sort lexicográfico con keys priorizados de menor a mayor importancia:

```
keys = (random_noise, goals_for, goal_diff, points)
order = np.lexsort(keys)[::-1]                  // descendente
```

`np.lexsort` ordena por el último key primero y rompe empates con los anteriores. La inversión final convierte a descendente. La inclusión de ruido aleatorio resuelve empates restantes de forma uniforme.

### 7.6. Selección de los 8 mejores terceros lugares

Tras ordenar cada grupo, los terceros lugares de los 12 grupos se ordenan globalmente con las mismas keys (puntos, goal diff, goles a favor, ruido) y se eligen los 8 con mayor ranking para completar los 32 clasificados a la primera ronda eliminatoria.

### 7.7. Construcción del bracket de R32

`build_round_of_32_bracket` ([`src/simulation/bracket_generator.py`](src/simulation/bracket_generator.py)) construye un emparejamiento determinístico:

- Ganador del grupo A vs. segundo del grupo B
- Ganador del grupo B vs. segundo del grupo C
- ... (rotación)
- Los 8 mejores terceros llenan los slots restantes según una asignación pre-definida por FIFA.

> *Por confirmar:* la implementación específica del orden de los terceros sigue la regla oficial FIFA 2026 publicada por la federación. El código asume un orden determinado en la rotación; conviene validar contra el reglamento oficial cuando esté publicado.

### 7.8. Simulación de un partido con penaltis

`_decide_match` ([`src/simulation/knockout.py`](src/simulation/knockout.py)) modela penaltis como un proxy Bernoulli:

```
probs = predict_fn(a, b)
win_prob_a = p_home + 0.5 * p_draw
via_penalties = rng.uniform() < p_draw
winner = a si rng.uniform() < win_prob_a, sino b
```

El supuesto operacional: en caso de empate (que en fase eliminatoria iría a tiempos extras y eventualmente penaltis), la ganadora se decide proporcionalmente a la fuerza relativa estimada por el modelo, distribuyendo el peso del empate entre las dos partes.

**Cruces ya jugados (condicionamiento).** El proxy Bernoulli aplica solo a cruces *aún no jugados*. Para un cruce ya disputado, `_played_knockout_winners` fuerza el ganador real vía `known_winners` en lugar de re-simularlo. Si el partido terminó en empate y se definió por penaltis, el marcador (p. ej. 1-1) no revela al ganador, así que `_bracket_advancers` ([`scripts/simulate_tournament.py`](scripts/simulate_tournament.py)) lo **infiere del bracket**: el equipo que avanzó es el que aparece en el cruce de la ronda siguiente (vía `feeds_winner_into`) una vez que se llena. Así las probabilidades de campeón condicionan en el resultado real de los penaltis, no en una nueva moneda al aire.

> *Limitación conocida (documentada en README):* "Penalty shootouts are modeled as Bernoulli draws around a knockout-volatility prior" — aplica a los tiroteos *simulados* (cruces futuros). Una mejora natural sería incorporar tasas de conversión históricas de penaltis por selección.

### 7.9. Pre-cache de predicciones

Antes del bucle de simulaciones:

```
teams = sorted(unique(team_a ∪ team_b))
cache = {}
for a in teams:
    for b in teams:
        if a != b:
            cache[(a, b)] = predict_fn(a, b)
return lambda a, b: cache[(str(a), str(b))]
```

Reduce el costo de simulación de $O(N \cdot \text{matches\_per\_run} \cdot \text{cost}(predict\_fn))$ a $O(n^2 \cdot \text{cost}(predict\_fn) + N \cdot \text{matches\_per\_run})$.

En `simulate_tournament.py`, el `predict_fn` se construye sobre una matriz de features **por par** (`build_pairwise_feature_matrix`) scoreada una vez con el modelo completo, en lugar de `predict_single(a,b)` que caía a ratings+Poisson por ausencia de feature columns. Así el campeón refleja squad value y el resto de features engineered.

### 7.10. Métricas implementadas

[`src/utils/metrics.py`](src/utils/metrics.py):

- **Log-loss**: $-\sum_i y_i \log(\hat{p}_i)$ con clipping para evitar $\log(0)$.
- **Brier score multiclase**: $\frac{1}{K}\sum_c \frac{1}{n}\sum_i (y_{ic} - \hat{p}_{ic})^2$.
- **Accuracy top-1**: $\text{argmax}(\hat{p}_i) = y_i$.
- **Macro-F1**: promedio del F1 por clase.
- **Expected Calibration Error (ECE)**: $\sum_b \frac{|B_b|}{n}|\text{acc}(B_b) - \text{conf}(B_b)|$ con 10 bins por defecto.
- **Ranked Probability Score (RPS) ordinal**: sobre la escala ordenada $[H, D, A]$, penaliza más un error de dos categorías que de una (Constantinou & Fenton 2012). En [`src/training/backtester.py`](src/training/backtester.py).
- **Quiniela score (custom)**: 1 pt por 1X2 correcto, +2 pts por marcador exacto, +1 pt por acertar sorpresa.

### 7.11. As-of join del valor de plantilla

Para cada partido $(d, a, b)$, se toma por equipo el snapshot de valor de plantilla con `as_of_date` máxima $\le d$ (un `pd.merge_asof` con `by="team"`, dirección `backward`). Faltantes → mediana de la confederación a esa época; luego `log1p` y diferencia home−away. Date-aware → sin leakage en el backtest. [`src/features/squad_value_features.py`](src/features/squad_value_features.py).

### 7.12. Backtest cross-tournament leakage-free (el árbitro)

```
para cada año Y objetivo:
    train  ← partidos con año < Y
    test   ← partidos del torneo (== Y, competition)
    ratings ← RatingEnsemble(shrinker).fit(train)          // solo historia previa
    feats   ← build_match_feature_matrix(train ∪ test, ..., squad_values)  // as-of
    modelos ← Trainer.fit(train_feats)                     // LASSO + XGB/MN/Poisson
    proba   ← blend(ratings, modelos)(test_feats)
    métricas(Y) ← log_loss, brier, RPS, accuracy, ECE
```

Como ratings y features de test se derivan **solo** de datos pre-Y (as-of), no hay fuga de información. `run_tournament_backtest` agrega por año; `collect_holdout_predictions` devuelve `(proba, y_true)` de un torneo para la curva de calibración. [`src/training/backtester.py`](src/training/backtester.py), [`scripts/backtest_tournaments.py`](scripts/backtest_tournaments.py). **Es el criterio de aceptación** de todo cambio de modelado: A.2 (squad value) se aceptó y A.4 (mixture prior) se rechazó aquí.

---

## 8. Bibliografía y referencias teóricas

### 8.1. Bibliografía identificada en el código y en `CLAUDE.md`

Las siguientes referencias aparecen citadas explícitamente en docstrings o documentación del proyecto:

1. **Dixon, M. J. & Coles, S. G.** (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society: Series C, 46(2), 265–280. — citada en [`src/models/poisson_model.py`](src/models/poisson_model.py) y en el roadmap del CLAUDE.md (decaimiento temporal y corrección $\tau$).
2. **Constantinou, A. C. & Fenton, N. E.** (2013). *Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries.* Journal of Quantitative Analysis in Sports, 9(1), 37–50. — citada en [`src/ratings/pi_rating.py`](src/ratings/pi_rating.py).
3. **Karlis, D. & Ntzoufras, I.** (2009). *Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference.* IMA Journal of Management Mathematics, 20(2), 133–145. — base teórica del camino Skellam en `outcome_probabilities`.
4. **Groll, A., Schauberger, G. & Tutz, G.** (2015). *Prediction of major international soccer tournaments based on team-specific regularized Poisson regression: an application to the FIFA World Cup 2014.* Journal of Quantitative Analysis in Sports, 11(2), 97–115. — justifica el uso de LASSO para selección de features.
5. **Groll, A., Ley, C., Schauberger, G. & Van Eetvelde, H.** (2019). *A hybrid random forest to predict soccer matches in international tournaments.* Journal of Quantitative Analysis in Sports, 15(4), 271–287. — patrón híbrido rating-como-feature.
6. **Hubáček, O., Šourek, G. & Železný, F.** (2019). *Learning to predict soccer results from relational data with gradient boosted trees.* Machine Learning, 108, 29–47. — ganador del *Soccer Prediction Challenge* 2017; referencia comparativa.

### 8.2. Bibliografía sugerida (no citada explícitamente pero fundacional)

Las siguientes referencias **no aparecen en el código** pero sustentan teóricamente las técnicas empleadas. Se sugiere su inclusión en cualquier documento académico derivado del proyecto:

7. **Elo, A. E.** (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing. — definición original del sistema Elo.
8. **Maher, M. J.** (1982). *Modelling association football scores.* Statistica Neerlandica, 36(3), 109–118. — primer modelo Poisson independiente con parámetros de equipo.
9. **Skellam, J. G.** (1946). *The frequency distribution of the difference between two Poisson variates belonging to different populations.* Journal of the Royal Statistical Society, 109(3), 296. — definición de la distribución Skellam.
10. **Tibshirani, R.** (1996). *Regression shrinkage and selection via the lasso.* Journal of the Royal Statistical Society: Series B, 58(1), 267–288. — definición del LASSO.
11. **Krishnapuram, B., Carin, L., Figueiredo, M. A. T. & Hartemink, A. J.** (2005). *Sparse multinomial logistic regression: fast algorithms and generalization bounds.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(6), 957–968. — extensión multinomial del LASSO.
12. **Platt, J. C.** (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.* Advances in Large Margin Classifiers, 10(3), 61–74. — calibración sigmoidal de Platt.
13. **Zadrozny, B. & Elkan, C.** (2002). *Transforming classifier scores into accurate multiclass probability estimates.* Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 694–699. — calibración isotónica multiclase.
14. **James, W. & Stein, C.** (1961). *Estimation with quadratic loss.* Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1, 361–379. — base teórica del shrinkage.
15. **Baio, G. & Blangiardo, M.** (2010). *Bayesian hierarchical model for the prediction of football results.* Journal of Applied Statistics, 37(2), 253–264. — mixture priors y shrinkage jerárquico.
16. **Leitner, C., Zeileis, A. & Hornik, K.** (2010). *Forecasting sports tournaments by ratings of (prob)abilities: a comparison for the EURO 2008.* International Journal of Forecasting, 26(3), 471–481. — modelo de consenso de casas de apuestas como benchmark.
17. **Boshnakov, G., Kharrat, T. & McHale, I. G.** (2017). *A bivariate Weibull count model for forecasting association football scores.* International Journal of Forecasting, 33(2), 458–466. — alternativa al Poisson con dispersión flexible.
18. **Dietterich, T. G.** (2000). *Ensemble methods in machine learning.* Multiple Classifier Systems, LNCS 1857, 1–15. — fundamento del ensemble averaging.
19. **Metropolis, N. & Ulam, S.** (1949). *The Monte Carlo Method.* Journal of the American Statistical Association, 44(247), 335–341. — referencia clásica del método.
20. **Chen, T. & Guestrin, C.** (2016). *XGBoost: A scalable tree boosting system.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. — referencia original de XGBoost.
21. **Friedman, J. H.** (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics, 29(5), 1189–1232. — fundamento teórico del gradient boosting.

### 8.3. Documentación oficial relevante

22. Documentación de scikit-learn 1.4.x — particularmente las páginas de `LogisticRegression`, `IsotonicRegression` y `CalibratedClassifierCV`.
23. Documentación de scipy 1.11.x — `scipy.stats.poisson` y `scipy.stats.skellam`.
24. Documentación de XGBoost 2.0 — especialmente la sección de objetivos `multi:softprob`.
25. Documentación de FastAPI 0.110 — patrones de validación con Pydantic v2.
26. Reglamento oficial de la FIFA World Cup 2026 (clasificación, tiebreakers, terceros lugares). *Por confirmar* la versión exacta vigente.

---

## 9. Relación entre implementación y teoría

| Módulo | Principio teórico | Referencia |
|---|---|---|
| [`src/ratings/elo.py`](src/ratings/elo.py) | Sistema Elo con K competition-aware, goal-margin multiplier y time decay | Elo (1978); Dixon & Coles (1997) para el decaimiento exponencial |
| [`src/ratings/pi_rating.py`](src/ratings/pi_rating.py) | Dynamic dual-rating home/away con dos tasas de aprendizaje | Constantinou & Fenton (2013) |
| [`src/ratings/form_rating.py`](src/ratings/form_rating.py) | Indicador de momentum sobre ventana móvil | Inferencia: práctica estándar en el dominio (cf. FiveThirtyEight SPI) |
| [`src/ratings/rating_ensemble.py`](src/ratings/rating_ensemble.py) | Ensemble averaging tras normalización z | Dietterich (2000) |
| [`src/ratings/shrinkage.py`](src/ratings/shrinkage.py) | Shrinkage hacia priors confederacionales | James & Stein (1961); Baio & Blangiardo (2010) como extensión jerárquica |
| [`src/models/poisson_model.py`](src/models/poisson_model.py) — fit | Estimación ponderada de tasas Poisson por equipo | Maher (1982); Dixon & Coles (1997) (time decay) |
| [`src/models/poisson_model.py`](src/models/poisson_model.py) — outcome | Skellam para diferencia de goles | Skellam (1946); Karlis & Ntzoufras (2009) |
| [`src/models/poisson_model.py`](src/models/poisson_model.py) — score grid | Corrección Dixon-Coles $\tau$ para marcadores bajos | Dixon & Coles (1997) |
| [`src/models/multinomial_model.py`](src/models/multinomial_model.py) | Regresión logística multinomial con estandarización | Cox (1958); Krishnapuram et al. (2005) |
| [`src/models/xgboost_model.py`](src/models/xgboost_model.py) | Gradient boosting de árboles | Friedman (2001); Chen & Guestrin (2016) |
| [`src/models/calibration.py`](src/models/calibration.py) | Calibración por clase (one-vs-rest) isotónica o Platt | Platt (1999); Zadrozny & Elkan (2002) |
| [`src/training/trainer.py`](src/training/trainer.py) — `_lasso_select_features` | Selección L1 multinomial | Tibshirani (1996); Krishnapuram et al. (2005); Groll et al. (2015) |
| [`src/ensemble/blender.py`](src/ensemble/blender.py) | Ensamblaje ponderado de probabilidades | Dietterich (2000) |
| [`src/ensemble/pick_optimizer.py`](src/ensemble/pick_optimizer.py) | Selección de pronósticos diferenciada por perfil de riesgo | Inferencia: literatura aplicada de pari-mutuel betting |
| [`src/simulation/tournament_simulator.py`](src/simulation/tournament_simulator.py) | Monte-Carlo del proceso completo | Metropolis & Ulam (1949) |
| [`src/simulation/group_stage.py`](src/simulation/group_stage.py) — `VectorizedGroupStageEngine` | Vectorización numpy de standings y tiebreakers FIFA | Inferencia: técnica de ingeniería estándar |
| [`src/simulation/knockout.py`](src/simulation/knockout.py) | Propagación estocástica de ganadores con proxy de penaltis | Inferencia + práctica de la industria |
| [`src/prediction/predictor.py`](src/prediction/predictor.py) | Predictor con degradación elegante cuando faltan features | Inferencia: patrón de robustez típico de sistemas ML en producción |

---

## 10. Limitaciones, supuestos y oportunidades de mejora

### 10.1. Limitaciones técnicas reconocidas

1. **Densidad muestral.** Las selecciones nacionales disputan ~10 partidos oficiales por año. El histórico utilizable (2014–2026) ofrece del orden de $10^4$ partidos en total, con apenas decenas por selección relevante. Esto pone un techo duro a la complejidad útil de los modelos: las arquitecturas de deep learning son **inviables** en este régimen.
2. **Heterogeneidad de cobertura.** StatsBomb Open Data está disponible solo para un subconjunto histórico. Sin instrumentación de tracking, los proxies de fatiga y viaje son aproximaciones funcionales pero ruidosas.
3. **Penaltis modelados como Bernoulli proxy.** La conversión real de penaltis varía entre selecciones (cf. tasas históricas publicadas por FIFA). El proxy actual asume que la diferencia de fuerza determina probabilísticamente la ganadora del tiroteo.
4. **Calibración potencialmente con leak parcial.** El calibrador isotónico se ajusta sobre `val_features`. *Por confirmar*: si el flujo actual garantiza que `val_features` es estrictamente posterior al conjunto de entrenamiento de XGBoost (split temporal con `validation_year=2025`), no hay leak. Una aserción explícita en `Trainer.fit` cerraría la garantía.
5. **Mixture prior implementado pero desactivado.** La mixtura élite/regular por confederación (Baio & Blangiardo, 2010) está implementada, pero el backtest cross-tournament mostró que los priors calibrados a mano empeoran el log-loss held-out; queda `enabled: false` a la espera de un re-tuneo validado (§2.8).
6. **Sin consenso de casas de apuestas.** La literatura (Leitner-Zeileis-Hornik 2010; Zeileis 2018) reporta de forma unánime que el consenso agregado de ≥10 casas, des-overround, define el benchmark a superar. El sistema no consume odds y por tanto no puede medirse contra ese benchmark.
7. **Valor de plantilla por proxy de nacionalidad.** El sistema **sí** consume valor de plantilla (§2.10), pero la agregación es por `country_of_citizenship`, no por convocatoria real: incluye ciudadanos no convocados y omite nacionalizados. El `top-11` mitiga el ruido; reconstruir convocatorias históricas reales (vía `game_lineups`) queda pendiente.
8. **Sin features StatsBomb.** El cliente existe pero no produce features. Recurso desperdiciado, aunque con ROI marginal en selecciones por baja muestra.
9. **Sin autenticación en la API.** Aceptable para uso individual; insuficiente para despliegue público.

### 10.2. Supuestos identificados en el código

1. **Independencia de Poissons** (excepto donde se aplica $\tau$ o se usa Skellam). La correlación residual entre goles del equipo local y visitante se asume despreciable. La literatura (Karlis & Ntzoufras, 2003) sugiere que un modelo bivariate Poisson con parámetro de covarianza captura mejor esta dependencia.
2. **Estacionariedad atenuada**. El decaimiento temporal asume que la importancia relativa de un partido decae exponencialmente. Esto es una aproximación; la verdadera función de decaimiento podría no ser monótona (e.g., un partido de cuartos de un Mundial podría ser más informativo que un partido de Liga de Naciones más reciente).
3. **Venue neutro en el Mundial**. La función `predict(..., neutral=True)` se invoca por defecto en la simulación. Esto descarta cualquier *home advantage* para anfitriones (USA, México, Canadá). El sistema actual no diferencia.
4. **Tiebreakers FIFA simplificados**. El motor vectorizado usa la cadena (puntos, GD, GF, ruido). El reglamento FIFA real incluye criterios adicionales como *fair-play points* y *head-to-head* que el sistema omite por simplicidad.
5. **Mapeo team→confederación incompleto**. El diccionario de confederaciones en [`src/ratings/shrinkage.py`](src/ratings/shrinkage.py) cubre ~210 selecciones; selecciones no mapeadas reciben un prior por defecto. *Por confirmar*: cobertura completa de las 48 selecciones del Mundial 2026.
6. **El Kaggle dataset cubre hasta ~2024**. La actualización del dataset es manual; no hay un mecanismo automático de re-descarga.

### 10.3. Oportunidades de mejora (priorizadas)

| ID | Mejora | Impacto esperado | Esfuerzo | Referencia |
|---|---|---|---|---|
| O.1 | Integrar consenso de odds de casas como prior/feature | Alto (benchmark establecido) | Medio | Leitner-Zeileis-Hornik (2010); Zeileis (2018) |
| ~~O.2~~ ✅ | **Hecho** — valor de plantilla (Kaggle `player-scores`), aceptado por backtest (§2.10) | Alto | Medio | Groll, Schauberger & Tutz (2015) |
| ~~O.3~~ ⏸️ | **Implementado pero desactivado** — mixture prior; empeoró el backtest (§2.8) | Medio | Bajo | Baio & Blangiardo (2010) |
| O.4 | PageRank como rating ortogonal al Elo | Medio | Medio | Hubáček et al. (2019) |
| O.5 | Aserción explícita de no-leak entre train y val del calibrador | Bajo | Trivial | — |
| ~~O.6~~ ✅ | **Hecho** — backtest cross-tournament leakage-free (`run_tournament_backtest`, §7.12) | Alto valor diagnóstico | Bajo | Groll et al. (2015, 2019) |
| O.7 | Modelo bivariate Poisson con covarianza explícita | Medio | Medio | Karlis & Ntzoufras (2003) |
| O.8 | Tiebreakers FIFA completos (fair-play, head-to-head) | Bajo | Medio | Reglamento FIFA 2026 |
| O.9 | Home advantage para anfitriones (USA/MEX/CAN) | Bajo–medio | Bajo | Práctica de la industria |
| O.10 | Autenticación en la API si se expone públicamente | Crítico para producción | Medio | Estándares OAuth2 / API key |

### 10.4. Mejoras desde perspectiva de ingeniería de software

- **Orquestación**: migrar del `.bat`/`Makefile` a Prefect o Airflow para visibilidad de runs.
- **Almacenamiento**: para volúmenes mayores, considerar un *backend* en SQLite o DuckDB para la tabla canónica de partidos.
- **Pruebas**: aumentar cobertura. `src/training/backtester.py` y `src/ensemble/blender.py` ya tienen tests (RPS, `collect_holdout_predictions`, blend weights); `src/training/{cross_validation,evaluator}` y `src/utils/{metrics,plotting,dates}` siguen sin cobertura.
- **CI/CD**: integrar GitHub Actions con `pytest`, `pylint` y un *smoke test* del pipeline completo en cada PR.
- **Versionado de modelos**: hoy los pkls se sobrescriben. Un esquema `models/<timestamp>/*.pkl` con un *symlink* a `latest` permitiría rollback.

---

## 11. Conclusión

El sistema **FIFA World Cup 2026 Quiniela Predictor V2** constituye una implementación rigurosa y modular del estado del arte aplicado en *forecasting* de torneos internacionales de fútbol. Combina técnicas clásicas y bien fundamentadas —Elo, Poisson, Dixon-Coles, Skellam, calibración isotónica— con métodos contemporáneos de aprendizaje automático —XGBoost, regularización L1, ensemble probabilístico ponderado— en una arquitectura en capas con dependencias unidireccionales y responsabilidades bien delimitadas.

Las decisiones de diseño son coherentes con la literatura: el patrón híbrido de Groll-Ley (2019) que utiliza ratings como features de un clasificador discriminativo; la corrección Dixon-Coles (1997) para marcadores bajos; el shrinkage bayesiano hacia priors confederacionales para mitigar las distorsiones de *strength of schedule*; el decaimiento temporal exponencial sobre la historia de entrenamiento. La incorporación de Skellam para H/D/A y la selección LASSO de features siguen recomendaciones explícitas de la literatura más reciente (Karlis & Ntzoufras, 2009; Groll, Schauberger & Tutz, 2015).

La capa de simulación Monte-Carlo permite estimar probabilidades de eventos compuestos —clasificación, llegar a determinada ronda, campeonato— que no admiten forma cerrada. Las optimizaciones implementadas (cache de predicciones, vectorización numpy en la fase de grupos) hacen viable ejecutar miles de simulaciones del torneo completo en tiempos del orden de minutos.

La separación explícita entre **estimación probabilística** (capa de modelos + calibrador + blender) y **selección de pronósticos** (capa de `pick_optimizer` con cuatro perfiles de riesgo) refleja una comprensión madura del dominio aplicado: la probabilidad subyacente es la misma, lo que cambia es la función de utilidad del jugador. Esta separación facilita además experimentar con perfiles adicionales sin tocar la capa de modelado.

Desde el upgrade de modelado de junio 2026, el sistema incorpora además el **valor de plantilla** (Groll et al.) como señal de talento ortogonal a los resultados — aceptado tras validarlo en un **backtest cross-tournament leakage-free** que se erige como el árbitro de todo cambio de modelado — y un **simulador que scorea cada emparejamiento con el modelo completo**, de modo que la distribución de campeón refleja talento y no solo historia de resultados.

Las limitaciones identificadas en §10 son conocidas y, en su mayoría, corresponden a *trade-offs* explícitos del proyecto: priorización de un sistema entendible y mantenible sobre uno que persiga el último 1% de log-loss a costa de complejidad infraestructural adicional (scrapers de odds en vivo, reconstrucción de convocatorias históricas). El sistema deja claramente identificadas, en el anexo A del [`CLAUDE.md`](CLAUDE.md), las mejoras pendientes con su impacto y esfuerzo estimado, lo que constituye una práctica deseable de ingeniería de software académica.

En conjunto, el proyecto demuestra que un sistema de pronóstico deportivo de calidad académica no requiere arquitecturas ML exóticas, sino la composición disciplinada de técnicas estadísticas bien fundamentadas, una ingeniería de software sólida y la conciencia explícita de las limitaciones del dominio aplicado.

---

**Fin del documento.**
