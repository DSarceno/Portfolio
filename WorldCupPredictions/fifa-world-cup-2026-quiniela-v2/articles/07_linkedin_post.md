En mi pronosticador del Mundial 2026, XGBoost no es EL modelo. Le di un voto de exactamente 0.30 dentro de un blend de cuatro. Y esa decisión de acotarlo es más defendible que la de usarlo.

El fútbol de selecciones es un problema de datos pequeños disfrazado de big data: ~10 partidos oficiales al año por equipo, ~10⁴ partidos útiles en total. Eso deja fuera al deep learning y apunta al estado del arte tabular: gradient boosted trees. Pero conviene entender DE DÓNDE sale su poder antes de confiarle el sistema.

Gradient boosting es descenso de gradiente… en el espacio de funciones. Construyes el modelo aditivo F_M = Σ ν·f_m de a un árbol por vez; cada árbol nuevo se ajusta al gradiente negativo de la pérdida (los pseudo-residuos), es decir aprende dónde y en qué dirección el ensemble se equivoca. XGBoost lo afina con un paso de segundo orden (gradiente g_i Y hessiano h_i) y un regularizador explícito Ω(f) = γT + ½λΣw².

Lo bonito: la regularización VIVE dentro de la matemática. Para una estructura fija, el peso óptimo de hoja es w_j* = −G_j/(H_j+λ) — λ encoge cada hoja hacia cero — y una división solo se acepta si su Gain supera γ (poda incorporada). No es un parche; es la fórmula.

En datos pequeños el enemigo es la varianza, así que la configuración va a contracorriente del instinto "modelo potente": árboles poco profundos (max_depth=5), shrinkage fuerte (ν=0.05) compensado con 400 árboles, 85% de submuestreo de filas y columnas, L2=1.0. Y aun así pesa solo 0.30: XGBoost no sabe NADA de cómo se generan los goles (eso es Poisson/Skellam) ni tiene un prior de fuerza (eso es Elo). Solo y sin frenos sobreajustaría rarezas de muestra pequeña.

Aprendizajes:

1. Boosting = descenso de gradiente funcional; el shrinkage es un tamaño de paso sub-relajado.

2. En XGBoost la regularización está en las ecuaciones: el peso de hoja −G/(H+λ) y el umbral −γ del split SON las penalizaciones.

3. En datos tabulares pequeños la pregunta no es "árboles vs deep learning" sino "cuánto regularizar y cuánto confiar".

4. El peso en el blend es una decisión de modelado tanto como la profundidad del árbol — y se valida igual: con el backtest sin fugas.

5. Calibra DESPUÉS de mezclar: óptimo en log-loss no implica calibrado cuando juntas cuatro modelos.

Cuando usas un learner potente, ¿lo coronas o lo acotas? ¿Qué peso le darías a tu mejor modelo dentro de un ensemble?

#MachineLearning #XGBoost #DataScience #Statistics #GradientBoosting
