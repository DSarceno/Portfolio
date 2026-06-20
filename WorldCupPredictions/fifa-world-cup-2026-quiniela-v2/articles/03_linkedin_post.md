Durante meses, un bloque de configuración de mi sistema ML no hacía absolutamente nada. Podías editar el YAML todo el día y no cambiaba ni una probabilidad.

Los modelos rara vez matan a un sistema de ML. Lo mata la arquitectura: una feature que se vuelve NaN tras un dropna, un config que nadie lee, un artefacto que no se deserializa, una etapa que sobreescribe el archivo del que todo depende.

En mi pronosticador del Mundial 2026 (60 módulos Python, pipeline end-to-end que se re-corre cada jornada), las decisiones de arquitectura importan más que cualquier ecuación. Las cuatro que me sostienen:

1. Capas con dependencias estrictamente unidireccionales (utils → data → ratings → features → models → ensemble → simulation → api). Cualquier import "hacia atrás" es code smell. El grafo es un DAG y eso permite testear y reemplazar cada capa por separado.

2. Una sola fuente de verdad: matches_unified.csv. Todo lo demás se deriva. Concentra el riesgo en un solo lugar — por eso se valida con validate_match_dataframe antes de cada escritura, y el dedup por (date, team_a, team_b) keep="last" hace que un resultado real sobreescriba al fixture placeholder.

3. Aislamiento por archivos: cada etapa lee de disco, escribe a disco, sale. Re-ejecutable, auditable, reproducible. A 50 MB no necesitas base de datos ni Airflow — sobre-diseñar también es un fallo.

4. Degradación elegante como feature: el predictor salta XGBoost/multinomial si faltan columnas; la API responde aunque no existan los .pkl. Una probabilidad un poco peor es más útil que una excepción.

¿Por qué tanta disciplina? Por las cicatrices, todas silenciosas:
- ensemble.weights era config muerta: Config.get() solo leía config.yaml, pero los pesos vivían en model_params.yaml. Un knob que no hace nada es PEOR que no tener knob: fabrica confianza falsa.
- out["col"] = raw["col"] tras un dropna sin reset_index(drop=True): pandas alinea por índice y casi todo queda NaN. Sin excepción.
- defaultdict(lambda: ...) rompe joblib.dump. Hubo que implementar __getstate__/__setstate__.

Todos comparten ADN: fallan en silencio. El trabajo real de la arquitectura es volver ruidosos los fallos silenciosos.

¿Cuál fue su bug "silencioso" más caro: config muerta, misalignment de índices, o un pickle roto?

#MLOps #SoftwareEngineering #MachineLearning #DataEngineering #SystemDesign #Reproducibility
