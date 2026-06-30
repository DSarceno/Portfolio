Mi modelo del Mundial 2026 estaba perfectamente calibrado y validado por un backtest sin fugas. Y aun así, en producción, mintió. El culpable no fue la estadística: fue el álgebra de cómo se mezclan los datos nuevos con los viejos.

Durante un torneo en vivo el pipeline ya no corre una vez sobre datos congelados. Corre cada jornada, contra una tabla que editas a mano y con prisa. Ahí la correctitud deja de depender de tu función de pérdida y pasa a depender de tres propiedades: que tus operaciones de datos sean idempotentes, estables en la clave y validadas por esquema. Tres fallas reales me lo enseñaron.

1) El CSV que mentía (huérfanos). El merge original era un upsert por clave (date, team_a, team_b): U(E,N) = {e ∈ E : k(e) ∉ k(N)} ∪ N. Es idempotente al re-aplicar el mismo N… pero NO ante una corrección. Si arreglas una fecha o un nombre, la fila vieja queda con una clave que ya no está en N, así que el upsert no puede borrarla: sobrevive como huérfana y el equipo se cuenta doble. Re-correr no lo arregla. La solución: purgar por partición de origen y reingerir — R_s(E,N) = {e ∈ E : source(e) ≠ s} ∪ N — idempotente Y libre de huérfanos por construcción.

2) La zona horaria que forjaba partidos. La fecha canónica es utcDate truncado a 10 caracteres. Un partido nocturno en la costa oeste (20:00 UTC−7) es 03:00 UTC del día siguiente. Si capturas la fecha LOCAL, tu clave no colisiona con la del fixture: el resultado cae en un partido fantasma y el real queda "por jugar" para siempre.

3) La coma que se volvió TypeError. El CSV es posicional: una coma escrita como punto corre todas las columnas y mete un string en una columna numérica. Defensa en profundidad: coerción al calcular (to_numeric errors="coerce") + validación de aridad/dominio al ingerir.

Aprendizajes:

1. La idempotencia se diseña, no se asume. Si re-correr cambia el resultado, tu merge no es idempotente.

2. Una clave primaria editable es una clave que crea huérfanos. Purga por origen en vez de upsert por clave.

3. Las zonas horarias son aritmética y el truncamiento pierde información. Una zona canónica (UTC) y haz que la clave correcta sea la fácil.

4. Correcto hoy ≠ correcto mañana: un estimador cuyo soporte depende de datos en vivo puede responder otra pregunta sin que una sola línea cambie.

¿Cuál ha sido tu bug de integridad de datos más caro en producción — el huérfano, la zona horaria, o el esquema que nadie validó?

#DataEngineering #MLOps #DataScience #Python #SoftwareEngineering
