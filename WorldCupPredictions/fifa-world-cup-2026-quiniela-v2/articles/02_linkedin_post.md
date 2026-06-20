Ajusté un prior, re-corrí el pipeline, y la tabla de campeones se veía mejor: España #1, Brasil de vuelta en el top 5. Se sentía como progreso. Era la trampa más peligrosa del proyecto.

En mi sistema de pronóstico del Mundial 2026 escribí una regla incómoda: la validez aparente no es evidencia. Que la lista de campeones se vea "realista" no dice NADA sobre si las probabilidades por partido mejoraron.

El problema: un rating basado en resultados sube a un equipo que sobre-rinde (Marruecos) y hunde a una potencia en mala racha (Brasil). Eso no es un bug — es el output honesto del modelo. Pero invita a "arreglarlo" a ojo, y ahí empieza el desastre.

La solución: un backtest cross-tournament SIN fuga de datos como único árbitro. Entreno todo el stack solo con partidos de año < Y, evalúo en el torneo == Y, y garantizo que ratings y features de test salen solo de datos pre-Y (as-of join con merge_asof backward). Métrica de decisión fijada de antemano: log-loss held-out. No accuracy. No intuición.

El árbitro dio dos veredictos que contradijeron la intuición en direcciones opuestas:

1. Valor de plantilla (Transfermarkt): ACEPTADO. Log-loss 0.9994 → 0.9905, Brier 0.1976 → 0.1955, RPS 0.2117 → 0.2082. La accuracy NO se movió: si esa hubiera sido la métrica, una buena feature muere.

2. Mixture prior élite/regular: RECHAZADO. Hacía la tabla MÁS realista (España #1) pero empeoraba log-loss +2.3% y accuracy −2.4 pts. Mejoró lo que no importa, degradó lo que sí. Quedó enabled: false.

3. Barrido de pesos del ensemble y del shrinkage: NEUTRO. Lección deflacionaria y valiosa: los knobs ya estaban casi óptimos. La palanca real era señal ortogonal (talento), no tunear.

4. La prevención de leakage es mecánica, no aspiracional: as-of join + ratings fit-on-train/apply-on-test. No es "tener cuidado", es un mecanismo testeable.

La moraleja general: en cualquier dominio donde el output sea lo bastante plausible para engañarte, construye el árbitro PRIMERO y luego deja que te lleve la contraria.

¿Cómo separan ustedes "se ve bien" de "está mejor calibrado" en sus proyectos?

#MachineLearning #MLOps #DataScience #ModelEvaluation #Backtesting #DataLeakage
