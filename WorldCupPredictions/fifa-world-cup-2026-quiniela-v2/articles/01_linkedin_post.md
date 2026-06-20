Para sacar P(Local/Empate/Visita) de un modelo de goles, casi todo el mundo integra una matriz de marcadores. Es justo donde se esconden los peores bugs.

En mi sistema de pronóstico para el Mundial 2026, la decisión matemática más importante fue dejar de integrar el grid de marcadores y usar la distribución de Skellam.

El problema: cada selección juega ~10 partidos oficiales al año. El histórico útil es del orden de 10⁴ partidos, decenas por equipo relevante. Ese régimen de muestra pequeña descarta el deep learning y obliga a modelos estructurales: Poisson para goles, Elo para fuerza, un poco de boosting encima.

El insight: si los goles de cada equipo son Poisson independientes, la diferencia de goles D = G_a − G_b sigue una Skellam, que tiene forma cerrada. P(Local)=P(D>0), P(Empate)=P(D=0), P(Visita)=P(D<0). Tres llamadas a scipy.stats.skellam. Sin truncar la matriz, sin renormalizar, sin corrección τ ad-hoc — y calibra mejor los empates (Karlis & Ntzoufras, 2009), que es donde estos modelos fallan.

Aprendizajes que me llevo:

1. Una forma cerrada no solo es elegante: elimina una clase entera de bugs. El grid tenía los triángulos invertidos (Local se leía del triángulo equivocado) y el Poisson pesaba ~50% en la simulación. Resultado: Paraguay y Panamá por encima de España y Brasil para campeón. Un bug probabilístico NO truena, produce tonterías confiadas y plausibles.

2. Hibridiza por pregunta, no por dogma: Skellam para 1X2, grid Dixon-Coles para marcador exacto (la quiniela paga ambos).

3. El sentido de un ensemble es la ortogonalidad: Elo (largo plazo), PI rating (asimetría local/visita), forma (momentum) y valor de plantilla (talento). El z-score los hace sumables.

4. En muestra pequeña, el shrinkage de James-Stein hacia priors por confederación no es opcional: sin él, la fuerza de calendario infla a los equipos débiles.

5. Solo medible: la mejora del valor de plantilla (log-loss 0.9994 → 0.9905) se aceptó por backtest, no por cómo se "veía" la tabla.

¿Ustedes integran el grid o se van directo a Skellam para outcomes? ¿Han cazado bugs probabilísticos silenciosos en producción?

#DataScience #MachineLearning #Estadistica #SportsAnalytics #Football #Poisson
