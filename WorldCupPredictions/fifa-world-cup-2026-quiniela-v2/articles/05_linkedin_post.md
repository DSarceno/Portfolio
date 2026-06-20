Mi pronosticador del Mundial 2026 entrega CUATRO hojas de pronóstico distintas a partir de un único forecast idéntico. No es indecisión: es teoría de decisión.

La mayoría de los data scientists dirían que un buen sistema de forecasting "produce probabilidades calibradas". Es necesario, pero es la mitad del sistema. Una probabilidad es una creencia sobre el mundo; una decisión es la acción que tomas dada esa creencia y lo que está en juego. No son el mismo objeto, y confundirlos es uno de los errores silenciosos más comunes en ML aplicado.

El objetivo del sistema no es accuracy académica, sino maximizar el valor esperado en quinielas. Y la apuesta óptima depende de tu utilidad y de lo que apuestan los demás. Por eso: una capa de probabilidad y una capa de selección de picks, deliberadamente desacopladas.

El insight: pick* = argmax_a E_P[U(a, resultado)]. Si U fuera "1 si aciertas, 0 si no", el pick óptimo es argmax(p) y la decisión colapsa en la probabilidad. Pero en una quiniela tu pago depende de qué eligieron los demás (pari-mutuel): acertar lo que todos acertaron paga poco; acertar lo que pocos acertaron paga mucho. La utilidad es COMPETITIVA. Puedes ser el jugador más preciso del pozo y aun así perder.

Un forecast calibrado (p_H, p_D, p_A) alimenta cuatro políticas de decisión:
- safe: argmax estricto, favoritos claros.
- balanced: empate si ningún favorito supera 0.50.
- aggressive: toma la sorpresa solo si co-disparan upset_window ≥ 0.5 Y fragilidad ≥ 0.5 Y el underdog supera la tolerancia. Apuesta disciplinada, no capricho.
- contrarian: fade al equipo con mayor sesgo público; sacrifica accuracy por diferenciación. Pari-mutuel puro en una regla.

Aprendizajes:

1. argmax(p) es óptimo SOLO bajo pérdida 0/1. En cualquier pozo, mercado o subasta, la mejor acción se aparta del resultado más probable.

2. El desacople mantiene honesta tu calibración: un solo forecast validado, muchas políticas. Nunca doblas las probabilidades para servir a una estrategia.

3. Mide cada capa con su métrica: reglas de scoring propias (log-loss) para la creencia; un score de quiniela específico del juego para la decisión.

4. El patrón Strategy ES teoría de decisión en código: cuatro perfiles son cuatro funciones de utilidad, no cuatro modelos.

Cuando el output de tu modelo alimenta una decisión real, ¿qué función de utilidad la gobierna de verdad? Casi nunca es la 0/1 que tu argmax asume en silencio.

#DataScience #DecisionTheory #MachineLearning #GameTheory #SportsAnalytics #ExpectedValue
