Una tabla de memoización y una reescritura en NumPy convirtieron una simulación intratable en un trabajo de pausa-para-café: 200× a 1500× más rápido.

No existe forma cerrada para "P(este equipo gana el Mundial)". El cuadro es endógeno: contra quién juegas en 32avos depende de cómo quedó la fase de grupos, que depende de los mismos resultados que intentas propagar. Sin forma analítica, Monte-Carlo: simula el proceso miles de veces y cuenta.

El problema: con 48 equipos, miles de corridas y un modelo de probabilidad costoso en el loop interno, eso se vuelve un trabajo de toda la noche. Y el error de Monte-Carlo cae solo como O(N^-1/2): para reducirlo a la mitad necesitas 4× las muestras. No puedes comprar precisión barata subiendo N — cada factor constante del costo por simulación pesa muchísimo.

El insight clave: solo hay 48 equipos, o sea n(n-1)=2256 enfrentamientos ordenados distintos. Las miles de simulaciones le hacen al modelo la MISMA pregunta una y otra vez. Memoización de manual.

Precomputo la matriz completa una vez antes del loop. La transformación de complejidad es toda la historia:
O(N · m · cost(predict)) → O(n² · cost(predict) + N · m)
Dentro del loop, cada "predicción" es un lookup O(1) en diccionario.

Aprendizajes:

1. Busca la pregunta repetida. La memoización es la optimización de mayor palanca cuando el espacio de entrada es pequeño y la función es pura.

2. Vectoriza el bookkeeping, no solo la matemática. np.lexsort hizo los desempates FIFA (puntos, dif. de gol, goles a favor + ruido aleatorio) más rápidos Y más legibles que un loop de ranking en Python.

3. El rendimiento compra correctitud. El cache no solo aceleró: hizo asequible correr el MODELO COMPLETO en el loop (antes caía a ratings+Poisson por features ausentes). Resultado: Brasil #15→#5, España #1 al 12%. La distribución por fin reflejó talento, no solo resultados.

4. Amortiza warmups reusando estado: el cache tarda ~45s en construirse; reusa el simulador, no lo reconstruyas por corrida.

5. Rápido y mal sigue siendo mal: un bug de setdefault dejaba a TODOS los equipos congelados en "32avos". Un simulador veloz que registra lo incorrecto solo está equivocado, más rápido.

¿Memoización o paralelización como primer recurso cuando el loop interno es el cuello de botella?

#Python #Performance #MonteCarlo #DataScience #NumPy #Optimization
