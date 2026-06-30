# Example Session (Worked Example)

This is a complete worked example of the skill in action. The intake and section headings
are in English for clarity; **all publish-ready post copy is in Spanish**, per the language
policy. Numbers and examples below are illustrative and labeled as such.

---

## 1. User Request

> "Use the LinkedIn growth skill. My topic is *machine learning en producción para equipos
> pequeños* (MLOps for small data teams)."

## 2. Clarifying Questions (asked by the skill)

The skill asks the 10 questions. The user answers:

1. **Topic:** Llevar modelos de ML a producción en equipos pequeños (1–3 personas).
2. **Main goal:** Build authority (and attract a few consulting clients).
3. **Target audience:** Data scientists y fundadores técnicos de startups sin equipo de
   plataforma dedicado.
4. **LinkedIn level:** Intermediate (~900 connections).
5. **Frequency:** 3x per week.
6. **Tone:** Educational + bold.
7. **Depth:** Mixed.
8. **Formats:** Carousels + text.
9. **Offer:** Asesoría de MLOps por horas + plantillas open-source.
10. **Avoid:** Nada de "hype" de AGI, sin promesas de resultados garantizados.

`period_days`: 30 (default).

---

## 3. Generated Output

### A. Strategy Summary

- **Positioning:** El ingeniero que hace que el ML *funcione fuera del notebook* sin un
  equipo de plataforma.
- **Audience angle:** Habla a quien ya entrena modelos pero se atasca al desplegar,
  monitorear y mantener.
- **Main message:** En equipos pequeños, la disciplina operativa vale más que el modelo
  más sofisticado.
- **Differentiation:** Foco en lo "aburrido pero decisivo" (reproducibilidad, monitoreo,
  rollback) frente al contenido de moda sobre modelos nuevos.
- **Growth objective:** Autoridad + flujo constante de consultas de asesoría.

### B. Content Pillars

| # | Pillar | Purpose | Audience value | Example ideas | Best format |
|---|--------|---------|----------------|---------------|-------------|
| 1 | Fundamentos de MLOps | Educar | Bases claras | "Qué es un pipeline reproducible y por qué tu notebook no lo es" | Carrusel |
| 2 | Errores en producción | Autoridad/POV | Evitar dolores | "El modelo no falló: falló tu monitoreo" | Texto |
| 3 | Pruebas de la vida real | Prueba/Historia | Confianza | "Cómo recuperé un despliegue roto en 20 min" | Texto |
| 4 | Herramientas mínimas viables | Educar | Stack accesible | "El stack de MLOps más barato que de verdad funciona" | Carrusel |
| 5 | Conversación de comunidad | Engage | Diálogo | Encuesta: "¿Qué te cuesta más: desplegar o monitorear?" | Encuesta |
| 6 | De experimento a negocio (Oferta) | Convertir | Pasar a la acción | "Cuándo conviene traer ayuda externa de MLOps" | Texto |

### C. Chronogram (excerpt — first week of a 30-day, 3x/week plan)

| Day | Pillar | Topic | Hook | Format | Goal | CTA | Image idea |
|-----|--------|-------|------|--------|------|-----|------------|
| Día 1 (Lun) | P1 Fundamentos | Pipeline reproducible vs. notebook | "Tu notebook no es un pipeline. Y eso te está costando despliegues." | Carrusel | Saves + autoridad | "Guárdalo antes de tu próximo deploy." | Carrusel: notebook caótico → pipeline ordenado |
| Día 3 (Mié) | P2 Errores | Monitoreo, no el modelo | "El 80% de los 'modelos malos' en realidad son modelos sin monitoreo." (*cifra ilustrativa*) | Texto | Comentarios | "¿Cómo monitoreas tú? Cuéntame abajo." | Panel de monitoreo simple |
| Día 5 (Vie) | P4 Herramientas | Stack mínimo viable | "No necesitas Kubernetes para tu primer modelo en producción." | Carrusel | Saves + clics | "Te dejo la plantilla en comentarios." | Diagrama de stack de 4 piezas |

> The remaining 23 days are scheduled in the full chronogram with rotated pillars; full
> posts are drafted in batches. Below are the **three full posts for week 1**.

### D. Full Posts (Spanish, ready to copy)

---

**Post 1 — Carrusel (Día 1, Pilar: Fundamentos)**

```
Caption:

Tu notebook no es un pipeline. Y eso te está costando despliegues.

Un notebook prueba que un modelo PUEDE funcionar.
Un pipeline garantiza que VUELVE a funcionar mañana, sin ti delante.

La diferencia no es sofisticación. Es disciplina.

Guárdalo antes de tu próximo deploy. 👇

¿Tu última puesta en producción la hiciste desde un notebook? Cuéntame cómo te fue.

#MLOps #MachineLearning #DataScience #MLEngineering #IA

---

Slides:

1. (Portada) "Tu notebook no es un pipeline"
2. El problema: ejecuciones manuales, orden de celdas, rutas locales.
3. Qué hace un pipeline: pasos versionados + datos versionados + entorno fijo.
4. Reproducibilidad = mismo input → mismo output, en otra máquina.
5. Mínimo viable: script + control de versiones + entorno declarado.
6. (Cierre) "Si no se reproduce, no está en producción. Guárdalo. 👇"
```

*Quality check:* hook específico ✓, una idea ✓, valor concreto ✓, CTA claro ✓, sin datos
inventados ✓, en español ✓.

---

**Post 2 — Texto (Día 3, Pilar: Errores en producción)**

```
El 80% de los "modelos malos" en realidad son modelos sin monitoreo.
(Es mi estimación por experiencia, no un estudio formal.)

Lo he visto muchas veces:
el modelo se entrenó bien, pasó validación, se desplegó…
y tres semanas después "dejó de servir".

Pero el modelo no cambió. Cambió el mundo:
— Llegaron datos con otra distribución.
— Una columna empezó a venir vacía.
— El comportamiento del usuario se movió.

Sin monitoreo, te enteras por una queja, no por una alerta.

Lo mínimo que deberías vigilar:
1. Distribución de las entradas (data drift).
2. Tasa de errores y latencia del servicio.
3. Una métrica de negocio ligada al modelo.

No necesitas una plataforma cara. Necesitas mirar, a propósito y a tiempo.

¿Qué monitoreas tú hoy? ¿Y qué te gustaría monitorear pero aún no? Cuéntame abajo.

#MLOps #MachineLearning #DataScience #MLEngineering
```

*Quality check:* afirmación etiquetada como estimación ✓ (no inventa estudio), valor
accionable ✓, pregunta de engagement ✓.

---

**Post 3 — Carrusel (Día 5, Pilar: Herramientas mínimas viables)**

```
Caption:

No necesitas Kubernetes para tu primer modelo en producción.

La mayoría de equipos pequeños se paraliza eligiendo infraestructura
que no usará hasta tener 100x más tráfico.

Aquí tienes un stack mínimo que sí funciona hoy.
Te dejo la plantilla en comentarios. 👇

¿Qué pieza de tu stack te costó más decidir?

#MLOps #MachineLearning #DataScience #IA

---

Slides:

1. (Portada) "Stack de MLOps mínimo viable"
2. Versionado: Git + un registro simple de modelos.
3. Servicio: una API ligera (no orquestador completo).
4. Monitoreo: logs + una métrica de drift.
5. Automatización: un job programado, no un clúster.
6. (Cierre) "Empieza simple. Escala cuando duela, no antes. Plantilla 👇"
```

*Quality check:* postura clara (bold) ✓, formato variado respecto al post 1 ✓, CTA hacia la
oferta (plantilla) ✓.

### E. Image / Carousel Prompts

**For Post 1 (cover slide):**
> **Visual concept:** A messy tangle of notebook cells on the left transforming into a
> clean, ordered pipeline of connected blocks on the right.
> **Style:** clean minimal flat illustration, editorial.
> **Layout:** left-to-right transformation, focal arrow in center.
> **Text overlay (ES):** "Notebook ≠ Pipeline".
> **Color palette:** deep navy, white, accent teal.
> **Audience mood:** clarifying, credible.
> **Avoided elements:** stock photos, robots, fake metrics, clutter.

**For Post 2:**
> **Visual concept:** A calm dashboard with one alert quietly lighting up — the early
> warning the team almost missed.
> **Style:** minimal UI illustration.
> **Layout:** single centered panel, one highlighted metric.
> **Text overlay (ES):** "El modelo no falló. Falló el monitoreo."
> **Color palette:** navy, white, amber accent.
> **Audience mood:** analytical, slightly cautionary.
> **Avoided elements:** dense fake charts, precise invented numbers.

**For Post 3 (cover slide):**
> **Visual concept:** Four simple building blocks stacked into a small, sturdy structure —
> minimalism over complexity.
> **Style:** flat geometric illustration.
> **Layout:** four labeled blocks, generous negative space.
> **Text overlay (ES):** "Stack mínimo viable".
> **Color palette:** navy, white, teal (consistent with Post 1).
> **Audience mood:** reassuring, practical.
> **Avoided elements:** Kubernetes logos, overwhelming architecture diagrams.

### F. Growth Strategy

- **Best times (general):** martes a jueves, mañana o mediodía; confirmar con tus analíticas
  tras 2–3 semanas.
- **Commenting:** 10–15 min antes y después de publicar, comentarios sustantivos en posts de
  data science y startups.
- **Networking:** conectar con DS y fundadores técnicos con nota breve y específica.
- **Repurposing:** convertir el carrusel del Día 1 en post de texto a las 3 semanas; dividir
  un artículo largo en 3 posts.
- **Engagement loops:** responder cada comentario en la primera hora; usar la encuesta del
  Pilar 5 y luego publicar un análisis de resultados.
- **Weekly metrics:** impresiones, tasa de engagement, comentarios + guardados, vistas de
  perfil, seguidores, DMs/consultas de asesoría.
- **Improve:** duplicar el pilar/formato del mejor post; diagnosticar los flojos (¿hook,
  tema, hora o CTA?).

### G. Assumptions

- Posting days set to Mon/Wed/Fri (user said 3x/week but not which days).
- Target audience refined to "data scientists y fundadores técnicos de startups" from the
  user's answer.
- The "80%" figure in Post 2 is explicitly labeled as the author's experience-based
  estimate, not a cited statistic.
- Visual identity standardized on navy + teal across the calendar for feed consistency.

**What to do next:** review the strategy and week-1 posts; tell me which to adjust, and I'll
draft week 2 (Días 8–12) next.
