# Sprint 3 — Orquestacion Agentica con LangGraph

**Grupo 3: Implementacion de busqueda hibrida con ontologia**

## Objetivos

- Configurar el grafo en LangGraph
- Implementar busqueda hibrida con filtrado por ontologia
- El agente usa la ontologia para filtrar el espacio de busqueda antes de aplicar similitud de coseno

## Implementacion

### Grafo LangGraph (`src/agents/graph.py`)

El grafo de estado (`StateGraph[AgentState]`) tiene 8 nodos y 3 edges condicionales:

```
START
  -> inicializar
      |-> [tiene imagen] -> procesar_imagen -> clasificar
      |-> [no imagen] --------------------> clasificar
                                              |-> [en temario] -> generar_consulta -> buscar -> filtrar -> generar_respuesta -> finalizar
                                              |-> [fuera temario] -> finalizar
```

### Busqueda Hibrida con Ontologia

El nodo `clasificar` usa `ClasificadorSemantico` para:
1. Generar embedding CLIP del query
2. Comparar con embeddings de las 10 anclas semanticas del temario
3. Si max_sim >= 0.45: query dentro del dominio, procede a busqueda
4. Si max_sim < 0.45: query fuera del dominio, respuesta generica

Esto asegura que solo se busque en Qdrant cuando la consulta es relevante al dominio de histologia.

### Nodo `buscar` (search.py)

El nodo de busqueda hibrida:
1. Recibe el text_embedding y opcionalmente image_embedding
2. Ajusta thresholds segun el modo (texto/multimodal/solicitud_imagenes)
3. Ejecuta busqueda en colecciones de texto e imagenes en paralelo
4. Retorna resultados combinados

### Modos de Operacion

| Modo | Trigger | Comportamiento |
|---|---|---|
| texto | Sin imagen, sin solicitud | Solo busqueda de texto |
| multimodal | Con imagen subida | Busqueda texto + imagen |
| solicitud_imagenes | "mostrame imagen de..." | Busqueda texto->imagen |

### Filtrado por Entidades

El nodo `clasificar` extrae entidades histologicas (tejidos, estructuras, tinciones) via LLM, que se usan para refinar la consulta de busqueda en `generar_consulta`.

## Mejoras Futuras (Sprint 4+)

- A2A Protocol: agente de busqueda <-> agente de explicacion medica
- CopilotKit: UI con Canvas para co-edicion
- RAGAS: evaluacion de Faithfulness y Context Recall
