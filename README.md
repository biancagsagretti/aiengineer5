# Agente de razonamiento cíclico con memoria persistente

Pre-entrega 5. Agente **ReAct** construido con **LangGraph** que decide por sí mismo cuándo usar herramientas, razona con el resultado (reintentando si hay error) y recuerda la conversación entre mensajes gracias a un checkpointer **SQLite**.

## Arquitectura

```
START → [modelo] ──(tools_condition)──→ [tools] ─┐
           ↑            │ sin tool_calls          │
           └────────────┼─────────────────────────┘
                        ↓
                       END
```

| Archivo | Contenido |
|---|---|
| `src/tools.py` | Herramientas `@tool` async: `buscar_pedidos` y `obtener_ultimo_pedido` (BD simulada) |
| `src/graph.py` | `AgentState(MessagesState)`, nodo del modelo con `llm.bind_tools()`, `ToolNode`, `tools_condition`, recorte de historial |
| `src/main.py` | `asyncio` + `AsyncSqliteSaver`, escenarios de prueba, chat interactivo y generación de la traza |
| `traces/ejemplo_traza.json` | Traza del ciclo ReAct |

## Cómo levantar el entorno

Requiere **Python 3.12+**.

```bash
# 1. Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # o: poetry install / pip install -e .

# 2. Variables de entorno (las claves NO van al repo)
cp .env.example .env
# editá .env y elegí LLM_PROVIDER (anthropic, openai o gemini) y completá la clave
# correspondiente: ANTHROPIC_API_KEY, OPENAI_API_KEY o GOOGLE_API_KEY

# 3. Correr los escenarios de prueba (genera traces/ejemplo_traza.json)
python -m src.main

# 4. Chat interactivo con memoria
python -m src.main --chat --thread ana
```

Si cerrás el chat y volvés a abrirlo con el mismo `--thread`, el agente retoma la conversación porque el estado quedó guardado en `checkpoints.sqlite`.

## Pruebas incluidas

1. **Multi-paso:** *"¿Cuál de los clientes 102 y 205 gastó más, y cuánto?"* → el agente llama a `buscar_pedidos` dos veces (una por cliente) y compara.
2. **Memoria (mismo `thread_id`):** *"¿Y cuál fue el último pedido de ese cliente?"* → no nombra al cliente; el agente lo recuerda y llama a `obtener_ultimo_pedido`.
3. **Ciclo de retorno ante error:** *"¿Cuántos pedidos tuvo el cliente 999?"* → la herramienta devuelve un error con los clientes disponibles y el agente lo explica o pide aclaración.

## Criterios de aceptación

- **Autonomía:** no hay `if/else` manual; `tools_condition` enruta según los `tool_calls` que emite el modelo.
- **Ciclo de retorno:** la arista `tools → modelo` devuelve el resultado (o el error) al LLM, que decide reintentar, responder o aclarar.
- **Resiliencia de estado:** `AsyncSqliteSaver` + `thread_id` en `config["configurable"]`.
- **Código limpio:** Python 3.12, type hints y `async/await` en herramientas, nodos y ejecución.

## Decisiones de diseño

- **`recursion_limit = 10`** en cada invocación (`RECURSION_LIMIT` en `src/main.py`); si se alcanza, se captura `GraphRecursionError` y se informa en vez de entrar en un bucle.
- **Control del tamaño del estado:** el estado se acumula en SQLite, pero al modelo solo se le envían los últimos 30 mensajes (`trim_messages`, `MAX_MENSAJES_CONTEXTO`). Una mejora posible es resumir los mensajes viejos con el propio LLM.
- **Errores accionables:** las herramientas devuelven `{"error": ..., "clientes_disponibles": [...]}` para que el agente pueda corregir el ID y reintentar.
- **Docstrings descriptivos:** el LLM elige la herramienta solo por su descripción; cada docstring indica cuándo usarla, cuándo no y qué devuelve.
- `SqliteSaver` es para desarrollo local; en producción se reemplazaría por `AsyncPostgresSaver`.

## Traza de ejecución

`traces/ejemplo_traza.json` contiene, por escenario y turno, cada paso del ciclo (decisión de herramienta, resultado, respuesta final) y la cantidad de llamadas a herramientas. Ejemplo resumido:

```
Usuario: ¿Cuál de los clientes 102 y 205 gastó más, y cuánto?
  → El agente decide usar: buscar_pedidos({'cliente_id': 102})
  ← La herramienta devuelve: {"cliente_id": 102, "pedidos": 3, "total": 14500}
  → El agente decide usar: buscar_pedidos({'cliente_id': 205})
  ← La herramienta devuelve: {"cliente_id": 205, "pedidos": 2, "total": 22000}
  ✔ Respuesta: El cliente 205 gastó más: $22.000 ...
```

## Seguridad

`.env` y `*.sqlite` están en `.gitignore`. Solo se versiona `.env.example`, sin claves.
