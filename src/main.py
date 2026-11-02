"""Punto de entrada: corre los escenarios de prueba o un chat interactivo.

Uso:
    python -m src.main                      # demo + guarda la traza en traces/
    python -m src.main --chat --thread ana  # chat interactivo con memoria
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.graph import build_graph, get_llm

RECURSION_LIMIT = 10  # techo de pasos del grafo: evita bucles infinitos y costos

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("agente")


def _texto(content: str | list[Any]) -> str:
    """Normaliza el contenido de un mensaje (Gemini puede devolver bloques)."""
    if isinstance(content, str):
        return content
    partes = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(partes)


def _serializar(msg: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje a un dict simple para el log / JSON."""
    data: dict[str, Any] = {"tipo": msg.type, "contenido": _texto(msg.content)}
    if isinstance(msg, AIMessage) and msg.tool_calls:
        data["tool_calls"] = [
            {"nombre": tc["name"], "args": tc["args"]} for tc in msg.tool_calls
        ]
    if isinstance(msg, ToolMessage):
        data["herramienta"] = msg.name
    return data


def _loguear(paso: dict[str, Any]) -> None:
    """Imprime un paso con el estilo de la traza ReAct."""
    for tc in paso.get("tool_calls", []):
        log.info("  → El agente decide usar: %s(%s)", tc["nombre"], tc["args"])
    if paso["tipo"] == "tool":
        log.info("  ← La herramienta devuelve: %s", paso["contenido"])
    if paso["tipo"] == "ai" and not paso.get("tool_calls"):
        log.info("  ✔ Respuesta: %s", paso["contenido"])


async def ejecutar_turno(
    graph: CompiledStateGraph, thread_id: str, pregunta: str
) -> dict[str, Any]:
    """Envía una pregunta al agente y devuelve la traza del turno."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    log.info("\nUsuario [%s]: %s", thread_id, pregunta)

    pasos: list[dict[str, Any]] = []
    respuesta_final = ""
    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(content=pregunta)]},
            config,
            stream_mode="updates",
        ):
            for nodo, payload in update.items():
                for msg in (payload or {}).get("messages", []):
                    paso = {"nodo": nodo, **_serializar(msg)}
                    pasos.append(paso)
                    _loguear(paso)
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        respuesta_final = _texto(msg.content)
    except GraphRecursionError:
        respuesta_final = f"[Cortado: se alcanzó recursion_limit={RECURSION_LIMIT}]"
        log.warning("  ⚠ %s", respuesta_final)

    return {
        "usuario": pregunta,
        "pasos": pasos,
        "llamadas_a_herramientas": sum(len(p.get("tool_calls", [])) for p in pasos),
        "respuesta_final": respuesta_final,
    }


async def correr_demo(graph: CompiledStateGraph, salida: Path) -> None:
    """Corre los escenarios de prueba y guarda la traza en JSON."""
    sufijo = uuid.uuid4().hex[:6]  # thread nuevo en cada corrida
    escenarios: list[dict[str, Any]] = []

    # 1) Multi-paso + memoria: mismo thread_id en ambos turnos.
    thread = f"demo-multipaso-{sufijo}"
    log.info("\n=== Escenario 1: razonamiento multi-paso + memoria ===")
    turnos = [
        await ejecutar_turno(
            graph, thread, "¿Cuál de los clientes 102 y 205 gastó más, y cuánto?"
        ),
        # No nombra al cliente: solo funciona si recuerda el turno anterior.
        await ejecutar_turno(
            graph, thread, "¿Y cuál fue el último pedido de ese cliente?"
        ),
    ]
    escenarios.append(
        {"nombre": "multipaso_y_memoria", "thread_id": thread, "turnos": turnos}
    )

    # 2) Ciclo de retorno: la herramienta devuelve error (cliente inexistente).
    thread = f"demo-error-{sufijo}"
    log.info("\n=== Escenario 2: manejo de error de herramienta ===")
    turnos = [
        await ejecutar_turno(graph, thread, "¿Cuántos pedidos tuvo el cliente 999?")
    ]
    escenarios.append(
        {"nombre": "error_y_reintento", "thread_id": thread, "turnos": turnos}
    )

    traza = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "ilustrativa": False,
        "proveedor": os.getenv("LLM_PROVIDER", "gemini"),
        "recursion_limit": RECURSION_LIMIT,
        "escenarios": escenarios,
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("\nTraza guardada en %s", salida)


async def chat(graph: CompiledStateGraph, thread_id: str) -> None:
    """Chat interactivo: reusar el mismo --thread retoma la conversación."""
    log.info("Chat (thread_id=%s). Escribí 'salir' para terminar.", thread_id)
    while True:
        pregunta = (await asyncio.to_thread(input, "\nVos: ")).strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if pregunta:
            await ejecutar_turno(graph, thread_id, pregunta)


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", action="store_true", help="modo interactivo")
    parser.add_argument("--thread", default="sesion-1", help="thread_id (modo chat)")
    parser.add_argument(
        "--output", type=Path, default=Path("traces/ejemplo_traza.json")
    )
    args = parser.parse_args()

    db_path = os.getenv("SQLITE_PATH", "checkpoints.sqlite")
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(get_llm(), checkpointer)
        if args.chat:
            await chat(graph, args.thread)
        else:
            await correr_demo(graph, args.output)


if __name__ == "__main__":
    asyncio.run(main())
