"""Definición del StateGraph (ciclo ReAct) y configuración del LLM."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools import TOOLS

SYSTEM_PROMPT = (
    "Sos un asistente de análisis de ventas. Respondé en español rioplatense, "
    "de forma breve. Para obtener datos de pedidos usá SIEMPRE las herramientas; "
    "nunca inventes cifras. Si necesitás datos de varios clientes, consultá cada "
    "uno por separado. Si una herramienta devuelve un error, corregí el dato y "
    "reintentá, o pedile una aclaración al usuario si no podés resolverlo. "
    "Mostrá los montos con formato $14.500."
)

# Máximo de mensajes que se envían al modelo (evita que el contexto crezca sin control).
MAX_MENSAJES_CONTEXTO = 30


class AgentState(MessagesState):
    """Estado del agente.

    Hereda de MessagesState, que ya define `messages` con el reducer
    `add_messages` (los mensajes se acumulan en vez de pisarse).
    """


def get_llm() -> BaseChatModel:
    """Crea el LLM según la variable de entorno LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Lee la clave de GOOGLE_API_KEY (o GEMINI_API_KEY, que se mapea abajo).
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=api_key,
            
        )
    raise ValueError(
        f"LLM_PROVIDER inválido: {provider!r} (usá 'anthropic', 'openai' o 'gemini')"
    )


def build_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Construye y compila el grafo ReAct.

    START -> modelo --(tools_condition)--> tools -> modelo ... -> END
    """
    llm_con_tools = llm.bind_tools(TOOLS)

    async def llamar_modelo(state: AgentState) -> dict[str, list]:
        # Recorta el historial: se queda con los últimos N mensajes, empezando
        # siempre en un mensaje del usuario para no separar tool_call/tool_result.
        historial = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,  # cuenta mensajes, no tokens
            max_tokens=MAX_MENSAJES_CONTEXTO,
            start_on="human",
            include_system=False,
        )
        respuesta = await llm_con_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *historial]
        )
        return {"messages": [respuesta]}

    builder = StateGraph(AgentState)
    builder.add_node("modelo", llamar_modelo)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "modelo")
    # Si el último mensaje del modelo trae tool_calls -> "tools"; si no -> END.
    builder.add_conditional_edges("modelo", tools_condition)
    # Después de ejecutar las herramientas se vuelve al modelo (el ciclo).
    builder.add_edge("tools", "modelo")
    return builder.compile(checkpointer=checkpointer)
