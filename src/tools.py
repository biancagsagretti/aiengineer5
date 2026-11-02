"""Herramientas del agente.

Simulan una base de datos de ventas. El LLM NO ve el código: solo el nombre,
la firma y el docstring de cada herramienta, por eso los docstrings son
deliberadamente detallados.
"""

from typing import Any

from langchain_core.tools import tool

# "Base de datos" falsa: cliente_id -> lista de pedidos.
PEDIDOS_DB: dict[int, list[dict[str, Any]]] = {
    102: [
        {"id": "P-1001", "fecha": "2026-08-03", "monto": 4500},
        {"id": "P-1007", "fecha": "2026-09-12", "monto": 5000},
        {"id": "P-1015", "fecha": "2026-10-21", "monto": 5000},
    ],
    205: [
        {"id": "P-2003", "fecha": "2026-07-15", "monto": 12000},
        {"id": "P-2011", "fecha": "2026-10-02", "monto": 10000},
    ],
    310: [
        {"id": "P-3002", "fecha": "2026-09-30", "monto": 3200},
    ],
}


def _cliente_inexistente(cliente_id: int) -> dict[str, Any]:
    """Respuesta de error con información útil para que el agente pueda reintentar."""
    return {
        "error": f"El cliente {cliente_id} no existe en la base de datos de ventas.",
        "clientes_disponibles": sorted(PEDIDOS_DB),
    }


@tool
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Consulta la base de datos de ventas y devuelve el RESUMEN de pedidos de UN cliente.

    Usala siempre que necesites saber cuántos pedidos hizo un cliente o cuánto
    dinero gastó en total. Recibe un único ID numérico de cliente (por ejemplo 102).
    Si necesitás datos de varios clientes, llamala una vez por cada cliente.

    Devuelve un diccionario con: "cliente_id", "pedidos" (cantidad de pedidos) y
    "total" (suma de los montos en pesos). Si el cliente no existe devuelve
    {"error": ..., "clientes_disponibles": [...]}; en ese caso revisá el ID,
    reintentá con uno válido o pedile una aclaración al usuario.
    """
    pedidos = PEDIDOS_DB.get(cliente_id)
    if pedidos is None:
        return _cliente_inexistente(cliente_id)
    return {
        "cliente_id": cliente_id,
        "pedidos": len(pedidos),
        "total": sum(p["monto"] for p in pedidos),
    }


@tool
async def obtener_ultimo_pedido(cliente_id: int) -> dict[str, Any]:
    """Devuelve el DETALLE del pedido más reciente de UN cliente.

    Usala cuando el usuario pregunte por el último pedido, el pedido más nuevo o
    la última compra de un cliente. Recibe un único ID numérico de cliente.
    NO sirve para contar pedidos ni para calcular totales (para eso usá
    buscar_pedidos).

    Devuelve un diccionario con "cliente_id" y "ultimo_pedido" (con "id",
    "fecha" en formato AAAA-MM-DD y "monto" en pesos). Si el cliente no existe
    devuelve {"error": ..., "clientes_disponibles": [...]}.
    """
    pedidos = PEDIDOS_DB.get(cliente_id)
    if pedidos is None:
        return _cliente_inexistente(cliente_id)
    ultimo = max(pedidos, key=lambda p: p["fecha"])
    return {"cliente_id": cliente_id, "ultimo_pedido": ultimo}


TOOLS = [buscar_pedidos, obtener_ultimo_pedido]
