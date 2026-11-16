"""
Herramientas de Antojo.

Cada herramienta es una función de Python que trabaja sobre el menú (data/menu.csv)
y devuelve un texto JSON. El modelo de lenguaje decide cuándo llamarlas, pero
los precios y los cálculos siempre salen de aquí, nunca de la "memoria" del modelo.
"""

import json
import math
import os
import unicodedata

import pandas as pd

# ── Carga del menú ──────────────────────────────────────────────────────────

_RUTAS_POSIBLES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "menu.csv"),
    "data/menu.csv",
    "menu.csv",
]


def cargar_menu(ruta: str | None = None) -> pd.DataFrame:
    """Lee el CSV del menú y agrega una columna normalizada para buscar por nombre."""
    rutas = [ruta] if ruta else _RUTAS_POSIBLES
    for r in rutas:
        if r and os.path.exists(r):
            df = pd.read_csv(r)
            df["nombre_norm"] = df["nombre"].apply(normalizar)
            return df
    raise FileNotFoundError("No se encontró menu.csv. Revisa que esté en la carpeta data/.")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin espacios sobrantes: 'Buñuelo ' -> 'bunuelo'."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.lower().split())


MENU = cargar_menu()

CATEGORIAS = sorted(MENU["categoria"].unique().tolist())
RESTRICCIONES = ["vegetariano", "sin_lactosa"]


def _a_json(datos) -> str:
    return json.dumps(datos, ensure_ascii=False)


def _buscar_producto(nombre: str):
    """
    Busca un producto por nombre de forma tolerante:
    1) coincidencia exacta normalizada, 2) singular (empanadas -> empanada),
    3) coincidencia parcial única. Devuelve la fila o None.
    """
    n = normalizar(nombre)
    candidatos = [n]
    if n.endswith("es"):
        candidatos.append(n[:-2])
    if n.endswith("s"):
        candidatos.append(n[:-1])

    for c in candidatos:
        fila = MENU[MENU["nombre_norm"] == c]
        if len(fila) == 1:
            return fila.iloc[0]

    for c in candidatos:
        parcial = MENU[MENU["nombre_norm"].str.contains(c, regex=False)]
        if len(parcial) == 1:
            return parcial.iloc[0]
    return None


# ── Herramienta 1 ───────────────────────────────────────────────────────────

def buscar_platos(categoria: str = None, restriccion: str = None, precio_max: float = None) -> str:
    """Filtra el menú por categoría, restricción alimentaria y precio máximo."""
    df = MENU

    if categoria:
        cat = normalizar(categoria)
        if cat not in CATEGORIAS:
            return _a_json({"error": f"Categoría '{categoria}' no existe.",
                            "categorias_validas": CATEGORIAS})
        df = df[df["categoria"] == cat]

    if restriccion:
        r = normalizar(restriccion).replace(" ", "_")
        if r not in RESTRICCIONES:
            return _a_json({"error": f"Restricción '{restriccion}' no reconocida.",
                            "restricciones_validas": RESTRICCIONES})
        df = df[df[r] == "si"]

    if precio_max is not None:
        df = df[df["precio"] <= float(precio_max)]

    platos = (df.sort_values("precio")[["nombre", "categoria", "precio", "vegetariano", "sin_lactosa"]]
                .to_dict(orient="records"))
    return _a_json({"cantidad": len(platos), "platos": platos})


# ── Herramienta 2 ───────────────────────────────────────────────────────────

def calcular_pedido(items: list, cantidades: list = None) -> str:
    """Calcula el total de un pedido. Si falta un producto en el menú, lo reporta sin inventar precio."""
    if not items:
        return _a_json({"error": "El pedido no tiene productos."})
    if cantidades is None:
        cantidades = [1] * len(items)
    if len(items) != len(cantidades):
        return _a_json({"error": "La lista de productos y la de cantidades deben tener el mismo tamaño."})

    detalle, no_encontrados, total = [], [], 0
    for nombre, cant in zip(items, cantidades):
        cant = int(cant)
        if cant <= 0:
            return _a_json({"error": f"Cantidad inválida para '{nombre}': {cant}."})
        fila = _buscar_producto(nombre)
        if fila is None:
            no_encontrados.append(nombre)
            continue
        subtotal = int(fila["precio"]) * cant
        total += subtotal
        detalle.append({"producto": fila["nombre"], "cantidad": cant,
                        "precio_unitario": int(fila["precio"]), "subtotal": subtotal})

    resultado = {"detalle": detalle, "total": total}
    if no_encontrados:
        resultado["no_encontrados"] = no_encontrados
        resultado["aviso"] = "Estos productos no están en el menú; no se incluyeron en el total."
    return _a_json(resultado)


# ── Herramienta 3 ───────────────────────────────────────────────────────────

def dividir_cuenta(total: float, personas: int) -> str:
    """Divide un total entre varias personas, redondeando hacia arriba al peso."""
    personas = int(personas)
    if personas <= 0:
        return _a_json({"error": "El número de personas debe ser mayor que cero."})
    if float(total) < 0:
        return _a_json({"error": "El total no puede ser negativo."})
    por_persona = math.ceil(float(total) / personas)
    return _a_json({"total": float(total), "personas": personas, "por_persona": por_persona})


# ── Esquemas que ve el modelo (formato OpenAI tool use) ─────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_platos",
            "description": (
                "Busca productos en el menú de la cafetería. Úsala para saber qué platos hay, "
                "cuáles cumplen una restricción alimentaria o cuáles caben en un presupuesto. "
                "Todos los filtros son opcionales y se pueden combinar. Devuelve los productos "
                "ordenados del más barato al más caro."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "categoria": {"type": "string", "enum": CATEGORIAS,
                                  "description": "Tipo de producto."},
                    "restriccion": {"type": "string", "enum": RESTRICCIONES,
                                    "description": "Restricción alimentaria que debe cumplir."},
                    "precio_max": {"type": "number",
                                   "description": "Precio máximo en pesos colombianos, por unidad."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_pedido",
            "description": (
                "Calcula el valor total de un pedido con los precios del menú. Úsala SIEMPRE "
                "que el usuario pregunte cuánto cuesta algo; no sumes precios por tu cuenta. "
                "Si un producto no existe en el menú, la respuesta lo indica en 'no_encontrados'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"},
                              "description": "Nombres de los productos, por ejemplo ['Empanada', 'Jugo natural']."},
                    "cantidades": {"type": "array", "items": {"type": "integer"},
                                   "description": "Cantidad de cada producto, en el mismo orden que items."},
                },
                "required": ["items", "cantidades"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dividir_cuenta",
            "description": (
                "Divide un total entre varias personas. Úsala cuando el usuario pregunte "
                "cuánto paga cada uno. Normalmente el total viene de calcular_pedido."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "total": {"type": "number", "description": "Valor total en pesos."},
                    "personas": {"type": "integer", "description": "Número de personas que pagan."},
                },
                "required": ["total", "personas"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "buscar_platos": buscar_platos,
    "calcular_pedido": calcular_pedido,
    "dividir_cuenta": dividir_cuenta,
}
