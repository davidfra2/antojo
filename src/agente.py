"""
Ciclo del agente de Antojo, adaptado del agentic loop de Lecture 11.

  usuario -> LLM -> (¿pide herramienta?) -> se ejecuta -> resultado al LLM -> ... -> respuesta

Diferencias con el loop de clase:
  * Límite de 5 llamadas a herramientas por pregunta (MAX_LLAMADAS).
  * Guarda una traza (herramienta, argumentos, resultado) para documentar las pruebas.
  * Captura errores de las herramientas y se los devuelve al modelo en vez de romperse.
"""

import json
import os

from openai import OpenAI

from herramientas import TOOLS, TOOL_REGISTRY

# ── Configuración del modelo (NVIDIA NIM, endpoint compatible con OpenAI) ───

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = os.getenv("ANTOJO_MODEL", "mistralai/mistral-large-4-1t-a52b")
MAX_LLAMADAS = 5

MENSAJE_LIMITE = (
    "No pude completar tu solicitud con las consultas disponibles. "
    "¿Puedes reformularla o hacerla más específica?"
)

SYSTEM_PROMPT = """Eres Antojo, el asistente de la cafetería de la universidad.
Ayudas a estudiantes y profesores a elegir qué comer y a saber cuánto pagarán.

Reglas:
- Usa SIEMPRE las herramientas para consultar el menú y para cualquier precio, suma o división.
  Nunca inventes productos, precios ni totales.
- Si un producto no está en el menú, dilo claramente y no le asignes precio.
- Si la pregunta es ambigua (por ejemplo "algo barato"), explica el criterio que usas
  o pide una aclaración breve.
- Solo respondes sobre la cafetería. Si te preguntan otra cosa, dilo amablemente.
- Responde en español, de forma breve y amable. Escribe los precios así: $14.500.
"""


def crear_cliente(api_key: str | None = None) -> OpenAI:
    """Crea el cliente. La clave se lee de NVIDIA_API_KEY; nunca se escribe en el código."""
    api_key = api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Falta la variable NVIDIA_API_KEY (o el secreto de Colab con ese nombre).")
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def _ejecutar_herramienta(nombre: str, argumentos_json: str) -> tuple[dict, str]:
    """Ejecuta una herramienta y devuelve (argumentos, resultado). Nunca lanza excepción."""
    try:
        args = json.loads(argumentos_json or "{}") or {}
    except json.JSONDecodeError:
        return {}, json.dumps({"error": "Argumentos con formato JSON inválido."})

    fn = TOOL_REGISTRY.get(nombre)
    if fn is None:
        return args, json.dumps({"error": f"La herramienta '{nombre}' no existe."})
    try:
        return args, fn(**args)
    except Exception as e:  # argumentos con tipo equivocado, etc.
        return args, json.dumps({"error": f"Error al ejecutar {nombre}: {e}"}, ensure_ascii=False)


def run_agent(pregunta: str, client: OpenAI, model: str = MODEL,
              max_llamadas: int = MAX_LLAMADAS, verbose: bool = True) -> dict:
    """
    Ejecuta el agente para una pregunta.

    Devuelve un dict con:
      respuesta        texto final para el usuario
      traza            lista de {herramienta, argumentos, resultado}
      llamadas         número de herramientas ejecutadas
      limite_alcanzado True si se detuvo por el límite
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pregunta},
    ]
    traza = []

    if verbose:
        print(f"\n{'─' * 70}\n👤 {pregunta}")

    while True:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0,
        )
        msg = response.choices[0].message

        # ── Sin herramientas: es la respuesta final ─────────────────────
        if not msg.tool_calls:
            respuesta = msg.content or ""
            if verbose:
                print(f"🤖 {respuesta}")
            return {"respuesta": respuesta, "traza": traza,
                    "llamadas": len(traza), "limite_alcanzado": False}

        # ── El modelo pide herramientas ─────────────────────────────────
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            # Límite de llamadas: se revisa ANTES de ejecutar cada herramienta
            if len(traza) >= max_llamadas:
                if verbose:
                    print(f"⛔ Límite de {max_llamadas} llamadas alcanzado.\n🤖 {MENSAJE_LIMITE}")
                return {"respuesta": MENSAJE_LIMITE, "traza": traza,
                        "llamadas": len(traza), "limite_alcanzado": True}

            args, resultado = _ejecutar_herramienta(tc.function.name, tc.function.arguments)
            traza.append({"herramienta": tc.function.name, "argumentos": args, "resultado": resultado})

            if verbose:
                print(f"🔧 {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
                print(f"   → {resultado[:300]}{'...' if len(resultado) > 300 else ''}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": resultado})
