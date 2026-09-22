"""Ejecución de acciones locales (System 1): domótica, GPIO, APIs internas.

Sin dependencias de red ni de LLM. Cada acción debe responder en el acto.
"""

from __future__ import annotations

from datetime import datetime

_LIGHTS_STATE = {"on": False}

_DAYS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


def _describe_time() -> str:
    now = datetime.now()  # hora local del sistema, sin forzar timezone.
    hour12 = now.hour % 12 or 12
    if now.hour < 12:
        periodo = "de la mañana"
    elif now.hour < 19:
        periodo = "de la tarde"
    else:
        periodo = "de la noche"
    return f"Son las {hour12}:{now.minute:02d} {periodo}."


def _describe_date() -> str:
    now = datetime.now()
    dia_semana = _DAYS_ES[now.weekday()]
    mes = _MONTHS_ES[now.month - 1]
    return f"Hoy es {dia_semana}, {now.day} de {mes}."


def execute_local_command(action: str, target: str) -> str:
    """Ejecuta una acción local y devuelve la frase de confirmación para el TTS.

    Args:
        action: Verbo de la acción ("TURN_ON" / "TURN_OFF" / "QUERY").
        target: Dispositivo/dato objetivo ("LIGHTS", "TIME", "DATE", ...).

    Returns:
        Texto a sintetizar como confirmación.
    """
    if target == "LIGHTS":
        _LIGHTS_STATE["on"] = action == "TURN_ON"
        estado = "encendidas" if _LIGHTS_STATE["on"] else "apagadas"
        # TODO(hardware): sustituir por la llamada real a GPIO/Zigbee/MQTT.
        return f"Luces {estado}."

    if target == "TIME":
        return _describe_time()

    if target == "DATE":
        return _describe_date()

    return "Hecho."
