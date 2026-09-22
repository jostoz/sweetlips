"""Ejecución de acciones locales (System 1): domótica, GPIO, APIs internas.

Sin dependencias de red ni de LLM. Cada acción debe responder en el acto.
"""

from __future__ import annotations

from datetime import datetime

_LIGHTS_STATE = {"on": False}

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _describe_time() -> str:
    now = datetime.now()  # local system time, no forced timezone.
    hour12 = now.hour % 12 or 12
    period = "AM" if now.hour < 12 else "PM"
    return f"It's {hour12}:{now.minute:02d} {period}."


def _describe_date() -> str:
    now = datetime.now()
    weekday = _DAYS[now.weekday()]
    month = _MONTHS[now.month - 1]
    return f"Today is {weekday}, {month} {now.day}."


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
        state = "on" if _LIGHTS_STATE["on"] else "off"
        # TODO(hardware): sustituir por la llamada real a GPIO/Zigbee/MQTT.
        return f"Lights are {state}."

    if target == "TIME":
        return _describe_time()

    if target == "DATE":
        return _describe_date()

    return "Done."
