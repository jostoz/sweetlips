"""Ejecución de acciones locales (System 1): domótica, GPIO, APIs internas.

Sin dependencias de red ni de LLM. Cada acción debe responder en el acto.
"""

from __future__ import annotations

_LIGHTS_STATE = {"on": False}


def execute_local_command(action: str, target: str) -> str:
    """Ejecuta una acción local y devuelve la frase de confirmación para el TTS.

    Args:
        action: Verbo de la acción ("TURN_ON" / "TURN_OFF").
        target: Dispositivo objetivo ("LIGHTS", ...).

    Returns:
        Texto a sintetizar como confirmación.
    """
    if target == "LIGHTS":
        _LIGHTS_STATE["on"] = action == "TURN_ON"
        estado = "encendidas" if _LIGHTS_STATE["on"] else "apagadas"
        # TODO(hardware): sustituir por la llamada real a GPIO/Zigbee/MQTT.
        return f"Luces {estado}."

    return "Hecho."
