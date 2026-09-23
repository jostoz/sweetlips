"""Ejecución de acciones locales (System 1): domótica, GPIO, APIs internas.

Sin dependencias de red ni de LLM. Cada acción debe responder en el acto.
Parametrizado por idioma (mismo string que JevSystem1Processor.language)
para las frases de confirmación habladas.
"""

from __future__ import annotations

from datetime import datetime

_LIGHTS_STATE = {"on": False}

_DAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_DAYS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


def _describe_time(language: str) -> str:
    now = datetime.now()  # hora local del sistema, sin forzar timezone.
    hour12 = now.hour % 12 or 12
    if language == "Spanish":
        if now.hour < 12:
            periodo = "de la mañana"
        elif now.hour < 19:
            periodo = "de la tarde"
        else:
            periodo = "de la noche"
        return f"Son las {hour12}:{now.minute:02d} {periodo}."
    period = "AM" if now.hour < 12 else "PM"
    return f"It's {hour12}:{now.minute:02d} {period}."


def _describe_date(language: str) -> str:
    now = datetime.now()
    if language == "Spanish":
        dia_semana = _DAYS_ES[now.weekday()]
        mes = _MONTHS_ES[now.month - 1]
        return f"Hoy es {dia_semana}, {now.day} de {mes}."
    weekday = _DAYS_EN[now.weekday()]
    month = _MONTHS_EN[now.month - 1]
    return f"Today is {weekday}, {month} {now.day}."


def execute_local_command(action: str, target: str, language: str = "English") -> str:
    """Ejecuta una acción local y devuelve la frase de confirmación para el TTS.

    Args:
        action: Verbo de la acción ("TURN_ON" / "TURN_OFF" / "QUERY").
        target: Dispositivo/dato objetivo ("LIGHTS", "TIME", "DATE", ...).
        language: "English"/"Spanish" -- idioma de la frase de confirmación.

    Returns:
        Texto a sintetizar como confirmación.
    """
    if target == "LIGHTS":
        _LIGHTS_STATE["on"] = action == "TURN_ON"
        if language == "Spanish":
            estado = "encendidas" if _LIGHTS_STATE["on"] else "apagadas"
            # TODO(hardware): sustituir por la llamada real a GPIO/Zigbee/MQTT.
            return f"Luces {estado}."
        state = "on" if _LIGHTS_STATE["on"] else "off"
        return f"Lights are {state}."

    if target == "TIME":
        return _describe_time(language)

    if target == "DATE":
        return _describe_date(language)

    return "Hecho." if language == "Spanish" else "Done."
