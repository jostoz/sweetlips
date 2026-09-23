# Copia las voces OneCore clasicas (David/Zira/Mark en ingles, Raul/
# Sabina en espanol -- ya instaladas por Windows, ver registro
# HKLM:\software\Microsoft\Speech_OneCore\Voices\Tokens) al namespace
# clasico de SAPI5 (HKLM:\SOFTWARE\Microsoft\Speech\Voices\Tokens), que
# System.Speech/SAPI.SpVoice/pyttsx3 SI soportan seleccionar por nombre
# (a diferencia del binding de Python para WinRT
# Windows.Media.SpeechSynthesis, que no expone AllVoices en la version
# actual de winrt-Windows.Media.SpeechSynthesis de PyPI).
#
# Uso: fallback si alguna vez NaturalVoiceSAPIAdapter (ver README, "TTS
# nativo de Windows") deja de funcionar tras una actualizacion de
# Windows -- las voces OneCore clasicas via este puente siguen siendo
# gratis, locales, rapidas (30-61ms medido) y no dependen de ningun hack
# externo, aunque suenan mas robotizadas que las voces "Natural".
#
# Fuente: https://github.com/microsoft/VibeVoice -- aportado por el
# usuario en la sesion de sweetlips/pipecat-local-audio-edge.
#
# Requiere PowerShell elevado (admin): escribe en HKLM.

$sourcePath = 'HKLM:\software\Microsoft\Speech_OneCore\Voices\Tokens'
$destinationPath = 'HKLM:\SOFTWARE\Microsoft\Speech\Voices\Tokens'          # apps de 64 bits
$destinationPath2 = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\SPEECH\Voices\Tokens'  # apps de 32 bits

$listVoices = Get-ChildItem $sourcePath
foreach ($voice in $listVoices) {
    $source = $voice.PSPath
    Copy-Item -Path $source -Destination $destinationPath -Recurse -Force
    Copy-Item -Path $source -Destination $destinationPath2 -Recurse -Force
}

Write-Host "Copiadas $($listVoices.Count) voces OneCore a SAPI5 clasico."
