"""Namluví staniční hlášení do web/audio/hlaseni_N.mp3.

Hlas: cs-CZ-VlastaNeural (ženský, Microsoft neural TTS přes edge-tts; vyžaduje internet,
texty se posílají službě Microsoftu). Instalace: pip install edge-tts
Zvuk dostane charakter palubního rozhlasu (pásmová propust, lehký dozvuk, normalizace hlasitosti).
Texty musí odpovídat poli ANNOUNCE ve web/index.html (stejné pořadí).
"""
import asyncio
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "web" / "audio"
VOICE = "cs-CZ-VlastaNeural"

ANNOUNCE = [
    "Vítejte na palubě lanové dráhy Pisárky – Kampus. Během jízdy se nevyklánějte z oken a nekrmte havrany.",
    "Příští zastávka Riviéra. Cestující jsou upozorňováni na zvýšený výskyt havranů.",
    "Vážení cestující, lanovka vede nad nocovištěm chráněných ptáků. Neotvírejte okna, neházejte drobečky a nepodávejte žaloby.",
    "Mezistanice Kampus. Přestup na tramvaj, sanitku a pohřební službu.",
    "Konečná stanice Kampus – terminál. Děkujeme, že jste využili lanovku, která neexistuje.",
]
PA_FILTER = ("highpass=f=220,lowpass=f=5200,aecho=0.85:0.6:35|70:0.16|0.07,loudnorm=I=-16:TP=-1.5,"
             "adelay=150,apad=pad_dur=0.3")


def synth_edge(text, path):
    import edge_tts
    asyncio.run(edge_tts.Communicate(text, VOICE, rate="-4%").save(str(path)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(ANNOUNCE):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.mp3"
            synth_edge(text, raw)
            mp3 = OUT / f"hlaseni_{i + 1}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af", PA_FILTER,
                            "-ac", "1", "-ar", "44100", "-b:a", "64k", str(mp3)], check=True)
            print(mp3.name, f"{mp3.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
