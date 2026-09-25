# Video jízdy lanovkou Pisárky – Kampus: postup

## Co je hotové
| soubor | co to je |
|---|---|
| `out/nahled_720p.mp4` | náhled celé jízdy z otevřených dat (ortofoto ČÚZK na terénu), ~2:45 |
| `out/01_dron.esp` | projekt Google Earth Studia: přelet dronem od kampusu k nástupní stanici (48 s) |
| `out/02_kabina.esp` | projekt Google Earth Studia: jízda z kabiny, zrychlená 4×, a závěrečné vystoupání (117 s) |
| `out/lanovka_trasa.kml` | lano, stanice a podpěry pro kontrolu v GES |
| `out/nahled_trasy.png` | mapa trasy a podélný profil |

Oba projekty navazují: poslední snímek dronu = první snímek jízdy.

## Render v Google Earth Studiu (udělá uživatel, potřebuje Google účet)
1. Otevřete https://earth.google.com/studio/ v Chromu a přihlaste se.
2. **Import projektu:** na úvodní obrazovce „Blank Project“ nebo v menu File → Import
   (případně přetáhněte `01_dron.esp` do okna).
3. **Kontrola (volitelná):** v panelu *Overlays* importujte `lanovka_trasa.kml`. Tmavá čára je lano.
   Projeďte časovou osu: kamera má letět nad trasou a na konci stát v nástupní stanici u Lipové,
   pár metrů nad zemí. **Před renderem KML vypněte**, lanovku dokreslí skript.
4. **Render** (tlačítko Render vpravo nahoře):
   - Local render (ne Cloud), rozlišení podle projektu 1920×1080, 30 fps, formát JPEG
   - v *Advanced* zapněte **3D Camera Export**, formát **JSON**, souřadnice **Global (ECEF)**
   - Attribution: ponechte zapnutou (podmínka Google)
   - karta prohlížeče musí zůstat otevřená a na popředí
5. Totéž pro `02_kabina.esp`.
6. Stažené ZIPy rozbalte do:
   ```
   /mnt/projects/lanovka/ges/01_dron/     (footage/*.jpeg + 01_dron.json)
   /mnt/projects/lanovka/ges/02_kabina/
   ```

Pokud je kamera pod zemí nebo příliš vysoko, dejte vědět, posunu výšky a vygeneruju projekty znovu.
FOV nechte výchozí; když ho změníte, skript ho přečte z tracking dat.

## Složení finálního videa
```
python3 composite.py            # → out/lanovka_ges.mp4 (1080p, ~1 s/snímek na tomto stroji)
```
Skript z tracking dat vezme přesnou kameru každého snímku, dokreslí lano, podpěry, stanice a jedoucí kabiny
(zakryté terénem a stromy podle DMP 1G) a přidá popisky stanic, minimapu a upozornění, že jde o záměr.

## Úpravy
- trasa, stanice, výšky podpěr: `geo.py` (STATIONS, PYLONS) → `python3 make_esp.py` → znovu render v GES
- rychlost / zrychlení / průjezd stanicemi / pohled z kabiny: konstanty v `make_esp.py`
- vzhled lanovky a popisky: `render.py`
- náhled bez GES: `python3 render.py --out out/nahled.mp4` (~0,6–0,9 s/snímek)

## Omezení
- Polohy stanic a podpěr jsou rekonstrukce z textových popisů (DPMB, územní studie) a OSM, ±50–100 m.
  Modelová trasa má 1 762 m, projekt uvádí 1 701–1 731 m. Tři podpěry jsou rozmístěné odhadem tak,
  aby lano vedlo nad korunami stromů.
- Video zobrazuje záměr, který není schválený (stavební povolení zrušeno 9/2024, projekt odložen).
