# Pipeline-Requests für `v2 synthetic.duckdb`

Anforderungen an die Build-Pipeline (`analysis/webmap_export/`), damit die
webmap-backend Provider vollständig & schnell laufen.

## Umsetzungsstatus

| Punkt | Status | Wo |
|-------|--------|-----|
| **B** Kanton-Spalten (`trips.origin/dest_canton_id`, `network_links.canton_id`) | ✅ implementiert + Fixture-getestet | `schema.py`, `canton.py`, `__init__.py` |
| **C** Eine kanonische Kanton-Zuordnung | ✅ Spatial-Join gegen swisstopo-Kantonspolygone (überschreibt synth. `persons.canton_id`) | `canton.py` |
| **D** `spider_link_index` nach `link_id` clustern | ✅ `ORDER BY link_id` beim Insert | `spider.py` |
| **D** `spider_routes` nach `person_id` clustern | ⚪️ bewusst ausgelassen (Vollkopie-Rebuild nötig, geringer Hebel) | — |
| **A** `municipalities.geojson` (static_asset) | ✅ implementiert + Fixture-getestet (gemeinde→WGS84+kantonsnum) | `static_assets.py` |
| **A** `nodes_by_canton/{c}_nodes.geojson` (static_asset) | ✅ implementiert + Fixture-getestet (network_nodes × cantons) | `static_assets.py` |
| **A** `link_speeds` befüllen | ✅ aus `output_events` (LinkEnter/Leave → Reisezeit → Speed/Stunde) | `events_extras.py` |
| **A** `boarding_data_by_line.json` | ✅ aus `output_events` (Boardings) + `output_transitSchedule` (Linien/Stops), inkl. cantons-Array | `events_extras.py` + `transit.py` |
| **A** `stop_municipality.json` | ✅ stops × swisstopo gemeinde/canton | `transit.py` |
| **A** `stop_transfer_data_by_canton.json` | ⚪️ offen — Umsteige-Logik aus Events; bei Bedarf nachrüsten | — |

**JSON-Schema-Hinweis:** Form von `boarding_data_by_line.json` /
`stop_municipality.json` aus altem Monolith abgeleitet (Backend-`providers/` war
leer) — vom Backend-Agenten bestätigen/anpassen. Eingänge (events/schedule) sind
Standard-MATSim, nicht geraten.

**Architektur:** alles in der einen `webmap_export`-Stage (Package
`analysis/webmap_export/`, getriggert via Config) — keine separaten synpp-Stages,
keine losen Dateien. static_assets werden als BLOBs (key/content_type/payload)
in der `static_assets`-Tabelle abgelegt.

**eqasim-Quellen für A** (aus alter Stage `webmap-preprocessing`):
`ITERS/it.<N>/<N>.linkstats.txt.gz` (link volumes; speed unklar),
`output_transitSchedule.xml.gz` (stops/lines), `pt_passenger_counts.csv.gz`
(boardings/alightings), swisstopo (gemeinde/canton).

End-to-End-Build erfordert einen MATSim/eqasim-Run (Output liegt aktuell nicht
auf Disk); Build nur via `sbatch`.

---


Alles **additiv** (neue Spalten / gefüllte Tabellen) → bricht die aktuellen
Provider nicht, die fangen Fehlen bereits graceful ab.

## A) Fehlende Daten füllen → repariert die 7 „data-gap"-Features

Aktuell leer/nicht gebaut in `v2 synthetic.duckdb`:

- **`link_speeds`-Tabelle ist leer (0 Zeilen)** → `link_speeds.json` +
  `speed_dashboard.json` liefern nichts.
  → Befüllen mit `(link_id, time_bucket, speed)` pro Link & Zeitfenster.
- **`static_assets`-Tabelle ist leer (0 Zeilen)** bzw. der `json_preview/`-Ordner
  fehlt → diese Provider haben keine Quelle:
  - `boarding_data_by_line.json` (Linien-Boardings, mit `cantons`-Array)
  - `stop_transfer_data_by_canton.json`
  - `stop_municipality.json` (Stop → Gemeinde-Lookup)
  - `municipalities.geojson` (Gemeinde-Polygone, WGS84, mit `kantonsnum`)
  - `nodes_by_canton/{canton}_nodes.geojson` (Netz-Knoten pro Kanton)
  → Entweder als `static_assets`-BLOBs (`key` / `content_type` / `payload`)
  oder als `json_preview/`-Dateien neben der duckdb.

## B) Kanton-Spalten vorberechnen → macht `zone_flows` sauber & schnell

Die neue `trips`-Tabelle hat keine Kanton-Spalten, `network_links` auch nicht.
Aktuell zur Laufzeit über einen H3-Cache gelöst (funktioniert, aber ~6–10 s
Cache-Build beim ersten Request + ~99,6 % statt 100 % genau). Wenn die Pipeline
das vorberechnet, fällt das weg:

- **`trips`**: Spalten `origin_canton_id` + `dest_canton_id`
  (Spatial-Join `origin_pt`/`dest_pt` gegen Kanton-Polygone beim Build — wie es
  die alte `spider.duckdb` schon hatte).
- **`network_links`**: Spalte `canton_id` (Link-Centroid → Kanton).

→ Damit wird `zone_flows` ein trivialer schneller Integer-Join ohne App-Cache;
der Provider wird dann auf diese Spalten angepasst.

## C) Wichtig: eine kanonische Kanton-Zuordnung

Gemessen: `persons.canton_id` und `ST_Within(home_pt, hot_polygons-Kanton)`
widersprechen sich bei **12,4 %** der Personen → konkurrierende Methoden.
Bitte **eine** Methode/Grenzquelle festlegen und überall gleich anwenden:

- `persons.canton_id` (existiert)
- die neuen `trips.origin_canton_id` / `dest_canton_id` (B)
- `network_links.canton_id` (B)

So sehen alle Provider (`mode_share`, `histogram`, `pt_sub`, `zone_flows` …)
dieselbe Kantonszugehörigkeit.

## D) Optional, für 5-Mio-Tempo

- `spider_link_index` beim Schreiben nach `link_id` sortieren/clustern und
  `spider_routes` nach `person_id` — beschleunigt die Spider-Filter bei
  300 Mio+ Zeilen.

## Kurzfassung

> `v2 synthetic.duckdb`: (1) `link_speeds` befüllen, (2) `json_preview`/
> static-assets bauen (boarding / stop_transfer / stop_municipality /
> municipalities / nodes_by_canton), (3) `origin_canton_id` + `dest_canton_id`
> auf `trips` und `canton_id` auf `network_links` vorberechnen — konsistent mit
> `persons.canton_id`.

Sobald die Pipeline **B** liefert, wird `zone_flows` entsprechend vereinfacht
(weg vom Laufzeit-Cache).
