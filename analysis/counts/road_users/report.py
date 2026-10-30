"""A small standalone report linking the diagnosis to the interactive viewer."""

import sqlite3
import json
from pathlib import Path

import pandas as pd


def write_report(output, observed_results, sample):
    output = Path(output)
    diagnosis = json.loads((output / 'corridor_diagnosis.json').read_text())
    audit_path = output / 'cross_border_source_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    table = []
    for station in ['24', '505', '504']:
        observed = observed_results.loc[observed_results.id.astype(str).eq(station)].iloc[0]
        modelled = sum(row['scaled_passages'] for row in diagnosis['stations'] if row['station'] == station)
        table.append({'Station': station, 'Observed vehicles/day': round(observed.flow),
                      'Modelled vehicles/day': round(modelled), 'Difference (%)': round(100 * (modelled / observed.flow - 1), 1)})
    flow_rows = []
    for name, label in [('south_exit', '505 but not 504 · toward Geneva'),
                        ('south_entry', '504 but not 505 · toward Geneva'),
                        ('north_entry', '505 but not 504 · toward Lausanne'),
                        ('north_exit', '504 but not 505 · toward Lausanne')]:
        c = diagnosis['cohorts'][name]
        flow_rows.append({'Observed simulated journey cohort': label, 'Sample journeys': c['sample_trips'],
                          'Scaled journeys': round(c['scaled_trips']),
                          'Counterpart passed after midnight': c['counterpart_crossed_after_midnight'],
                          'Stuck at some later point': c['event_status'].get('stuck', 0)})
    contributions = pd.DataFrame([{'Agent group': group, 'Contribution to 505−504 (vehicles/day)': round(value)}
                                 for group, value in diagnosis['contribution_to_505_minus_504_full_population'].items()])
    south = diagnosis['conservation']['south_505_minus_504']
    north = diagnosis['conservation']['north_505_minus_504']
    total_scaled = (south + north) / sample
    crossborder_scaled = diagnosis['contribution_to_505_minus_504_full_population']['Other cross-border']
    crossborder_share = 100 * crossborder_scaled / total_scaled if total_scaled else 0
    audit_section = ''
    if audit:
        positive = [audit['by_cohort'][key] for key in ['south_exit', 'north_entry']]
        total = sum(row['trips'] for row in positive)
        outside = sum(row['assigned_outside_metric_support'] for row in positive)
        audited = pd.read_csv(output / 'cross_border_source_audit.csv', dtype={'person': str})
        examples = audited[audited.cohort.eq('south_exit') & audited.export_cache_verified & ~audited.origin_projected & audited.assigned_outside_metric_support]
        example_html = ''
        if len(examples):
            example = examples.sort_values('extra_geometric_distance_km', ascending=False).iloc[0]
            with sqlite3.connect(output / 'corridor.sqlite') as connection:
                trip = connection.execute('SELECT journey_id FROM journeys WHERE person=? AND sequence=?', (example.person, int(example.sequence))).fetchone()
            if trip:
                example_html = f'<p>Example simulated agent <code>{example.person}</code> is constrained through <code>{example.assigned_point}</code>. Using the stored OD endpoints, this adds {example.extra_geometric_distance_km:.1f} km to the geometric origin→crossing→destination path compared with the nearest candidate. <a href="http://127.0.0.1:8765/?journey={trip[0]}">Inspect the actual traced journey</a>.</p>'
        audit_section = f'''<h2>Likely upstream cause: crossing-point selection uses mixed coordinate systems</h2>
        <p>The 2021 survey reader imports origin/destination longitude and latitude. Its crossing-point selector
        compares those degree coordinates directly with EPSG:2056 crossing geometries in metres. As a result,
        the intended “within 20% of the minimum detour” rule can admit distant points throughout a broad survey group.</p>
        <p>We checked source caches only where exported age, home and assigned crossing agree. Among
        <strong>{total:,} verified, nonprojected cross-border journeys</strong> in the two cohorts adding traffic at station 505,
        <strong>{outside:,} ({100 * outside / total:.1f}%)</strong> have an assigned crossing outside that eligibility rule when
        distances are recomputed in metres using the stored OD endpoints. The dominant survey label is
        <code>Gruppe O: VD/NE → F. Dép. Jura. Doubs</code>.</p>
        <p>This is an eligibility diagnostic, not a rerun of the traffic model. It supports a focused coordinate-system
        correction and a check of grouped border locations before tuning demand or motorway capacities.</p>
        {example_html}
        <p>Correcting artificial detours may reduce traffic at stations 24 and 505. Their currently good totals may
        contain compensating errors; this does not establish that traffic at 504 itself should simply be increased.</p>
        <p class="small">Source: <code>data/cross_border/generate_od.py</code>, <code>read_2021_data</code>, nested
        <code>sample_point</code>. Original coordinates and source caches were read only; no demand or assignment code was changed.</p>'''
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Nyon corridor traffic diagnosis · iteration 120</title><style>
    body{{font:16px/1.6 system-ui,sans-serif;color:#233b4b;max-width:1050px;margin:40px auto;padding:0 24px;background:#f8fafb}}
    h1{{font-size:30px;line-height:1.2}}h2{{font-size:21px;margin-top:30px}}a{{color:#096a96}}code{{background:#edf2f5;padding:2px 5px;border-radius:3px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce6ec;text-align:left}}th{{background:#e7eff4}}.lead{{font-size:19px}}.small{{font-size:13px;color:#637b89}}.button{{display:inline-block;padding:10px 18px;background:#176d8f;color:white;border-radius:6px;text-decoration:none}}li{{margin:6px 0}}
    </style><h1>Where traffic goes between Gland and Nyon</h1>
    <p class="small">Iteration 120 · {100 * sample:g}% sample · daily window 00:00–24:00 · observed reference: cached ASTRA weekday comparison</p>
    <p class="lead">Most of the simulated decrease is traffic using the Nyon interchange, with foreign cross-border agents
    making the largest contribution. It is not principally traffic vanishing through stuck-agent removal.</p>
    <p><a class="button" href="http://127.0.0.1:8765/">Open interactive road-user map</a></p>
    {pd.DataFrame(table).to_html(index=False,border=0)}
    <h2>The directional accounting balances</h2>
    <p>Toward Geneva, there are {south:,} more sample journeys at 505 than at 504. Toward Lausanne, there are {north:,} more
    at 505 than at 504. Combined, that is {south + north:,} sample vehicle passages, or {total_scaled:,.0f} on the full-population scale.
    Almost all the main exit/entry cohorts use the Nyon interchange ramps. A handful cross the counterpart detector after midnight.</p>
    {pd.DataFrame(flow_rows).to_html(index=False,border=0)}
    <p>“Stuck at some later point” does not mean the motorway lost that vehicle between the detectors. In particular,
    a trip may exit at Nyon normally and get stuck on a subsequent local road.</p>
    <h2>Which agents account for the difference?</h2>{contributions.to_html(index=False,border=0)}
    <p>The cross-border-demand group contributes {crossborder_scaled:,.0f} vehicles/day: {crossborder_share:.1f}% of the modelled 505−504 difference.
    “French residents” denotes the separate external-France population; the cross-border-demand group also contains French residents.</p>
    {audit_section}
    <h2>Map contents and validation</h2><ul>
    <li>Visible roads are fetched by viewport and road class, with an explicit 12,000-link display limit.</li>
    <li>Clicking overlapping geometry lists all nearby visible directed links and their individual statistics.</li>
    <li>Daily traffic uses the supplied count CSV scaled by 1/{sample:g}; user-group colours count exact distinct model agents.</li>
    <li>Age, residence, employment/student status where exported, daily activities, home cells and example journey paths are available.</li>
    <li>Employment/student status remains unknown when absent; activity presence is not substituted for status.</li>
    <li>The selected-plan leg matched the actual vehicle and sequence for {100 * diagnosis['selected_plan_match_fraction']:.2f}% of corridor journeys.</li>
    <li>Events last 30 hours, but daily statistics exclude crossings at or after 24:00. Full journeys were retained for midnight and completion checks.</li>
    <li>The supplied Java count CSV includes events exactly at 24:00; user profiles use 0 ≤ time &lt; 24:00. The six study-detector entries agree exactly. Across directly comparable major-road links, 80 differ by a single midnight passage; the map retains the supplied CSV volumes.</li>
    </ul><h2>Artifacts</h2><ul>
    <li><a href="corridor_diagnosis.json">Directional diagnosis and ramp coordinates</a></li>
    <li><a href="corridor_trip_profiles.csv">Journey profiles</a></li>
    <li><a href="cross_border_source_audit.csv">Read-only source audit</a></li>
    <li><a href="browser_validation.json">Browser interaction validation</a></li>
    </ul><p class="small">The map server runs only on localhost. Restart it with <code>launch_map.command</code> in this folder.</p></html>'''
    (output / 'diagnosis.html').write_text(content)
