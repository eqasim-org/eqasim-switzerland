"""Transfer comparisons from saved journeys; no routing or API requests."""
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

BACKENDS = {'rust': 'Eqasim Rust', 'raptor': 'SwissRailRaptor', 'flash': 'FLASH-TB', 'google': 'Saved Google'}
DISTANCE_EDGES = [0, 1, 2, 5, 10, 20, 50, 100, np.inf]
DISTANCE_LABELS = ['0–1', '1–2', '2–5', '5–10', '10–20', '20–50', '50–100', '100+']


def statistics(trips, routes):
    inputs = trips.select('trip_id', 'origin_x', 'origin_y', 'destination_x', 'destination_y',
                          'departure_time', 'weight_person').to_pandas()
    inputs['distance_km'] = np.hypot(inputs.destination_x - inputs.origin_x,
                                     inputs.destination_y - inputs.origin_y) / 1000
    inputs['distance_band'] = pd.cut(inputs.distance_km, DISTANCE_EDGES, labels=DISTANCE_LABELS, right=False)
    hours = np.floor(inputs.departure_time / 3600)
    inputs['hour'] = hours.where(np.isfinite(hours) & (hours >= 0)).astype('Int64')
    frames = {}
    for name in BACKENDS:
        route = routes[name].select('trip_id', 'pt_transfers', 'journey_found').to_pandas()
        frame = inputs.merge(route, on='trip_id', validate='one_to_one')
        valid = frame.journey_found.fillna(False) & np.isfinite(frame.pt_transfers) & (frame.pt_transfers >= 0)
        frames[name] = frame.loc[valid].copy()
    common = set.intersection(*(set(f.trip_id) for f in frames.values()))
    records = []
    for cohort in ['common', 'available']:
        for backend, original in frames.items():
            frame = original.loc[original.trip_id.isin(common)] if cohort == 'common' else original
            groups = [('overall', 'All trips', frame)]
            groups += [('distance', str(key), group) for key, group in frame.groupby('distance_band', observed=True)]
            groups += [('hour', str(key), group) for key, group in frame.groupby('hour')]
            for dimension, group, data in groups:
                weights = data.weight_person.to_numpy(dtype=float)
                values = data.pt_transfers.to_numpy(dtype=float)
                weighted = np.isfinite(weights) & (weights > 0)
                mass = weights[weighted].sum()
                records.append({'cohort': cohort, 'backend': backend, 'dimension': dimension, 'group': group,
                                'n': len(values), 'mean': float(values.mean()) if len(values) else np.nan,
                                'weighted_n': int(weighted.sum()), 'weight_sum': float(mass),
                                'weighted_mean': float(np.average(values[weighted], weights=weights[weighted])) if mass else np.nan,
                                'zero_transfer_pct': float((values == 0).mean() * 100) if len(values) else np.nan,
                                'two_or_more_pct': float((values >= 2).mean() * 100) if len(values) else np.nan})
    return pd.DataFrame(records)


def generate(trips, routes, output):
    output = Path(output)
    stats = statistics(trips, routes)
    stats.to_csv(output / 'transfers_by_distance_hour.csv', index=False)
    for cohort in ['common', 'available']:
        data = stats.loc[stats.cohort.eq(cohort)]
        for metric in ['mean', 'weighted_mean']:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1.3, 1.3, 1]}, constrained_layout=True)
            for colour, (backend, label) in zip(['#2878B5', '#D46B39', '#2D9475', '#8A61B3'], BACKENDS.items()):
                selected = data.loc[data.backend.eq(backend)]
                distance = selected.loc[selected.dimension.eq('distance')].set_index('group').reindex(DISTANCE_LABELS)
                axes[0].plot(range(len(DISTANCE_LABELS)), distance[metric], marker='o', ms=4, color=colour, label=label)
                hourly = selected.loc[selected.dimension.eq('hour')].copy()
                if not hourly.empty:
                    hourly['hour'] = hourly.group.astype(int)
                    # Missing hours remain gaps, not interpolated or zero-filled.
                    hourly = hourly.set_index('hour').reindex(range(int(hourly.hour.min()), int(hourly.hour.max()) + 1))
                    axes[1].plot(hourly.index, hourly[metric], marker='o', ms=4, color=colour, label=label)
                overall = selected.loc[selected.dimension.eq('overall')].iloc[0]
                x = list(BACKENDS).index(backend)
                value = overall[metric]
                if np.isfinite(value):
                    axes[2].bar(x, value, color=colour, width=.65)
                    axes[2].annotate(f'{value:.2f}\nn={int(overall.n):,}', (x, value), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9)
            axes[0].set(xticks=range(len(DISTANCE_LABELS)), xticklabels=DISTANCE_LABELS, xlabel='Euclidean O–D distance (km)', title='Transfers by distance band')
            axes[1].set(xlabel='Departure hour (service-day clock)', title='Transfers by departure hour')
            axes[2].set(xticks=range(len(BACKENDS)), xticklabels=['Rust', 'Raptor', 'FLASH', 'Google'], title='Overall mean', xlabel='Router')
            for ax in axes:
                ax.set_ylabel('Mean number of transfers')
                ax.set_ylim(0, max(ax.get_ylim()[1] * 1.2, .2))
                ax.grid(axis='y', alpha=.2)
            axes[0].legend(fontsize=9)
            fig.suptitle(('Same trips available in all four sources' if cohort == 'common' else 'Each router’s available trips') +
                         (' · person-weighted means' if metric == 'weighted_mean' else ' · unweighted trip means'), fontsize=13)
            fig.savefig(output / f'transfers_{cohort}_{metric}.png', dpi=160)
            plt.close(fig)
    tables = []
    for cohort in ['common', 'available']:
        frame = stats.loc[stats.cohort.eq(cohort) & stats.dimension.eq('overall')].copy()
        frame['backend'] = frame.backend.map(BACKENDS)
        frame = frame[['backend', 'n', 'mean', 'weighted_n', 'weighted_mean', 'zero_transfer_pct', 'two_or_more_pct']]
        frame.columns = ['Router', 'Trips', 'Mean transfers', 'Trips with valid weights', 'Person-weighted mean', 'Zero transfers (%)', '2+ transfers (%)']
        tables.append(f'<div class="transfer-cohort-table" data-cohort="{cohort}"' + (' hidden' if cohort == 'available' else '') + '>' + frame.to_html(index=False, float_format=lambda x:f'{x:.3f}', na_rep='—', border=0) + '</div>')
    return '''<section id="transfer-comparison"><h2>Number of transfers by distance and departure hour</h2>
<p>Comparison of three routers and saved Google results. No Google API calls. A transfer is a change between PT vehicle legs; walking access and egress are not transfers. Euclidean distance is recalculated from projected origin/destination coordinates. Pure-walking, missing and unroutable journeys, and invalid transfer counts, are excluded rather than counted as zero.</p>
<label>Trip cohort <select id="transfer-cohort"><option value="common">Same trips available in all four sources</option><option value="available">Each router’s available trips</option></select></label>
<label style="margin-left:1em">Averaging <select id="transfer-weighting"><option value="mean">Unweighted trips</option><option value="weighted_mean">Microcensus person weights</option></select></label>
<p>The common cohort gives an identical trip sample for Rust, Raptor, FLASH and saved Google. Available-trip means can differ because the routers serve different trips. Google requires a verified, available saved PT journey; missing Google coverage is excluded, not counted as zero transfers. Weighting uses finite positive person weights only; tables and CSV report the sample sizes. Hours 24+ remain on the service-day clock, without folding into the morning. Empty groups are gaps.</p>
<img id="transfer-figure" src="transfers_common_mean.png" alt="Mean transfer counts by Euclidean distance, departure hour and router">
<div class="scroll">''' + ''.join(tables) + '''</div><p><a href="transfers_by_distance_hour.csv">Download all transfer averages, distance/hour bins and sample sizes (CSV)</a></p>
<details><summary>Detailed distance/hour statistics</summary><div class="scroll">''' + stats.assign(backend=stats.backend.map(BACKENDS)).to_html(index=False, float_format=lambda x:f'{x:.3f}', na_rep='—', border=0) + '''</div></details>
<script>(()=>{const cohort=document.getElementById('transfer-cohort'),weighting=document.getElementById('transfer-weighting');function update(){document.getElementById('transfer-figure').src='transfers_'+cohort.value+'_'+weighting.value+'.png';document.querySelectorAll('.transfer-cohort-table').forEach(el=>el.hidden=el.dataset.cohort!==cohort.value);}cohort.addEventListener('change',update);weighting.addEventListener('change',update);})();</script></section>'''
