"""Paper-sized vector figures and 600-dpi previews, using offline basemaps."""
from pathlib import Path
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.collections import LineCollection
import geopandas as gpd
import shapely
from cohorts import LABELS, PURPOSES, PT_CLASSES

COLORS = {'survey': '#30343B', 'rda': '#009E73', 'nn_distance': '#0072B2', 'nn_no_distance': '#D55E00'}
PURPOSE_LABELS = {'shop': 'Shopping', 'leisure': 'Leisure', 'other': 'Other',
    'work_secondary': 'Secondary work', 'home_secondary': 'Secondary home', 'education_secondary': 'Secondary education'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'legend.frameon': False})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(directory / (name + '.' + extension), dpi=600, bbox_inches='tight')
    plt.close(fig)


def panel(ax, label):
    ax.text(-.12, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color='#E5E7EB', linewidth=.5)
    ax.set_axisbelow(True)


def ecdf(ax, values, weights, label, color, linestyle='-'):
    order = np.argsort(np.asarray(values))
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    # Keep endpoints and all essential quantiles without rendering millions of vertices.
    probability = np.cumsum(w) / w.sum()
    indices = np.unique(np.r_[0, np.searchsorted(probability, np.linspace(0, 1, 2501)).clip(0, len(v)-1), len(v)-1])
    ax.plot(np.maximum(v[indices], .001), probability[indices], label=label, color=color, lw=1.5, ls=linestyle)


def distance_figure(datasets, summary, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), gridspec_kw={'width_ratios': [1, 1.25]})
    for name, f in datasets.items():
        ecdf(axes[0], f.distance_km, f.weight, LABELS[name], COLORS[name])
    axes[0].set(xscale='log', xlim=(.01, 200), ylim=(0, 1), xlabel='Secondary-leg distance (km; logarithmic)', ylabel='Weighted cumulative share')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[0].axvline(1.5, color='#A0A0A0', ls=':', lw=.8)
    order = PURPOSES
    for offset, name in zip(np.linspace(-.25, .25, len(datasets)), datasets):
        f = summary.loc[(summary.dataset == name) & (summary.dimension == 'purpose')].set_index('group').reindex(order)
        axes[1].errorbar(f.mean_km, np.arange(len(order)) + offset,
            xerr=[f.mean_km - f.mean_low_km, f.mean_high_km - f.mean_km], fmt='o', markersize=3.3,
            color=COLORS[name], capsize=1.7, elinewidth=.8)
    axes[1].set(yticks=np.arange(len(order)), yticklabels=[PURPOSE_LABELS[p] for p in order], xlabel='Mean secondary-leg distance (km)')
    axes[1].invert_yaxis()
    for i, ax in enumerate(axes): panel(ax, string.ascii_lowercase[i])
    fig.tight_layout(w_pad=2)
    save(fig, directory, '01_distance_distributions_and_purpose')


def behavior_figure(summary, datasets, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), sharey=True)
    for ax, car in zip(axes, ['No car access', 'Car access']):
        for name in datasets:
            f = summary.loc[(summary.dataset == name) & (summary.dimension == 'car_x_home_pt')].set_index('group').reindex([car + ' / ' + p for p in PT_CLASSES])
            ax.errorbar(np.arange(5), f.mean_km, yerr=[f.mean_km-f.mean_low_km, f.mean_high_km-f.mean_km],
                        color=COLORS[name], label=LABELS[name], fmt='o-', lw=1.2, markersize=3, capsize=2)
        ax.set(title=car, xticks=range(5), xticklabels=['A', 'B', 'C', 'D', 'No class'], xlabel='Residential public-transport accessibility')
    axes[0].set_ylabel('Mean secondary-leg distance (km)')
    axes[1].legend(fontsize=7)
    for i, ax in enumerate(axes): panel(ax, string.ascii_lowercase[i])
    fig.tight_layout()
    save(fig, directory, '02_car_access_and_residential_accessibility')


def contrast_figure(car, pt, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9))
    for ax, table, title in zip(axes, [car, pt], ['Car access − no car access', 'Home access: A/B − D/no class']):
        for y, row in enumerate(table.itertuples(index=False)):
            if row.status == 'ok':
                ax.errorbar(row.difference_km, y, xerr=[[row.difference_km-row.low_km], [row.high_km-row.difference_km]],
                            fmt='o', color=COLORS[row.dataset], markersize=4, capsize=2)
        ax.set(yticks=range(len(table)), yticklabels=[LABELS[n] for n in table.dataset], xlabel='Standardized mean distance difference (km)', title=title)
        ax.axvline(0, color='#A0A0A0', ls=':', lw=.8)
        ax.invert_yaxis()
    for i, ax in enumerate(axes): panel(ax, string.ascii_lowercase[i])
    fig.tight_layout(w_pad=2)
    save(fig, directory, '03_standardized_behavioral_contrasts')


def transition_figure(table, datasets, directory):
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.3, 1.65*len(datasets)), sharex=True, sharey=True, squeeze=False)
    for i, name in enumerate(datasets):
        for j, car in enumerate(['No car access', 'Car access']):
            ax = axes[i, j]
            f = table.loc[(table.dataset == name) & (table.car_group == car)]
            matrix = f.pivot(index='home_pt', columns='destination_pt', values='share').reindex(index=PT_CLASSES, columns=PT_CLASSES)
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for iy in range(5):
                for ix in range(5):
                    value = matrix.iloc[iy, ix]
                    if np.isfinite(value):
                        ax.text(ix, iy, f'{value:.0%}', ha='center', va='center', fontsize=6.5, color='white' if value > .58 else '#30343B')
            ax.set(xticks=range(5), yticks=range(5), xticklabels=['A', 'B', 'C', 'D', 'None'], yticklabels=['A', 'B', 'C', 'D', 'None'])
            if i == 0: ax.set_title(car)
            if j == 0: ax.set_ylabel(LABELS[name].replace('distance input', 'distance').replace('learned selector', 'selector') + '\nHome class')
            if i == len(datasets)-1: ax.set_xlabel('Destination accessibility class')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Destination share within each home-class row', fraction=.025, pad=.03)
    fig.subplots_adjust(hspace=.20, wspace=.10, right=.87)
    save(fig, directory, '04_home_to_destination_accessibility')


def activity_figure(person_tables, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0))
    for name, all_people in person_tables.items():
        f = all_people.loc[all_people.secondary_legs >= 2]
        if f.empty: continue
        ecdf(axes[0], f.radius_gyration_km, f.weight, LABELS[name], COLORS[name])
        ecdf(axes[1], f.revisit_share_250m, f.weight, LABELS[name], COLORS[name])
    axes[0].set(xscale='log', xlim=(.01, 100), xlabel='Secondary activity-space radius of gyration (km)', ylabel='Weighted share of persons')
    axes[1].set(xlim=(0, 1), xlabel='Repeated-visit share (250 m spatial cells)')
    axes[1].legend(fontsize=7)
    for i, ax in enumerate(axes): panel(ax, string.ascii_lowercase[i])
    fig.tight_layout()
    save(fig, directory, '05_person_level_activity_spaces')


def replay_figure(observed, predictions, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0))
    ecdf(axes[0], observed.distance_km, observed.weight, 'Observed · same survey trips', COLORS['survey'])
    for name, f in predictions.groupby('dataset', observed=True):
        ecdf(axes[0], f.distance_km, f.weight, LABELS[str(name)], COLORS[str(name)])
    axes[0].set(xscale='log', xlim=(.01, 200), xlabel='Secondary-leg distance (km)', ylabel='Weighted cumulative share')
    axes[0].legend(fontsize=6.7)
    averaged = predictions.groupby(['dataset', 'person_id', 'trip_id'], observed=True).distance_km.mean().unstack('dataset')
    if {'nn_distance', 'nn_no_distance'} <= set(averaged):
        difference = averaged.nn_no_distance - averaged.nn_distance
        weights = observed.set_index(['person_id', 'trip_id']).weight.reindex(difference.index)
        axes[1].hist(difference.clip(-40, 40), bins=np.linspace(-40, 40, 65), weights=weights / weights.sum(), color=COLORS['nn_no_distance'], alpha=.85)
        axes[1].axvline(0, color='#555', lw=.8)
    axes[1].set(xlabel='Paired NN distance difference (km)\nLearned selector − distance input', ylabel='Weighted share (tails pooled at ±40 km)')
    for i, ax in enumerate(axes): panel(ax, string.ascii_lowercase[i])
    fig.tight_layout(w_pad=2)
    save(fig, directory, '06_observed_origin_survey_replay')


def prepare_basemap(caches, regions):
    reference = caches[0]
    municipalities = reference.stage('data.spatial.municipalities')[0].to_crs(2056)
    lakes_path = Path(reference.config('data_path')) / 'spatial/lakes'
    lake_files = sorted(lakes_path.glob('*.shp'))
    lakes = gpd.read_file(lake_files[0]).to_crs(2056) if lake_files else None
    roads_path = None
    for cache in caches:
        c = cache.yaml['config']
        path = Path(c['output_path']) / c['output_id'] / (c.get('output_prefix', 'switzerland_') + 'detailed_network.csv')
        if path.exists():
            roads_path = path
            break
    roads = []
    if roads_path:
        for chunk in pd.read_csv(roads_path, usecols=['Geometry'], chunksize=150000):
            first = chunk.Geometry.str.extract(r'LINESTRING\s*\(\s*([-\d.]+)\s+([-\d.]+)').astype(float)
            selected = np.zeros(len(chunk), dtype=bool)
            for r in regions.values():
                x0, y0, x1, y1 = r['bounds']
                selected |= first[0].between(x0-2000, x1+2000) & first[1].between(y0-2000, y1+2000)
            roads.extend(shapely.from_wkt(chunk.loc[selected, 'Geometry'].to_numpy()).tolist())
    if roads:
        normalized = shapely.normalize(np.array(roads, dtype=object))
        _, indices = np.unique(shapely.to_wkb(normalized), return_index=True)
        roads = list(shapely.simplify(normalized[indices], 15))
    provenance = dict(road_geometry=str(roads_path) if roads_path else None, lake_geometry=str(lake_files[0]) if lake_files else None, municipality_artifact=reference.used.get('data.spatial.municipalities'))
    return municipalities, lakes, roads, provenance


def geographic_background(ax, region, basemap):
    municipalities, lakes, roads, _ = basemap
    x0, y0, x1, y1 = region['bounds']
    bounds = shapely.box(x0, y0, x1, y1)
    ax.set_facecolor('#F7F7F2')
    local = municipalities.loc[municipalities.intersects(bounds)]
    local.boundary.plot(ax=ax, color='#BAC2C8', linewidth=.45, zorder=1, rasterized=True)
    if lakes is not None:
        water = lakes.loc[lakes.intersects(bounds)]
        if not water.empty: water.plot(ax=ax, color='#D9EAF0', edgecolor='#A6CBD9', linewidth=.4, zorder=1, rasterized=True)
    if roads:
        local_roads = [g for g in roads if g.intersects(bounds)]
        ax.add_collection(LineCollection([np.asarray(g.coords) for g in local_roads if g.geom_type == 'LineString'],
                                         linewidths=.13, colors='#ADB3B6', alpha=.7, zorder=2, rasterized=True))
    ax.set(xlim=(x0,x1), ylim=(y0,y1), aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values(): spine.set_visible(False)
    # A map scale in projected metres (LV95), shared by every panel.
    length = 2000.
    sx, sy = x0 + .06*(x1-x0), y0 + .06*(y1-y0)
    ax.plot([sx, sx+length], [sy, sy], color='#333', lw=1.5, zorder=5)
    ax.text(sx+length/2, sy+.018*(y1-y0), '2 km', ha='center', fontsize=6, zorder=5,
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.5))
    ax.annotate('N', xy=(.93,.94), xytext=(.93,.85), xycoords='axes fraction',
                ha='center', fontsize=7, arrowprops=dict(arrowstyle='-|>', color='#333', lw=.7), zorder=5)


def density_grid(frame, bounds, resolution=500):
    x0, y0, x1, y1 = bounds
    inside = frame.origin_x.between(x0, x1) & frame.origin_y.between(y0,y1)
    f = frame.loc[inside]
    xb = np.arange(x0, x1 + resolution, resolution)
    yb = np.arange(y0, y1 + resolution, resolution)
    if f.empty:
        return np.zeros((len(xb)-1,len(yb)-1)), xb,yb,0,0.
    grid, _, _ = np.histogram2d(f.destination_x, f.destination_y, bins=[xb,yb], weights=f.weight)
    return grid / f.weight.sum() * 1000, xb,yb,f.person_id.nunique(),grid.sum()/f.weight.sum()


def maps(datasets, observed, predictions, regions, purposes, basemap, directory):
    if predictions.empty:
        return []
    outputs = []
    for city, region in regions.items():
        for purpose in purposes:
            frames = [observed, predictions.loc[predictions.dataset == 'nn_distance'], predictions.loc[predictions.dataset == 'nn_no_distance'],
                      datasets.get('rda', pd.DataFrame()), datasets.get('nn_distance', pd.DataFrame()), datasets.get('nn_no_distance', pd.DataFrame())]
            if any(f.empty for f in frames): continue
            frames = [f.loc[f.purpose == purpose] for f in frames]
            grids = [density_grid(f, region['bounds']) for f in frames]
            vmax = max(float(g[0].max()) for g in grids)
            if vmax <= 0: continue
            norm = PowerNorm(gamma=.5, vmin=0, vmax=vmax)
            fig, axes = plt.subplots(2, 3, figsize=(7.3, 6.2))
            titles = ['Observed microcensus', 'NN · distance input', 'NN · learned selector', 'RDA', 'NN · distance input', 'NN · learned selector']
            for i,(ax,title,grid) in enumerate(zip(axes.flat,titles,grids)):
                geographic_background(ax, region, basemap)
                z,xb,yb,n,visible = grid
                image = ax.pcolormesh(xb,yb,np.ma.masked_where(z.T<=0,z.T), norm=norm, cmap='YlOrRd', alpha=.83, shading='flat', rasterized=True,zorder=3)
                ax.set_title(f'{string.ascii_lowercase[i]}   {title}', loc='left', fontsize=8)
                ax.text(.03,.975,f'n = {n:,} people\n{visible:.0%} of destination weight visible',transform=ax.transAxes,
                        va='top', fontsize=6, zorder=6,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
            fig.suptitle(f"{region['label']} · {PURPOSE_LABELS[purpose]}",fontsize=11, x=.08, ha='left')
            fig.text(.01,.73,'Same survey trips\nObserved origins',rotation=90,va='center',fontsize=8)
            fig.text(.01,.36,'Synthetic population\nCity-origin trips',rotation=90,va='center',fontsize=8)
            fig.subplots_adjust(left=.07,right=.99,bottom=.20,top=.91,hspace=.12,wspace=.05)
            cax=fig.add_axes([.27,.115,.48,.016])
            colorbar=fig.colorbar(image,cax=cax,orientation='horizontal')
            colorbar.set_label('Destination weight per 1,000 city-origin trips / 500 m cell',fontsize=7,labelpad=4)
            colorbar.ax.tick_params(labelsize=6.5)
            fig.text(.07,.015,'LV95 · Roads: scenario network (© OpenStreetMap contributors)\nBoundaries and lakes: supplied Swiss spatial data · identical extent, grid and color scale in all panels',fontsize=5.8)
            name=f'07_map_{city}_{purpose}'
            grid_tables=[]
            for panel_index, (title, grid) in enumerate(zip(titles, grids)):
                z, xb, yb, _, _ = grid
                xc, yc = np.meshgrid((xb[:-1]+xb[1:])/2, (yb[:-1]+yb[1:])/2, indexing='ij')
                grid_tables.append(pd.DataFrame(dict(panel_id=string.ascii_lowercase[panel_index], cohort='survey_replay' if panel_index < 3 else 'synthetic_population', panel=title, x=xc.ravel(), y=yc.ravel(), destinations_per_1000=z.ravel())))
            table_directory=Path(directory).parent/'tables'
            table_directory.mkdir(exist_ok=True)
            pd.concat(grid_tables,ignore_index=True).to_csv(table_directory/(name+'_grid.csv'),index=False)
            save(fig,directory,name)
            outputs.append(name)
    return outputs


def selector_figure(trips, directory):
    bins = [.01,.25,.5,1,1.5,3,5,10,20,1000]
    labels = ['.01–.25','.25–.5','.5–1','1–1.5','1.5–3','3–5','5–10','10–20','>20']
    fig,axes=plt.subplots(1,2,figsize=(7.3,3),sharey=True)
    for ax,car in zip(axes,['No car access','Car access']):
        for name,frame in trips.loc[trips.car_group==car].groupby('dataset',observed=True):
            f=frame.copy()
            f['bin']=pd.cut(f.observed_km,bins,labels=labels,include_lowest=True)
            f['short_weight']=f.selected_short.astype(float)*f.weight
            table=f.groupby('bin',observed=False)[['weight','short_weight']].sum().reindex(labels)
            ax.plot(range(len(labels)),table.short_weight/table.weight,color=COLORS[str(name)],marker='o',ms=3,lw=1.3,label=LABELS[str(name)])
        ax.axvline(3.5,color='#AAA',ls=':',lw=.8)
        ax.set(title=car,xticks=range(len(labels)),xticklabels=labels,ylim=(-.03,1.03),xlabel='Observed trip distance bin (km)')
        ax.tick_params(axis='x',rotation=45)
    axes[0].set_ylabel('Weighted share selecting short-range model')
    axes[1].legend(fontsize=7)
    for i,ax in enumerate(axes):panel(ax,string.ascii_lowercase[i])
    fig.tight_layout()
    save(fig,directory,'08_model_selection_diagnostic')
