"""Render sampled NPVM demand by zone, traffic scope and crossing source."""

import json
from pathlib import Path

from analysis.freight.trips_html import prepare, zone_features


def render(frames, locations, metadata, output):
    data = {}
    for model, frame in frames.items():
        payload = prepare(frame, "Freight")
        if len(payload["rows"]) != len(frame):
            raise ValueError("Demand-map rows must align with the generated trip table")
        for row, trip in zip(payload["rows"], frame.itertuples()):
            row.extend([trip.scope, bool(trip.origin_foreign), bool(trip.destination_foreign)])
        if not data:
            payload["zones"] = zone_features(frame, "Freight", {"npvm_locations": locations[model]},
                                              {"config": {"use_npvm_freight": True}})
        payload["zone_system"] = "NPVM"
        data[model] = payload
    return render_data(data, metadata, output)


def render_data(data, metadata, output):
    # Both variants use the same OD geography; embed zone shapes once.
    data = {model: dict(payload) for model, payload in data.items()}
    for model in list(data)[1:]:
        data[model].pop("zones", None)
    template = Path(__file__).with_name("trips.html.template").read_text()
    template = template.replace('Freight &amp; LCV', 'NPVM freight demand')
    template = template.replace('1. Demand</label>', '1. Crossing source</label>')
    # Keep polygon hit targets above the full-map canvas used for crossing markers.
    template = template.replace("map.createPane('zones').style.zIndex=350;", "map.createPane('zones').style.zIndex=450;")
    template = template.replace("{pane:'zones',bubblingMouseEvents:false,pointToLayer:", "{pane:'zones',renderer:L.svg({pane:'zones'}),bubblingMouseEvents:false,pointToLayer:")
    template = template.replace("fillColor:'#8068c0',fillOpacity:.35", "fillColor:'#9333ea',fillOpacity:.65")
    template = template.replace('const el=id=>', 'const sharedZones=DATA[Object.keys(DATA)[0]].zones;for(const d of Object.values(DATA))d.zones=sharedZones;\nconst el=id=>')
    template = template.replace('<option value="Freight">Freight</option>', '<option value="Network">Network links</option>')
    template = template.replace('<option value="LCV">LCV — light commercial vehicles</option>', '<option value="Interview">Interview places</option>')
    controls = '''<label for="trafficScope">Traffic demand</label><select id="trafficScope">
<option value="all">Both domestic and cross-border</option><option value="domestic">Domestic only</option>
<option value="international">All cross-border, including transit</option><option value="incoming">Entering Switzerland</option>
<option value="outgoing">Leaving Switzerland</option><option value="transit">Transit only</option></select>
<p class="muted">Click a zone for its demand and connected zones. Incoming/outgoing below selects direction relative to that zone. Crossing markers show sampled endpoint counts; transit contributes entry and exit.</p>'''
    template = template.replace('<label>Departure hours [start, end)</label>', controls + '<label>Departure hours [start, end)</label>')
    template = template.replace('.rows.filter(r=>allTimes', '.rows.filter(matchesScope).filter(r=>allTimes')
    template = template.replace('let zoneLookup=', 'let demandCounts=new Map(), demandMaximum=1;\nlet zoneLookup=')
    template = template.replace('function updateConnections(zoneTrips){', "function updateConnections(zoneTrips){demandCounts=new Map();for(const r of timeRows())demandCounts.set(r[selectedIndex()],(demandCounts.get(r[selectedIndex()])||0)+1);demandMaximum=Math.max(1,...demandCounts.values());")
    template = template.replace("fillOpacity:selectedZone ? .015 : .06", "fillOpacity:selectedZone ? .015 : .04+.6*Math.sqrt((demandCounts.get(id)||0)/demandMaximum)")
    template = template.replace("' selected zone':''}", "' selected zone':' · '+fmt(demandCounts.get(id)||0)+' sampled trips'}")
    template = template.replace('vehicle_type,origin_zone,destination_zone', 'vehicle_type,origin_zone,destination_zone,traffic_scope,origin_foreign,destination_foreign')
    template = template.replace('<div class="zone-controls" id="zoneControls">', '<section id="zoneStats" aria-live="polite" style="padding:12px;background:#eef3f8;margin-top:12px">Click a zone to see freight totals.</section><div class="zone-controls" id="zoneControls">')
    template = template.replace('zoneLayers.set(id,layer);layer.on', 'zoneLayers.set(id,layer);layer.bindPopup(()=>zonePopup(id),{maxWidth:400});layer.on')
    template = template.replace("el('selectedZone').value=id;update();", "el('selectedZone').value=id;update();layer.bringToFront();layer.openPopup();")
    script = '''
function matchesScope(r){const scope=el('trafficScope').value;return scope==='all'||(scope==='international'?r[9]!=='domestic':r[9]===scope);}
function zoneTotals(id){
 const counts={domestic:0,incoming:0,outgoing:0,transit:0,total:0,origins:0,destinations:0};
 const start=Number(el('start').value)*3600,end=Number(el('end').value)*3600;
 for(const r of DATA[el('dataset').value].rows){
  if(!allTimes&&!(r[5]!==null&&r[5]>=start&&r[5]<end))continue;
  if(r[7]!==id&&r[8]!==id)continue;
  counts[r[9]]++;counts.total++;if(r[7]===id)counts.origins++;if(r[8]===id)counts.destinations++;
 }
 counts.international=counts.incoming+counts.outgoing+counts.transit;return counts;
}
function zonePopup(id){
 const counts=zoneTotals(id),box=document.createElement('div'),title=document.createElement('strong');
 title.textContent=zoneName(id);box.append(title);const table=document.createElement('table');
 for(const [label,key] of [['Cross-border including transit','international'],['Entering Switzerland','incoming'],['Leaving Switzerland','outgoing'],['Transit OD trips','transit'],['Local / domestic trips','domestic'],['Total unique trips','total'],['Origins in this zone','origins'],['Destinations in this zone','destinations']]){
  const tr=document.createElement('tr');for(const value of [label,fmt(counts[key])]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);
 }
 box.append(table);const note=document.createElement('p');note.textContent=(100*META.input_downsampling)+'% sample: trips with an origin or destination in this source zone, counted once. All traffic types; current time range and crossing source. Transit counts foreign OD endpoints, not routed passage through Swiss zones.';note.style.fontSize='11px';box.append(note);return box;
}
function showZoneStats(){const panel=el('zoneStats');panel.replaceChildren();if(selectedZone)panel.append(zonePopup(selectedZone));else panel.textContent='Click a zone to see entering, leaving, transit and domestic freight totals.';}
const entranceMarkers=L.layerGroup().addTo(map);
function drawEntrances(){
 entranceMarkers.clearLayers();const counts=new Map();
 for(const r of visible){for(const [flag,index,role] of [[10,1,'Entry'],[11,3,'Exit']]){
  if(!r[flag])continue;const key=role+'|'+r[index]+'|'+r[index+1];
  if(!counts.has(key))counts.set(key,{lat:r[index],lon:r[index+1],role,n:0});counts.get(key).n++;
 }}
 for(const p of counts.values())L.circleMarker([p.lat,p.lon],{radius:Math.min(16,4+Math.log2(1+p.n)),color:p.role==='Entry'?'#138478':'#c05a13',fillOpacity:.8,weight:2}).bindTooltip(p.role+' · '+p.n.toLocaleString('en-GB')+' sampled endpoints').addTo(entranceMarkers);
}
const originalUpdate=update;update=function(){originalUpdate();drawEntrances();showZoneStats();};
el('trafficScope').onchange=update;drawEntrances();showZoneStats();
'''
    template = template.replace('</script></body>', script + '</script></body>')
    assets = (output / "crossings.html").read_text()
    values = {"__LEAFLET_CSS__": assets.split('<style>', 1)[1].split('</style>', 1)[0],
              "__LEAFLET_JS__": assets.split('<script>', 1)[1].split('</script>', 1)[0],
              "__DATA__": json.dumps(data, separators=(",", ":"), allow_nan=False).replace('<', '\\u003c'),
              "__META__": json.dumps({**metadata, "demand_map_basis": "Sampled NPVM trips; source OD zones define traffic scope. Transit is one trip and two crossing endpoints."}, allow_nan=False).replace('<', '\\u003c')}
    for key, value in values.items():
        template = template.replace(key, value)
    target = output / "demand.html"
    target.write_text(template)
    return target
