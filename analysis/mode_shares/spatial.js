/* Leaflet controls and an accessible SVG comparison; no data are sent to a server. */
'use strict';
const payload=JSON.parse(document.getElementById('payload').textContent);
document.getElementById('payload').remove();
const $=id=>document.getElementById(id);
const modeNames=['Car driver','Public transport','Walk','Bike','Car passenger'];
const fmt=new Intl.NumberFormat('en-CH');
let ready=false, sequence=0, latest=null, shape=null, shapeName='All filtered trips', pendingCentre=false;
let resultView='shares';
let activeDraw=null, activeLayer=null, selectionMode='all';
const regions={
  municipality:{title:'Municipality',key:'municipalities',id:'municipality_id',name:'municipality_name',features:[],layer:null},
  canton:{title:'Canton',key:'cantons',id:'canton_id',name:'canton_name',features:[],layer:null}
};
const map=L.map('map',{preferCanvas:true}).setView([46.82,8.23],8);
const tiles=L.tileLayer(payload.tiles,{maxZoom:18,opacity:.7,attribution:'© swisstopo, geo.admin.ch'}).addTo(map);
tiles.on('tileerror',()=>{$('mapNote').textContent='Background tiles are unavailable. Drawing, circles, municipality/canton selection and statistics still work. Map source: © swisstopo.';});
const selected=L.featureGroup().addTo(map);
const drawing=new L.Control.Draw({edit:{featureGroup:selected,remove:false},draw:false});
map.addControl(drawing);
function setSelection(layer,name,geometry) {
  selected.clearLayers();selected.addLayer(layer);activeLayer=layer;
  shape=geometry;shapeName=name;pendingCentre=false;map.getContainer().style.cursor='';
  $('selection').textContent=name;query();
}
function circle(center,radius) {
  activateMode('circle',true);
  $('latitude').value=center.lat.toFixed(7);$('longitude').value=center.lng.toFixed(7);
  $('radius').value=Math.round(radius);
  setSelection(L.circle(center,{radius,color:'#176b94',weight:2,fillOpacity:.12}),`Circle · ${center.lat.toFixed(5)}, ${center.lng.toFixed(5)} · ${Math.round(radius).toLocaleString()} m`,{type:'circle',center:[center.lng,center.lat],radius});
}
map.on(L.Draw.Event.CREATED,event=>{
  if(event.layerType==='circle') circle(event.layer.getLatLng(),event.layer.getRadius());
  else {activateMode('polygon',true);setSelection(event.layer,'Drawn polygon',event.layer.toGeoJSON().geometry);}
});
map.on(L.Draw.Event.EDITED,event=>event.layers.eachLayer(layer=>{
  if(layer instanceof L.Circle) circle(layer.getLatLng(),layer.getRadius());
  else setSelection(layer,'Edited polygon',layer.toGeoJSON().geometry);
}));
map.on('click',event=>{if(pendingCentre) circle(event.latlng,Number($('radius').value));});
$('drawPolygon').onclick=()=>{activateMode('polygon');activeDraw=new L.Draw.Polygon(map,{allowIntersection:false,shapeOptions:{color:'#176b94'}});activeDraw.enable();};
$('pickCentre').onclick=()=>{activateMode('circle');pendingCentre=true;map.getContainer().style.cursor='crosshair';$('selection').textContent='Click the map to place the circle centre, or enter coordinates below';};
$('municipalityMode').onclick=()=>activateMode('municipality');
$('cantonMode').onclick=()=>activateMode('canton');
$('applyCircle').onclick=()=>{
  const lat=Number($('latitude').value),lon=Number($('longitude').value),r=Number($('radius').value);
  if(!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180||!Number.isFinite(r)||r<=0) {alert('Enter valid coordinates and a positive radius.');return;}
  circle(L.latLng(lat,lon),r);map.fitBounds(activeLayer.getBounds(),{maxZoom:14});
};
$('radius').onchange=()=>{if(shape?.type==='circle' && Number($('radius').value)>0)circle(activeLayer.getLatLng(),Number($('radius').value));};
$('national').onclick=()=>activateMode('all');
$('direction').onchange=query;
$('modeShareView').onclick=()=>setResultView('shares');
$('distanceView').onclick=()=>setResultView('distances');
$('opacity').oninput=()=>tiles.setOpacity(Number($('opacity').value));
function query() {
  if(!ready)return;
  $('status').textContent='Computing selected trips…';$('download').disabled=true;
  worker.postMessage({type:'query',id:++sequence,shape,direction:$('direction').value});
}
const workerURL=URL.createObjectURL(new Blob([$('workerSource').textContent],{type:'application/javascript'}));
const worker=new Worker(workerURL);
worker.onerror=error=>{$('status').textContent='Could not initialise the spatial worker: '+error.message;};
worker.onmessage=event=>{
  const message=event.data;
  if(message.type==='ready') {
    ready=true;for(const id of ['drawPolygon','pickCentre','national','applyCircle'])$(id).disabled=false;
    updateRegionButtons();$('status').textContent='Ready';query();
  } else if(message.type==='error') $('status').textContent='Error: '+message.message;
  else if(message.type==='result' && message.id===sequence) {
    latest=message.result;render(latest);$('status').textContent=shapeName+' · '+$('direction').selectedOptions[0].text;
    $('download').disabled=false;
  }
};
worker.postMessage({type:'init',datasets:payload.datasets});
const coverage=Object.entries(payload.datasets).map(([name,data])=>`${name}: ${fmt.format(data.missing_locations.origin)} trips without origin coordinates, ${fmt.format(data.missing_locations.destination)} without destination coordinates, ${fmt.format(data.missing_locations.home)} without home coordinates.`).join(' ');
$('locationCoverage').textContent=coverage+' Missing coordinates cannot meet a spatial condition; the unrestricted total still includes all retained trips.';
// Release compressed trip strings in the main window after the worker receives them.
for(const dataset of Object.values(payload.datasets))delete dataset.data;
function share(value){return value===null?'—':(100*value).toFixed(1)+'%';}
function setResultView(view) {
  resultView=view;
  for(const [id,value] of [['modeShareView','shares'],['distanceView','distances']]) {
    $(id).classList.toggle('primary',view===value);$(id).setAttribute('aria-pressed',String(view===value));
  }
  if(latest)render(latest);
}
function km(value) {return value===null?'—':value.toFixed(2);}
function table(headers,rows,titles=[]) {
  const head=$('numberHeaders'),body=$('numbers');head.replaceChildren();body.replaceChildren();
  for(const label of headers) {const th=document.createElement('th');th.textContent=label;head.append(th);}
  rows.forEach((values,index)=>{
    const row=document.createElement('tr');if(titles[index])row.title=titles[index];
    for(const text of values) {const cell=document.createElement('td');cell.textContent=text;row.append(cell);}
    if(index===0&&resultView==='distances')row.style.fontWeight='700';
    body.append(row);
  });
}
function render(result) {
  const survey=result.Microcensus,sim=result.MATSim,distances=resultView==='distances';
  $('surveyMetricLabel').textContent=distances?'Microcensus · mean distance':'Microcensus observations';
  $('simMetricLabel').textContent=distances?'MATSim · mean distance':'MATSim simulated trips';
  $('surveyTrips').textContent=distances?(survey.distances[0].mean===null?'—':km(survey.distances[0].mean)+' km'):fmt.format(survey.trips);
  $('simTrips').textContent=distances?(sim.distances[0].mean===null?'—':km(sim.distances[0].mean)+' km'):fmt.format(sim.trips);
  $('surveySample').textContent=distances?fmt.format(survey.distances[0].n)+' trips with valid distance':'';
  $('simSample').textContent=distances?fmt.format(sim.distances[0].n)+' trips with valid distance':'';
  $('surveyPeople').textContent=fmt.format(survey.people)+' distinct respondents';
  $('simPeople').textContent=fmt.format(sim.people)+' distinct simulated people';
  $('surveyWeight').textContent=fmt.format(Math.round(survey.total*10)/10)+' weighted selected trips';
  const notes=[];
  if(survey.people<30)notes.push('Small Microcensus sample: fewer than 30 distinct respondents. Interpret these results cautiously.');
  if(!survey.trips||!sim.trips)notes.push('One dataset has no retained trips in this selection; its statistics are undefined.');
  if(distances) {
    $('viewExplanation').textContent='Euclidean distance in km · boxes: 25th–75th percentiles; line: median; whiskers: 10th–90th percentiles; diamond: mean. Microcensus statistics use person weights.';
    const rows=[],titles=[],labels=['All modes',...modeNames];
    for(let m=0;m<6;m++) {
      const a=survey.distances[m],b=sim.distances[m];
      const delta=a.mean===null||b.mean===null?'—':(b.mean-a.mean).toFixed(2);
      rows.push([labels[m],km(a.mean),km(b.mean),delta,km(a.median),km(b.median),fmt.format(a.n),fmt.format(b.n)]);
      titles.push(['Microcensus: '+distanceDescription(a),'MATSim: '+distanceDescription(b)].join(' | '));
    }
    table(['Mode','Mz mean km','Sim. mean km','Δ km','Mz median','Sim. median','Mz valid trips','Sim. valid trips'],rows,titles);
    distanceChart(survey.distances,sim.distances);
    if(survey.distances[0].n<survey.trips||sim.distances[0].n<sim.trips)notes.push('Some selected trips have invalid distances and are excluded from distance statistics only.');
  } else {
    $('viewExplanation').textContent='Weighted survey trip shares compared with simulated trip shares.';
    const rows=[];
    for(let m=0;m<5;m++) {
      const delta=survey.shares[m]===null||sim.shares[m]===null?'—':((sim.shares[m]-survey.shares[m])*100).toFixed(1)+' pp';
      rows.push([modeNames[m],share(survey.shares[m]),share(sim.shares[m]),delta,fmt.format(survey.counts[m]),fmt.format(sim.counts[m])]);
    }
    table(['Mode','Mz share','MATSim share','Difference','Mz trips','Sim. trips'],rows);
    chart(survey.shares,sim.shares);
  }
  $('smallSample').className=notes.length?'warning':'';$('smallSample').textContent=notes.join(' ');
}
function distanceDescription(s) {
  return `n=${s.n}, mean=${km(s.mean)} km, P10=${km(s.p10)}, P25=${km(s.p25)}, median=${km(s.median)}, P75=${km(s.p75)}, P90=${km(s.p90)}`;
}
function distanceChart(survey,sim) {
  const width=730,height=500,left=122,right=25,top=25,bottom=45,plotW=width-left-right;
  const maximum=Math.max(.1,...[...survey,...sim].filter(s=>s.n).map(s=>Math.max(s.p90,s.mean)));
  const step=10**Math.floor(Math.log10(maximum/4));
  const tick=Math.ceil(maximum/4/step)*step,limit=tick*4;
  const scale=value=>left+plotW*value/limit;
  let svg=`<svg role="img" aria-label="Weighted Euclidean distance distributions in kilometres" viewBox="0 0 ${width} ${height}" style="width:100%">`;
  svg+=`<rect x="0" y="${top}" width="${width}" height="65" rx="6" fill="#f0f5fa"/>`;
  for(let i=0;i<=4;i++) {
    const value=i*tick,x=scale(value);
    svg+=`<line x1="${x}" y1="${top}" x2="${x}" y2="${height-bottom}" stroke="#e0e8f0"/><text x="${x}" y="${height-25}" text-anchor="middle" font-size="11" fill="#53677e">${value<1?value.toFixed(2):value.toFixed(1)}</text>`;
  }
  const labels=['All modes',...modeNames];
  for(let m=0;m<6;m++) {
    const center=top+35+m*67;
    svg+=`<text x="${left-12}" y="${center+4}" text-anchor="end" font-size="12" font-weight="${m===0?'700':'400'}" fill="#182b45">${labels[m]}</text>`;
    for(const [j,stats,color] of [[0,survey,'#176b94'],[1,sim,'#dc5675']]) {
      const s=stats[m];if(!s.n)continue;
      const y=center+(j===0?-11:11),lo=scale(s.p10),hi=scale(s.p90),q1=scale(s.p25),q3=scale(s.p75),median=scale(s.median),mean=scale(s.mean);
      svg+=`<g><title>${j?'MATSim':'Microcensus'} ${labels[m]}: ${distanceDescription(s)}</title><line x1="${lo}" y1="${y}" x2="${hi}" y2="${y}" stroke="${color}" stroke-width="2"/><line x1="${lo}" y1="${y-5}" x2="${lo}" y2="${y+5}" stroke="${color}"/><line x1="${hi}" y1="${y-5}" x2="${hi}" y2="${y+5}" stroke="${color}"/><rect x="${q1}" y="${y-7}" width="${Math.max(1,q3-q1)}" height="14" fill="${color}" fill-opacity=".2" stroke="${color}"/><line x1="${median}" y1="${y-7}" x2="${median}" y2="${y+7}" stroke="${color}" stroke-width="2"/><path d="M ${mean} ${y-5} l 5 5 l -5 5 l -5 -5 Z" fill="${color}"/></g>`;
    }
  }
  svg+=`<text x="${left+plotW/2}" y="${height-5}" text-anchor="middle" font-size="12" fill="#182b45">Euclidean distance (km)</text>`;
  $('chart').innerHTML=svg+'</svg>';
}
function chart(survey,sim) {
  const width=600,height=280,left=45,bottom=45,top=20,plotH=height-bottom-top,plotW=width-left-10;
  let svg=`<svg role="img" aria-label="Selected mode shares" viewBox="0 0 ${width} ${height}" style="width:100%">`;
  for(let tick=0;tick<=100;tick+=20) {
    const y=top+plotH*(1-tick/100);
    svg+=`<line x1="${left}" y1="${y}" x2="${width-10}" y2="${y}" stroke="#e7edf4"/><text x="${left-7}" y="${y+4}" text-anchor="end" font-size="11" fill="#53677e">${tick}%</text>`;
  }
  for(let m=0;m<5;m++) {
    const center=left+plotW*(m+.5)/5;
    for(const [j,values,color] of [[0,survey,'#176b94'],[1,sim,'#dc5675']]) {
      if(values[m]===null)continue;
      const h=values[m]*plotH,x=center+(j-.5)*30-13,y=top+plotH-h;
      svg+=`<rect x="${x}" y="${y}" width="26" height="${h}" rx="3" fill="${color}"><title>${j?'MATSim':'Microcensus'} ${modeNames[m]}: ${share(values[m])}</title></rect><text x="${x+13}" y="${y-5}" text-anchor="middle" font-size="10" fill="${color}">${(100*values[m]).toFixed(1)}</text>`;
    }
    const label=['Car driver','PT','Walk','Bike','Car passenger'][m];
    svg+=`<text x="${center}" y="${height-18}" text-anchor="middle" font-size="11" fill="#182b45">${label}</text>`;
  }
  $('chart').innerHTML=svg+'</svg>';
}
// Selection modes are exclusive: only the relevant administrative overlay is shown.
function activateMode(mode,keepSelection=false) {
  if(activeDraw) {activeDraw.disable();activeDraw=null;}
  pendingCentre=false;map.getContainer().style.cursor='';
  selectionMode=mode;
  for(const [kind,region] of Object.entries(regions)) {
    if(region.layer) {if(kind===mode)region.layer.addTo(map);else map.removeLayer(region.layer);}
  }
  for(const [id,kind] of [['drawPolygon','polygon'],['pickCentre','circle'],['municipalityMode','municipality'],['cantonMode','canton']]) {
    $(id).classList.toggle('primary',kind===mode);$(id).setAttribute('aria-pressed',String(kind===mode));
  }
  $('circleControls').hidden=mode!=='circle';
  $('municipalityControls').hidden=mode!=='municipality';
  $('cantonControls').hidden=mode!=='canton';
  // Administrative boundaries are reference data, not editable drawn polygons.
  if(mode==='municipality'||mode==='canton') {if(drawing._map)map.removeControl(drawing);}
  else if(!drawing._map)map.addControl(drawing);
  if(!keepSelection) {
    selected.clearLayers();activeLayer=null;shape=null;shapeName='All filtered trips';query();
  }
  if(!keepSelection) {
    $('selection').textContent=regions[mode]
      ?`Click a ${mode} on the map or choose one below. Until selected, statistics show all filtered trips.`
      :mode==='all'?'All filtered trips · no geographical restriction'
      :mode==='polygon'?'Draw a polygon on the map':'Click the map to place a circle centre';
  }
}
function updateRegionButtons() {
  for(const [kind,region] of Object.entries(regions))$(kind+'Mode').disabled=!ready||!region.features.length;
}
async function loadRegions(kind) {
  const region=regions[kind];
  try {
    const raw=Uint8Array.from(atob(payload[region.key]),c=>c.charCodeAt(0));
    const geo=await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
    region.features=geo.features.sort((a,b)=>String(a.properties[region.name]).localeCompare(String(b.properties[region.name])));
    payload[region.key]=null;
    region.layer=L.geoJSON(geo,{style:{color:kind==='canton'?'#394f6b':'#58718d',weight:kind==='canton'?2:1,fillOpacity:.05},smoothFactor:2,onEachFeature:(feature,layer)=>{
      layer.bindTooltip(feature.properties[region.name]+' · '+feature.properties[region.id]);
      layer.on('click',event=>{L.DomEvent.stopPropagation(event);chooseRegion(kind,feature);});
    }});
    if(selectionMode===kind)region.layer.addTo(map);
    updateOptions(kind);updateRegionButtons();
  } catch(error) {$('mapNote').textContent=region.title+' boundaries could not load: '+error.message;}
}
function updateOptions(kind) {
  const region=regions[kind];
  const q=kind==='municipality'?$('municipalitySearch').value.toLocaleLowerCase():'';
  const select=$(kind);select.replaceChildren(new Option('Choose…',''));
  for(const feature of region.features) {
    const p=feature.properties,label=p[region.name]+' · '+p[region.id];
    if(!q||label.toLocaleLowerCase().includes(q))select.add(new Option(label,String(p[region.id])));
  }
}
function chooseRegion(kind,feature) {
  activateMode(kind,true);
  const region=regions[kind],p=feature.properties;
  const layer=L.geoJSON(feature,{style:{color:'#176b94',weight:3,fillOpacity:.15}});
  setSelection(layer,region.title+' · '+p[region.name]+' ('+p[region.id]+')',feature.geometry);
  map.fitBounds(layer.getBounds(),{maxZoom:14});
}
$('municipalitySearch').oninput=()=>updateOptions('municipality');
for(const [kind,region] of Object.entries(regions))$(kind).onchange=()=>{
  const feature=region.features.find(f=>String(f.properties[region.id])===$(kind).value);
  if(feature)chooseRegion(kind,feature);
};
$('download').onclick=()=>{
  if(!latest)return;
  const distanceView=resultView==='distances';
  const rows=[distanceView
    ?['dataset','mode','mean_km','p10_km','p25_km','median_km','p75_km','p90_km','minimum_km','maximum_km','valid_distance_trips','weighted_distance_trips','selected_trips','selected_people','selection','direction']
    :['dataset','mode','share_percent','raw_trips','weighted_trips','selected_trips','selected_people','selection','direction']];
  for(const [name,data] of Object.entries(latest)) {
    if(distanceView) {
      for(let m=0;m<6;m++) {
        const s=data.distances[m];
        rows.push([name,m===0?'all_modes':payload.modes[m-1],s.mean,s.p10,s.p25,s.median,s.p75,s.p90,s.min,s.max,s.n,s.weight,data.trips,data.people,shapeName,$('direction').value]);
      }
    } else for(let m=0;m<5;m++)rows.push([name,payload.modes[m],data.shares[m]===null?'':100*data.shares[m],data.counts[m],data.weighted[m],data.trips,data.people,shapeName,$('direction').value]);
  }
  const csv=rows.map(row=>row.map(v=>'"'+(v===null?'':String(v)).replaceAll('"','""')+'"').join(',')).join('\n');
  const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  link.download=distanceView?'selected_distances.csv':'selected_mode_shares.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
loadRegions('municipality');
loadRegions('canton');
