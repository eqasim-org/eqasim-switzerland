'use strict';
// Geometry, Path2D construction, rasterization and hit testing stay off the UI thread.
const scenes=new Map(),analyses=new Map(),jobs=new Map();
const MAX_SCENE_BYTES=48*1024*1024;
let clock=0,totalBytes=0,lastHit=null;
const hitStates=new Map();
const fetches=new Map();
let peer=null;
function road(link,all){return link.modes.split(',').some(m=>m.trim()==='car'||m.trim()==='truck')&&link.hierarchy<=(all?5:4)}
function canvasContext(width,height,dpr){const canvas=new OffscreenCanvas(Math.max(1,Math.round(width*dpr)),Math.max(1,Math.round(height*dpr))),ctx=canvas.getContext('2d',{alpha:true,desynchronized:true});ctx.lineCap='round';ctx.lineJoin='round';return {canvas,ctx}}
function prepareScene(message){
 if(scenes.has(message.key)){scenes.get(message.key).used=++clock;return}
 const links=new Map();let bytes=0;
 for(const link of message.links){const path=new Path2D(),origin=link.points[0]||[0,0];let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;for(let i=0;i<link.points.length;i++){const p=link.points[i];i?path.lineTo(p[0]-origin[0],p[1]-origin[1]):path.moveTo(p[0]-origin[0],p[1]-origin[1]);x0=Math.min(x0,p[0]);y0=Math.min(y0,p[1]);x1=Math.max(x1,p[0]);y1=Math.max(y1,p[1])}links.set(link.index,{link,path,origin,box:[x0,y0,x1,y1]});bytes+=link.points.length*32+320}
 scenes.set(message.key,{links,bytes,used:++clock});totalBytes+=bytes;
 while(totalBytes>MAX_SCENE_BYTES&&scenes.size>2){const oldest=[...scenes].filter(([key])=>key!==message.key).sort((a,b)=>a[1].used-b[1].used)[0];if(!oldest)break;scenes.delete(oldest[0]);totalBytes-=oldest[1].bytes;postMessage({type:'evicted',key:oldest[0]})}
}
function prepareAnalysis(message){if(analyses.has(message.key))return;const parent=new Map(message.parent),nested=new Map(message.nested);const ordered=(map,direction)=>{const out=[];for(const [index,f] of map){const sides=direction==='both'?[f.upstream>=f.downstream?'upstream':'downstream']:[direction];for(const side of sides)if(f[side]>0)out.push({index,side,count:f[side]})}out.sort((a,b)=>a.count-b.count||a.index-b.index);return out};const orders=map=>Object.fromEntries(['both','upstream','downstream'].map(d=>[d,ordered(map,d)]));analyses.set(message.key,{parent,nested,parentOrder:orders(parent),nestedOrder:orders(nested)});while(analyses.size>4)analyses.delete(analyses.keys().next().value)}
function render(message){
 const start=performance.now(),scene=scenes.get(message.scene),analysis=analyses.get(message.analysis);if(!scene||!analysis){postMessage({type:'missing',role:message.role,id:message.id});return}scene.used=++clock;
 const v=message.view,s=message.style,b=[-v.ox/v.scale,(v.oy-v.height)/v.scale,(v.width-v.ox)/v.scale,v.oy/v.scale];
 const visible=item=>{const q=item.box,pad=(s.maximum+8)/v.scale;return q[0]<=b[2]+pad&&q[2]>=b[0]-pad&&q[1]<=b[3]+pad&&q[3]>=b[1]-pad};
 const active=f=>f&&((s.direction!=='downstream'&&f.upstream>=s.minimum)||(s.direction!=='upstream'&&f.downstream>=s.minimum));
 const setup=ctx=>ctx.setTransform(v.dpr*v.scale,0,0,-v.dpr*v.scale,v.dpr*v.ox,v.dpr*v.oy);
 if(message.role==='network'){
  const out=canvasContext(v.width,v.height,v.dpr);setup(out.ctx);out.ctx.strokeStyle=s.networkColor;out.ctx.lineWidth=s.networkWidth/v.scale;out.ctx.globalAlpha=.76;
  out.ctx.setTransform(v.dpr,0,0,v.dpr,0,0);out.ctx.lineWidth=s.networkWidth;const path=new Path2D();let count=0;for(const [index,item] of scene.links){if(!road(item.link,s.allLinks)||item.link.hierarchy>s.level||!visible(item)||active(analysis.parent.get(index))||active(analysis.nested.get(index)))continue;path.addPath(item.path,new DOMMatrix([v.scale,0,0,-v.scale,v.ox+item.origin[0]*v.scale,v.oy-item.origin[1]*v.scale]));count++}out.ctx.stroke(path);const bitmap=out.canvas.transferToImageBitmap();postMessage({type:'frame',role:'network',id:message.id,analysis:message.analysis,view:v,bitmap,metrics:{milliseconds:performance.now()-start,links:count}},[bitmap]);return;
 }
 const parent=canvasContext(v.width,v.height,v.dpr),nested=canvasContext(v.width,v.height,v.dpr);setup(parent.ctx);setup(nested.ctx);
 const hitGrid=new Map(),strokes=[],cell=48;
 const indexStroke=(item,width,layer)=>{
  const points=item.link.points,pad=width/2+4,minX=Math.max(0,Math.floor((item.box[0]*v.scale+v.ox-pad)/cell)),maxX=Math.min(Math.ceil(v.width/cell),Math.floor((item.box[2]*v.scale+v.ox+pad)/cell)),minY=Math.max(0,Math.floor((-item.box[3]*v.scale+v.oy-pad)/cell)),maxY=Math.min(Math.ceil(v.height/cell),Math.floor((-item.box[1]*v.scale+v.oy+pad)/cell));
  if(minX>maxX||minY>maxY)return;
  const index=strokes.length;strokes.push({item,width,layer});
  // One reference per link/cell, not one per geometry vertex.
  for(let x=minX;x<=maxX;x++)for(let y=minY;y<=maxY;y++){const key=x+','+y;if(!hitGrid.has(key))hitGrid.set(key,[]);hitGrid.get(key).push(index)}
 };
 let count=0;
 for(const [order,ctx,layer] of [[analysis.parentOrder[s.direction],parent.ctx,'main'],[analysis.nestedOrder[s.direction],nested.ctx,'nested']]){
  for(const stroke of order){if(s.direction!=='both'&&s.direction!==stroke.side||stroke.count<s.minimum)continue;const item=scene.links.get(stroke.index);if(!item||!road(item.link,s.allLinks)||!visible(item))continue;let width=1+s.maximum*Math.pow(stroke.count/Math.max(1,layer==='nested'?s.nestedTotal:s.total),s.power);if(layer==='nested')width=Math.max(1.3,.72*width);ctx.setTransform(v.dpr*v.scale,0,0,-v.dpr*v.scale,v.dpr*(v.ox+item.origin[0]*v.scale),v.dpr*(v.oy-item.origin[1]*v.scale));ctx.lineWidth=width/v.scale;ctx.strokeStyle=layer==='nested'?(stroke.side==='upstream'?s.nestedColor:s.nestedDownColor):stroke.side==='upstream'?s.upColor:s.downColor;ctx.stroke(item.path);if(layer==='nested'||s.parentVisible)indexStroke(item,width,layer);count++}
 }
 if(s.nestedSelected>=0){const item=scene.links.get(s.nestedSelected);if(item&&road(item.link,s.allLinks)&&visible(item))indexStroke(item,5,'nested')}
 if(s.selected>=0){const item=scene.links.get(s.selected);if(item&&road(item.link,s.allLinks)&&visible(item))indexStroke(item,7,'main')}
 lastHit={view:v,strokes,grid:hitGrid,analysis:message.analysis};hitStates.set(message.key,lastHit);while(hitStates.size>4)hitStates.delete(hitStates.keys().next().value);
 const parentBitmap=parent.canvas.transferToImageBitmap(),nestedBitmap=nested.canvas.transferToImageBitmap();postMessage({type:'frame',role:'flows',id:message.id,analysis:message.analysis,view:v,parent:parentBitmap,nested:nestedBitmap,metrics:{milliseconds:performance.now()-start,strokes:count,hitEntries:strokes.length}},[parentBitmap,nestedBitmap]);
}
function hit(message){
 const state=hitStates.get(message.frameKey)||lastHit;if(!state||state.analysis!==message.analysis){postMessage({type:'hit',id:message.id,links:[]});return}
 const old=state.view,v=message.view,world=[(message.x-v.ox)/v.scale,-(message.y-v.oy)/v.scale],x=world[0]*old.scale+old.ox,y=-world[1]*old.scale+old.oy;
 const candidates=new Map(),ratio=v.scale/old.scale,radius=(40/Math.min(1,ratio)),loX=Math.floor((x-radius)/48),hiX=Math.floor((x+radius)/48),loY=Math.floor((y-radius)/48),hiY=Math.floor((y+radius)/48),ids=new Set();
 for(let ix=loX;ix<=hiX;ix++)for(let iy=loY;iy<=hiY;iy++)for(const id of state.grid.get(ix+','+iy)||[])ids.add(id);
 for(const id of ids){const stroke=state.strokes[id],points=stroke.item.link.points;let distance=Infinity;for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((world[0]-a[0])*dx+(world[1]-a[1])*dy)/(dx*dx+dy*dy||1)));distance=Math.min(distance,Math.hypot(world[0]-a[0]-t*dx,world[1]-a[1]-t*dy)*v.scale)}if(distance<=stroke.width/2+4){const key=stroke.layer+':'+stroke.item.link.index,previous=candidates.get(key);if(!previous||distance<previous.distance)candidates.set(key,{link:{...stroke.item.link,traceLayer:stroke.layer},distance})}}
 const links=[...candidates.values()].sort((a,b)=>(b.link.traceLayer==='nested')-(a.link.traceLayer==='nested')||a.distance-b.distance).slice(0,4).map(v=>v.link);postMessage({type:'hit',id:message.id,links});
}
onmessage=async event=>{const m=event.data;try{
 if(m.type==='peer'){peer=m.port;peer.onmessage=e=>{if(e.data.type==='scene'){prepareScene(e.data);postMessage({type:'sceneReady',key:e.data.key})}};}
 else if(m.type==='scene')prepareScene(m);
 else if(m.type==='analysis')prepareAnalysis(m);
 else if(m.type==='render')render(m);
 else if(m.type==='hit')hit(m);
 else if(m.type==='fetch'){
  const controller=new AbortController();fetches.set(m.id,controller);try{const response=await fetch(m.url,{signal:controller.signal});if(!response.ok)throw Error(await response.text());if(!(response.headers.get('content-type')||'').includes('json'))throw Error('Server returned a page instead of geometry');const data=await response.json();if(controller.signal.aborted)throw new DOMException('Aborted','AbortError');for(const link of data.links||[]){const b=[Infinity,Infinity,-Infinity,-Infinity];for(const p of link.points){b[0]=Math.min(b[0],p[0]);b[1]=Math.min(b[1],p[1]);b[2]=Math.max(b[2],p[0]);b[3]=Math.max(b[3],p[1])}link.bounds=b}const key='worker-scene-'+m.id;data.workerSceneKey=key;prepareScene({key,links:data.links||[]});postMessage({type:'sceneReady',key});if(peer)peer.postMessage({type:'scene',key,links:data.links||[]});postMessage({type:'fetched',id:m.id,data})}finally{fetches.delete(m.id)}
 }else if(m.type==='cancel')fetches.get(m.id)?.abort();
 }catch(error){postMessage({type:'error',id:m.id,role:m.role,message:error.message,aborted:error.name==='AbortError'})}};
