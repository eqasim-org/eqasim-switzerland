/* Spatial selection runs off the UI thread, directly on columnar trip arrays. */
const MODES = ['car','pt','walk','bike','car_passenger'];
const MISSING = -2147483648;
let datasets = {};
async function inflate(encoded) {
  const raw = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}
function boundary(a,b,x,y) {
  const cross=(x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0]);
  return Math.abs(cross)<1e-12 && x>=Math.min(a[0],b[0])-1e-10 && x<=Math.max(a[0],b[0])+1e-10 && y>=Math.min(a[1],b[1])-1e-10 && y<=Math.max(a[1],b[1])+1e-10;
}
function prepareRing(ring) {
  // Index edges by latitude band: a horizontal ray only needs nearby edges.
  // This retains the original geometry, including holes and boundary points.
  let left=Infinity,right=-Infinity,bottom=Infinity,top=-Infinity;
  for(const [x,y] of ring) {left=Math.min(left,x);right=Math.max(right,x);bottom=Math.min(bottom,y);top=Math.max(top,y);}
  const count=Math.min(256,Math.max(1,Math.ceil(Math.sqrt(ring.length))));
  const scale=top>bottom?count/(top-bottom):0;
  const band=y=>Math.max(0,Math.min(count-1,Math.floor((y-bottom)*scale)));
  const buckets=Array.from({length:count},()=>[]);
  for(let i=0,j=ring.length-1;i<ring.length;j=i++) {
    const lo=band(Math.min(ring[j][1],ring[i][1]));
    const hi=band(Math.max(ring[j][1],ring[i][1]));
    for(let k=lo;k<=hi;k++)buckets[k].push(i);
  }
  return {left,right,bottom,top,test:(x,y)=>{
    if(x<left||x>right||y<bottom||y>top)return 0;
    let inside=false;
    for(const i of buckets[band(y)]) {
      const a=ring[(i+ring.length-1)%ring.length],b=ring[i];
      if(boundary(a,b,x,y))return 2;
      if((a[1]>y)!==(b[1]>y) && x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])inside=!inside;
    }
    return inside?1:0;
  }};
}
function preparePolygon(p) {
  const rings=p.map(prepareRing),outer=rings[0];
  return {left:outer.left,right:outer.right,bottom:outer.bottom,top:outer.top,test:(x,y)=>{
    const hit=outer.test(x,y);
    if(hit===2)return true;
    if(!hit)return false;
    for(let i=1;i<rings.length;i++) {const hole=rings[i].test(x,y);if(hole===2)return true;if(hole===1)return false;}
    return true;
  }};
}
function selector(shape) {
  if(!shape) return ()=>true;
  if(shape.type==='circle') {
    const [lon,lat]=shape.center, rad=Math.PI/180, r=shape.radius;
    return (x,y)=>{
      if(x===MISSING || y===MISSING) return false;
      x/=1e6;y/=1e6;
      const a=Math.sin((y-lat)*rad/2)**2 + Math.cos(lat*rad)*Math.cos(y*rad)*Math.sin((x-lon)*rad/2)**2;
      return 2*6371000*Math.asin(Math.min(1,Math.sqrt(a))) <= r+1e-6;
    };
  }
  const polygons=shape.type==='Polygon'?[shape.coordinates]:shape.coordinates;
  const prepared=polygons.map(preparePolygon);
  return (x,y)=>{
    if(x===MISSING || y===MISSING) return false;
    x/=1e6;y/=1e6;
    return prepared.some(b=>x>=b.left && x<=b.right && y>=b.bottom && y<=b.top && b.test(x,y));
  };
}
function summarize(data, shape, direction) {
  const contains=selector(shape), weights=Array(5).fill(0), counts=Array(5).fill(0);
  const people=new Uint8Array(data.people);
  let n=0, persons=0;
  for(let i=0;i<data.rows;i++) {
    let keep;
    if(!shape) keep=true;
    else if(direction==='residents') keep=contains(data.home_lon[i],data.home_lat[i]);
    else if(direction==='arrivals') keep=contains(data.destination_lon[i],data.destination_lat[i]);
    else if(direction==='departures') keep=contains(data.origin_lon[i],data.origin_lat[i]);
    else {
      const a=contains(data.origin_lon[i],data.origin_lat[i]);
      const b=contains(data.destination_lon[i],data.destination_lat[i]);
      keep=direction==='both'?(a&&b):(a||b);
    }
    if(!keep) continue;
    n++; const id=data.person[i]; if(!people[id]) {people[id]=1;persons++;}
    counts[data.mode[i]]++;weights[data.mode[i]]+=data.weight[i];
  }
  const total=weights.reduce((a,b)=>a+b,0);
  return {trips:n,people:persons,total,counts,weighted:weights,shares:weights.map(v=>total?v/total:null)};
}
onmessage=async(event)=>{
  const message=event.data;
  try {
    if(message.type==='init') {
      for(const [name,meta] of Object.entries(message.datasets)) {
        const buffer=await inflate(meta.data), data={rows:meta.rows,people:meta.people};
        for(const [key,col] of Object.entries(meta.columns)) {
          const Type=col.dtype==='|u1'?Uint8Array:col.dtype==='<u4'?Uint32Array:col.dtype==='<f8'?Float64Array:Int32Array;
          data[key]=new Type(buffer,col.offset,meta.rows);
        }
        datasets[name]=data;
      }
      postMessage({type:'ready'});
    } else if(message.type==='query') {
      const result={};
      for(const [name,data] of Object.entries(datasets)) result[name]=summarize(data,message.shape,message.direction);
      postMessage({type:'result',id:message.id,result});
    }
  } catch(error) {postMessage({type:'error',message:error.message,id:message.id});}
};
