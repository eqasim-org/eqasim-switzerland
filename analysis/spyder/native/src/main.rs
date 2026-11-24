use anyhow::{bail, Context, Result};
use flate2::{read::MultiGzDecoder,write::GzEncoder,Compression};
use std::{borrow::Cow,sync::{Arc,Mutex,RwLock,atomic::{AtomicU64,Ordering},mpsc},thread};
use quick_xml::{events::{BytesStart, Event}, Reader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, HashSet}, fs::{self, File}, io::{BufRead, BufReader, BufWriter, Read, Write}, net::{TcpListener, TcpStream}, path::Path};

#[derive(Serialize, Deserialize)]
struct Link { id: String, points: Vec<[f64; 2]>, modes: String, #[serde(default)] highway: String }
#[derive(Serialize, Deserialize, Default)]
struct Trip { mode: String, #[serde(default)] group: u8, links: Vec<u32>, times: Vec<f64> }
#[derive(Serialize)]
struct Flow { link: u32, upstream: u64, downstream: u64 }

fn input(path: &str) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("Opening {path}"))?;
    let stream: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(MultiGzDecoder::new(file))
    } else if path.ends_with(".zst") {
        Box::new(zstd::stream::read::Decoder::new(file).with_context(|| format!("Opening Zstandard stream {path}"))?)
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::with_capacity(1024 * 1024, stream)))
}
fn attrs(e: &BytesStart<'_>) -> Result<HashMap<String, String>> {
    e.attributes().map(|a| {
        let a = a?;
        Ok((String::from_utf8(a.key.as_ref().to_vec())?, a.unescape_value()?.into_owned()))
    }).collect()
}
fn json_write<T: Serialize>(path: &Path, data: &T) -> Result<()> {
    let temp = path.with_extension("tmp");
    let mut out = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer(&mut out, data)?;
    out.flush()?;
    fs::rename(temp, path)?;
    Ok(())
}
fn network(path: &str, geometry: &str) -> Result<Vec<Link>> {
    let mut reader = Reader::from_reader(input(path)?);
    let mut buf = Vec::new();
    let mut nodes = HashMap::<String, [f64; 2]>::new();
    let mut links = Vec::<Link>::new();
    let mut current_link=None;
    let mut highway_attribute=false;
    loop {
        let event=reader.read_event_into(&mut buf).with_context(|| format!("Reading XML {path} at decompressed byte {}", reader.buffer_position()))?;
        let empty=matches!(&event,Event::Empty(_));
        match event {
            Event::Start(e) | Event::Empty(e) if e.name().as_ref() == b"node" => {
                let a = attrs(&e)?;
                nodes.insert(a["id"].clone(), [a["x"].parse()?, a["y"].parse()?]);
            }
            Event::Start(e) | Event::Empty(e) if e.name().as_ref() == b"link" => {
                let a = attrs(&e)?;
                links.push(Link { id: a["id"].clone(), points: vec![*nodes.get(&a["from"]).context("Missing from node")?, *nodes.get(&a["to"]).context("Missing to node")?], modes: a.get("modes").cloned().unwrap_or_default(), highway:a.get("osm:way:highway").or_else(||a.get("highway")).cloned().unwrap_or_default() });
                current_link=if empty {None} else {Some(links.len()-1)};
            }
            Event::Start(e) if e.name().as_ref()==b"attribute" && current_link.is_some() => {
                highway_attribute=attrs(&e)?.get("name").is_some_and(|name|is_highway_name(name));
            }
            Event::Text(e) if highway_attribute => {if let Some(i)=current_link {links[i].highway.push_str(&e.unescape()?);}}
            Event::CData(e) if highway_attribute => {if let Some(i)=current_link {links[i].highway.push_str(std::str::from_utf8(e.as_ref())?);}}
            Event::End(e) if e.name().as_ref()==b"attribute" => highway_attribute=false,
            Event::End(e) if e.name().as_ref()==b"link" => {current_link=None;highway_attribute=false;}
            Event::Eof => break,
            _ => ()
        }
        buf.clear();
    }
    let indices: HashMap<_, _> = links.iter().enumerate().map(|(i, l)| (l.id.clone(), i)).collect();
    let mut matched = 0;
    let mut invalid = 0;
    if geometry != "-" {
        let mut csv = csv::Reader::from_reader(input(geometry)?);
        let headers = csv.headers()?.clone();
        let id = headers.iter().position(|x| x == "LinkId" || x == "link_id").context("Geometry CSV needs LinkId or link_id")?;
        let geo = headers.iter().position(|x| x == "Geometry" || x == "geometry").context("Geometry CSV needs Geometry or geometry")?;
        for row in csv.records() {
            let row = row?;
            if let Some(&i) = indices.get(&row[id]) {
                let wkt = row[geo].trim();
                let points: Option<Vec<[f64; 2]>> = wkt.find('(').zip(wkt.rfind(')')).and_then(|(a,b)| {
                    if !wkt[..a].trim().eq_ignore_ascii_case("LINESTRING") { return None; }
                    wkt[a+1..b].split(',').map(|p| {
                        let xy: Vec<_> = p.split_whitespace().collect();
                        Some([xy.first()?.parse().ok()?, xy.get(1)?.parse().ok()?])
                    }).collect()
                });
                if let Some(mut points) = points.filter(|p| p.len() >= 2 && p.iter().flatten().all(|v| v.is_finite())) {
                    let start = links[i].points[0];
                    let distance = |p: [f64; 2]| (p[0]-start[0]).hypot(p[1]-start[1]);
                    if distance(*points.last().unwrap()) < distance(points[0]) { points.reverse(); }
                    // Only accept shapes aligned to the current network, not stale merged IDs.
                    let end = links[i].points[1];
                    let last = points.last().unwrap();
                    if distance(points[0]) <= 5.0 && (last[0]-end[0]).hypot(last[1]-end[1]) <= 5.0 {
                        points[0] = start;
                        let last_index = points.len()-1;
                        points[last_index] = end;
                        links[i].points = points;
                        matched += 1;
                    } else { invalid += 1; }
                } else { invalid += 1; }
            }
        }
    }
    eprintln!("Network: {} links; {matched} detailed shapes; {invalid} invalid/stale shapes; {} endpoint fallbacks", links.len(), links.len()-matched);
    Ok(links)
}
// Model population groups, not nationality inferred from identifiers.
const GROUPS: [&str; 6] = ["Unknown / unmatched", "Swiss resident population", "French external population", "Cross-border population", "Freight", "Airport"];

fn population_group(attributes: &HashMap<String, String>) -> u8 {
    let value = |key: &str| attributes.get(key).map(|s| s.trim()).unwrap_or("");
    let flag = |key: &str| match value(key).to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    };
    let sub = value("subpopulation").to_ascii_lowercase();
    if flag("isFreight") == Some(true) || sub == "freight" { return 4; }
    if sub == "airport" { return 5; }
    if flag("isCrossBorder") == Some(true) || sub == "crossborder" { return 3; }
    if flag("isExternalFR") == Some(true) { return 2; }
    let canton = value("cantonName").to_ascii_uppercase();
    let swiss_canton = ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS", "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE", "GE", "JU"].contains(&canton.as_str())
        // Lower-case 'fr' is the pipeline's external-France sentinel, not Fribourg.
        && value("cantonName") != "fr";
    let swiss_id = value("cantonId").parse::<f64>().is_ok_and(|n| n >= 1.0 && n <= 26.0 && n.fract() == 0.0);
    let ordinary = sub.is_empty() || matches!(sub.as_str(), "ordinary" | "normal" | "person");
    if ordinary && (swiss_canton || swiss_id || (flag("isCrossBorder") == Some(false) && flag("isExternalFR") == Some(false))) {
        return 1;
    }
    0
}
fn population(path: &str) -> Result<HashMap<String, u8>> {
    let mut people = HashMap::new();
    if path == "-" {
        eprintln!("No population source: all vehicle trips will be Unknown / unmatched");
        return Ok(people);
    }
    let format_path = path.strip_suffix(".gz").or_else(|| path.strip_suffix(".zst")).unwrap_or(path);
    if format_path.ends_with(".csv") {
        let mut source = input(path)?;
        let header = source.fill_buf()?;
        let header = &header[..header.iter().position(|b| *b == b'\n').unwrap_or(header.len())];
        let delimiter = if header.contains(&b';') { b';' } else { b',' };
        let mut csv = csv::ReaderBuilder::new().delimiter(delimiter).from_reader(source);
        let headers = csv.headers()?.clone();
        let id = headers.iter().position(|s| matches!(s, "person" | "person_id" | "id")).context("Persons CSV needs person, person_id, or id column")?;
        let columns: Vec<_> = headers.iter().enumerate().filter(|(_,k)| matches!(*k, "isExternalFR" | "isCrossBorder" | "isFreight" | "subpopulation" | "cantonName" | "cantonId")).map(|(i,k)| (i,k.to_string())).collect();
        for row in csv.records() {
            let row = row?;
            let a = columns.iter().map(|(i,k)| (k.clone(), row[*i].to_string())).collect();
            if people.insert(row[id].to_string(), population_group(&a)).is_some() { bail!("Duplicate person ID in persons CSV"); }
        }
    } else {
        // Read only person-level attributes; plan/leg/activity attributes cannot override them.
        let mut reader = Reader::from_reader(input(path)?);
        let mut buf = Vec::new();
        let mut depth = 0usize;
        let mut person_depth = None;
        let mut person_id = String::new();
        let mut attributes = HashMap::new();
        let mut attribute = None;
        let mut text = String::new();
        loop {
            match reader.read_event_into(&mut buf).with_context(|| format!("Reading XML {path} at decompressed byte {}", reader.buffer_position()))? {
                Event::Start(e) => {
                    depth += 1;
                    if e.name().as_ref() == b"person" {
                        person_depth = Some(depth);
                        person_id = attrs(&e)?.remove("id").context("Population person missing id")?;
                        attributes.clear();
                    } else if e.name().as_ref() == b"attribute" && person_depth.is_some_and(|d| depth == d + 2) {
                        attribute = attrs(&e)?.remove("name");
                        text.clear();
                    }
                }
                Event::Text(e) if attribute.is_some() => text.push_str(&e.unescape()?),
                Event::CData(e) if attribute.is_some() => text.push_str(std::str::from_utf8(e.as_ref())?),
                Event::End(e) => {
                    if e.name().as_ref() == b"attribute" {
                        if let Some(name) = attribute.take() { attributes.insert(name, text.trim().to_string()); }
                    } else if e.name().as_ref() == b"person" {
                        if people.insert(person_id.clone(), population_group(&attributes)).is_some() { bail!("Duplicate person ID in population XML"); }
                        person_depth = None;
                    }
                    depth = depth.saturating_sub(1);
                }
                Event::Empty(e) if e.name().as_ref() == b"person" => {
                    let id = attrs(&e)?.remove("id").context("Population person missing id")?;
                    if people.insert(id, 0).is_some() { bail!("Duplicate person ID in population XML"); }
                }
                Event::Eof => break,
                _ => ()
            }
            buf.clear();
        }
    }
    let mut counts = [0u64; 6];
    for &group in people.values() { counts[group as usize] += 1; }
    eprintln!("Population: {} persons; groups {:?}", people.len(), counts);
    Ok(people)
}
fn events(path: &str, links: &[Link], people: &HashMap<String, u8>) -> Result<Vec<Trip>> {
    let index: HashMap<&str, u32> = links.iter().enumerate().map(|(i,l)| (l.id.as_str(), i as u32)).collect();
    let mut reader = Reader::from_reader(input(path)?);
    let mut buf = Vec::new();
    let mut active = HashMap::<String, Trip>::new();
    let mut trips = Vec::new();
    let mut seen = 0u64;
    let mut unknown = 0u64;
    loop {
        match reader.read_event_into(&mut buf).with_context(|| format!("Reading XML {path} at decompressed byte {}", reader.buffer_position()))? {
            Event::Start(e) | Event::Empty(e) if e.name().as_ref() == b"event" => {
                // Ignore person/activity events before allocating their attribute maps.
                let kind = e.try_get_attribute("type")?.map(|a| a.value.into_owned()).unwrap_or_default();
                if !matches!(kind.as_slice(), b"vehicle enters traffic" | b"vehicle leaves traffic" | b"entered link") { buf.clear(); continue; }
                let a = attrs(&e)?;
                let vehicle = a.get("vehicle").context("Traffic event missing vehicle")?;
                let entering = kind == b"vehicle enters traffic";
                let leaving = kind == b"vehicle leaves traffic";
                if entering {
                    if let Some(old) = active.remove(vehicle) { if !old.links.is_empty() { trips.push(old); } }
                    active.insert(vehicle.clone(), Trip { mode: a.get("networkMode").cloned().unwrap_or_else(|| "unknown".into()), group: a.get("person").and_then(|id| people.get(id)).copied().unwrap_or(0), ..Trip::default() });
                }
                let trip = active.entry(vehicle.clone()).or_insert_with(|| Trip { mode: "unknown".into(), ..Trip::default() });
                if !leaving {
                    if let Some(id) = a.get("link") {
                        if let Some(&i) = index.get(id.as_str()) {
                            // Some event writers emit entered-link for the initial link as well.
                            let time: f64 = a.get("time").context("Missing event time")?.parse()?;
                            if !time.is_finite() { bail!("Nonfinite event time"); }
                            if trip.links.last() != Some(&i) || trip.times.last() != Some(&time) {
                                trip.links.push(i); trip.times.push(time);
                            }
                        } else { unknown += 1; }
                    }
                }
                if leaving { if let Some(trip) = active.remove(vehicle) { if !trip.links.is_empty() { trips.push(trip); } } }
                seen += 1;
                if seen % 5_000_000 == 0 { eprintln!("Read {seen} traffic/link events..."); }
            }
            Event::Eof => break,
            _ => ()
        }
        buf.clear();
    }
    // EOF may contain vehicles still in traffic. Sort to make the cache reproducible.
    let mut tail: Vec<_> = active.into_iter().collect();
    tail.sort_by(|a,b| a.0.cmp(&b.0));
    trips.extend(tail.into_iter().map(|(_,t)| t).filter(|t| !t.links.is_empty()));
    eprintln!("Events: {seen} relevant events; {} vehicle trips; {unknown} references outside network", trips.len());
    Ok(trips)
}
fn is_highway_name(name:&str)->bool {matches!(name,"osm:way:highway"|"osm:highway"|"highway")}
fn read_highways(path:&str)->Result<HashMap<String,String>> {
    let mut reader=Reader::from_reader(input(path)?);
    let mut buf=Vec::new();let mut id=None;let mut value=String::new();let mut capture=false;let mut result=HashMap::new();
    loop {
        match reader.read_event_into(&mut buf).with_context(||format!("Reading highway attributes in {path} at byte {}",reader.buffer_position()))? {
            Event::Start(e) if e.name().as_ref()==b"link" => {
                let a=attrs(&e)?;id=a.get("id").cloned();value=a.get("osm:way:highway").or_else(||a.get("highway")).cloned().unwrap_or_default();
            }
            Event::Empty(e) if e.name().as_ref()==b"link" => {
                let a=attrs(&e)?;if let Some(key)=a.get("id") {result.insert(key.clone(),a.get("osm:way:highway").or_else(||a.get("highway")).cloned().unwrap_or_default());}
            }
            Event::Start(e) if e.name().as_ref()==b"attribute" && id.is_some() => capture=attrs(&e)?.get("name").is_some_and(|name|is_highway_name(name)),
            Event::Text(e) if capture =>value.push_str(&e.unescape()?),
            Event::CData(e) if capture =>value.push_str(std::str::from_utf8(e.as_ref())?),
            Event::End(e) if e.name().as_ref()==b"attribute" =>capture=false,
            Event::End(e) if e.name().as_ref()==b"link" => {if let Some(key)=id.take() {result.insert(key,value.trim().to_string());}capture=false;value.clear();}
            Event::Eof=>break,
            _=>(),
        }
        buf.clear();
    }
    eprintln!("Read highway attributes for {} links (events untouched)",result.len());
    Ok(result)
}
fn hierarchy_level(highway:&str)->u8 {
    highway.to_ascii_lowercase().split(|c:char|!c.is_ascii_alphabetic()&&c!='_').map(|kind|match kind {
        "motorway"|"motorway_link"=>0,"trunk"|"trunk_link"=>1,"primary"|"primary_link"=>2,
        "secondary"|"secondary_link"=>3,"tertiary"|"tertiary_link"=>4,_=>5,
    }).min().unwrap_or(5)
}
fn write_hierarchy(dir:&Path,highways:&HashMap<String,String>)->Result<()> {
    let levels:HashMap<_,_>=highways.iter().map(|(id,kind)|(id,hierarchy_level(kind))).collect();
    json_write(&dir.join("hierarchy.json"),&levels)
}

#[derive(Clone,Copy)]
enum ViaSide {Upstream,Downstream,Either}
#[derive(Clone,Copy)]
struct Via {link:u32,side:ViaSide}
fn via_matches(trip:&Trip,position:usize,via:Via)->bool {
    match via.side {ViaSide::Upstream=>trip.links[..position].contains(&via.link),ViaSide::Downstream=>trip.links[position+1..].contains(&via.link),ViaSide::Either=>trip.links[..position].contains(&via.link)||trip.links[position+1..].contains(&via.link)}
}
fn matches_traffic(trip: &Trip, mode: &str, road_traffic: bool) -> bool {
    (mode.is_empty() || trip.mode == mode) && (!road_traffic || matches!(trip.mode.as_str(), "car" | "truck"))
}
fn query(trips: &[Trip], candidates: &[usize], selected: u32, start: f64, end: f64, mode: &str, groups: u8, road_traffic: bool, via: Option<Via>, anchor_via: bool) -> serde_json::Value {
    let anchor = if anchor_via {via.unwrap().link} else {selected};
    let (count, up, down, group_counts) = candidates.par_iter().fold(
        || (0u64, HashMap::<u32,u64>::new(), HashMap::<u32,u64>::new(), [0u64;6]),
        |(mut count, mut up, mut down, mut group_counts), &i| {
            let t = &trips[i];
            if !matches_traffic(t,mode,road_traffic) { return (count, up, down, group_counts); }
            if let Some(p) = t.links.iter().enumerate().position(|(p,l)| *l == selected && t.times[p] >= start && t.times[p] < end) {
                if via.is_some_and(|via|!via_matches(t,p,via)) {return (count,up,down,group_counts);}
                group_counts[t.group as usize] += 1;
                if groups & (1 << t.group) != 0 {
                    count += 1;
                    // Time filtering remains at the first target; split the complete path at the second.
                    let p = if anchor_via {t.links.iter().position(|l| *l == anchor).unwrap()} else {p};
                    for l in t.links[..p].iter().copied().filter(|l| *l != anchor).collect::<HashSet<_>>() { *up.entry(l).or_default() += 1; }
                    for l in t.links[p+1..].iter().copied().filter(|l| *l != anchor).collect::<HashSet<_>>() { *down.entry(l).or_default() += 1; }
                }
            }
            (count, up, down, group_counts)
        }
    ).reduce(|| (0, HashMap::new(), HashMap::new(), [0u64;6]), |(a,mut b,mut c,mut g),(d,e,f,h)| {
        for (k,v) in e { *b.entry(k).or_default() += v; }
        for (k,v) in f { *c.entry(k).or_default() += v; }
        for i in 0..6 {g[i]+=h[i];}
        (a+d,b,c,g)
    });
    let mut keys: Vec<_> = up.keys().chain(down.keys()).copied().collect(); keys.sort_unstable(); keys.dedup();
    let flows: Vec<_> = keys.into_iter().map(|link| Flow {link, upstream: *up.get(&link).unwrap_or(&0), downstream: *down.get(&link).unwrap_or(&0)}).collect();
    serde_json::json!({"selected":anchor,"primary_selected":selected,"trips":count,"flows":flows,"group_counts":group_counts,"all_group_trips":group_counts.iter().sum::<u64>(),"via":via.map(|via|via.link)})
}
fn parse_groups(params: &HashMap<&str,&str>) -> Result<u8> {
    if params.contains_key("group") && params.contains_key("groups") {bail!("Use groups or group, not both");}
    let value=params.get("groups").or_else(||params.get("group"));
    let Some(value)=value else {return Ok(63);};
    if *value=="none" && params.contains_key("groups") {return Ok(0);}
    let mut mask=0;
    for token in value.split(',') {
        let group: u8=token.parse().context("Invalid population group")?;
        if group as usize>=GROUPS.len() {bail!("Invalid population group");}
        mask|=1<<group;
    }
    Ok(mask)
}
struct Overlay {
    ids: HashSet<usize>,
    counts: HashMap<usize,(u64,u64)>,
    selected: usize,
    trips: u64,
    groups: u8,
    mode: String,
    road_traffic: bool,
}
fn is_road(link:&Link) -> bool {
    link.modes.split(',').any(|mode|matches!(mode.trim(),"car"|"truck"))
}

struct Connection {stream:TcpStream,gzip:bool}
impl Write for Connection {
    fn write(&mut self,data:&[u8])->std::io::Result<usize>{self.stream.write(data)}
    fn flush(&mut self)->std::io::Result<()>{self.stream.flush()}
}
fn respond(stream:&mut Connection,status:&str,mime:&str,body:&[u8])->Result<()> {
    let compressed=stream.gzip && body.len()>=4096 && mime.starts_with("application/json");
    let payload=if compressed {let mut encoder=GzEncoder::new(Vec::new(),Compression::fast());encoder.write_all(body)?;Cow::Owned(encoder.finish()?)} else {Cow::Borrowed(body)};
    write!(stream,"HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\nVary: Accept-Encoding\r\n",payload.len())?;
    if compressed {write!(stream,"Content-Encoding: gzip\r\n")?;}
    if mime.starts_with("text/html")||mime.starts_with("application/javascript") {write!(stream,"Cache-Control: no-cache\r\n")?;}
    write!(stream,"\r\n")?;stream.write_all(&payload)?;Ok(())
}

// Swisstopo CH1903+/LV95 -> WGS84 approximate formula, then Web Mercator.
// https://www.swisstopo.admin.ch/dam/en/sd-web/KLRCX9XIdXDu/ch1903wgs84-EN.pdf
fn display_point([east,north]: [f64;2]) -> [f64;2] {
    let e=(east-2_600_000.0)/1_000_000.0;
    let n=(north-1_200_000.0)/1_000_000.0;
    let lon=(2.6779094+4.728982*e+0.791484*e*n+0.1306*e*n*n-0.0436*e*e*e)*100.0/36.0;
    let lat=(16.9023892+3.238272*n-0.270978*e*e-0.002528*n*n-0.0447*e*e*n-0.0140*n*n*n)*100.0/36.0;
    [6_378_137.0*lon.to_radians(),6_378_137.0*(std::f64::consts::FRAC_PI_4+lat.clamp(-85.0,85.0).to_radians()/2.0).tan().ln()]
}
fn link_box(link: &Link) -> [f64;4] {
    link.points.iter().fold([f64::INFINITY,f64::INFINITY,f64::NEG_INFINITY,f64::NEG_INFINITY],|mut b,p| {
        b[0]=b[0].min(p[0]);b[1]=b[1].min(p[1]);b[2]=b[2].max(p[0]);b[3]=b[3].max(p[1]);b
    })
}
struct DisplayIndex {
    cells: [HashMap<(i32,i32),Vec<usize>>;6],
    by_level: [Vec<usize>;6],
    boxes: Vec<[f64;4]>,
    long_links: [Vec<usize>;6],
    roads: Vec<bool>,
    major: Vec<bool>,
    levels: Vec<u8>,
    bounds: [f64;4],
}
impl DisplayIndex {
    fn new(links:&[Link],cached_levels:&HashMap<String,u8>)->Self {
        let mut index=Self {cells:std::array::from_fn(|_|HashMap::new()),by_level:std::array::from_fn(|_|Vec::new()),boxes:vec![[0.0;4];links.len()],long_links:std::array::from_fn(|_|Vec::new()),roads:Vec::with_capacity(links.len()),major:Vec::with_capacity(links.len()),levels:Vec::with_capacity(links.len()),bounds:[f64::INFINITY,f64::INFINITY,f64::NEG_INFINITY,f64::NEG_INFINITY]};
        for (i,link) in links.iter().enumerate() {
            let level=cached_levels.get(&link.id).copied().filter(|rank|*rank<=5).unwrap_or_else(||hierarchy_level(&link.highway));
            let road=is_road(link);index.levels.push(level);index.roads.push(road);index.major.push(road&&level<5);
            if !road {continue;}
            let level=level as usize;let b=link_box(link);index.boxes[i]=b;index.by_level[level].push(i);
            for k in 0..4 {index.bounds[k]=if k<2 {index.bounds[k].min(b[k])} else {index.bounds[k].max(b[k])};}
            let cells=Self::cell_bounds(b);
            if (i64::from(cells[2])-i64::from(cells[0])+1)*(i64::from(cells[3])-i64::from(cells[1])+1)>100 {index.long_links[level].push(i);continue;}
            for x in cells[0]..=cells[2] {for y in cells[1]..=cells[3] {index.cells[level].entry((x,y)).or_default().push(i);}}
        }
        index
    }
    fn cell_bounds(b:[f64;4])->[i32;4] {b.map(|v|(v/5000.0).floor() as i32)}
    fn candidates(&self,b:[f64;4],level:u8)->Vec<usize> {
        let cells=Self::cell_bounds(b);
        let intersects=|i:usize|{let a=self.boxes[i];a[0]<=b[2]&&a[2]>=b[0]&&a[1]<=b[3]&&a[3]>=b[1]};
        let mut result=if (i64::from(cells[2])-i64::from(cells[0])+1)*(i64::from(cells[3])-i64::from(cells[1])+1)>10000 {
            self.by_level[..=level as usize].par_iter().flat_map_iter(|ids|ids.iter().copied().filter(|i|intersects(*i))).collect()
        } else {
            let mut candidates=HashSet::new();
            for rank in 0..=level as usize {
                for x in cells[0]..=cells[2] {for y in cells[1]..=cells[3] {if let Some(ids)=self.cells[rank].get(&(x,y)) {candidates.extend(ids.iter().copied());}}}
                candidates.extend(self.long_links[rank].iter().copied());
            }
            candidates.into_iter().filter(|i|intersects(*i)).collect::<Vec<_>>()
        };
        result.sort_unstable();result
    }
}

#[derive(Serialize)]
struct DisplayLink<'a> {index:usize,id:&'a str,points:Cow<'a,[[f64;2]]>,modes:&'a str,highway:&'a str,hierarchy:u8}
fn display_link(link:&Link,index:usize,tolerance:f64)->DisplayLink<'_> {
    let points=if tolerance<=0.0 {Cow::Borrowed(link.points.as_slice())} else {
        let mut points=Vec::new();
        for (i,p) in link.points.iter().enumerate() {if i==0||i+1==link.points.len()||points.last().is_none_or(|last:&[f64;2]|(last[0]-p[0]).hypot(last[1]-p[1])>=tolerance) {points.push(*p);}}
        Cow::Owned(points)
    };
    DisplayLink {index,id:&link.id,points,modes:&link.modes,highway:&link.highway,hierarchy:hierarchy_level(&link.highway)}
}
#[derive(Serialize)]
struct ViewportResponse<'a> {links:Vec<DisplayLink<'a>>,total:usize,truncated:bool,requested_level:u8,effective_level:u8,coarsened:bool,spider_only:bool}

fn link_distance(link:&Link,x:f64,y:f64) -> f64 {
    link.points.windows(2).map(|p| {
        let dx=p[1][0]-p[0][0];let dy=p[1][1]-p[0][1];
        let t=if dx*dx+dy*dy>0.0 {((x-p[0][0])*dx+(y-p[0][1])*dy)/(dx*dx+dy*dy)} else {0.0};
        let t=t.clamp(0.0,1.0);(x-p[0][0]-t*dx).hypot(y-p[0][1]-t*dy)
    }).fold(f64::INFINITY,f64::min)
}
fn decode_query(value:&str)->String {
    let bytes=value.as_bytes();let mut decoded=Vec::new();let mut i=0;
    while i<bytes.len() {
        if bytes[i]==b'%' && i+2<bytes.len() {
            if let (Some(a),Some(b))=((bytes[i+1] as char).to_digit(16),(bytes[i+2] as char).to_digit(16)) { decoded.push((a*16+b) as u8);i+=3;continue; }
        }
        decoded.push(if bytes[i]==b'+' {b' '} else {bytes[i]});i+=1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}
fn serve(dir: &Path, address: &str, budget:usize) -> Result<()> {
    let mut links: Vec<Link> = serde_json::from_reader(BufReader::new(File::open(dir.join("network.json"))?))?;
    if let Ok(file)=File::open(dir.join("highways.json")) {
        let mut highways:HashMap<String,String>=serde_json::from_reader(BufReader::new(file))?;
        for link in &mut links {if let Some(highway)=highways.remove(&link.id) {link.highway=highway;}}
    }
    let trips: Vec<Trip> = serde_json::from_reader(BufReader::new(File::open(dir.join("trips.json"))?))?;
    let mut inverted = vec![Vec::<usize>::new(); links.len()];
    let mut modes: Vec<_> = trips.iter().map(|t| t.mode.clone()).collect(); modes.sort(); modes.dedup();
    for (i,t) in trips.iter().enumerate() {
        for l in t.links.iter().copied().collect::<HashSet<_>>() { inverted[l as usize].push(i); }
    }
    // Project only the display copy; processed LV95 network and trip caches stay unchanged.
    links.par_iter_mut().for_each(|link| {
        for point in &mut link.points { *point = display_point(*point); }
    });
    let cached_levels:HashMap<String,u8>=match File::open(dir.join("hierarchy.json")) {Ok(file)=>serde_json::from_reader(BufReader::new(file))?,Err(_)=>HashMap::new()};
    let spatial = DisplayIndex::new(&links,&cached_levels);
    let ids: HashMap<_,_> = links.iter().enumerate().map(|(i,l)| (l.id.as_str(),i)).collect();
    let html = fs::read(dir.join("index.html"))?;
    let worker=fs::read(dir.join("viewer.worker.js")).unwrap_or_default();
    let runtime=fs::read(dir.join("viewer.runtime.js")).unwrap_or_default();
    let background = fs::read(dir.join("background.json")).unwrap_or_else(|_| br#"{"layers":{},"attribution":""}"#.to_vec());
    let mut group_counts = [0u64; 6];
    for trip in &trips { group_counts[trip.group as usize] += 1; }
    let groups: Vec<_> = GROUPS.iter().enumerate().map(|(i,label)| serde_json::json!({"id":i,"label":label,"trips":group_counts[i]})).collect();
    let meta = serde_json::to_vec(&serde_json::json!({"modes":modes,"groups":groups,"trips":trips.len(),"threads":budget,"compute_threads":rayon::current_num_threads(),"http_workers":http_workers(budget),"links":spatial.roads.iter().filter(|road|**road).count(),"network_links":links.len(),"major_links":spatial.major.iter().filter(|major|**major).count(),"bounds":spatial.bounds,"display_crs":"EPSG:3857"}))?;
    let listener = TcpListener::bind(address)?;
    eprintln!("Viewer ready at http://{address}/ (Ctrl+C to stop)");
    let overlay_counter=AtomicU64::new(0);
    let stored_overlays=RwLock::new(HashMap::<u64,Arc<Overlay>>::new());
    let handle=|raw:TcpStream|->Result<()> {
            raw.set_nodelay(true)?;
            raw.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
            raw.set_write_timeout(Some(std::time::Duration::from_secs(30)))?;
            let mut reader=BufReader::new((&raw).take(32768));let mut line=String::new();reader.read_line(&mut line)?;
            let mut gzip=false;let mut header_size=line.len();
            loop {let mut header=String::new();if reader.read_line(&mut header)?==0||header=="\r\n"||header=="\n" {break;}header_size+=header.len();if header_size>32700 {bail!("Request headers too large");}
                if let Some((name,value))=header.split_once(':') {if name.eq_ignore_ascii_case("accept-encoding") {gzip=value.split(',').any(|part|{let mut fields=part.trim().split(';');fields.next()==Some("gzip")&&!fields.any(|f|f.trim()=="q=0")});}}
            }
            drop(reader);let mut stream=Connection {stream:raw,gzip};
            let overlays=stored_overlays.read().map_err(|_|anyhow::anyhow!("Overlay lock poisoned"))?.clone();
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.first() != Some(&"GET") { return respond(&mut stream, "405 Method Not Allowed", "text/plain", b"GET required"); }
            let url = parts.get(1).copied().unwrap_or("/");
            match url {
                "/" | "/index.html" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", &html),

                "/viewer.worker.js" if !worker.is_empty()=>respond(&mut stream,"200 OK","application/javascript; charset=utf-8",&worker),
                "/viewer.runtime.js" if !runtime.is_empty()=>respond(&mut stream,"200 OK","application/javascript; charset=utf-8",&runtime),
                "/api/background" => respond(&mut stream,"200 OK","application/json",&background),
                "/api/meta" => respond(&mut stream, "200 OK", "application/json", &meta),
                _ if url.starts_with("/api/network?") || url.starts_with("/api/pick?") || url.starts_with("/api/link?") => {
                    let params: HashMap<_,_> = url.split_once('?').unwrap().1.split('&').filter_map(|p| p.split_once('=')).collect();
                    let show_all=params.get("all_links")!=Some(&"0");
                    let requested_level=match params.get("level") {Some(value)=>match value.parse::<u8>() {Ok(level) if level<=5=>level,_=>return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid hierarchy level")},None=>5};
                    let allowed_level=if show_all {5} else {4};
                    let level=requested_level.min(allowed_level);
                    let main_overlay=params.get("overlay").and_then(|v|v.parse::<u64>().ok()).and_then(|id|overlays.get(&id));
                    let nested_overlay=params.get("nested").and_then(|v|v.parse::<u64>().ok()).and_then(|id|overlays.get(&id));
                    let in_spider=|i:usize|main_overlay.is_some_and(|o|o.selected==i||o.ids.contains(&i))||nested_overlay.is_some_and(|o|o.selected==i||o.ids.contains(&i));
                    let spider_only=params.get("spider_only")==Some(&"1");
                    let show_link=|i:usize| spatial.roads[i] && spatial.levels[i]<=allowed_level && (in_spider(i)||spatial.levels[i]<=level) && (!spider_only||in_spider(i));
                    let candidates_in_view=|bounds:[f64;4]| {
                        let mut result=if spider_only {Vec::new()} else {spatial.candidates(bounds,level)};
                        // Spider members bypass background zoom ranks, without scanning minor roads.
                        for overlay in main_overlay.into_iter().chain(nested_overlay) {
                            for i in overlay.ids.iter().copied().chain(std::iter::once(overlay.selected)) {
                                if !show_link(i) {continue;}
                                let b=spatial.boxes[i];
                                if b[0]<=bounds[2]&&b[2]>=bounds[0]&&b[1]<=bounds[3]&&b[3]>=bounds[1] {result.push(i);}
                            }
                        }
                        result.retain(|i|show_link(*i));result.sort_unstable();result.dedup();result
                    };
                    let number = |name: &str| params.get(name).and_then(|s| s.parse::<f64>().ok()).filter(|n| n.is_finite());
                    let result = if url.starts_with("/api/link?") {
                        let id = params.get("id").map(|id| decode_query(id));
                        let i = params.get("index").and_then(|s| s.parse::<usize>().ok()).filter(|i| *i < links.len())
                            .or_else(|| id.as_ref().and_then(|id| ids.get(id.as_str()).copied()));
                        match i {
                            Some(i) if show_link(i) => serde_json::to_value(display_link(&links[i],i,0.0))?,
                            Some(_) => {return respond(&mut stream,"400 Bad Request","text/plain",b"Link hidden by road filter; enable Show all car/truck links to include minor roads");}
                            None => { respond(&mut stream,"404 Not Found","text/plain",b"Link ID not found")?; return Ok(()); }
                        }
                    } else if url.starts_with("/api/pick?") {
                        let (Some(x),Some(y),Some(radius)) = (number("x"),number("y"),number("radius")) else {
                            return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid pick coordinates");
                        };
                        if radius <= 0.0 || radius > 100000.0 { return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid pick radius"); }
                        let mut candidates: Vec<_> = candidates_in_view([x-radius,y-radius,x+radius,y+radius]).into_iter().filter_map(|i| {
                            if !show_link(i) {return None;}
                            let distance = link_distance(&links[i],x,y);
                            (distance <= radius).then_some((distance,i))
                        }).collect();
                        let focused=params.get("overlay").and_then(|v|v.parse::<u64>().ok()).and_then(|id|overlays.get(&id));
                        candidates.sort_by(|a,b| focused.is_some_and(|o|o.ids.contains(&b.1)).cmp(&focused.is_some_and(|o|o.ids.contains(&a.1))).then(a.0.total_cmp(&b.0)).then(a.1.cmp(&b.1)));
                        let total = candidates.len();
                        let choices: Vec<_> = candidates.into_iter().take(60).map(|(_,i)| display_link(&links[i],i,0.0)).collect();
                        serde_json::json!({"links":choices,"total":total})
                    } else {
                        let (Some(x0),Some(y0),Some(x1),Some(y1)) = (number("x0"),number("y0"),number("x1"),number("y1")) else {
                            return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid viewport bounds");
                        };
                        if x0 >= x1 || y0 >= y1 { return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid viewport bounds"); }
                        let candidates=candidates_in_view([x0,y0,x1,y1]);
                        // Reduce density by whole highway classes, never by individual-link sampling.
                        let mut counts=[0usize;6];for &i in &candidates {if !in_spider(i) {counts[spatial.levels[i] as usize]+=1;}}
                        let mut cumulative=0usize;for value in &mut counts {cumulative+=*value;*value=cumulative;}
                        let mut effective_level=level;
                        for rank in (0..=level).rev() {if counts[rank as usize]>0 {effective_level=rank;if counts[rank as usize]<=40000 {break;}}}
                        let tolerance=number("tolerance").unwrap_or(0.0).clamp(0.0,10000.0);
                        let visible:Vec<_>=candidates.par_iter().filter(|&&i|in_spider(i)||spatial.levels[i]<=effective_level).map(|&i|display_link(&links[i],i,if in_spider(i) {0.0} else {tolerance})).collect();
                        let response=ViewportResponse {links:visible,total:candidates.len(),truncated:false,requested_level,effective_level,coarsened:effective_level<level,spider_only};
                        return respond(&mut stream,"200 OK","application/json",&serde_json::to_vec(&response)?);
                    };
                    respond(&mut stream,"200 OK","application/json",&serde_json::to_vec(&result)?)
                }
                _ if url.starts_with("/api/flow?") => {
                    let params:HashMap<_,_>=url.split_once('?').unwrap().1.split('&').filter_map(|p|p.split_once('=')).collect();
                    let i=params.get("link").and_then(|v|v.parse::<usize>().ok()).filter(|i|*i<links.len());
                    let overlay=params.get("overlay").and_then(|v|v.parse::<u64>().ok()).and_then(|token|overlays.get(&token));
                    let (Some(i),Some(overlay))=(i,overlay) else {return respond(&mut stream,"400 Bad Request","text/plain",b"Unknown link or expired spider selection; select the target again");};
                    // Count each trip once, with the same modes/groups. All-time link traffic
                    // makes the denominator a true superset of target-window matching trips.
                    let link_trips=inverted[i].par_iter().filter(|&&t|overlay.groups&(1<<trips[t].group)!=0 && matches_traffic(&trips[t],&overlay.mode,overlay.road_traffic)).count() as u64;
                    let (upstream,downstream)=overlay.counts.get(&i).copied().unwrap_or((0,0));
                    let percentage=|part:u64,total:u64|if total>0 {100.0*part as f64/total as f64} else {0.0};
                    let value=serde_json::json!({"link":i,"target":i==overlay.selected,"upstream":upstream,"downstream":downstream,"link_trips":link_trips,"target_trips":overlay.trips,"upstream_percent_of_target":percentage(upstream,overlay.trips),"downstream_percent_of_target":percentage(downstream,overlay.trips),"upstream_percent_of_link":percentage(upstream,link_trips),"downstream_percent_of_link":percentage(downstream,link_trips),"link_time_scope":"all_recorded_times"});
                    respond(&mut stream,"200 OK","application/json",&serde_json::to_vec(&value)?)
                }
                _ if url.starts_with("/api/spider?") => {
                    let params: HashMap<_,_> = url.split_once('?').unwrap().1.split('&').filter_map(|p| p.split_once('=')).collect();
                    let groups=match parse_groups(&params) {
                        Ok(groups)=>groups,
                        Err(error)=>return respond(&mut stream,"400 Bad Request","text/plain",error.to_string().as_bytes()),
                    };
                    let road_traffic=params.get("road_traffic")==Some(&"1");
                    let selected = params.get("link").and_then(|p| p.parse::<usize>().ok());
                    let start = params.get("start").and_then(|p| p.parse::<f64>().ok()).unwrap_or(0.0);
                    let end = params.get("end").and_then(|p| p.parse::<f64>().ok()).unwrap_or(f64::MAX);
                    let mode_index = params.get("mode").and_then(|p| p.parse::<usize>().ok());
                    if selected.is_none() || selected.unwrap() >= links.len() || !start.is_finite() || !end.is_finite() || start >= end || mode_index.is_some_and(|i| i >= modes.len()) {
                        return respond(&mut stream, "400 Bad Request", "text/plain", b"Invalid link, time range, or mode");
                    }
                    let i = selected.unwrap();
                    let mode = mode_index.map(|i| modes[i].as_str()).unwrap_or("");
                    let via=if let Some(value)=params.get("via") {
                        let link=match value.parse::<usize>() {Ok(link) if link<links.len()&&link!=i=>link,_=>return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid nested via link")};
                        let side=match params.get("via_side").copied().unwrap_or("either") {"upstream"=>ViaSide::Upstream,"downstream"=>ViaSide::Downstream,"either"=>ViaSide::Either,_=>return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid via direction")};
                        Some(Via {link:link as u32,side})
                    } else {None};
                    let anchor_via=match params.get("anchor").copied().unwrap_or("target") {
                        "target"=>false,"via" if via.is_some()=>true,
                        _=>return respond(&mut stream,"400 Bad Request","text/plain",b"Invalid trace anchor"),
                    };
                    let mut result = query(&trips, &inverted[i], i as u32, start, end, mode, groups, road_traffic, via, anchor_via);
                    let overlay_counter=overlay_counter.fetch_add(1,Ordering::Relaxed)+1;
                    let counts:HashMap<_,_>=result["flows"].as_array().unwrap().iter().map(|f|(f["link"].as_u64().unwrap() as usize,(f["upstream"].as_u64().unwrap(),f["downstream"].as_u64().unwrap()))).collect();
                    let mut overlays=stored_overlays.write().map_err(|_|anyhow::anyhow!("Overlay lock poisoned"))?;
                    overlays.insert(overlay_counter,Arc::new(Overlay {ids:counts.keys().copied().collect(),counts,selected:result["selected"].as_u64().unwrap() as usize,trips:result["trips"].as_u64().unwrap(),groups,mode:mode.to_string(),road_traffic}));
                    if overlays.len()>8 {let parent=params.get("parent_overlay").and_then(|value|value.parse::<u64>().ok());if let Some(oldest)=overlays.keys().filter(|id|Some(**id)!=parent).min().copied() {overlays.remove(&oldest);}}
                    drop(overlays);
                    result["overlay_token"]=serde_json::json!(overlay_counter);
                    respond(&mut stream, "200 OK", "application/json", &serde_json::to_vec(&result)?)
                }
                _ => respond(&mut stream, "404 Not Found", "text/plain", b"Not found")
            }
    };
    let (sender,receiver)=mpsc::sync_channel::<TcpStream>(http_workers(budget)*4);let receiver=Mutex::new(receiver);
    thread::scope(|scope|->Result<()> {
        for _ in 0..http_workers(budget) {let handle=&handle;let receiver=&receiver;scope.spawn(move||loop {let incoming=receiver.lock().unwrap().recv();match incoming {Ok(stream)=>{if let Err(error)=handle(stream) {let cancelled=error.chain().filter_map(|e|e.downcast_ref::<std::io::Error>()).any(|e|matches!(e.kind(),std::io::ErrorKind::BrokenPipe|std::io::ErrorKind::ConnectionReset|std::io::ErrorKind::ConnectionAborted|std::io::ErrorKind::TimedOut|std::io::ErrorKind::WouldBlock));if !cancelled {eprintln!("Request failed: {error:#}");}}},Err(_)=>break}});}
        for stream in listener.incoming() {if sender.send(stream?).is_err() {break;}}
        drop(sender);Ok(())
    })
}
fn align_cached_endpoints(source:&str,dir:&Path)->Result<()> {
    let endpoints:HashMap<_,_>=network(source,"-")?.into_iter().map(|link|(link.id,link.points)).collect();
    let mut links:Vec<Link>=serde_json::from_reader(BufReader::new(File::open(dir.join("network.json"))?))?;
    let mut changed=0;
    for link in &mut links {
        if let Some(nodes)=endpoints.get(&link.id) {
            if link.points.len()<2 {continue;}
            let last=link.points.len()-1;
            for (index,node) in [(0,nodes[0]),(last,nodes[1])] {
                let p=link.points[index];
                if (p[0]-node[0]).hypot(p[1]-node[1])<=5.0 && p!=node {link.points[index]=node;changed+=1;}
            }
        }
    }
    if changed>0 {json_write(&dir.join("network.json"),&links)?;}
    eprintln!("Aligned {changed} cached geometry endpoints to MATSim nodes (events untouched)");
    Ok(())
}
fn http_workers(budget:usize)->usize {(budget/2).clamp(1,2)}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let threads: usize = args.last().context("Missing arguments")?.parse()?;
    let compute=if args.get(1).map(String::as_str)==Some("serve") {threads.saturating_sub(http_workers(threads)).max(1)} else {threads.max(1)};
    rayon::ThreadPoolBuilder::new().num_threads(compute).build_global()?;
    match args.get(1).map(String::as_str) {
        Some("prepare") if args.len() == 7 || args.len() == 8 => {
            let links = network(&args[2], &args[4])?;
            let people = population(if args.len() == 8 { &args[6] } else { "-" })?;
            let trips = events(&args[3], &links, &people)?;
            let dir = Path::new(&args[5]);
            json_write(&dir.join("network.json"), &links)?;
            json_write(&dir.join("trips.json"), &trips)?;
            let highways:HashMap<_,_>=links.iter().map(|link|(&link.id,&link.highway)).collect();
            json_write(&dir.join("highways.json"), &highways)?;
            let levels:HashMap<_,_>=links.iter().map(|link|(&link.id,hierarchy_level(&link.highway))).collect();
            json_write(&dir.join("hierarchy.json"),&levels)?;
            Ok(())
        }
        Some("align-endpoints") if args.len()==5 => align_cached_endpoints(&args[2],Path::new(&args[3])),
        Some("highways") if args.len()==5 => {let highways=read_highways(&args[2])?;let dir=Path::new(&args[3]);json_write(&dir.join("highways.json"),&highways)?;write_hierarchy(dir,&highways)},
        Some("hierarchy") if args.len()==4 => {let dir=Path::new(&args[2]);let highways:HashMap<String,String>=serde_json::from_reader(BufReader::new(File::open(dir.join("highways.json"))?))?;write_hierarchy(dir,&highways)},
        Some("serve") if args.len() == 5 => serve(Path::new(&args[2]), &args[3], threads),
        _ => bail!("Usage: eqasim-spyder prepare NETWORK EVENTS GEOMETRY WORK [PERSONS_CSV_OR_POPULATION_XML] THREADS | serve WORK ADDRESS THREADS")
    }
}
