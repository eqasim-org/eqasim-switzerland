"""Bounded disk-backed metadata and route-share allocation for national assignments."""

from array import array
from collections import OrderedDict
import hashlib
import json
import logging
import math
import random
import sqlite3
import time

from sumo.scenario.tools import elements

logger = logging.getLogger('synpp')
BATCH = 10000
EXCLUDED = {'from','to','via','fromLonLat','toLonLat','viaLonLat','fromXY','toXY',
            'fromJunction','toJunction','fromTaz','toTaz'}


class AssignmentSpool:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('PRAGMA cache_size=-131072')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.executescript('''
            CREATE TABLE sources(id TEXT PRIMARY KEY, attrs TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE groups(gid INTEGER PRIMARY KEY, signature BLOB UNIQUE, choices TEXT NOT NULL);
            CREATE TABLE members(seq INTEGER PRIMARY KEY, gid INTEGER NOT NULL, depart REAL NOT NULL,
                                 id TEXT UNIQUE, attrs TEXT NOT NULL, choice INTEGER NOT NULL DEFAULT 0);
        ''')
        self.count = 0
        self.phase = 'indexing source metadata'
        self.last_heartbeat = time.monotonic()
        self.db.set_progress_handler(self._heartbeat, 1000000)

    def _heartbeat(self):
        now=time.monotonic()
        if now-self.last_heartbeat>=30:
            logger.info('SUMO realization: %s; %d assigned vehicles indexed',self.phase,self.count)
            self.last_heartbeat=now
        return 0

    def close(self):
        self.db.close()

    def read_sources(self, path):
        types = {}; rows = []; count = 0; started = time.monotonic()
        from lxml import etree
        for element in elements(path, ('trip','vType')):
            if element.tag == 'vType':
                types[element.get('id')] = etree.fromstring(etree.tostring(element))
                continue
            rows.append((element.get('id'), json.dumps({k:v for k,v in element.attrib.items() if k not in EXCLUDED})))
            count += 1
            if len(rows) >= BATCH:
                self.db.executemany('INSERT INTO sources VALUES (?,?)', rows); rows.clear()
                if count % 100000 == 0: logger.info('SUMO realization: indexed %d source trips; %.1fs', count, time.monotonic()-started)
        if rows: self.db.executemany('INSERT INTO sources VALUES (?,?)', rows)
        self.db.commit()
        logger.info('SUMO realization: source indexing complete: %d trips; %.1fs', count, time.monotonic()-started)
        return types

    def read_assignments(self, path, begin, interval):
        self.phase = 'indexing assignments'
        pending = []; cache = OrderedDict(); started = time.monotonic()
        for element in elements(path, 'vehicle'):
            attrs = dict(element.attrib)
            choices = []
            for route in element.findall('routeDistribution/route') or element.findall('route'):
                probability = float(route.get('probability','1'))
                edges = tuple(route.get('edges','').split())
                if not math.isfinite(probability) or probability < 0 or not edges:
                    raise ValueError(f"Invalid assigned route distribution for {attrs['id']}")
                if probability: choices.append((edges,probability))
            if not choices: raise ValueError(f"No positive assigned route probability for {attrs['id']}")
            pending.append((attrs,choices))
            if len(pending) >= BATCH:
                self._insert(pending, cache, begin, interval); pending.clear()
                if self.count % 100000 == 0: logger.info('SUMO realization: indexed %d assigned vehicles; %.1fs',self.count,time.monotonic()-started)
        if pending: self._insert(pending,cache,begin,interval)
        self.db.commit()
        if not self.count: raise ValueError('Macroscopic assignment produced no individual vehicles')
        logger.info('SUMO realization: indexing group members for %d vehicles',self.count)
        self.phase = 'building group index on disk'
        self.db.execute('CREATE INDEX member_groups ON members(gid,seq)');self.db.commit()
        logger.info('SUMO realization: assignment indexing complete; %.1fs',time.monotonic()-started)

    def _insert(self, pending, cache, begin, interval):
        # Resolve a batch of source IDs, including native scale suffixes, without a national Python dictionary.
        ids = set()
        for attrs,_ in pending:
            key=attrs['id'];ids.add(key)
            parent,separator,clone=key.rpartition('.')
            if separator and clone.isdigit():ids.add(parent)
        sources={}; keys=list(ids)
        for start in range(0,len(keys),400):
            part=keys[start:start+400]
            sources.update(self.db.execute('SELECT id,attrs FROM sources WHERE id IN ('+','.join('?' for _ in part)+')',part))
        rows=[]
        for attrs,choices in pending:
            key=attrs['id']; parent=key
            if key not in sources:
                parent,separator,clone=key.rpartition('.')
                if not separator or not clone.isdigit(): parent=key
            if parent not in sources: raise ValueError(f'Cannot match loaded vehicle {key} to its source trip')
            attrs={**json.loads(sources[parent]),**attrs}
            depart=float(attrs['depart'])
            if not math.isfinite(depart) or depart<0: raise ValueError(f'Invalid assigned departure for {key}')
            payload=json.dumps(choices,separators=(',',':'))
            signature=hashlib.sha256(json.dumps([math.floor((depart-begin)/interval),attrs.get('type'),choices],separators=(',',':')).encode()).digest()
            gid=cache.get(signature)
            if gid is None:
                self.db.execute('INSERT OR IGNORE INTO groups(signature,choices) VALUES (?,?)',(signature,payload))
                gid=self.db.execute('SELECT gid FROM groups WHERE signature=?',(signature,)).fetchone()[0]
                cache[signature]=gid
                if len(cache)>512:cache.popitem(last=False)
            self.count+=1
            rows.append((self.count,gid,depart,key,json.dumps(attrs)))
        self.db.executemany('INSERT INTO members(seq,gid,depart,id,attrs) VALUES (?,?,?,?,?)',rows)
        self.db.commit()

    def allocate(self, seed):
        self.phase = 'allocating route shares'
        rng=random.Random(seed); completed=0; started=time.monotonic(); last_log=started
        group_count=self.db.execute('SELECT count(*) FROM groups').fetchone()[0]
        for gid,payload in self.db.execute('SELECT gid,choices FROM groups ORDER BY gid'):
            choices=json.loads(payload)
            members=array('q',(row[0] for row in self.db.execute('SELECT seq FROM members WHERE gid=? ORDER BY seq',(gid,))))
            total=math.fsum(p for _,p in choices)
            targets=[len(members)*p/total for _,p in choices];counts=[math.floor(t) for t in targets]
            ranking=sorted(range(len(choices)),key=lambda i:(targets[i]-counts[i],rng.random()),reverse=True)
            for i in ranking[:len(members)-sum(counts)]:counts[i]+=1
            rng.shuffle(members)
            offset=0; updates=[]
            for choice,count in enumerate(counts):
                if choice == 0:
                    offset += count
                    continue
                for index in range(offset,offset+count):
                    updates.append((choice,members[index]))
                    if len(updates)>=BATCH:
                        self.db.executemany('UPDATE members SET choice=? WHERE seq=?',updates);updates.clear()
                offset+=count
            if updates:self.db.executemany('UPDATE members SET choice=? WHERE seq=?',updates)
            completed+=1
            if time.monotonic()-last_log>=30:
                self.db.commit();logger.info('SUMO realization: allocated %d/%d distributions; %.1fs',completed,group_count,time.monotonic()-started);last_log=time.monotonic()
        self.db.commit()
        logger.info('SUMO realization: sorting departure index on disk for %d vehicles',self.count)
        self.phase = 'building departure index on disk'
        self.db.execute('CREATE INDEX departure_order ON members(depart,id)');self.db.commit()
        logger.info('SUMO realization: allocated %d distributions; %.1fs',group_count,time.monotonic()-started)
        return group_count

    def selected(self):
        self.phase = 'streaming selected routes'
        cached=OrderedDict()
        for gid,choice,attrs in self.db.execute('SELECT gid,choice,attrs FROM members ORDER BY depart,id'):
            choices=cached.get(gid)
            if choices is None:
                choices=json.loads(self.db.execute('SELECT choices FROM groups WHERE gid=?',(gid,)).fetchone()[0]);cached[gid]=choices
                if len(cached)>128:cached.popitem(last=False)
            yield json.loads(attrs),choices[choice][0]
