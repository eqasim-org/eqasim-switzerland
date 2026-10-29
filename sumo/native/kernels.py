"""ctypes batches release the GIL during native route arithmetic."""

import ctypes as ct

import numpy as np

from sumo.scenario.network.tools import position

I = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')


class Fragment(ct.Structure):
    _fields_ = [('piece', ct.c_int64), ('visit', ct.c_int64), *[(name, ct.c_double) for name in
        ('begin', 'end', 'depart_pos', 'arrival_pos', 'distance')], *[(name, ct.c_int64) for name in
        ('new_entry', 'new_exit', 'interpolated')]]


class RouteKernel:
    def __init__(self, library, mapping=None, network=None, transitions=None):
        self.library = ct.CDLL(str(library))
        self.library.edge_clocks.argtypes = [ct.c_int64,D,D,ct.c_double,ct.c_double,ct.c_double,D,ct.POINTER(ct.c_double)]
        self.library.edge_clocks.restype = ct.c_int
        self.library.cut_route.argtypes = [ct.c_int64,I,D,ct.c_double,ct.c_double,ct.c_double,D,I,D,D,D,I,I,ct.c_int64,ct.POINTER(Fragment)]
        self.library.cut_route.restype = ct.c_int64
        if mapping is None:
            return
        self.ids = {key:index for index,key in enumerate(network['edges'])}
        self.lengths = np.array([edge['length'] for edge in network['edges'].values()],dtype=np.float64)
        self.pieces, offsets = [], [0]
        for edge in self.ids:
            self.pieces.extend(mapping.get(edge, [])); offsets.append(len(self.pieces))
        if not np.isfinite(self.lengths).all() or (self.lengths<=0).any():
            raise ValueError('Native cutter requires finite positive edge lengths')
        if any(not (0<=p['start']<p['end']<=1) or not np.isfinite(p['length']) or p['length']<=0 for p in self.pieces):
            raise ValueError('Invalid regional edge fragment mapping')
        self.offsets = np.array(offsets,dtype=np.int64)
        self.starts,self.ends,self.piece_lengths = [np.array([p[name] for p in self.pieces],dtype=np.float64)
            for name in ('start','end','length')]
        identifiers = {}
        for index,piece in enumerate(self.pieces):
            identifiers.setdefault(piece['id'],[]).append(index)
        pairs = sorted((a,b) for source,target in transitions for a in identifiers.get(source,[]) for b in identifiers.get(target,[]))
        self.allowed_from = np.array([a for a,b in pairs],dtype=np.int64)
        self.allowed_to = np.array([b for a,b in pairs],dtype=np.int64)

    def clocks(self, lengths, costs, depart, start, stop):
        lengths,costs = np.ascontiguousarray(lengths,dtype=np.float64),np.ascontiguousarray(costs,dtype=np.float64)
        if lengths.ndim!=1 or costs.ndim!=1 or not len(lengths) or len(lengths)!=len(costs):
            raise ValueError('Loaded clock arrays must be nonempty, one-dimensional and aligned')
        if not np.isfinite([depart,start,stop]).all() or depart<0 or start<0 or stop<0 or start>lengths[0] or stop>lengths[-1]:
            raise ValueError('Invalid loaded route departure or endpoint positions')
        exits=np.empty(len(lengths)); distance=ct.c_double()
        if self.library.edge_clocks(len(lengths),lengths,costs,depart,start,stop,exits,ct.byref(distance)):
            raise ValueError('Invalid loaded route lengths, costs or endpoint positions')
        return exits.tolist(),distance.value

    def visits(self, vehicle):
        attrs=vehicle['attributes']
        edges=np.fromiter((self.ids[key] for key in vehicle['edges']),dtype=np.int64)
        exits=np.asarray(vehicle['exit_times'],dtype=np.float64)
        if not len(edges) or len(exits)!=len(edges) or not np.isfinite(exits).all():
            raise ValueError('Missing or invalid route edge exit times')
        first=position(attrs.get('departPos','0'),self.lengths[edges[0]])
        last=position(attrs.get('arrivalPos','max'),self.lengths[edges[-1]])
        capacity=int(np.sum(self.offsets[edges+1]-self.offsets[edges]))
        output=(Fragment*capacity)()
        count=self.library.cut_route(len(edges),edges,exits,float(attrs['depart']),first,last,
            self.lengths,self.offsets,self.starts,self.ends,self.piece_lengths,self.allowed_from,self.allowed_to,
            len(self.allowed_from),output)
        if count<0:
            raise ValueError(f"Invalid endpoint positions for vehicle {attrs['id']}")
        visits=[]
        previous=None
        for index in range(count):
            part=output[index]
            first=len(visits)<=part.visit
            if first:
                if previous is not None:
                    visits[-1][-1].update(arrival_pos=previous.arrival_pos,new_exit=previous.new_exit)
                visits.append([])
            # Interior fragments only need edge identity, exit time and aggregate fields.
            item={'id':self.pieces[part.piece]['id'],'end':part.end,
                  'distance':part.distance,'interpolated':part.interpolated}
            if first:
                item.update(begin=part.begin,depart_pos=part.depart_pos,new_entry=part.new_entry)
            visits[-1].append(item)
            previous=part
        if previous is not None:
            visits[-1][-1].update(arrival_pos=previous.arrival_pos,new_exit=previous.new_exit)
        return visits
