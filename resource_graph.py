from collections import defaultdict,deque
from copy import deepcopy
import numpy as np
import random
import pickle

def invert_simple_map(x):
    assert type(x) in {dict,defaultdict}
    x2 = defaultdict(x.default_factory)

    for (k,v) in x.items():
        x2[v] = k 
    return x2 

def pairseq_to_dict(ps):
    d = defaultdict(None)
    for p in ps: d[p[0]] = p[1]
    return d

def mean_safe_division(seq):
    if len(seq) == 0: return 0.
    return sum(seq) / len(seq)

def is_valid_range(r,r2=None):
    s1 = len(r) == 2
    s2 = r[0] <= r[1]
    s3 = True

    if type(r2) != type(None):
        assert is_valid_range(r2)
        s3 = r[0] >= r2[0] and r[1] <= r2[1]
    return s1 and s2 and s3 

def default_rg_value_assignment(rg,scale_range):
    sort_function = sort_cst1
    map_function = node_health_score_assignment_type_1
    edge_function = es_func_1
    fs(rg,sort_function,map_function,scale_range,edge_function)
    return rg 

########################################## metrics used to assemble a graph by subgraph composition
# max function: positive and negative v/e score
def ve_fitscore_type1(G,G_):
    assert type(G) == type(G_)
    assert type(G) == MicroGraph

    # positive ve score
    vs1 = G.sub_ve_score(G_)

    # negative ve score
    vs2 = G_.sub_ve_score(G) 
    return (vs1[0] + vs1[1]) + (vs2[0] + vs2[1])

# max function: negative v/e score 
def ve_fitscore_type2(G,G_):
    # negative ve score
    vs = G.sub_ve_score(G_) 
    return (vs[0] + vs[1])

######################################### function used to assign health scores to vertices and edges

# NOTE: function used to make instances of `value_gen` in <ResourceGraph.set_ve_health_values>.
def fs(rg,sort_function,map_function,scale_range,edge_function,
       additional_args__sf=(),additional_args__mf=(),additional_args__ef=()):
    mg = MicroGraph.from_ResourceGraph(rg) 
    
    # assign the health scores for the nodes
    svs = sort_function(mg,*additional_args__sf)
    if len(svs) == 0:
        return
    ##print("SORTED LEN: ",len(svs))
    
    ranj = [svs[0][1],svs[-1][1]]
    ranj = sorted(ranj)

    for q in svs:
        ##print("[0] node {} score {}".format(q[0],q[1]))
        q[1] = map_function(q[1],ranj,scale_range,*additional_args__mf)
        ##print("before:\t",q[0], rg.node_health_map[q[0]])
        rg.node_health_map[q[0]] = q[1]
        ##print("after:\t",q[0], rg.node_health_map[q[0]])

        ##print("[1] node {} score {}".format(q[0],q[1]))


    # assign the health scores for the edges
    for k in rg.edges_health_map.keys():
        s = k.split(",")
        assert len(s) == 2
        qe = [rg.node_health_map[s[0]],rg.node_health_map[s[1]]]
        rg.edges_health_map[k]  = edge_function(qe,*additional_args__ef)
    
    return

def node_health_score_assignment_type_1(node_score,r1,r2):
    assert type(node_score) in [int,float] 
    return scale_x_to_x2(node_score,r1,r2)

# sorts vertices by connectivity_score_type1
#
# ordring := +, ascending order
#            -, descending order 
#            +, ascending distance from mean
#            -, descending distance from mean
def sort_cst1(mg,vertex_distance=1,ordring="+"):
    assert ordring in ["+","-","0+","0-"], "invalid ordering scheme"

    q = []
    for k,v in mg.dg.items(): 
        q.append([k,connectivity_score_type1(k,mg,vertex_distance)])

    if ordring == "+":
        return sorted(q,key=lambda x: x[1])
    elif ordring == "-":
        return sorted(q,key= lambda x:x[1])[::-1]
    else:
        # get the average of the numbers
        m = [q_[1] for q_ in q]
        m_ = np.mean(np.array(m))
        for i in range(len(q)):
            q[i] = (q[i][0],abs(m_ - q[i][1]))

        q = sorted(q,key=lambda x: x[1])

        if ordring == "0-":
            return q[::-1]
        return q

"""
capture the cumulative connectivity of a node's neighbors 
at the n'th distance, typically 1.
"""
def connectivity_score_type1(n_idn,mg,vertex_distance=1):
    connectivities = defaultdict(int) 
    accessed = deque([(n_idn,0)])

    while len(accessed) > 0:
        ax = accessed.popleft()

        if ax[1] == vertex_distance:
            connectivities[ax[0]] = len(mg.dg[ax[0]])
        else:
            q = [(x,ax[1] + 1) for x in mg.dg[ax[0]]]
            accessed.extend(q)

    return sum(list(connectivities.values()))

    ###### START: edge-score assignments

# average
def es_func_1(node_health_pair):
    assert len(node_health_pair) == 2
    return (node_health_pair[0] + node_health_pair[1]) / 2.0

# average * multiple
def es_func_2(node_health_pair,multiple=2.0):
    return es_func_1(node_health_pair) * multiple

    ###### END: edge-score assignments
    
def scale_x_to_x2(x,r,r2):
    assert len(r) == 2
    assert len(r) == len(r2)
    assert x >= r[0] and x <= r[1]
    assert r2[1] >= r2[0]
    ##print("R: ", r, "  R2: ",r2) 
    # zero-length range case
    if r[1] == r[0]:
        return r2[1]
     
    return r2[0] + (x - r[0]) / float(r[1] - r[0]) * (r2[1] - r2[0]) 

    ############## functions for default score assignments to ResourceGraph's nodes and edges

def rg_score_assignment_type1(rg,scale_range=[10,200]):
    assert type(rg) == ResourceGraph
    sort_function = sort_cst1
    map_function = node_health_score_assignment_type_1
    edge_function = es_func_1
    fs(rg,sort_function,map_function,scale_range,edge_function)
    return rg

def rg_score_assignment_type_uniform(rg,default_health=1):

    for k in rg.node_health_map.keys():
        rg.node_health_map[k] = default_health
    
    for k in rg.edges_health_map.keys():
        rg.edges_health_map[k] = default_health

    return rg 

"""
default function used to calculate a d-network from the
input rg



"""
def rg_score_assignment_type_deception__default(rg_ref,rg_dec):

    # assign rg_dec node health values
    for k in rg_dec.node_health_map.keys():
        rg_dec.node_health_map[k] = rg_ref.node_health_map[k]

    # get the mean edge-health for each node
    meh = defaultdict(float)
    for k in rg_ref.node_health_map.keys():
        meh[k] = rg_ref.node_mean_eh(k)

    # iterate through the edges of rg_dec and modify
    mg = MicroGraph.from_ResourceGraph(rg_dec) 
    for k in rg_dec.edges_health_map.keys():
        if k in rg_ref.edges_health_map:
            rg_dec.edges_health_map[k] = rg_ref.edges_health_map[k]
        else:
            vx = k.split(",")
            val = (meh[vx[0]] + meh[vx[1]]) / 2.
            rg_dec.edges_health_map[k] = val
    return rg_dec

# class used to iterate node identifier in variable graph.
class DefaultNodeIdnCounter:

    def __init__(self,nidn):
        try:
            int(nidn)
        except:
            raise ValueError
        self.nidn = nidn
    
    def __next__(self):
        self.nidn = str(int(self.nidn) + 1)
        return self.nidn


"""
graph designed for small-scale use (<= 5000 nodes)
"""
class MicroGraph:

    def __init__(self,dgraph=defaultdict(set)):
        assert type(dgraph) == defaultdict
        self.dg = dgraph
        return

    def __str__(self):
        return str(self.dg) 

    def subgraph_nodeset_exclusion(self,ns):
        dg2 = deepcopy(self.dg)
        for (k,v) in dg2.items():
            if k in ns:
                del self.dg[k]
            else:
                v_ = set([s for s in v if s not in ns])
                self.dg[k] = v

    def subgraph_by_nodeset_(self,ns):
        mg2 = deepcopy(self)
        q = set(mg2.dg.keys()) - ns
        mg2.subgraph_nodeset_exclusion(q) 
        return mg2

    """
    outputs the MicroGraph of minimal v,e-score based on the variables
    `wanted_nodes` and `wanted_edges`. 
    """
    @staticmethod
    def minimal_MG_by_nodes_and_edges(wanted_nodes,wanted_edges):
        dg = defaultdict(set)

        for x in wanted_nodes:
            dg[x] = set()

        for x in wanted_edges:
            q = wanted_edges.split(",")
            assert len(q) == 2
            dg[q[0]] |= dg[q[1]]
        return MicroGraph(dg)

    @staticmethod
    def from_ResourceGraph(rg,directed=False):
        
        ##print("XXXX")
        mg = MicroGraph(defaultdict(set))
        for k in rg.node_health_map.keys():
            mg.dg[k] = set()
        
        for k in rg.edges_health_map.keys(): 
            s = k.split(",") 
            mg.dg[s[0]] |= {s[1]}
            if not directed:
                mg.dg[s[1]] |= {s[0]}
        return mg

    @staticmethod
    def isotransform_MG(mg,isomap):
        dg = defaultdict(set)

        for (k,v) in mg.dg.items():
            v2 = set()
            for v_ in v:
                assert v_ in isomap
                v2 = v2 | {isomap[v_]}
            assert k in isomap 
            dg[isomap[k]] = v2
        return MicroGraph(dg) 

    def ve_score(self):
        # count up the number of nodes
        # count up the number of unique edges
        ns = len(self.dg) 
        es = self.edge_count()
        return (ns,es)

    def __add__(self,mg):
        q = deepcopy(self.dg)
        for (k,v) in mg.dg.items():
            q[k] = q[k] | v
        return MicroGraph(q) 

    def __sub__(self,mg):
        # delete all edges
        mg1 = deepcopy(self)
        for (k,v) in mg.dg.items():
            mg1.dg[k] = mg1.dg[k] - v
        
        # delete all nodes
        for k in mg.dg.keys():
            del mg1.dg[k] 
        return mg1

    # caution: not tested
    def __eq__(self,mg):
        if len(self.dg) != len(mg.dg): 
            return False
        for (k,v) in self.dg.items():
            if mg.dg[k] != v: return False
        return True

    def edge_count(self):
        es = set()
        for (k,v) in self.dg.items():
            for v_ in v:
                l = sorted([k,v_])
                es |= {l[0]+","+l[1]}
        return len(es)

    # vertex-edge score of subtraction operation with mg
    def sub_ve_score(self,mg):
        # delete all edges of mg from self
        mg1 = deepcopy(self)

        ##print("to delete")
        ##print(mg.dg) 

        ##print("before deletion of nodes")
        ##print(mg1.dg) 

        for (k,v) in mg.dg.items():
            ##print("deleting edge {}-{}".format(k,v))  
            mg1.dg[k] = mg1.dg[k] - v 

        ##print("after deletion of nodes")
        ##print(mg1.dg) 

        # count the edges
        es = mg1.edge_count()
        ##print("edges: ", es) 

        # delete all nodes
        for k in mg.dg.keys():
            del mg1.dg[k]

        # count the nodes
        ns = len(mg1.dg)
        ##print("nodes: ", ns) 
        return (ns,es)

    """
    alternative subtraction scheme.

    return:
    - [0] node set of self not of mg 
    - [1] edge set of self not of mg
    """
    def alt_subtract(self,mg):
        # node set
        ns = set(self.dg.keys()) - set(mg.dg.keys())

        # edge set
        es = self.edge_set() - mg.edge_set()
        return ns,es 

    def neighbor_count(self):
        q = defaultdict(int)
        for (k,v) in self.dg.items():
            q[k] = len(v)
        return q

    def edge_set(self):
        es = set()
        for (k,v) in self.dg.items():
            for v_ in v:
                es = es | {k + "," + v_} 
        return es 

    ################# greatest common subgraph problem

    """
    calculates the sequence of components belonging to this
    MicroGraph.

    return: 
    - list<set<node identifier>>
    """
    def component_seq(self):
        remaining_nodeset = set(self.dg.keys())
        if len(remaining_nodeset) == 0:
            return []
        search_sol = []
        heads_and_component = [{remaining_nodeset.pop()},set()]

        while len(remaining_nodeset) > 0 or\
            len(heads_and_component[0]) > 0:

            if len(heads_and_component[0]) == 0:
                heads_and_component[0] = {remaining_nodeset.pop()}

            # pop one head and gather its neighbors
            nd = heads_and_component[0].pop()
            heads_and_component[1] |= {nd} 
            q = deepcopy(self.dg[nd])
            heads_and_component[1] = heads_and_component[1] | q
            q2 = set([q_ for q_ in q if q_ in remaining_nodeset])
            remaining_nodeset = remaining_nodeset - q2 - {nd}
            heads_and_component[0] = heads_and_component[0] | q2 

            if len(heads_and_component[0]) == 0:
                search_sol.append(deepcopy(heads_and_component[1]))
                heads_and_component[1].clear()
                if len(remaining_nodeset) == 0: continue 
                heads_and_component[0] = heads_and_component[0] | {remaining_nodeset.pop()}
        return search_sol

class ResourceGraph:

    def __init__(self,nhm={},ehm ={},public_health_stat=False):
        self.node_health_map = nhm
        self.edges_health_map = ehm
        self.phs = public_health_stat
        self.nic = DefaultNodeIdnCounter("0")
        ##self.neg_chips = []

    def __str__(self):
        s = "\t* nodes\n"
        for (k,v) in self.node_health_map.items():
            s += k + "|" + str(v) + "\n"
        s += "\n"
        s += "\t* edges\n"
        for (k,v) in self.edges_health_map.items():
            s += k + "|" + str(v) + "\n"
        s += "\n"
        return s 

    """
    i := int, deterministic seed identifier
    d := int, number of nodes for rg
    connectivity_range := range, values in [0,1]
    mpr := health range for nodes
    """
    @staticmethod
    def generate__type_stdrand(i:int,d,connectivity_range,hr):
        assert type(i) in {type(None),int}
        if type(i) == int:
            random.seed(i)
        """
        print("D: ",d)
        """
        assert type(d) == int
        assert is_valid_range(connectivity_range,[0.,1.]) 
        assert is_valid_range(hr)
        assert hr[0] > 0

        dg = defaultdict(set)
        for i in range(d):
            x = str(i)
            d_ = len(dg[x])
            r = float(d_) / d
            nd = random.uniform(r,connectivity_range[1])
            # get the maximum remaining number of edges to add
            # based on nd
            rem = round((nd - r) * d)

            # get the remaining candidates for edges
            rem_cand = [str(j) for j in range(i+1,d)]
            random.shuffle(rem_cand)
            for j in range(rem):
                if len(rem_cand) == 0: break 
                x2 = rem_cand.pop()
                dg[x] = dg[x] | {x2}
                dg[x2] = dg[x2] | {x}

        mg = MicroGraph(dg)
        rg = ResourceGraph.from_MicroGraph(mg)
        default_rg_value_assignment(rg,hr)
        return rg

    """
    sets the DefaultNodeIdnCounter to the highest-scoring vertex
    """
    def set_nic(self):
        if len(self.node_health_map) == 0:
            self.nic = DefaultNodeIdnCounter("0")
            return

        v = list(self.node_health_map.values()) 
        v = max([int(v_) for v_ in v])
        self.nic = DefaultNodeIdnCounter(str(v))
        return

    ############################ graph complement functions ################
    
    # TODO: test
    """
    used in the case of NegoChips, 
    """
    def deception_complement(self,deceptor_nodes):
        # construct a new MicroGraph
        mg_ref = MicroGraph.from_ResourceGraph(self)
        mgx = MicroGraph(defaultdict(set))
        ns = set(mg_ref.dg.keys())

        for (k,v) in mg_ref.dg.items():
            if k in deceptor_nodes:
                mgx.dg[k] = ns - deepcopy(v)
            else:
                mgx.dg[k] = deepcopy(v)
    
        rgx = ResourceGraph.from_MicroGraph(mgx)
        return rg_score_assignment_type_deception__default(self,rgx)

    def edge_complement(self,score_assignment = rg_score_assignment_type1):
        # convert to MicroGraph and get all the edges
        mg = MicroGraph.from_ResourceGraph(self)
        dg2 = defaultdict(set) 
        q = set(mg.dg.keys())

        for (k,v) in mg.dg.items():
            dg2[k] = q - v 
        mg2 = MicroGraph(dg2)

        # convert back to ResourceGraph and assign scores
        # for nodes and edges
        rg = ResourceGraph.from_MicroGraph(mg2)
        rg = score_assignment(rg)
        return rg

    def partial_edge_complement(self,nodeset,score_assignment=rg_score_assignment_type1):
        assert type(nodeset) == set
        mg = MicroGraph.from_ResourceGraph(self)
        mg2 = mg.subgraph_by_nodeset_(nodeset)

        rg2 = ResourceGraph.from_MicroGraph(mg2)
        rg2 = rg2.edge_complement(rg_score_assignment_type_uniform)
        mgx = MicroGraph.from_ResourceGraph(rg2)

        for (k,v) in mg.dg.items():
            for v_ in v:
                if v_ not in nodeset:
                    mgx.dg[k] = mgx.dg[k] | {v_}
        rgx = ResourceGraph.from_MicroGraph(mgx)
        if type(score_assignment) != type(None):
            rgx = score_assignment(rgx)
        return rgx

    ############################ translation functions ################

    # default value for node and edge health is 1
    @staticmethod
    def from_MicroGraph(mg,default_health=1):
        rg = ResourceGraph({},{})
        ##print("micrograph")
        ##print(mg.dg) 
        ##print("-- from micro")
        for (k,v) in mg.dg.items():
            rg.node_health_map[k] = default_health
            for v_ in v:
                es = k + "," + v_
                ##print("adding edge ",es)
                rg.edges_health_map[es] = default_health
        ##print("done adding edges...")
        return rg 

    # pickle method
    @staticmethod
    def write_resource_graph():
        return -1 

    @staticmethod
    def load_resource_graph():
        return -1 

    ############################ graph modification functions ################

    def add_node(self,node_info):
        assert len(node_info) == 2
        assert type(node_info[0]) is str
        assert type(node_info[1]) is float
        self.node_health_map[node_info[0]] = node_info[1]
        return

    def delete_node(self,node_idn):
        del self.node_health_map[node_idn]
        return

    def add_edge(self,edge_pair,edge_health):
        s = edge_pair.split(",")
        assert len(s) == 2
        self.edges_health_map[edge_pair] = edge_health
        return

    def delete_edge(self,edge_pair):
        s = edge_pair.split(",")
        assert len(s) == 2
        del self.edges_health_map[edge_pair] 
        return

    def update_node(self,node,delta):
        self.node_health_map[node] = self.node_health_map[node] + delta
        return

    def update_edge(self,edge_pair,delta):
        s = edge_pair.split(",")
        assert len(s) == 2
        self.edges_health_map[edge_pair] = self.edges_health_map[edge_pair] + delta
        return

    """
    cleans the instance of all nodes and edges of health <= 0.
    """
    def clean_graph(self):
        delnodes,deledges = set(),set()
        for (k,v) in self.node_health_map.items():
            if v <= 0.: delnodes |= {k}

        for (k,v) in self.edges_health_map.items():
            if v <= 0.: 
                deledges |= {k}
                continue
            q = k.split(",")
            stat = q[0] in delnodes or q[1] in delnodes
            if stat:
                deledges |= {k}

        for x in delnodes: self.delete_node(x)
        for x in deledges: self.delete_edge(x)
        return delnodes,deledges

    ############################## health value info. ################
    """
    """
    def update_health_values(self,node_dm,edge_dm):
        for (k,v) in node_dm.items():
            self.node_health_map[k] = self.node_health_map[k] + v

        for (k,v) in edge_dm.items():
            self.edges_health_map[k] = self.edges_health_map[k] + v
        return

    """
    average of edge health pertaining to node `n`
    """
    def node_mean_eh(self,n:str):
        c = 0.
        d = 0

        # iterate through the edges
        for (k,v) in self.edges_health_map.items():
            eds = k.split(",")
            stat = n in eds 

            if stat:
                c += v
                d += 1

        if d == 0:
            return 0.
        return c / d

    # TODO: caution
    def cumulative_ne_health_by_mg(self,mg):
        assert type(mg) == MicroGraph
        x = 0.

        for (k,v) in mg.dg.items():
            x += self.node_health_map[k]
            for v_ in v:
                q = k + "," + v_
                x += self.edges_health_map[q]
        return x

    def ne_extremum(self):
        nl = list(self.node_health_map.values())
        el = list(self.edges_health_map.values()) 
        q1 = (min(nl),max(nl))
        q2 = (min(el),max(el))
        return (q1,q2)

    ############################ isomorphism functions ################

    # the subgraph isomorphism problem
    # use with caution!! 
    """
    return:
    - if all_iso:
        list<dict, node self -> node other>
      otherwise: 
    """
    def subgraph_isomorphism(self,mg,all_iso=False):#,include_extra=0):
        search_candidates = []
        
        # get the initial candidates for each 
        q = {}
            # rank the nodes of dg from smallest to largest degree
            # element in l := (node, node degree) of mg
        l = [(k,len(v)) for (k,v) in mg.dg.items()]
        l = sorted(l, key=lambda x: x[1])
        lx = [l_[1] for l_ in l]
        dl = [l_ for (i,l_) in enumerate(lx) if lx[:i].count(l_) == 0]    
        mg2 = MicroGraph.from_ResourceGraph(self)

            # do minumum deletion if no include_extras
        ml = dl[0]
        stat = True
        while stat:
            qx = set() 
            for (k,v) in mg2.dg.items():
                if len(v) < ml:
                    qx |= {k}
            stat = len(qx) != 0
            mg2.subgraph_nodeset_exclusion(qx)

            # iterate through mg and determine qualifying nodes of mg2
        qualifying = defaultdict(set) # node of mg -> set of nodes of mg2
        for l_ in l:
            for (k,v) in mg2.dg.items():
                if len(v) >= l_[1]:
                    qualifying[l_[0]] |= {k}
        
        # continue on with search 

        # each element is of the form
        ## node of mg2 --> node of mg
        ## remaining nodes of mg
        search_list = deque()
        l = l[::-1]

        # get the candidates for the first
        for xs in qualifying[l[0][0]]:
            sl1 = [[xs,l[0][0]]]
            sl2 = set(mg.dg.keys()) - {l[0][0]}
            search_list.append([sl1,sl2])
        
        # continue on with search
        stat = len(search_list) != 0 
        results = []
        ##results_extra = []
        while stat:
            # pop the first candidate
            candidate = search_list.popleft()
            cand_exc = set([cdes[0] for cdes in candidate[0]])
            stat2 = len(candidate[1]) != 0
            if not all_iso and not stat2:
                if len(candidate[1]) == 0:
                    return candidate[0] 
            if not stat2:# or include_extra > 0:
                """
                if len(candidate[0]) != len(mg.dg):
                    if include_extra > 0:
                        results.append(candidate[0])
                        include_extra -= 1
                        stat = len(search_list) != 0
                        continue 
                """
                results.append(candidate[0])
                stat = len(search_list) != 0
                #include_extra -= 1
                continue                

            q = candidate[1].pop()
            # get neighbors of candidate in mg
            nmg = mg.dg[q]
            # determine counterparts of mg neighbors to mg2 in candidate map
            counternmg = set([x[0] for x in candidate[0] if x[1] in nmg])

            # get possible counterparts of q to mg2
            qual = deepcopy(qualifying[q])
            qual -= cand_exc

            # iterate through the possible counterparts and determine 
            # which ones would work
            c = 0 
            for qn in qual:
                possible_neighbors = mg2.dg[qn]
                inter = counternmg.issubset(possible_neighbors)
                if inter:
                    cand1 = deepcopy(candidate[0])
                    cand2 = deepcopy(candidate[1])
                    cand1.append([qn,q])
                    search_list.appendleft([cand1,cand2])
                    c += 1

            # add extra, and rank it in ascending order by score
            """
            if c == 0:## and include_extra > 0:
                
                candidate[1] |= {q} 
                j = 0
                for i in range(len(results_extra)):
                    if len(results_extra[i][1]) > len(candidate[1]):
                        j = i
                results_extra.append(candidate[0])
                results_extra = results_extra[:include_extra]
                include_extra -= 1
                #search_list.append(candidate)
            """

            stat = len(search_list) != 0
        return results# + results_extra

    # TODO: test
    # NOTE: should be a isomorphism (no partial solutions)
    """
    isomap := dict, node of self -> node of wanted_mg
    """
    def isomap_to_isograph(self,wanted_mg:MicroGraph,isomap):
        output = MicroGraph(defaultdict(set))
        mgslf = MicroGraph.from_ResourceGraph(self)
        isomap_inv = invert_simple_map(isomap) 

        # iterate through (node of mgslf --> node of wanted_mg)
        for (k,v) in isomap.items():
            # get the neighbors of v in wanted_mg
            neighbors = wanted_mg.dg[v]

            # iterate through and fetch corresponding
            # node to neighbor
            for n in neighbors:
                try: output.dg[k] |= {isomap_inv[n]}
                except: pass
        return output

    ############################ partial isomorphism functions ################

    def additions_from_partial_n2n_map(self,wanted_graph:MicroGraph,reversos=False):
        assert type(reversos) == bool 
        assert type(wanted_graph) == MicroGraph
        
        self.set_nic()
        nmap = self.nearest_partial_n2n_map(wanted_graph,reversos)

        # for each of the values that are question marks, reassign it as
        # a stringized integer
        additional_nodes = set() 
        additional_edges = set()
        nmap_ = defaultdict(str)
        for k in nmap.keys():
            if nmap[k] == "?":
                nk = next(self.nic)
                nmap[k] = nk
                additional_nodes = additional_nodes | {nk} 
            else:
                nmap_[nmap[k]] = k

        # get the partial isomorphism
        nmap = invert_simple_map(nmap) 
        mgx = self.isomap_to_isograph(wanted_graph,nmap_)

        # add the additional nodes to the partial
        for a in additional_nodes:
            mgx.dg[nmap[a]] = set() 

        # perform sub. op.
        e1,additional_edges = wanted_graph.alt_subtract(mgx)
        return additional_nodes,additional_edges

    ## TODO: 
    """
    outputs a node-to-node mapping that is a partial isomorphism: 
    node of wanted graph -> node of this resource graph OR ? [if none available]
    """
    def nearest_partial_n2n_map(self,wanted_graph:MicroGraph,reversos = True):
        assert type(wanted_graph) == MicroGraph

        # each element is of the form
        # map<node of wanted graph -> node of RG>
        # sequence<candidates of RG>
        # sequence<candidates of wanted_graph>
        scache = deque()
        nc1 = wanted_graph.neighbor_count()
        ncseq1 = sorted([(k,v) for k,v in nc1.items()],key=lambda x:x[1])
        ncseq1 = [x[0] for x in ncseq1]
        best_soln = defaultdict(str)

        mg2 = MicroGraph.from_ResourceGraph(self) 
        nc2 = mg2.neighbor_count()
        ncseq2 = sorted([(k,v) for k,v in nc2.items()],key=lambda x:x[1],reverse=reversos)
        ncseq2 = [x[0] for x in ncseq2]

        # 0-case
        if len(ncseq2) == 0:
            for x in ncseq1:  
                best_soln[x] = "?"
            return best_soln

        # start the cache off with assigning each 
        for x in ncseq2:
            m = defaultdict(str)
            m[ncseq1[0]] = x
            ncseq1_ = deepcopy(ncseq1)
            ncseq1_.pop(0)
            ncseq2_ = deepcopy(ncseq2)
            ncseq2_.remove(x)

            ##print("[1] ",ncseq1_)
            ##print("[2] ",ncseq2_)

            scache.append([m,ncseq1_,ncseq2_])

        # build the cache until soln found
        while len(scache) > 0:
            q = scache.popleft()

            # case: finished mapping, check
            if len(q[1]) == 0:
                ism = invert_simple_map(q[0])
                mgx = self.isomap_to_isograph(wanted_graph,ism)
                stat = mgx == wanted_graph
                if not stat:
                    return q[0]
            
            # case: no more candidates from wanted to self
            if len(q[2]) == 0 and len(q[1]) > 0:
                for x in q[1]:
                    q[0][x] = "?"
                return q[0]
            
            c1 = q[1].pop(0)
            for x in q[2]:
                qx = deepcopy(q[0])
                qx[c1] = x
                qx2 = deepcopy(q[2]) 
                qx2.remove(x)
                scache.appendleft([qx,deepcopy(q[1]),qx2])
        return None