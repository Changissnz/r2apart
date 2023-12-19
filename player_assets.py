from rules import *

MMOVE_TYPES = {"MInfo#1","MInfo#2","MInfo#3"}

def merge_dictionaries__additive(dseq):
    if len(dseq) == 0:
        return defaultdict(float)

    dx = deepcopy(dseq[0])
    for x in dseq[1:]:
        for (k,v) in x.items():
            dx[k] = dx[k] + v
    return dx

def merge_dictionaries__concatenate(d1,d2):
    q = defaultdict(None)
    d2x = deepcopy(d2)

    for (k,v) in d1.items():
        v2 = [v]
        if k in d2x:
            v2.append(d2x[k])
            del d2x[k]
        q[k] = v2

    for (k,v) in d2x.items():
        v2 = [v]
        q[k] = v2
    return q



"""
used primarily for dictionaries with int|float values
"""
def dict_subtraction(d1,d2):
    dx = set(d1.keys()) | set(d2.keys())
    d3 = defaultdict(float)
    for d in dx:
        d3[d] = d1[d] - d2[d] 
    return d3

def std_linear_combination(wx,sample):
    ls = len(sample)


    assert ls > 0
    assert len(wx) - 1 == len(sample)
    q = 0
    for i in range(ls):
        q += (wx[i] * sample[i])
    q += wx[ls]
    return q

# TODO: test 
"""
rank values are in the range [1,len(d)]. Tied
samples have the same rank.

return:
- dict, key of d -> rank of key 
"""
def rank_stddict_floatvalues(d,reverse=False):
    assert type(d) in {dict,defaultdict}
    """
    print("D: ", d)
    """

    # convert d to a list
    dx = [(k,v) for (k,v) in d.items()]
    # sort it by the second index
    dx = sorted(dx,key=lambda x:x[1],reverse=reverse)
    if len(dx) == 0:
        return {}

    # iterate through and assign each one a
    # ranking number,
    dranks = defaultdict(float)
    r = 1
    x = dx[0][1]
    dranks[dx[0][0]] = r

    for q in dx[1:]:
        """
        print("X: ", x)
        print("Q: ", q)
        """
        if abs(x - q[1]) > 10 ** -5:
            r += 1 
            x = q[1]
        dranks[q[0]] = r
    ##print("RANKINGSSS: ",dranks)

    return dranks

## functions used to condense the delta values of a PMove on a 
## particular Player
################################################################

"""
Function calculates a potency score for a move of interest given
a health value (of a node or edge) and the delta of the health value.

c := a float coefficient, typically the max multiplier of the
     move of interest
p := float, typically the payoff of the move of interest. 
"""
def public_potency_continuous_default_function(h,h_delta,c,p): 
    d = abs(c * p)
    assert d > 0
    return float(h + h_delta) / d

"""
Must use the output from `public_potency_continuous_default_function`.
Input is the output from the function `public_potency_continuous_default_function`.

"""
def public_potency_boolean_default_function(h_post,c,p):
    return int(h_post <= 0.) 

def private_potency_continuous_default_function(h,h_delta,c,p):
    d = abs(c * p)
    assert d > 0
    return abs(h_delta) / d

# must use the output from `private_potency_continuous_default_function`
def private_potency_boolean_default_function(h_post,c,p):
    return int(h_post <= c * p / 2.) 

################################################################

# the co-payoff command that allows for effects to take place between two or more players
# from the reference of one of the players that takes this `PMove` instance. 
class PMove:

    def __init__(self,payoff:float, antipayoff:float,payoff_target:ResourceGraph,
        antipayoff_target:ResourceGraph,is_public_info:bool,pm_idn:str="NEIN"):

        # float
        self.payoff = payoff
        # float
        self.antipayoff = antipayoff
        # ResourceGraph,subgraph
        self.payoff_target = payoff_target
        # ResourceGraph,subgraph
        self.antipayoff_target = antipayoff_target
        # TODO: delete this
        self.is_public_info = is_public_info
        # identity value of instance
        self.pm_idn = pm_idn 

    """
    """
    def __str__(self):
        s = "* PMove identifier: " + self.pm_idn + "\n"
        s += "* payoff: " + str(self.payoff) + "\n"
        s += "* antipayoff: " + str(self.antipayoff) + "\n"
        s += "* payoff graph: " + "\n" + str(self.payoff_target) + "\n"
        s += "* antipayoff graph: " + "\n" + str(self.antipayoff_target) + "\n"
        return s

    @staticmethod
    def generate__type_default(i:int,por,pidn:str):
        if type(i) != type(None):
            random.seed(i)
        assert is_valid_range(por)
        assert por[0] < 0 and por[1] > 0

        payoff = random.randint(1,por[1])
        antipayoff = random.randint(por[0],-1)
        ti = random.randint(0,len(DEFAULT_SAMPLE_RESOURCEGRAPHS) - 1)
        ati = random.randint(0,len(DEFAULT_SAMPLE_RESOURCEGRAPHS) - 1)
        rg1 = DEFAULT_SAMPLE_RESOURCEGRAPHS[ti]()
        rg2 = DEFAULT_SAMPLE_RESOURCEGRAPHS[ati]()
        return PMove(payoff,antipayoff,rg1,rg2,True,pidn)

    """
    expected delta maps
    [0] node -> health delta
    [1] edge -> health delta

    possible edge additions for higher payoff
    [2] iterable(stringized edges)
    [3] node -> node frequency
    """
    ## TODO: test
    def gauge_payoff_seq_on_RG(self,rg:ResourceGraph,is_target:bool):
        """
        print("GAUGING PAYOFF SEQ ON RG")
        """
        nhdelta = defaultdict(float)
        ehdelta = defaultdict(float)
        iter_stringizededges = []

        # collect isomorphisms
        q = deepcopy(self.payoff_target) if is_target else\
            deepcopy(self.antipayoff_target)
        mgrg = MicroGraph.from_ResourceGraph(q)#rg)
        si = rg.subgraph_isomorphism(mgrg,all_iso=True)#,include_extra=include_extras)

            ######
        """
        print("DONE COLLECTING ISOS")
        for si_ in si: 
            print(si_)
            print()
        """
            ######

        # iterate through each of the isomorphisms, in reverse order
            # start with the possible extraneous
        sk = set(mgrg.dg.keys())
        mx = max([int(i) for i in sk])
        nic = DefaultNodeIdnCounter(mx)

        # $^ 
        delta_node = defaultdict(int)
        delta_edge = defaultdict(int)
        suggested_imp = []
        node_relevance = defaultdict(int) 

        while len(si) > 0:
            x = si.pop(-1)
            si2 = pairseq_to_dict(x)
            g2 = rg.isomap_to_isograph(deepcopy(mgrg),si2)
            delta_node,delta_edge,node_relevance = self.update_delta_map_for_move(\
                delta_node,delta_edge,g2,node_relevance,is_target)

        addit = None
        if is_target:
            qx = MicroGraph.from_ResourceGraph(q) 
            addit = rg.additions_from_partial_n2n_map(qx,reversos=False)#mgrg,reversos=False)
        return delta_node,delta_edge,addit,node_relevance

    """
    occurrence of a node n -> apply payoff to n
    every edge e := (n1,n2) -> apply payoff to e,n1,n2
    """
    # NOTE: looks only at undirected graph
    def update_delta_map_for_move(self,dnm:defaultdict,dem:defaultdict,\
        mg:MicroGraph,node_relevance:defaultdict,is_target:bool):
        q = self.payoff if is_target else self.antipayoff
        # iterate through nodes
        for (k,v) in mg.dg.items():
            dnm[k] += q
            node_relevance[k] += 1
            for v_ in v:
                x = sorted([k,v_])
                dem[x[0] + "," + x[1]] += q 
                dnm[k] += q
                dnm[v_] += q
                node_relevance[k] += 1
                node_relevance[v_] += 1
        return dnm,dem,node_relevance


    # TODO: test 
    """

    nic := a node identifier counter for 

    return:
    - map: node of ?anti?target graph -> node of rg
    - seq: <e_1,e_2,...,e_j>; e_i := edge of rg
    """
    def suggested_additions_on_RG(self,isomap:defaultdict,partial_mg:MicroGraph,\
        is_target:bool,nic:DefaultNodeIdnCounter):
        
        # minus partial from
        pt = self.payoff_target if is_target else self.antipayoff_target 
        q1,q2 = pt.alt_subtract(partial_mg) 
        
        # add the additional nodes
            # payoff graph node -> resource graph identifier
        nm = defaultdict(str) 
        for q in q1:
            nid = next(nic)
            nm[nid] = q
        
        nm2 = deepcopy(nm)
        nm2.update(isomap)
        nm2 = invert_simple_map(nm2)
        ##
        ##print("+ ISOMAP")
        ##print(nm2)
        ##

        em = set()

        # add the payoff edges
        while len(q2) > 0:
            # pop one
            e1 = q2.pop()
            e1q = e1.split(",")

            # remove its duplicate
            e1_ = e1q[1] + "," + e1q[0]
            q2 = q2 - {e1_}

            # add undirected edge
            undi1 = nm2[e1q[0]] + "," + nm2[e1q[1]]
            em = em | {undi1}
        return nm,em,nic


#stores the payoff information for one affected player p2 given 
#another player's PMove 
class PInfo:

    def __init__(self,ne_cpotency_pair,ne_bpotency_pair,ne_additions,node_frequency,
        nodes_delta,edges_delta):
        # node continuous, edge continuous
        self.ne_cpotency_pair = ne_cpotency_pair
        self.ne_bpotency_pair = ne_bpotency_pair
        self.ne_additions = ne_additions
        self.nf = node_frequency
        self.nd = nodes_delta
        self.ed = edges_delta  
        return

    """
    condenses itself into 1-d vector form of length 11:
    2 + 2 + (1 for |ne_additions[0]|) + (1 for |ne_additions[1]|) +
    (1 for mean(nf)) + (1 for min(nf)) + (1 for max(nf)) + (1 for node delta) + 
    (1 for edge delta) = 11
     """
    def condense(self):
        q = []
        q.extend(deepcopy(self.ne_cpotency_pair))
        q.extend(deepcopy(self.ne_bpotency_pair))
        q.append(len(self.ne_additions[0]))
        q.append(len(self.ne_additions[1]))

        v = set(self.nf.values())
        if len(v) != 0:
            average = sum(v)
            minumum = min(v)
            maximum = max(v)
        else:
            average,minumum,maximum = 0,0,0
        q.extend([average,minumum,maximum]) 
        q.extend([self.nd,self.ed])
        return q

    def std_condense(self):
        return self.condense() 

    def __str__(self):
        s = "continuous score:\t" + str(self.ne_cpotency_pair) 
        s += "\nboolean score:\t" + str(self.ne_bpotency_pair)
        s += "\n+nodes:" + str(self.ne_additions[0])
        s += "\n+edges:" + str(self.ne_additions[1])
        s += "\nfrequency:\n" + str(self.nf)# + "\n" 
        s += "\nnodes delta: " + str(self.nd) + " edges delta: " + str(self.ed) + "\n"
        return s 

#####################################################################################

# the `Modify` move command; the following are its uses:
# - "withdraw" from nodes and edges (deletes those nodes and edges
#       belonging to the owner's subgraph, and adds the cumulative
#       health from those deletions to the owner's excess health). 
#   * set(nodes),set(edges)
# -  instructions for adding nodes and edges to fulfill an additional
#    isomorphism of one of the owner's default moves. 
#   * set(additional nodes), set(additional edges)
# X X X X TODO: 
# -  instructions for creating a new move according to a default
#    combinative addition scheme between two or three moves.
#   * list(AMoves)
#
#  arguments for each MMove type
# - #1: "add n.e": (move idn, (c1,c2,c3))
# - #2: "withdraw": (1-hit nodes, 1-hit edges), (2-hit nodes, 2-hit edges)
# - #3: "make move": <iterable::(move idn)>
class MMove:

    def __init__(self,move_type,move_data):
        assert move_type in MMOVE_TYPES, "move type {} invalid".format(move_type)
        self.move_type = move_type
        self.move_data = move_data
        self.check_data()
        return 

    def check_data(self):
        if self.move_type == "MInfo#1":
            assert len(self.move_data) == 2
            assert type(self.move_data[0]) == str
            assert len(self.move_data[1]) == 3
        elif self.move_type == "MInfo#2":
            assert len(self.move_data) == 2
            assert len(self.move_data[0]) == len(self.move_data[1])
            assert len(self.move_data[0]) == 2
        else:
            assert False, "not yet coded"
        return


class MInfo:

    """
    msd := defaultdict, move idn -> (c1,c2,c3)
    ne1 := (1-hit nodes, 1-hit edges)
    ne2 := (2-hit nodes, 2-hit edges)
    mhsr := float, minumum hit survival rate
    """
    def __init__(self, mmove_score_dict,ne1,ne2,\
        mhsr):
        assert type(mmove_score_dict) == defaultdict
        assert len(ne1) == len(ne2)
        assert len(ne1) == 2
        self.msd = mmove_score_dict
        self.ne1 = ne1
        self.ne2 = ne2
        self.mhsr = mhsr

    def __str__(self):
        s = str(self.msd) + "\n\n" + str(self.ne1) +\
            "\n\n" + str(self.ne2) + "\n* min hit: " +\
                str(self.mhsr) + "\n\n"
        return s

    """
    return: 
    *2 vectors* 
    [0] 3 values X # of moves (in ascending order by move idn.)
    [1] # of 1-hit nodes, # of 1-hit edges, # of 2-hit nodes,
        # of 2-hit edges, minumum hit survival rate
    """
    def condense(self):
        x1 = []
        q = list(self.msd.keys())
        q = sorted([int(q_) for q_ in q])
        for q_ in q:
            x1.extend(deepcopy(self.msd[str(q_)])) 
        x2 = [len(self.ne1[0]),len(self.ne1[1]),\
            len(self.ne2[0]),len(self.ne2[1]),deepcopy(self.mhsr)]
        return x1,x2

    def std_condense(self):
        x2 = [len(self.ne1[0]),len(self.ne1[1]),\
            len(self.ne2[0]),len(self.ne2[1]),deepcopy(self.mhsr)]
        return deepcopy(self.msd),x2

    def to_MMove(self,mmove_type):
        assert "MInfo#1" in mmove_type or "MInfo#2" in mmove_type
        md = None
        mt = None
        if "MInfo#1" in mmove_type:
            q = mmove_type.split("-")
            assert len(q) == 2
            assert q[1] in self.msd
            md = (q[1],self.msd[q[1]])
            mt = "MInfo#1"
        else:
            md = (deepcopy(self.msd1),deepcopy(self.msd2))
            mt = "MInfo#2"
        return MMove(mt,md)

######################################################################################

class AMove:

    def __init__(self,payoff_target:MicroGraph,antipayoff_target:MicroGraph):
        assert type(payoff_target) == type(antipayoff_target)
        assert type(payoff_target) == MicroGraph
        self.pt = payoff_target
        self.at = antipayoff_target

class AInfo:

    def __init__(self,s1,s2,s3,s4,s5):
        assert type(s1) == type(s2)
        assert type(s1) == defaultdict
        assert type(s4) == defaultdict
        assert len(s5) == 2
        assert type(am1) == AMove and type(am1) == type(am2)

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5 = s5
        self.am1 = am1
        self.am2 = am2

    def __str__(self):
        s = "\texpected losses" + "\n"
        s += str(self.s1) + "\n\n"
        s += "\tv-e score / resource graph" + "\n"
        s += str(self.s2) + "\n\n"
        s += "\texpected gains for other" + "\n"
        s += str(self.s3) + "\n\n"
        s += "\thit survival rate of other player" + "\n"
        s += str(self.s4) + "\n\n"
        s += "\thit survival rate of self" + "\n"
        s += "\t\t-- 25 percentile" + "\n"
        s += str(self.s5[0]) + "\n"
        s += "\t\t-- 75 percentile" + "\n"
        s += str(self.s5[1]) + "\n"
        return s

    """
    [0] gains for self, 25-percentile min hit,25-percentile max hit,
        75-percentile min hit, 75-percentile max hit
    [1] 4 X # of anti-players,
        (expected losses, ratio of loss, min hit survival rate, max hit survival rate)
    """
    def condense(self):
        # player info
        v = [deepcopy(self.s3)]
        v.extend(deepcopy(self.s5[0]))
        v.extend(deepcopy(self.s5[1]))

        # anti-player info
        vx = list(self.s1.keys())
        vx = sorted([int(vx_) for vx_ in vx])

        v2 = []
        for vx_ in vx:
            v2.extend([self.s1[str(vx_)],self.s2[str(vx_)]])
            v2.extend(self.s4[str(vx_)])
        return v,v2

    def std_condense(self):
        # player info
        v = [deepcopy(self.s3)]
        v.extend(deepcopy(self.s5[0]))
        v.extend(deepcopy(self.s5[1]))

        # anti-player info
            # mean of s1
        s1mean = mean_safe_division(list(set(self.s1.values())))
            # mean of s2
        s2mean = mean_safe_division(list(set(self.s2.values())))
            # mean of s4[0] and s4[1]
        s4seq = mean_safe_division(list(set(self.s4.values())))
        s40mean = mean_safe_division([x[0] for x in s4seq])
        s41mean = mean_safe_division([x[1] for x in s4seq])
        v2 = [s1mean,s2mean,s40mean,s41mean]
        return v,v2

#######################################################################################

class NMove:

    def __init__(self,is_nego,destination_player,chipinfo_seq):
        assert type(is_nego) == bool 
        assert type(destination_player) == str 
        assert type(chipinfo_seq) == list
        self.is_nego = is_nego
        self.destination_player = destination_player
        # delta,loc,type
        self.chipinfo_seq = chipinfo_seq
        return

class NInfo:

    def __init__(self,is_nego,destination_player,chipinfo_seq):
        # check for assertion
        q = NMove(is_nego,destination_player,chipinfo_seq) 
        self.is_nego = is_nego
        self.destination_player = destination_player
        # delta,loc,type
        self.chipinfo_seq = chipinfo_seq
        return 

    def __str__(self):
        s = "is nego: {}".format(self.is_nego)
        s += "\n" + "affected player: {}".format(self.destination_player)
        s += "\n" + "chip info:\n" + str(self.chipinfo_seq)
        return s

    """
    return: 
    - two-tuple, [0|1 for is_nego, cumulative delta from `chipinfo_seq`]
    """
    def std_condense(self):
        x1 = [int(self.is_nego)]
        x = sum([q[0] for q in self.chipinfo_seq])
        x1.append(x)
        return x1

    def to_NMove(self):
        return NMove(self.is_nego,self.destination_player,self.chipinfo_seq)

#######################################################################################

STD_DEC_WEIGHT_INDEXSIZE_MAP = {"PInfo":(0,11),\
    "AInfo#1":(1,5),\
    "AInfo#2":(2,4),\
    "MInfo#1":(3,3),\
    "MInfo#2":(4,5),\
    "NInfo":(5,2)} 

STD_DEC_WEIGHT_SEQLABELS = ["PInfo","AInfo#1","AInfo#2",\
        "MInfo#1","MInfo#2","NInfo"] 

# standard decision function used by players in <r2apart>. 
# The following variables are to be used in the decision 
# function:
# - <PInfo>, used for <PMove>
# 
class StdDecFunction:

    """
    weights := None or 
        [0] PInfo weights vec, size 11
        [1] AInfo weights vec #1, size 5
        [2] AInfo weights vec #2, size 4
        [3] MInfo weights vec #1, size 3
        [4] MInfo weights vec #2, size 5
        [5] NInfo weights vec #1, size 2
    """
    def __init__(self,weights=None):
        self.weights = weights

        if type(self.weights) == type(None):
            self.instantiate_default_weights()
        return

    # TODO: 
    """
    sets `weights` to 1-vector in the event that null weights are given 
    """
    def instantiate_default_weights(self):
        q = None
        self.weights = []
        for x in STD_DEC_WEIGHT_SEQLABELS:
            q = [1 for i in range(STD_DEC_WEIGHT_INDEXSIZE_MAP[x][1] + 1)]
            self.weights.append(q)
        return

    """
    rcm_vec := reduced-condensed vector repr. of move
    move_type := str, key in STD_DEC_WEIGHT_INDEXSIZE_MAP
    """
    def output(self,rcm_vec,move_type):
        assert move_type in STD_DEC_WEIGHT_INDEXSIZE_MAP
        i = STD_DEC_WEIGHT_INDEXSIZE_MAP[move_type][0]
        wgts = self.weights[i]
        return std_linear_combination(wgts,rcm_vec)

    """
    adjusts weights. 
    """
    def adjust(self):
        return -1

####################################################

class PContextDecision:

    def __init__(self,pproc,aproc,mproc,nproc):
        assert type(pproc) == dict
        assert type(pproc) == type(aproc) and type(pproc) == type(aproc)
        assert type(nproc) == list

        # dict, PMove idn -> float score
        self.pproc = pproc
        # dict, AMove type -> float score
        self.aproc = aproc
        # dict, MMove type -> float score 
        self.mproc = mproc
        # vec, float score corresponding to each NMove index
        self.nproc = nproc
        self.ranking = None

    def rank(self,verbose):
        q = self.to_one_vec()
        self.ranking = sorted(q,key=lambda x:x[1],reverse=True)
        if verbose: 
            print("-- ranking of choices")
            print(self.ranking)
        return deepcopy(self.ranking)

    """
    - return:
    vector of move descriptors; each element is one of
    * PInfo-`pmove idn`, float
    * AInfo`(#1|#2)`,float
    * MInfo`(#1|#2)`,float
    * NInfo-`nmove index`,float
    """
    def to_one_vec(self):
        q = []
        q.extend([("PInfo-" + k,v) for (k,v) in self.pproc.items()])
        q.extend([(k,v) for (k,v) in self.aproc.items()])
        q.extend([(k,v) for (k,v) in self.mproc.items()])
        q.extend([("NInfo-" + str(i),v) for (i,v) in enumerate(self.nproc)])
        return q

########################################################################################

def accumulate_ea_map(eam):
    assert type(eam) in {dict,defaultdict}
    expected,actual = 0.,0.

    for (k,v) in eam.items():
        expected += v[0]
        actual += v[1]
    return expected,actual

"""
container used to store the values for a Player's
defensive intelligence. Defense intelligence includes:

*based on execution and reception of PMoves*
- the self-side (antitarget's perspective)
    * node|edge of self -> other player idn -> <negative delta seq.>
    * node|edge -> hit survival rate; an estimate based on past history 
- the antiself-side (self's perspective)
    * PMove idn -> other player -> node -> expected/actual float diff
    * PMove idn -> other player -> edge -> expected/actual diff
- the self-side (target's perspective)
    * previous move idn., node|edge of self -> expected/actual float diff
"""
class DefInt:

    def __init__(self,mem_size:int):
        ## these variables are used to gauge other players' influence
        ## on the owner

        # node|edge -> other player idn -> <negative delta sequence>;
        # length of the negative delta sequence is `mem_size`. 
        self.node_delta = defaultdict(defaultdict)
        self.edge_delta = defaultdict(defaultdict)
        self.mem_size = mem_size
        self.player_idns = set()

        ## variables used to analyze owner's ResourceGraph
        self.node_hit_survival_rate = defaultdict(int)
        self.edge_hit_survival_rate = defaultdict(int)

        ## variables used to analyze other players' ResourceGraphs
        # pmove -> player -> node/edge -> expected,actual
        self.pmove_playernode_recep = defaultdict(defaultdict)
        self.pmove_playeredge_recep = defaultdict(defaultdict)

        # pmove -> node/edge -> expected,actual
        self.ea_self_target_node = defaultdict(defaultdict)
        self.ea_self_target_edge = defaultdict(defaultdict)
        return

    def remove_deceased_player(self,pidn):
        self.node_delta = self.remove_deceased_player_from_PMove_dict(pidn,self.node_delta)
        self.edge_delta = self.remove_deceased_player_from_PMove_dict(pidn,self.edge_delta)
        self.pmove_playernode_recep = self.remove_deceased_player_from_PMove_dict(pidn,self.pmove_playernode_recep)
        self.pmove_playeredge_recep = self.remove_deceased_player_from_PMove_dict(pidn,self.pmove_playeredge_recep)

    def remove_deceased_player_from_PMove_dict(self,pidn,d):
        for k in d.keys():
            v = d[k]
            if pidn in v:
                del v[pidn]
            d[k] = v
        return d






    def display(self):
        print("-- node delta")
        print(self.node_delta)
        print()
        print("-- edge delta")
        print(self.edge_delta)
        print()
        print("-- node hit survival rate")
        print(self.node_hit_survival_rate)
        print()
        print("-- edge hit survival rate")
        print(self.edge_hit_survival_rate)
        print()
        print("-- pmove player node reception")
        print(self.pmove_playernode_recep)
        print()
        print("-- pmove player edge reception")
        print(self.pmove_playeredge_recep)
        print()
        print("-- pmove self node reception")
        print(self.ea_self_target_node)
        print()
        print("-- pmove self edge reception")
        print(self.ea_self_target_edge)
        print()
        return

    def cumulative_nOrE_delta__most_recent(self,x,is_node):
        q = self.node_delta if is_node else self.edge_delta

        if x not in q: return 0.
        dx = q[x]
        c = 0.
        for (k,v) in dx.items():
            c += v[-1]
        return c

    """
    """
    def cumulative_expected_actual_of_move(self,pm_idn,is_self:bool):
        assert type(is_self) == bool
        exp,act = 0.,0.

        if is_self:
            q1 = self.ea_self_target_node[pm_idn]
            exp1,act1 = accumulate_ea_map(q1)
            exp += exp1
            act += act1

            q2 = self.ea_self_target_edge[pm_idn]
            exp1,act1 = accumulate_ea_map(q2)
            exp += exp1
            act += act1
        else:
            q1 = self.pmove_playernode_recep[pm_idn]
            for v in q1.values():
                exp1,act1 = accumulate_ea_map(v)
                exp += exp1
                act += act1

            q1 = self.pmove_playeredge_recep[pm_idn]
            for v in q1.values():
                exp1,act1 = accumulate_ea_map(v)
                exp += exp1
                act += act1
        return exp,act

    """
    """
    def cumulative_expected_actual_of_move_by_node_info(self,p_idn,node_idn):
        # case: self
        expected,actual = 0.,0.
        if type(p_idn) == type(None):
            for (k,v) in self.ea_self_target_node.items():
                if node_idn not in v: continue
                q = v[node_idn]
                expected += q[0]
                actual += q[1]
        # case: others
        else:
            for (k,v) in self.pmove_playernode_recep.items():
                if p_idn not in v: continue
                v2 = v[p_idn]
                if node_idn not in v2: continue
                q = v2[node_idn]
                expected += q[0]
                actual += q[1]
        return expected,actual 

    """
    """
    def add_sample(self,player_idn,node_delta,edge_delta):
        self.player_idns = self.player_idns | {player_idn}
        for (k,v) in node_delta.items():
            if player_idn not in self.node_delta[k]:
                self.node_delta[k][player_idn] = [] 

            self.node_delta[k][player_idn].append(v)
            x = len(self.node_delta[k][player_idn]) - self.mem_size
            if x > 0:
                self.node_delta[k][player_idn] = \
                    self.node_delta[k][player_idn][x:] 
        
        for (k,v) in edge_delta.items():
            if player_idn not in self.edge_delta[k]:
                self.edge_delta[k][player_idn] = []
            self.edge_delta[k][player_idn].append(v)
            x = len(self.edge_delta[k][player_idn]) - self.mem_size
            if x > 0:
                self.edge_delta[k][player_idn] = \
                    self.edge_delta[k][player_idn][x:] 
        return

    """
    Adds a sample of owner's PMove's expected/actual difference
    onto another player. 
    """
    def add_antisample(self,owner_idn,pm_idn,p_idn,ea_ndm,ea_edm):
        self.pmove_playernode_recep[pm_idn][p_idn] = ea_ndm
        self.pmove_playeredge_recep[pm_idn][p_idn] = ea_edm
        return

    """
    Adds sample of owner's PMove's expected/actual difference
    onto self.
    """
    def add_sample_ea(self,pm_idn,ea_ndm,ea_edm):
        self.ea_self_target_node[pm_idn] = ea_ndm
        self.ea_self_target_edge[pm_idn] = ea_edm
        return

    def remove_deceased(self,dn,de):
        for n in dn:
            del self.node_delta[n]

        for n in de:
            del self.edge_delta[n]
        return

    ############################ methods used for prediction

    def nodes_and_edges_by_hit_survival_rate(self,i:int):
        assert i > 0 and type(i) == int 

        nds = set([k for (k,v) in self.node_hit_survival_rate.items()\
            if v == i])
        eds = set([k for (k,v) in self.edge_hit_survival_rate.items()\
            if v == i])
        return (nds,eds) 

    def minimal_hit_survival_rate(self):
        nx,ex = float('inf'),float('inf')

        vn = [v for v in self.node_hit_survival_rate.values()]
        ve = [v for v in self.edge_hit_survival_rate.values()]
        
        vx = None if len(vn) == 0 else min(vn)
        ve = None if len(ve) == 0 else min(ve)
        return (vx,ve)

    def hit_survival_rate_extremum_on_MG(self,mg):
        assert type(mg) == MicroGraph 

        mini,maxi = float("inf"),0. 

        # iterate through nodes and edges to get hit survival rate
        for (k,v) in mgx.items():
            x = self.node_hit_survival_rate[k]
            mini = mini if mini < x else x
            maxi = maxi if maxi > x else x

            for v_ in v:
                q = k + "," + v_
                x = self.edge_hit_survival_rate[q]
                mini = mini if mini < x else x
                maxi = maxi if maxi > x else x
        return (mini,maxi)

    # TODO: untested 
    def hit_survival_rate_hypothesis(self,rg:ResourceGraph):
        nd,ed = self.expected_ne_delta()
        for (k,v) in rg.node_health_map.items():
            q = nd[k]
            if q == 0: q = 1
            self.node_hit_survival_rate[k] = \
                round(abs(v / q))

        for (k,v) in rg.edges_health_map.items():
            q = ed[k]
            if q == 0: q = 1
            self.edge_hit_survival_rate[k] = \
                round(abs(v / q))
        return
    
    """
    mean
    """
    def expected_ne_delta(self):
        dseq1 = []
        dseq2 = [] 

        for x in self.player_idns:
            nd,ed = self.expected_ne_delta_for_player(x)
            dseq1.append(nd)
            dseq2.append(ed)

        nd = merge_dictionaries__additive(dseq1)
        ed = merge_dictionaries__additive(dseq2)
        return nd,ed

    def expected_ne_delta_for_player(self,pidn):
        nd,ed = defaultdict(float),defaultdict(float)
        for (k,v) in self.node_delta.items(): 
            x = self.node_delta[k][pidn]
            l = len(x)
            nd[k] = 0 if l == 0 else sum(x) / l

        for (k,v) in self.edge_delta.items(): 
            x = self.edge_delta[k][pidn]
            l = len(x)
            ed[k] = 0 if l == 0 else sum(x) / l
        return nd,ed

"""
generates a sequence of <PMove>s that have four moves
comprising the unit:
- for every move of the unit, either the payoff XOR antipayoff
    is: (sample_resource_graph_(6|7)).
"""
def generate_PMoveSeq__type_assymetric_unit(num_moves:int,por):
    assert num_moves >= 4
    assert is_valid_range(por)
    assert por[0] < 0 and por[1] > 0

    x1 = deepcopy(DEFAULT_SAMPLE_RESOURCEGRAPHS)
    x2 = deepcopy(DEFAULT_SAMPLE_RESOURCEGRAPHS)

    u20 = x1.pop(5)()
    u21 = x1.pop(5)()

    u10 = x2.pop(5)()
    u11 = x2.pop(5)()

    ms = []
    x1,x2 = set(x1),set(x2)

    # make the payoff/antipayoff unit
    pmx = generate_PMove_helper__type_assymetric_unit(str(len(ms)),\
        por,u20,x2.pop()())
    ms.append(pmx)

    pmx = generate_PMove_helper__type_assymetric_unit(str(len(ms)),\
        por,u21,x2.pop()())
    ms.append(pmx)

    pmx = generate_PMove_helper__type_assymetric_unit(str(len(ms)),\
        por,x1.pop()(),u10)
    ms.append(pmx)

    pmx = generate_PMove_helper__type_assymetric_unit(str(len(ms)),\
        por,x1.pop()(),u11)
    ms.append(pmx)

    # make the remainder
    stat = len(ms) < num_moves and (len(x1) == 0 or len(x2) == 0)
    while stat:
        pmx = generate_PMove_helper__type_assymetric_unit(str(len(ms)),\
            por,x1.pop()(),x2.pop()())
        ms.append(pmx)
        stat = len(ms) < num_moves and (len(x1) == 0 or len(x2) == 0)
    return ms

def generate_PMove_helper__type_assymetric_unit(pidn,por,rg1,rg2):
    assert type(rg1) == ResourceGraph
    assert type(rg2) == ResourceGraph

    payoff = random.randint(1,por[1])
    antipayoff = random.randint(por[0],-1)
    return PMove(payoff,antipayoff,rg1,rg2,True,pidn)

########################################################################################

"""
Generic move descriptor used to search a <PMLog> instance for
samples that match the descriptor.

PMove: pm_idn
AMove: -
MMove: move_type
NMove: is_nego,nego_type,destination_player
"""
class GenericMoveDesc:

    def __init__(self,move_type,info_args):
        assert move_type in {PMove,AMove,MMove,NMove}
        self.mt = move_type
        self.ia = info_args
        if self.mt == PMove:
            assert len(self.ia) == 1
        elif self.mt == AMove:
            assert len(self.ia) == 0
        elif self.mt == MMove:
            assert len(self.ia) == 1
        else:
            assert len(self.ia) == 3

    @staticmethod
    def from_XInfo(move):
        t = type(move)
        assert t in {PMove,AMove,MMove,NMove}

        if t == PMove:
            return GenericMoveDesc(t,[move.pm_idn])
        elif t == AMove:
            return GenericMoveDesc(t,[])
        elif t == MMove:
            return GenericMoveDesc(t,[move.move_type])
        else:
            return GenericMoveDesc(t,[move.is_nego,\
                move.nego_type,move.destination_player])
        return 

    def __eq__(self,move):
        t = type(move)

        if t == PMove:
            if self.mt != t:
                return False
            return self.ia[0] == move.pm_idn
        elif t == AMove:
            return self.mt == t
        elif t == MMove:
            if self.mt != t:
                return False
            return self.ia[0] == move.move_type
        elif t == NMove:
            if self.mt != t:
                return False
            if self.ia[0] != move.is_nego:
                return False
            if self.ia[1] != move.nego_type:
                return False
            if self.ia[2] != move.destination_player:
                return False
            return True
        elif t == GenericMoveDesc:
            return self.ia == move.ia and \
                self.mt == move.mt
        return False 

# Player move log 
class PMLog:

    def __init__(self,log_type="full context"):
        assert log_type in {"full context", "specific"}
        self.mh = []
        self.lt = log_type
    
    def add_artifact(self,artifact):
        if self.lt == "full context":
            ##assert type(artifact) == PContext
            assert True 
        else:
            assert type(artifact) in {PMove,AMove,MMove,NMove}
        self.mh.append(artifact)
        return

    def most_recent_artifact(self,artifact):
        j = -1
        l = len(self.mh)
        fx = self.most_recent_artifact__full_context if \
            self.lt == "full context" else self.most_recent_artifact__specific

        for i in range(l - 1,-1,-1):
            if fx(artifact,i):
                return i
        return j

    def most_recent_artifact__full_context(self,artifact,i):
        q = self.mh[i]
        return artifact == q.selection

    def most_recent_artifact__specific(self,artifact,i):
        q = self.mh[i]
        return artifact == q

    def most_recent_PMove_gauge(self,pm_idn):
        assert self.lt == "full context"
        l = len(self.mh)
        for i in range(l - 1,-1,-1):
            if pm_idn in self.mh[i].pmove_prediction:
                return i,deepcopy(self.mh[i].pmove_prediction[pm_idn])
        return -1,None