# tools used by a Player to determine its next decision in the sequential game, r2apart.\
from resource_defaults import *

DEFAULT_NODE_HEALTH_RANGE = [10**6,2 * 10 ** 7]#[30,300]
DEFAULT_MOVE_PAYOFF_RANGE = [-5,5]
DEFAULT_AMOVE_COST_FUNCTION = mean_safe_division
DEFAULT_WMOVE_RATIO = 0.25
DEFAULT_GCS_SEARCH_TYPE = "full neighbor fit- type 2"
DEFAULT_BOT_FOLDER = "base_data/"
DEFAULT_TRAINING_FOLDER = "training_data/"
GAME_MODES = {"noneg","nego"}
GAME_MODES2 = {"public","private"} # ??

DEFAULT_PMOVE_MAX_GAUGES = 2
DEFAULT_DEFINT_MEMSIZE = 3
DEFAULT_NEGOCHIP_LIFESPAN = 7 
DEFAULT_NEGOCHIP_MULTIPLIER = 4


# the conversion rate from excess to ResourceGraph 
VALUE_EXPRESSION_RATE = 1.2


DEFAULT_SAMPLE_RESOURCEGRAPHS = [sample_resource_graph_1,\
                            sample_resource_graph_2,\
                            sample_resource_graph_3,\
                            sample_resource_graph_4,\
                            sample_resource_graph_5,\
                            sample_resource_graph_6,\
                            sample_resource_graph_7,\
                            sample_resource_graph_8]

# decep allows for a player's representation of its <ResourceGraph> as an edge complement
# distort allows for player to magnify its own payoff or to divide the payoff of its competitors
NEGO_TYPES = ["decep","distort"]

def ratio_x_in_range(x,r):
    assert r[0] <= r[1]
    if x < r[0]: return 0.
    if r[0] == r[1]: return 1. 
    return min([1.,float(x - r[0]) / (r[1] - r[0])]) 

# TODO: questionable in effectiveness
"""
ranks the nodes at risk, and the edges at risk, outputting
[0] ranked nodes (in descending order by risk score)
[0] ranked edges (in descending order by risk score)
"""
def default_resource_risk_function(rg, node_health_range = DEFAULT_NODE_HEALTH_RANGE):#,\
    #negative_move_payoff_range = [20,500]):
    assert node_health_range[0] <= node_health_range[1] and node_health_range[0] >= 0
    assert negative_move_payoff_range[0] <= negative_move_payoff_range[1] and\
        negative_move_payoff_range[0] >= 0

    # each element in the lists are
    # [0] (node|edge)
    # [1] risk score of node/edge
    nodes_at_risk = []
    edges_at_risk = []

    # iterate through the nodes
    for (k,v) in rg.node_health_map.items():
        r = ratio_x_in_range(v,node_health_range)
        nodes_at_risk.append((k,r))

    # iterate through the edges
    for (k,v) in rg.edges_health_map.items():
        r = ratio_x_in_range(v,node_health_range)
        edges_at_risk.append((k,r))
    
    nodes_at_risk = sorted(nodes_at_risk, key= lambda x:x[1],reverse=True) 
    edges_at_risk  = sorted(edges_at_risk, key= lambda x:x[1],reverse=True) 
    return nodes_at_risk, edges_at_risk

"""
- description
Basic deduction algorithm for locations of other players' Negochip
instances.

*case: is_self*
- distortion nodes (by others): will have a fraction of the expected

*case: not is_self*
- deceiver nodes (by others): big difference, close to 0 actual
- distorted nodes (by others): fraction of the expected

- return
node -> NegoChip type
"""
# NOTE: the expected - actual values do not take into consideration the 
#       gains of the nodes and edges by the players.
def negochip_locations__simple_deduction(ea_node_map,is_self):
    suspected_node_classifications = {}
    for (k,v) in ea_node_map.items():
        q = classify_node_by_ea_pair(v,is_self)
        if q != "noneg":
            suspected_node_classifications[k] = q 
    return suspected_node_classifications

"""
- return:
noneg|nego-distort|nego-deception
"""
def classify_node_by_ea_pair(self,ea_pair,is_self):

    if abs(ea_pair[0] - ea_pair[1]) < 3:
        return "noneg"

    if is_self:
        return "distort"
    
    if ea_pair[0] == 0:
        return "deception"

    q = ea_pair[0] % ea_pair[1]
    stat = q < 0.1

    if stat: 
        return "distort"
    return "deception"


####################################################################

# TODO: complete 
'''
negotiating chip placed on a node
'''
class NegoChip:

    def __init__(self,owner,nego_type,magnitude,loc):
        assert nego_type in NEGO_TYPES
        assert magnitude > 0 and type(magnitude) == int 
        self.owner = owner
        self.neg_type = nego_type
        self.magnitude = magnitude
        self.loc = loc
        self.lifespan = deepcopy(DEFAULT_NEGOCHIP_LIFESPAN)

    def __eq__(self,nc):
        stat1 = self.owner == nc.owner
        stat2 = self.neg_type == nc.neg_type
        stat3 = self.loc == nc.loc
        return stat1 and stat2 and stat3

    def activate(self,payoff=None):
        if self.neg_type == "distort":
            assert False
    
    def inc_one_timestamp(self):
        self.lifespan -= 1
        return self.lifespan > 0 

"""
chip used to negate a NegoChip
"""
class NegaChip:

    def __init__(self,owner,nega_type,loc):
        assert nega_type in NEGO_TYPES
        self.owner = owner
        self.neg_type = nega_type
        self.loc = loc

    def __eq__(self,nc):
        assert type(nc) == NegoChip
        stat2 = self.neg_type == nc.neg_type
        stat3 = self.loc == nc.loc
        return stat2 and stat3

'''
allows for a player to make an induction on what the <ResourceGraph> of another player P2 is
based on the isomorphisms of P2's presented <ResourceGraph> that it calculated.
'''
class GraphKnowledgeInducer:

    def __init__(self):
        self.mg = MicroGraph(defaultdict(set))

    def induce_one_isomorphism(self,mg):
        self.mg = self.mg + mg
        return

"""
player knowledge database
"""
class PKDB:

    def __init__(self):
        # player idn -> hypothesized MicroGraph
        self.other_mg = defaultdict(int)
        return

    """
    """
    def modify_MG(self,p_idn,updated_mg):
        assert type(p_idn) == str
        assert type(updated_mg) == MicroGraph
        self.other_mg[p_idn] = updated_mg
        return

"""
Container structure to hold Negochips for a 
Player's ResourceGraph.

Each node cannot contain duplicate Negochips.
"""
class NegoContainer:

    def __init__(self):
        # node location -> <negochip sequence>
        self.container = defaultdict(None)
        return

    """
    adds a Negochip `chip` to its designated node if
    there are no duplicates
    """
    def add_chip(self,chip):
        assert type(chip) == NegoChip
        q = self.container[chip.loc] if chip.loc \
            in self.container else []

        stat = True
        for (j,c) in enumerate(q):
            if c == chip:
                stat = False 
                break
        if stat:
            q.append(chip)
        return stat

    def inc_one_timestamp(self):
        delkeys = []
        for k in self.container.keys():
            v = self.container[k]
            vi = []
            for (i,v_) in enumerate(v):
                stat = v_.inc_one_timestamp()
                if stat:
                   vi.append(i)
            v_ = [v_ for (i,v_) in v if i in vi]
            self.container[k] = v_
            if len(v_) == 0:
                delkeys.append(k)

        for k in delkeys:
            del self.container[k]
        return 

    def remove_chip(self,chip):
        if chip.loc not in self.container:
            return False

        q = self.container[chip.loc] 
        j = None
        for (i,q_) in enumerate(q):
            if q_ == chip:
                j = i
                break
        if j != None:
            q.pop(j)
        self.container[chip.loc] = q
        return j != None

    """
    return: 
    - the available nodes from `starting_nodeset` based on the information
      of `player_idn` and `nego_type` that cannot be duplicated.
    """
    def available_nodes_by_info(self,player_idn,nego_type,starting_nodeset):
        assert nego_type in NEGO_TYPES
        assert type(starting_nodeset) == set

        available = set() 
        for n in starting_nodeset:
            q = self.container[n]

            # case: empty => available
            if type(n) == type(None):
                available |= {n}
                continue

            stat = True
            for q_ in q:
                l,m = q_.loc,q_.magnitude
                nc = NegoChip(player_idn,nego_type,m,l)
                
                if q_ == nc:
                    stat = False
                    break
            
            if stat:
                available |= {n}
        return available
    
    """
    Gathers sequence of active chips based on player identifier
    and `nego_type`. 
    
    NOTE: there can be exactly one chip per node in the sequence.
    """
    def active_chips_by_info(self,player_idn,nego_type):
        assert nego_type in NEGO_TYPES
        active = []

        for (k,v) in self.container.items():
            for v_ in v:
                l,m = v_.loc,v_.magnitude
                nc = NegoChip(player_idn,nego_type,m,l)

                if nc == v_:
                    active.append(deepcopy(v_))
                    break
        return active 

    """
    return:
    - sequence of active chips by `nego_type` for player 
    """
    def active_chips_by_node(self,n_idn,nego_type):
        q = self.container[n_idn]
        if type(q) == type(None): return []
        return [deepcopy(q_) for q_ in q if q_.nego_type == nego_type]
    
    """
    return: 
    - node idn -> multiplicative coefficient
    """
    def distort_coefficient_map(self,owner_idn):
        q = defaultdict(float)
        for k in self.container.keys():
            ac = self.active_chips_by_node(k,"distort")
            
            multiplier = 1.0
            for ac_ in ac:
                if ac_.owner == owner_idn:
                    multiplier = multiplier * ac_.magnitude
                else:
                    multiplier = multiplier / ac_.magnitude
            q[k] = multiplier
        return q