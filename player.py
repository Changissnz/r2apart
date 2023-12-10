from player_assets import *

PCONTEXT_DISPLAY_TYPES = {"PMove","AMove","MMove"} 

"""
NOTE: this function is designed to be the default, but program can use
       other functions (user's custom). 
"""
def potency_default_cumulative_function(player,nmap,emap,c,p):
    lnp = len(player.rg.node_health_map)
    if lnp == 0:
        return (0.,0.),(0.,0.)

    # set the potency functions
    potency_cfunc, potency_bfunc = None,None
    if player.rg.phs:
        potency_cfunc = public_potency_continuous_default_function
        potency_bfunc = public_potency_boolean_default_function
    else:
        potency_cfunc = private_potency_continuous_default_function
        potency_bfunc = private_potency_boolean_default_function

    # calculate the continuous and boolean potencies for nodes
    pnc,pnb = 0.,0.
    pec,peb = 0.,0.
    for (k,v) in player.rg.node_health_map.items():
        ptncy = potency_cfunc(k,nmap[k],c,p)
        ptncyb = potency_bfunc(ptncy,c,p) 
        pnc += ptncy
        pnb += ptncyb
    pnc /= lnp
    pnb /= lnp
    
    # do the same for edges
    lep = len(player.rg.edges_health_map)
    for (k,v) in player.rg.edges_health_map.items():
        ptncy = potency_cfunc(k,emap[k],c,p)
        ptncyb = potency_bfunc(ptncy,c,p)
        pec += ptncy
        peb += ptncyb

    if lep == 0:
        return (pnc,pec),(pnb,peb)
    
    pec /= lep
    peb /= lep 
    return (pnc,pec),(pnb,peb)

"""
scheme considers the cost of adding the required nodes+edges for each 
move. 

arguments:
- mmove_addition_dict := move idn -> (node additions,edge additions)
- mmove_payoff_dict := move idn -> payoff
- minumum_node_health := float
- minumum_edge_health := float 
- minimal_hit_survival_rate := int

return:
- dict, move idn -> (c1,c2,c3) 
*see output from <mmove_addition_score_move__type_1>*
"""
def mmove_addition_selection__type_1(mmove_addition_dict,mmove_payoff_dict,\
    minumum_node_health,minumum_edge_health,minimal_hit_survival_rate):

    assert type(rg) == ResourceGraph

    # get the node+,edge+ intersection 
    nas = [deepcopy(v) for v in mmove_addition_dict.values()]
    int1,int2 = ne_intersection(nas)

    mmove_scores = defaultdict(None)
    ks = [k for k in mmove_addition_dict.keys()]
    
    for k in ks:
        score = mmove_addition_score_move__type_1(mmove_addition_dict,k,mmove_payoff_dict[k],\
            (int1,int2),minimal_hit_survival_rate,minumum_node_health,minumum_edge_health)
        mmove_scores[k] = score
    return mmove_scores

"""
return:
- (float c1,float c2, float c3):
* c1 := score denoting the cost of adding the nodes and edges
* c2 := score corresponding to the estimated number of singleton hits the 
        node and edge additions can take
* c3 := score corresponding to the intersection ratio of the node
        and edge additions with the other moves 
"""
def mmove_addition_score_move__type_1(mmove_addition_dict,mmove_key,mmove_payoff,\
    intersects,minimal_hit_survival_rate,minumum_node_health,minumum_edge_health):

    q = mmove_addition_dict[mmove_key]
    x = len(q[0]) + len(q[1])
    c1 = (len(q[0]) * minumum_node_health * VALUE_EXPRESSION_RATE) +\
        (len(q[1]) * minumum_edge_health * VALUE_EXPRESSION_RATE)

    c2 = minimal_hit_survival_rate * x 
    c3 = 0 if x == 0 else (len(intersects[0]) + len(intersects[1])) / x
    return (c1,c2,c3) 

def ne_intersection(ne_addition_seq):
    if len(ne_addition_seq) == 0:
        return []

    q0 = ne_addition_seq[0][0]
    q1 = ne_addition_seq[0][1]

    for i in range(1,len(ne_addition_seq)):
        q0 = q0 & ne_addition_seq[i][0]
        q1 = q1 & ne_addition_seq[i][1]
    return (q0,q1)

######################################################

"""
container that holds a player's context with respect to 
all other players in a game.
"""
class PContext:

    def __init__(self): 
        # PMove -> player of interest -> PInfo
        self.pmove_prediction = defaultdict(defaultdict)
        # *see `load_amove_info* for variable info.
        self.amove_prediction = None

        self.mmove_prediction = defaultdict(float)

        # iterable<NInfo> 
        self.nmove_prediction = []

        # the decision that Player made
        self.selection = None
        return 

    ###################### display and write methods

    def write_to_outfile(self):
        return -1

    def display_pmove_prediction(self):

        return -1

    def player_PMove_gauge(self,pidn):
        d = defaultdict(None)
        for (k,v) in self.pmove_prediction.items():
            x = v[pidn]

            if type(x) != type(None):
                d[k] = x 
        return d

    def condensed_form(self):
        q = []

        if type(self.pmove_prediction) != type(None):
            q.append(self.pmove_prediction.condense())
        else:
            q.append(None)

        if type(self.amove_prediction) != type(None):
            q.append(self.amove_prediction.condense())
        else:
            q.append(None)

        if type(self.mmove_prediction) != type(None):
            q.append(self.mmove_prediction.condense())
        else:
            q.append(None)
        
        if type(self.nmove_prediction) != type(None):
            q.append(self.nmove_prediction.condense())
        else:
            q.append(None)
        return q

    def reduced_condensed_form(self):
        return -1

    ##################### update methods

    # TODO: write separate code for private vs. public context regarding
    #       visibility of ResourceGraph health information.
    # NOTE: delta maps are valid mappings, meaning the nodes/edges exist for the
    #       RG. 
    #   TODO: write a filter function to clean up the maps (in the case of NegoChip)
    """
    mv := PMove 
    player := Player
    dn := delta of nodes by PMove
    de := delta of edges by PMove
    si := suggested additions (nodes set, edges set)
    nr := node relevance
    is_target := boolean
    """
    def summarize_PMove(self,mv,player,dn,de,si,nr,is_target:bool):
        assert type(player) == Player 

        stat = len(nr) == 0 or len(dn) == 0
        if stat:
            print("NADA")
            return

        ##print("LENS:")
        ##print(len(dn),len(de),len(si),len(nr))
        c = max(nr.values())  
        p = mv.payoff if is_target else mv.antipayoff
        x1,x2 = potency_default_cumulative_function(player,dn,de,c,p)
        
        total_node_delta = sum(list(dn.values()))
        total_edge_delta = sum(list(de.values()))
        self.pmove_prediction[mv.pm_idn][player.idn] =  PInfo(x1,x2,si,nr,\
            total_node_delta,total_edge_delta)        
        return

    """
    arguments:
    - s1 := dict, player::non_owner -> expected losses
    - s2 := dict, player::non_owner -> (vertex-edge score of AMove / owner's knowledge of the player's MG);
            determines ratio of loss to player
    - s3 := float, expected gains for owner's payoff target
    - s4 := dict, other player -> (min hit survival,max hit survival)
    - s5 := [0] (min hit survival, max hit survival), AMove [25-percentile]
            [1] (min hit survival, max hit survival), AMove [75-percentile]
    """
    def load_AMove_info(self,s1,s2,s3,s4,s5):
        self.amove_prediction = AInfo(s1,s2,s3,s4,s5)
        return

    def load_MMove_info(self, x:MInfo):
        assert type(x) in {MInfo,type(None)}
        self.mmove_prediction = x
        return

    def summarize_NMove(self,player): 
        return -1

class PContextMapper:

    def __init__(self,sdf:StdDecFunction,context_file=None):
        self.sdf = sdf
        self.contexts = []
        return 

    """
    minimal calibration until wanted output is achieved. 
    """
    def calibrate_for_wanted_output(self,wanted_output):
        return -1

    """
    """
    def capture_context(self,pc:PContext):
        return -1

    """
    """
    def write_context_cache(self):
        return -1

"""
player decider; used by class<Player> as an utility in
determining the next best move. 
""" 
class PDEC:

    def __init__(self,pidn:str,pcontext_mapper:PContextMapper,\
        def_int =DefInt(DEFAULT_DEFINT_MEMSIZE),\
        pkdb = PKDB(),game_mode = "noneg"):
        assert game_mode in GAME_MODES, "invalid game mode"

        # TODO: delete this, not used
        self.other_RG = defaultdict(None)
        self.pidn = pidn

        # player p -> node -> suspected type of negochip
        self.suspected_negochips = defaultdict(defaultdict)

        # container that holds the active negochips for player
        self.nc = NegoContainer()

        # used for defensive intelligence
        self.def_int = def_int

        # used for machine-learning
        self.context_mapper = pcontext_mapper
        self.pkdb = pkdb
        self.pcontext = None

        # greatest common subgraph, used for <AMove> calculation
        self.gcs = [None,None,None]

        return

    ###################### predictive information for <PMove> ##################
    # TODO: add anti-summarize for other players
    """
    Offensive intelligence.

    Prediction using the PKDB of the effector of a PMove on an
    actor (<Player>)

    actor := owner of the PDEC instance, or a projected player
    """
    def gauge_pmove_payoff(self,actor,pmove): 
        assert type(actor) == Player
        assert type(pmove) == PMove

        # initialize if new round
        if type(self.pcontext) == type(None):
            self.pcontext = PContext() 
        
        x = None
        is_target = None
        if actor.idn == self.pidn:
            x = actor
            is_target = True
        else:
            x = actor.idn
            is_target = False

        q = self.payoff_info_for_player(x,pmove)
        dn,de,si,nr = None,None,None,None
        if type(q) != type(None):
            dn,de,si,nr = q
        else:
            print("no data")
            return 

        ###
        """
        print("SUMMARIZING:")
        print(dn)
        print()
        print(de)
        print()
        print(si)
        print()
        print(nr)
        print()
        """
        ###            
        self.pcontext.summarize_PMove(pmove,actor,dn,de,si,nr,is_target)
        return 

    """
    pidn := str, player identifier
    pmove := PMove
    """
    def payoff_info_for_player(self,pidn,pmove:PMove):
        assert type(pidn) in {str,Player} 

        print("PRINT TEST")
        print(self.pkdb.other_mg[pidn])
        print("-----")

        rg = None
        is_target = None
        # anti-target
        if type(pidn) == str:
            mgx = self.pkdb.other_mg[pidn]
            # case: null
            if mgx == 0:
                return None

            rg = ResourceGraph.from_MicroGraph(mgx)
            is_target = False

        # target
        else:
            rg = pidn.rg
            is_target = True

        # get the initial payoff info
        dn,de,at,nr = self.payoff_info_(rg,pmove,is_target)

        # distort-transform the payoff info
        dn,de = self.predictive_negochips_distorttransform(pidn,dn,de)#,rg)
        return dn,de,at,nr

    """
    helper method for `gauge_pmove_payoff`
    """
    def payoff_info_(self,rg:ResourceGraph,pmove:PMove,is_target:bool):

        # calculate the initial payoff sequence
        return pmove.gauge_payoff_seq_on_RG(rg,is_target)

    """
    A predictive function that transforms the predicted node 
    and edge payoffs of a PMove onto either another player (w/ 
    identity `pidn`) or the acting player w/ ResourceGraph `target_rg`, 
    based on the knowledge available to the owner with regards to its
    <NegoContainer>.
    """
    def predictive_negochips_distorttransform(self,pidn,dn,de):
        ##assert type(target_rg) in {ResourceGraph,type(None)}

        # convert the vector of distorted negochips to a map
        # node -> payoff
        mbm = defaultdict(float)
            # case: not self
        if self.pidn != pidn:
            q = self.suspected_negochips[pidn]
            print("Q")
            # player does not exact magnitude of negochip, assumes
            # DEFAULT_NEGOCHIP_MULTIPLIER 
            q2 = set()
            for (k,v) in q.items():
                if v == "distort":
                    mbm[k] = 1.0 / DEFAULT_NEGOCHIP_MULTIPLIER
            # case: self 
        else:
            qbm = self.nc.active_chips_by_info(pidn,"distort")
            for q_ in qbm:
                mbm[q_.loc] = q_.magnitude

        # distort-transform node payoffs
        for (k,v) in mbm.items():
            dn[k] = dn[k] * v

        # distort-transform edge payoffs
        for (k,v) in de.items():
            m1,m2 = v,v
            vx = k.split(",")
            if vx[0] in mbm:
                m1 = v * mbm[vx[0]]
            if vx[1] in mbm:
                m2 = v * mbm[vx[1]]
            m_ = (m1 + m2) / 2.
            de[k] = m_
        return dn,de

    """
    used for PMove hit registration
    """
    def actual_negochips_distorttransform(self,dn,de):
        dcm = self.nc.distort_coefficient_map(self.pidn)

        # transform the node delta values
        for k in dn.keys():
            f = 1.0
            if k in dcm: f = dcm[k]
            dn[k] = dn[k] * f

        # do the same for edge delta values
        for k in de.keys():
            m1,m2 = de[k],de[k]
            vx = k.split(",")
            if vx[0] in dcm:
                m1 = m1 * dcm[vx[0]]
            if vx[1] in dcm:
                m2 = m2 * dcm[vx[1]]
            de[k] = (m1 + m2) / 2.0
        return dn,de

    ##################### predictive information for <AMove> #############
    
    # TODO: rewrite
    """
    Defensive intelligence. 

    prediction of resource health is a pre-requisite to 
    calculating suggestions for improvement
    """
    def update_defint(self,owner_rg):
        return self.resource_risk_function(owner_rg)

    """
    performs predictive calculations for gauging the 
    effectiveness of two possible <AMoves> by the owner
    onto the other players, and stores them into
    <PDEC>
    """
    def gauge_amove_payoff(self,owner,other_players):

        # info for effects on other players
        q = self.prelim_gauge_amove_payoff(owner,other_players)
        if type(q) == type(None):
            print("NO AMOVE")
            return

        s1,s2,s3 = q
        s4 = self.amove_hitsurvivalrate__other_players(other_players)

        # declare the possible AMoves
        mg1,mg2 = self.generate_possible_AMove_target_graphs()
        am1 = AMove(mg1,deepcopy(self.gcs))
        am2 = AMove(mg2,deepcopy(self.gcs))

        s50 = self.amove_hitsurvivalrate__on_self(owner,am1)
        s51 = self.amove_hitsurvivalrate__on_self(owner,am2) 
        s5 = ((s50,am1),(s51,am2))

        self.pcontext.load_AMove_info(s1,s2,s3,s4,s5)
        return

    def amove_hitsurvivalrate__on_self(self,owner,am:AMove):
        
        # convert gcs to the player's MicroGraph
        q = owner.gcs[2]
        mgx = None
        if type(q) == type(None):
            mgx = deepcopy(owner.gcs[0])
        else:
            mgx = MicroGraph.isotransform_MG(owner.gcs[0],q)
        return self.def_int.hit_survival_rate_extremum_on_MG(mgx)

    """
    calculates the hit survival rate of the involved nodes&edges in `am`.
    """
    def amove_hitsurvivalrate__other_players(self,other_players:list):

        # obtain the (minimal,maximal) hit survival rate for all of `other_players`:
        #   player idn -> int::(hit survival) 
        dx = defaultdict(int)
            # case: private info
        if type(self.gcs[1]) == type(None):
            for o in other_players:
                dx[o.idn] = (None,None)

            # case: public info
        else:
            for o in other_players:
                x1,x2 = self.amove_hitsurvivalrate__on_player(o)
                dx[o.idn] = (x1,x2)
        return dx

    # TODO: inefficient
    """
    focuses on antipayoff target `at` belonging to `am` for the 
    player `p` of interest.

    Outputs the (min hit survival rate, max hit survival rate).
    """
    def amove_hitsurvivalrate__on_player(self,p):
        assert type(p) == Player

        # convert gcs to the player
        q = p.gcs[2]
        mgx = None
        if type(q) == type(None):
            mgx = deepcopy(p.gcs[0])
        else:
            mgx = MicroGraph.isotransform_MG(p.gcs[0],q)
        return p.pdec.def_int.hit_survival_rate_extremum_on_MG(mgx)

    # TODO: untested 
    """
    Preliminary gauge of AMove's effects on other players

    return:
    *if is_public*
    - dict, player::non_owner -> expected losses
    - dict, player::non_owner -> (vertex-edge score of AMove / owner's knowledge of the player's MG);
            determines ratio of loss to player
    - float, expected gains for owner's payoff target 
    *else*
    - None
    - dict, player -> (vertex-edge score of AMove / owner's knowledge of the player's MG)
    - None
    """
    def prelim_gauge_amove_payoff(self,owner,other_players):
        assert type(owner) == Player
        assert type(other_players) == list
        
        # case: no greatest common subgraph
        if type(self.gcs[0]) == type(None):
            return None

        s1,s2,s3 = None,defaultdict(float),None
        ves = self.gcs[0].ve_score()
        for o in other_players:
            mgx = self.pkdb.other_mg[o.idn]
            if type(mgx) == type(None):
                continue
            q = mgx.ve_score()
            q = q[0] + q[1]
            s2[o.idn] = ves / q if q != 0 else 0.

        # case: gauge of public info
        if type(self.gcs[1]) != type(None):
            s1,s3 = defaultdict(float),0.

            for o in other_players:
                q = self.gcs[1][o.idn]
                s1[o.idn] = q
                s3 += q
        return s1,s2,s3 

    """
    generates 2 possible AMove target graphs (MicroGraph). 
    """
    def generate_possible_AMove_target_graphs(self):
        
        # case: no hit survival rate info, so no generation.
        stat = len(self.def_int.node_hit_survival_rate) == 0
        if stat:
            return (None,None)

        # rank the nodes and edges in ascending order by
        # hit survival rate
        nhsr = sorted([(k,v) for (k,v) in self.def_int.node_hit_survival_rate.items()],\
            key=lambda x:x[1])
        ehsr = sorted([(k,v) for (k,v) in self.def_int.edge_hit_survival_rate.items()],\
            key=lambda x:x[1])

        # generate the 25-percentile graph
        l10 = round(0.25 * len(nhsr))
        l11 = round(0.25 * len(ehsr))
        mg1 = MicroGraph.minimal_MG_by_nodes_and_edges(\
            deepcopy(nhsr[:l10]),deepcopy(ehsr[:l11]))

        # generate the 75-th percentile graph
        l20 = round(0.75 * len(nhsr))
        l21 = round(0.75 * len(ehsr))
        mg2 = MicroGraph.minimal_MG_by_nodes_and_edges(\
            deepcopy(nhsr[l20:]),deepcopy(ehsr[l21:]))
        return (mg1,mg2)

    ########################## methods for <MMove>

    def gauge_mmove_payoff(self,minumum_node_health,minumum_edge_health,\
        mmove_payoff_dict):

        q = self.mmove_partial_iso_info(minumum_node_health,\
            minumum_edge_health,mmove_payoff_dict)

        if type(q) == type(None):
            self.pcontext.load_MMove_info(None)
            return 

        q2 = self.mmove_withdrawal_candidates()
        mhsr = self.def_int.minimal_hit_survival_rate()
        mds = MInfo(q,(q2[0],q2[1]),(q2[2],q2[3]),mhsr) 
        self.pcontext.load_MMove_info(mds)
        return
    
    """
    return:
    - 
    """
    def mmove_partial_iso_info(self,minumum_node_health,minumum_edge_health,\
        mmove_payoff_dict):

        # iterate through the possible moves and determine
        # the possible additions for each of the moves
        pinfs = self.pcontext.player_PMove_gauge(self.pidn)
        mxq = self.def_int.minimal_hit_survival_rate()

        if type(mxq[0]) == type(None):
            return None

        mx = min(mxq) 
        return mmove_addition_selection__type_1(pinfs,mmove_payoff_dict,\
            minumum_node_health,minumum_edge_health,mx)

    """
    return:
    - nodes with 1-hit survival rate
    - edges with 1-hit survival rate
    - nodes with 2-hit survival rate
    - edges with 2-hit survival rate
    """
    def mmove_withdrawal_candidates(self):
        # collect all nodes and edges with 1-hit survival rate
        ns1,es1 = self.def_int.nodes_and_edges_by_hit_survival_rate(1)

        # do the same for those w/ 2-hit survival rate
        ns2,es2 = self.def_int.nodes_and_edges_by_hit_survival_rate(2)
        return ns1,es1,ns2,es2

    ####################################################################

    # TODO: test. 
    """
    gauges a single player for NMove gains. 
    """
    def gauge_nmove_payoff(self,player,max_chip_number):
        ninego = self.gauge_nmove_negochip(player,max_chip_number)
        ninega = self.gauge_nmove_negachip(player,max_chip_number)    
        self.pcontext.pmove_prediction.extend([ninego,ninega])
        return

    # TODO: test
    def gauge_nmove_negochip(self,player,max_chip_number):
        ##return -1 
        q1,q2 = self.negochip_candidates(player)

        # gauge distort
        data = []
        for n in q1:
            x = self.gauge_negochip_improvement(player.idn,n,"distort")
            data.append((x,n,"distort"))

        # gauge deception
        if type(q2) != type(None):
            for n in q2:
                x = self.gauge_negochip_improvement(player.idn,n,"deception")
                data.append((x,n,"deception"))
        
        nhsr = sorted(data,key=lambda x: abs(x[0]),reverse=True)
        q = nhsr[:max_chip_number]
        nx1 = NInfo(True,player.idn,q)
        
    """
    gauge of <NegoChip> depends on `neg_type`:
    * `neg_type` = deception:
        uses DefInt.node_delta
    * `edge_type` = distort:
        uses DefInt.pmove_playernode_recep (other)
             OR DefInt.ea_self_target_node (self)

        if self: measures the delta of expected gains
        if other: measures the delta of expected losses 
    """
    def gauge_negochip_improvement(self,p_idn,n,neg_type):
        assert neg_type in NEGO_TYPES

        # program check
        if p_idn != self.pidn:
            assert neg_type != "deception"

        p_idn = None if p_idn == self.pidn else p_idn
        # case: deception
        if neg_type == "deception":
            return self.def_int.cumulative_nOrE_delta(n,True)
        # case: distort
        expected,actual = self.def_int.cumulative_expected_actual_of_move_by_node_info(p_idn,n)
        return (actual * DEFAULT_NEGOCHIP_MULTIPLIER) - actual

    # NOTE: gauge does not use precise mechanisms to produce
    #       numerical results
    """
    Helper method for `gauge_nmove_payoff`. Outputs the candidate
    nodes of `player` for NegoChip placement. 

    Used for analysis.

    - arguments:
    player := Player

    - return:
    available nodes for `distort`, available nodes for `deception`
    """
    def negochip_candidates(self,player):
        assert type(player) == Player

        # collect all known nodes for the player
        nl1,nl2 = None,None
            # case: self, distort and deception
            # starting nodeset is all nodes of player
        if player.idn == self.pidn:
            sn = set(player.rg.node_health_map.keys())
            nl1 = self.available_nodes_by_info(player.idn,"distort",sn)
            nl2 = self.available_nodes_by_info(player.idn,"deception",sn)
            # case: other
            # starting nodeset is known nodes of player
        else:
            qx = self.pkdb.other_mg[player.idn]
            sn = set()
            #   subcase: None
            if type(qx) == MicroGraph:
                sn = set(qx.dg.keys())
            nl1 = self.available_nodes_by_info(self.idn,"distort",sn)
        return nl1,nl2

    def gauge_nmove_negachip(self,player,max_chip_number):
        # get the negachip candidates
        q1 = self.negachip_candidates(player)
        
            # iterate through and calculate expected 
            # improvements
        expected_improvements = []
        for (k,v) in q1.items():
            f = self.gauge_negachip_improvement(player.idn,k,v)
            expected_improvements.append((f,k,v)) 

            # rank the samples
        nhsr = sorted(expected_improvements,key=lambda x: abs(x[0]),reverse=True)
        q = nhsr[:max_chip_number]
        nx1 = NInfo(False,player.idn,q)

    """
    fetches the cumulative (expected,actual) delta pair for the 
    node `n` belonging to `p_idn`. 

    Uses the global variable `DEFAULT_NEGOCHIP_DISTORT_MULT` to gauge
    the effects of a distortion NegoChip.

    If `neg_type` = distort:
        output (expected - expected / )
    Else:
        output (expected - actual)
    """
    def gauge_negachip_improvement(self,p_idn,n,neg_type):
        assert neg_type in NEGO_TYPES

        if self.pidn == p_idn:
            p_idn = None

        e,a = self.def_int.cumulative_expected_actual_of_move_by_node_info(\
            p_idn,n)

        if neg_type == "distort":
            return e - (e / DEFAULT_NEGOCHIP_MULTIPLIER)
        return e - a

    def negachip_candidates(self,player):
        candidates = deepcopy(self.pdec.suspected_negochips[player.idn])
        return candidates

    ##################### extraneous machine-learning functions #############
    # TODO: used to refine predictive capabilities.
    """
    expected-actual difference
    """
    def ea_diff(self):
        return -1

class Player:

    def __init__(self,rg,ms,idn = None,excess=1000,pcontext_mapper=None,\
        pml_type="full context"):
        # resource graph
        self.rg = rg
        # move sequence
        self.ms = ms
        # additional move sequence, of MMoves
        self.ams = []

        # register of isomorphisms from another player,
        # used to register another player's PMove hit
        # onto self. 
        self.iso_reg = None

        # NegoChip container
        # negochips held by the Player
        self.nc = NegoContainer()

        # player's idn
        self.idn = idn
        # excess resources
        self.excess_resources = excess 

        # the context mapper, used for learning
        assert type(pcontext_mapper) in {type(None),type(PContextMapper)}
        if type(pcontext_mapper) == type(None):
            self.pcontext_mapper = PContextMapper(StdDecFunction())
        else:
            self.pcontext_mapper = pcontext_mapper
        # player decision struct, used to store relevant information to make
        # decisions
        self.pdec = PDEC(idn,pcontext_mapper)
        # player move log
        self.pml = PMLog(pml_type)

    ########################## instantiation and display methods

    @staticmethod
    def generate(i:int,idn:int,nm:int,rg_args,excess,pcm):
        assert type(i) in {type(None),int}
        if type(i) == int:
            random.seed(i)

        # generate the resource graph
        rg = ResourceGraph.generate__type_stdrand(i,rg_args[0],rg_args[1],rg_args[2])
        rg = default_rg_value_assignment(rg,DEFAULT_NODE_HEALTH_RANGE)
        moveseq = [] 
        for j in range(nm):
            pidn = str(j)
            pm = PMove.generate__type_default(i,DEFAULT_MOVE_PAYOFF_RANGE,pidn)
            moveseq.append(pm)

        moveseq = generate_PMoveSeq__type_assymetric_unit(nm,DEFAULT_MOVE_PAYOFF_RANGE)

        # generate the moves 
        return Player(rg,moveseq,idn,excess,pcm)

    def display_context(self,context_displays=["PMove","AMove","MMove"]):
        assert type(context_displays) == list

        if type(self.pdec.pcontext) == type(None):
            print("NADA")
            return

        print(context_displays)
        for c in context_displays:
            assert c in PCONTEXT_DISPLAY_TYPES
            print("* CONTEXT: {}".format(c))
            if c == "PMove":
                for (k,v) in self.pdec.pcontext.pmove_prediction.items():
                    print("move idn {}".format(k))
                    for (k2,v2) in v.items():
                        print("affector {}".format(k2))
                        print(str(v2))
                    print("--------------------")
            elif c == "AMove":
                if type(self.pdec.pcontext.amove_prediction) == type(None):
                    print("NADA")
                    continue

            elif c == "MMove":
                if type(self.pdec.pcontext.mmove_prediction) == type(None):
                    print("NADA")
                    continue 
                print(str(self.pdec.pcontext.mmove_prediction))
                print("--------------------")
            else:
                print("TODO") 
                return -1
        return -1

    def display_PMoves(self):
        for x in self.ms:
            print(str(x)) 

    ########################## methods for decep/distort
    # NOTE: 
    """
    projects an outward appearance to another Player, used
    in deception.

    return:
    - Player, dummy visual
    """
    def project_out(self):
        chips = self.active_chips_by_info(self.idn,"deception")
        cx = [c.loc for c in chips]
        rgx = self.rg.deception_complement(cx)
        return Player(rgx,None,self.idn,0)

    def one_gauge_PMove(self,p,move_index:int):
        assert type(p) == Player 
        self.pdec.gauge_pmove_payoff(p,deepcopy(self.ms[move_index]))
        return 

    def one_gauge_AMove(self,other_players):
        self.pdec.gauge_amove_payoff(self,other_players)

    def one_gauge_MMove(self):

        # get the minumum health
        mnh = min([v for v in self.rg.node_health_map.values()])
        meh = min([v for v in self.rg.edges_health_map.values()])

        # get the mmove payoff
        x = [(m.pm_idn,m.payoff) for m in self.ms]
        mpd = defaultdict(None,x)
        self.pdec.gauge_mmove_payoff(mnh,meh,mpd) 

    def one_gauge_NMove(self,other_players):
        l = int(round(len(self.rg.node_health_map) / 3))

        for p in other_players:
            self.gauge_nmove_payoff(p,l)
        return
    
    ######### register XMoves
    #######################################################

    """
    register the delta values of PMove at `p_index` onto `rg`. 

    arguments:
    - p_index := index of PMove taken
    - node_dm := change of 
    """
    def register_PMove(self,p_index, exp_node_dm,exp_edge_dm,node_dm,edge_dm):
        self.rg.update_health_values(node_dm,edge_dm)

        # merge the dictionaries
        ea_nodes = merge_dictionaries__concatenate(exp_node_dm,node_dm) 
        ea_edges = merge_dictionaries__concatenate(exp_edge_dm,edge_dm)
        pm_idn = self.ms[p_index].pm_idn

        # add the expected/actual maps to <DefInt>    
        self.def_int.add_sample_ea(pm_idn,ea_nodes,ea_edges)

        return

    """
    observes the variables 
    `pdec.def_int` -> (pmove_playernode_recep,
                    pmove_playeredge_recep,
                    ea_self_target_node,
                    ea_self_target_edge) 
    to deduce the possible locations of negochips based on
    the expected and actual performance of nodes of interest (self|other). 
    """
    def post_analysis_PMove__negochip_deduction(self,p_index):

        q1 = self.pdec.def_int.pmove_playernode_recep[p_index]
        
        # calculate suspected chips for others
        for (player_idn,nddata) in q1.items():
            qx = negochip_locations__simple_deduction(deepcopy(nddata),False)
            self.pdec.suspected_negochips[player_idn] = qx

        # calculate suspected chips for self
        nddata = deepcopy(self.pdec.def_int.ea_self_target_node)
        qx = negochip_locations__simple_deduction(nddata,True)
        self.pdec.suspected_negochips[self.idn] = qx

    """
    registers the effects of PMove on another player
    """
    def register_PMove_anti(self,pm_idn,p_idn,ea_ndm,ea_edm):
        self.pdec.def_int.add_antisample(pm_idn,\
            p_idn,ea_ndm,ea_edm)
        return

    
    # INFO: 
    # NOTE: 
    # CAUTION
    # WRONG
    """
    register the delta values by another player's PMove (unknown) 
    onto `rg`. 

    return: 
    - (expected,actual) node deltas, (expected,actual) edge deltas,
       MicroGraph perceived by acting player (other) if `record_mg`=True. 
    *output is to be registered by another Player if in public mode *
    """
    def register_PMove_hit(self,attacker,pmove,record_mg=False):
        assert type(attacker) == Player 
        assert type(pmove) == PMove

        # fetch the image of the ResourceGraph
        rgx = self.project_out()
        mgx = MicroGraph.from_ResourceGraph(pmove.antipayoff_target)

        # calculate isomorphic attack on image
        self.iso_reg = rgx.subgraph_isomorphism(mgx,True)
        ndm,edm,endm,eedm,pmgx = self.ne_delta_map_of_iso_reg(mgx,pmove.antipayoff,record_mg)
        
        # log the deltas for self
        ndm_,edm_ = self.pdec.actual_negochips_distorttransform(ndm,edm)
        self.log_PMove_hit(attacker.idn,ndm_,edm_)

        # get the expected values by attacker
        ndmq,edmq = attacker.predictive_negochips_distorttransform(self.idn,ndm,edm)

        # construct the output
        node_eapair,edge_eapair = defaultdict(None),defaultdict(None)
        for (k,v) in ndmq.items():
            node_eapair[k] = (v,ndm_[k])

        for (k,v) in edmq.items():
            edge_eapair[k] = (v,edm_[k])
        return node_eapair,edge_eapair,pmgx

    """
    NOTE: method is similar to <PMove.gauge_payoff_seq_on_RG>,
          except does not register elements that do not actually
          exist in the player's ResourceGraph due to <NegoChip.deception>
          instances.
    """
    def ne_delta_map_of_iso_reg(self,mv_mgx,payoff,record_mg=False):
        node_dm = defaultdict(float)
        edge_dm = defaultdict(float) 
        exp_node_dm = defaultdict(float)
        exp_edge_dm = defaultdict(float)

        mgx = MicroGraph() if record_mg else None  
        while len(self.iso_reg) > 0:
            x = self.iso_reg.pop(-1)
            si2 = pairseq_to_dict(x)
            g2 = rg.isomap_to_isograph(deepcopy(mv_mgx),si2)
            ndm,edm,endm,eedm = self.delta_of_iso(g2,payoff)
            node_dm = merge_dictionaries__additive([node_dm,ndm])
            edge_dm = merge_dictionaries__additive([edge_dm,edm])
            exp_node_dm = merge_dictionaries__additive([exp_node_dm,endm])
            exp_edge_dm = merge_dictionaries__additive([exp_edge_dm,eedm])

            if record_mg:
                mgx = mgx + g2 
        return node_dm,edge_dm,exp_node_dm,exp_edge_dm,mgx

    """
    the rule for `iso_mgx` to register an effect on 
    Player's ResourceGraph:
    - for every node n present in `iso_mgx`, all of its neighbors 
      must exist in the Player's ResourceGraph. If any edge with 
      node n in `iso_mgx` is not present in `rg`, then `iso_mgx`
      does not register a delta with respect to target node n.
    - NOTE: the expected deltas do not consider the impacts of
            deceptive NegoChips.

    return:
    - node delta map, edge delta mpa,
      expected node delta map, expected edge delta map
    """
    def delta_of_iso(self,iso_mgx,payoff):
        mg_ref = MicroGraph.from_ResourceGraph(self.rg)
        ndm,edm = defaultdict(float),defaultdict(float)
        endm,eedm = defaultdict(float),defaultdict(float)
        tmpn,tmpe = defaultdict(float),defaultdict(float)

        for (k,v) in iso_mgx.dg.items():
            # check existence of nodes
            q = mg_ref.dg[k]
            stat = v.issubset(q)

            tmpn.clear()
            tmpe.clear()

            for v_ in v:
                tmpn[k] += payoff
                tmpn[v_] += payoff                
                tmpe[k + "," + v_] += payoff

            endm = merge_dictionaries__additive([tmpn,endm])
            eedm = merge_dictionaries__additive([tmpe,eedm])

            # register the hit if valid node recognition
            if stat:
                ndm = merge_dictionaries__additive([tmpn,ndm])
                edm = merge_dictionaries__additive([tmpe,edm])
        return ndm,edm,endm,eedm

    def log_PMove_hit(self,p_idn,node_dm,edge_dm):
        # add the sample to PDEC's var<DefInt>
        self.pdec.def_int.add_sample(p_idn,node_dm,edge_dm)

        # register the PMove hit
        self.rg.update_health_values(node_dm,edge_dm)

        # clean out all negative elements from <ResourceGraph>
        nr,er = self.rg.clean_graph() 

        # update PDEC's var<DefInt> according to the nodes,edges
        # removed
        self.pdec.def_int.remove_deceased(nr,er)
        return

    #############################################################

    def assign_GCS(self,mg:MicroGraph,player_health_impact,isomap):
        assert type(mg) == MicroGraph
        assert type(player_health_impact) in {type(None),defaultdict}
        assert type(isomap) in {type(None),defaultdict} 
        self.pdec.gcs[0] = mg
        self.pdec.gcs[1] = player_health_impact
        self.pdec.gcs[2] = isomap

    def cumulative_RG_health():
        return self.rg.cumulative_health()

    #################### remove or delete
    #######################################################

    def update_PKDB(self,other_player_pidn,pmgx):
        assert type(pmgx) == MicroGraph
        self.pdec.pkdb.modify_MG(other_player_pidn,pmgx)

    """
    ranks the PMove by priorities using the following
    variables:

    (1) V+E scores of move's target and antitarget graphs;
        (V+E scores correspond to isomorphic attack coverage).

    (2) Cumulative difference of expected and actual
    *DefInt*
    - pmove_playernode_recep
    - pmove_playeredge_recep
    - ea_self_target_node
    - ea_self_target_edge
    
    (3) Most recent occurrence of PMove (according to <PMLog>)

    For all three of the above variables, lower scores correspond to
    higher ranks.
    """
    def pmove_priorities(self):
        # fetch the values
        mpself,mpothers = self.ve_dualscores_of_moves()
        eaself,eaothers = self.diff_ea_of_moves()
        mro = self.most_recent_move_occurrences()

        # rank the values for
        rx1 = rank_stddict_floatvalues(mpself)
        rx2 = rank_stddict_floatvalues(mpothers)
        rx3 = rank_stddict_floatvalues(eaself)
        rx4 = rank_stddict_floatvalues(eaothers)
        rx5 = rank_stddict_floatvalues(mro)
        q = merge_dictionaries__additive([rx1,rx2,rx3,\
            rx4,rx5])
        rx = rank_stddict_floatvalues(q)
        return rx

    """
    """
    def ve_dualscores_of_moves(self):
        map_self,map_others = defaultdict(float),defaultdict(float)

        for m in self.ms:
            mg1 = MicroGraph.from_ResourceGraph(m.payoff_target)
            mg2 = MicroGraph.from_ResourceGraph(m.antipayoff_target)

            ns1,es1 = mg1.ve_score()
            ns2,es2 = mg2.ve_score()

            map_self[m.pm_idn] = (ns1,es1)
            map_others[m.pm_idn] = (ns2,es2)
        return map_self,map_others

    """
    return:
    - self difference of cumulative expected/actual from PMoves,
      others difference of cumulative expected/actual from PMoves.
    """
    def diff_ea_of_moves(self):
        # get all the PMove idns.
        pm_idns = [m.pm_idn for m in self.ms]

        # self
        eadiff_self = defaultdict(float)
        for pm in pm_idns:
            exp,act = self.pdec.def_int.cumulative_expected_actual_of_move(pm,True)
            eadiff_self[pm] = exp - act

        # others
        eadiff_others = defaultdict(float)
        for pm in pm_idns:
            exp,act = self.pdec.def_int.cumulative_expected_actual_of_move(pm,True)
            eadiff_others[pm] = exp - act
        return (eadiff_self,eadiff_others)

    def most_recent_move_occurrences(self):
        # iterate through each of the PMoves and check
        # for their most recent execution
        most_recent_move_indices = {}
        for x in self.ms:
            gmd = GenericMoveDesc.from_XInfo(x)
            i = self.pml.most_recent_artifact(gmd)
            most_recent_move_indices[x.pm_idn] = i
        return most_recent_move_indices

    def choose(self):
        # dummy choice is PMove
        return -1


    """
    used for testing
    """
    def dummy_move(self):
        # choose a random move
        ri = random.randrange()

    # TODO: add more
    def postmove_update(self):
        self.pdec.def_int.hit_survival_rate_hypothesis(self.rg)

    # TODO: 
    def apply_negachip(self):
        return -1