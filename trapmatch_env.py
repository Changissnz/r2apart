from rules import *
from pmove import * 

"""
outputs a random ordering of integers in the
range [0,i). 
"""
def random_ordering(i:int):
    q = [j for j in range(i)]
    random.shuffle(q)
    return q

class FARSEInfo:

    def __init__(self,parent_tmenv_idn,reference_timestamp,reference_timestamp_2,\
        training_player_decision_index,target_hopseq):
        self.pmi = parent_tmenv_idn
        self.rt = reference_timestamp
        self.rt2 = reference_timestamp_2
        self.tpdi = training_player_decision_index
        self.th = target_hopseq


"""
environment for the trap-matching multi-player
game
"""
class TMEnv:

    """
    """
    def __init__(self,players,game_mode_1,game_mode_2,verbose = True):
        assert game_mode_1 in GAME_MODES
        assert game_mode_2 in GAME_MODES2
        self.players = players
        self.game_mode_1 = game_mode_1
        self.game_mode_2 = game_mode_2
        self.verbose = verbose

        # ordering of players for the previous timestamp
        self.ts_ordering = None

        # preferred-move mode, used for testing
        self.preferred_move = None

        # set the player verbosity
        self.set_player_verbosity(self.verbose)
        self.idn = None
        self.timestamp = 1

        # used for training by FARSE
        self.fi = None
        return

    def set_player_verbosity(self,v):
        self.verbose = v
        for p in self.players:
            p.verbose = self.verbose
    
    def assign_idn(self,idn):
        self.idn = idn

    def assign_reference_timestamp(self,rt):
        self.reference_timestamp = rt

    """
    dumb generation assigns no PContextMapper to any player.
    """
    @staticmethod
    def generate__type_dumb(i:int,num_players,num_moves_range,\
        drange,connectivity_range,excess_range,game_modes,verbose):
        # TODO: refactor
        assert type(i) in {type(None),int}
        if type(i) == int:
            random.seed(i)
        
        players = []
        for j in range(num_players):
            nm = random.randint(num_moves_range[0],num_moves_range[1])
            rg_args = []
            rg_args.append(random.randint(drange[0],drange[1]))
            rg_args.append(deepcopy(connectivity_range))
            rg_args.append(deepcopy(DEFAULT_NODE_HEALTH_RANGE))
            excess = random.randint(excess_range[0],excess_range[1])
            print("generating for {} with args {}".format(j,rg_args))
            p = Player.generate(None,str(j),nm,rg_args,excess,None)
            players.append(deepcopy(p))
        return TMEnv(players,game_modes[0],game_modes[1],verbose) 

    def random_move_type_deterministic_assignment(self,num_moves_range=[0,4]):
        assert len(num_moves_range) == 2
        assert num_moves_range[0] <= num_moves_range[1]

        for p in self.players:
            q = deepcopy(DEFAULT_PLAYER_MOVE_INFO_TYPES)
            nm = random.randint(num_moves_range[0],num_moves_range[1])
            q1 = []
            while nm > 0:
                nmx = random.randint(0,len(q) - 1)
                q1.append(q[nmx])
                nm -= 1
            p.move_type_deterministic = q1 
            print("set player {}".format(p.idn))
        return

    def move_type_deterministic_assignment(self,move_types):

        for p in self.players:
            p.move_type_deterministic = deepcopy(move_types)

    def idn_to_player(self,idn):
        i = self.idn_to_index(idn)
        if i == -1:
            return None
        return self.players[i]
    
    def idn_to_index(self,idn):
        for (i,x) in enumerate(self.players):
            if x.idn == idn: return i
        return -1

    def set_ts_ordering(self):
        # set the ordering
        self.ts_ordering = random_ordering(len(self.players))

        # calculate the greatest common subgraph
        self.gcs_exec()

    """
    moves one timestamp by a random ordering of the Players
    in the game. 
    """
    def move_one_timestamp(self):
        self.set_ts_ordering()
        if len(self.ts_ordering) <= 1:
            print("DONE")
            return 
        # convert the ordering to identifiers
        idns = []
        for tso in self.ts_ordering:
            idns.append(self.players[tso].idn) 

        for i in idns:
            q = self.idn_to_index(i)
            # case: player has been terminated
            if q == -1:
                continue
            self.move_one_player(q)
        self.clear_timestamp_data()
        return

    """
    """
    def move_one_player(self,p_index:int,mi = None):
        # feed player info
        self.feed_moving_player_info(p_index)

        # case: move-type deterministic mode
        if type(self.players[p_index].move_type_deterministic) != type(None):
            print("type-deterministic for player {}".format(self.players[p_index].idn))
            mi = self.player_choice_move_deterministic_mode(p_index)
        # case: preferred-move mode 
        elif type(self.preferred_move) != type(None):
            # get the index of preferred move
            j = self.players[p_index].pdec.pcontext.pcd.index_of_move(self.preferred_move)
            # subcase: index is -1, allowed player to choose
            if j == -1:
                mi = self.players[p_index].choose()
            else:
                mi = self.players[p_index].pdec.pcontext.pcd.ranking[j]
        # case: allow player to decide
        elif type(mi) == type(None):
            mi = self.players[p_index].choose()

        self.exec_player_choice(p_index,mi)
        if self.verbose: print("====================================")
        return

    def player_choice_move_deterministic_mode(self,p_index):
        assert type(self.players[p_index].move_type_deterministic) != type(None)
        return self.random_player_choice_of_type(p_index,\
            self.players[p_index].move_type_deterministic)
        
    """
    iterates through and feed the player info.
    for its move
    """
    def feed_moving_player_info(self,p_index:int):
        # calculate PMove info
        if self.verbose: print("\t * GAUGING PMOVES FOR PLAYER {}".format(self.players[p_index].idn))
        self.all_pmove_AND_player_combos(p_index)

        # calculate AMove info
        other_players = [p for (i,p) in enumerate(self.players)]
        self.players[p_index].one_gauge_AMove(other_players)

        # calculate MMove info
        self.players[p_index].one_gauge_MMove()

        # calculate NMove info
        self.players[p_index].one_gauge_NMove(other_players)

        if self.verbose:
            print("-- Context acquired")
            self.players[p_index].display_context()

        # rank the moves by context-mapper
        sdf = self.players[p_index].pdec.context_mapper.sdf
        self.players[p_index].pdec.pcontext.std_dec_func_proc(sdf,self.verbose)


        return

    def all_pmove_AND_player_combos(self,p_index:int):
        p = self.players[p_index]

        # rank the priority of PMoves
        priorities = p.pmove_priorities()
        pseq = [(k,v) for (k,v) in priorities.items()]
        pseq = sorted(pseq,key=lambda x:x[1])
        
        # NOTE: var<DEFAULT_PMOVE_MAX_GAUGES> not used
        ##pseq = pseq[:DEFAULT_PMOVE_MAX_GAUGES]
        
        if self.verbose: 
            print("player {} PMove gauges: {}".format(self.players[p_index].idn,pseq))

        # iterate through the highest ranking PMoves and
        # gauge them
        for x in pseq:
            # get the move index of x[0]
            mi = p.pmove_idn_to_index(x[0])
            assert mi != -1
            for px in self.players:
                p.one_gauge_PMove(px,mi)

        # infer remaining moves if PMLog mode `full context` is on
        if p.pml.lt != "full context":
            if self.verbose: print("no inference for rest of PMove gauges")
            return

        idn_mvs = set([x.pm_idn for x in p.ms])
        q = set([x[0] for x in pseq])
        idn_mvs = idn_mvs - q

        for q in idn_mvs:
            index,pmp = p.pml.most_recent_PMove_gauge(q)
            if index == -1:
                continue
            p.pdec.pcontext.pmove_prediction[q] = pmp
        return

    # TODO: untested
    """
    Used to find the greatest common subgraph
    shared amongst the graphs. 
    """
    def gcs_exec(self,search_type = "full neighbor fit- type 2"):
        if self.verbose: print("* Finding the G.C.S")

        # convert each of the players' RG to their MG's.
        mgs = []
        for x in self.players:
            mg = MicroGraph.from_ResourceGraph(x.rg)
            mgs.append(mg)
        ##print("** number of micrographs: {}".format(len(mgs)))

        # obtain the solution for greatest
        # common subgraph
        gcsc = GCSContainer(deepcopy(mgs),search_type)
        gcsc.initialize_cache() 
        s1,s2 = gcsc.search(DEFAULT_GCS_SEARCH_CANDIDATE_SIZE)
        ##
        """
        print("S1")
        print(s1)
        print("S2")
        print(s2)
        """
        ##

        # insert a blank at the index of the reference
        if type(gcsc.reference_index) == type(None):
            print("no <AMove>")
            return
        if type(s1) == type(None): 
            print("no <AMove>")
            return

        s1[0].insert(gcsc.reference_index,None)

        # convert the solution to the reference micrograph
        mg = GCSContainer.solution_to_MG(s1)

        x1 = None
        if self.game_mode_2 == "public":
            x1 = defaultdict(float)

            # iterate through the search solution and
            # calculate the equivalent subgraph based
            # on mg, then determine the cumulative
            # health of each of those cumulative equivalents
            for (i,x) in enumerate(s1[0]):
                mg2 = deepcopy(mg)

                # case: not the reference
                if type(x) != type(None):
                    mg2 = MicroGraph.isotransform_MG(mg2,x)                
                h = self.players[i].rg.cumulative_ne_health_by_mg(mg2)
                x1[self.players[i].idn] = h

        # assign the result to each of the players
        for (i,p) in enumerate(self.players):
            p.assign_GCS(deepcopy(mg),deepcopy(x1),s1[0][i])
        return 

    #####################################

    def exec_player_choice(self,player_index,c):
        # case: PMove
        if self.verbose: 
            print("CHOISHA: ",c)

        x = self.players[player_index].parse_rank_decision(c)

        if "PInfo" in c[0]:
            self.exec_PMove(player_index,x)
        elif "AInfo" in c[0]:
            self.exec_AMove(player_index,x)
        elif "MInfo" in c[0]:
            self.exec_MMove(player_index,x)
        elif "NInfo" in c[0]:
            self.exec_NMove(player_index,x)
        else:
            assert False

        if self.verbose: print("RECORDIONOS")
        # record player move into its log
        self.players[player_index].record_into_pml()

        # remove all deceased players from move
        self.remove_deceased()

        # run post-round calculations for remaining players
        self.post_round_calculations()
        return

    """
    outputs a random player choice
    """
    def random_player_choice_of_type(self,player_index,wanted_types):
        indices = self.indices_for_possible_move_types(player_index,wanted_types)

        # case: no wanted types available, choose random
        if len(indices) == 0:
            ri = random.randint(0,len(self.players[player_index].pdec.pcontext.pcd.ranking) - 1)
        else:
            ri = random.randint(0,len(indices) - 1)
            ri = indices[ri]
        c = self.players[player_index].pdec.pcontext.pcd.ranking[ri]
        return c

    """
    search through the <PContext>s 
    """
    def indices_for_possible_move_types(self,player_index,wanted_types):
        indices = []
        for (i,m) in enumerate(self.players[player_index].pdec.pcontext.pcd.ranking):
            r = m[0].split("-")
            r = r[0]
            r = r.split("#")
            r = r[0] 
            if r in wanted_types:
                indices.append(i)
        return indices

    # TODO: test
    def exec_PMove(self,player_index,pmove_index):
        # register the PMove onto self
        rg = self.players[player_index].rg
        pmove = self.players[player_index].ms[pmove_index]
        edn,ede,at,nr = self.players[player_index].pdec.payoff_info_(\
            rg,pmove,True)
        dn,de = self.players[player_index].pdec.actual_negochips_distorttransform(\
            edn,ede,self.players[player_index].idn) 

        self.players[player_index].register_PMove(pmove_index,edn,ede,dn,de)

        # register the PMove onto all other active players
        record_mgx = True if pmove_index in {2,3} else False

        idn = self.players[player_index].idn
        for i in range(len(self.players)):
            if i == player_index: continue
        
            neap,eeap,pmgx = self.players[i].register_PMove_hit(self.players[player_index],\
                pmove,record_mgx)
            self.players[player_index].register_PMove_anti(\
                pmove_index,self.players[i].idn,neap,eeap)

            # update the PKDB if pmove_index in {2,3}
            if record_mgx:
                reupdate = True if pmove_index == 2 else False
                self.players[player_index].update_PKDB(self.players[i].idn,pmgx,reupdate)

        # conduct the post-analysis of the PMove to hypothesize on
        # negochip locations 
        self.players[player_index].post_analysis_PMove__negochip_deduction(pmove_index)
        return

    # TODO: test 
    def exec_AMove(self,player_index,amove:AMove):
        f = 0.
        for (i,x) in enumerate(self.players):
            if i == player_index: continue
            f += x.register_AMove_hit(amove)
        self.players[i].register_AMove(amove,f)
        return

    # TODO: test 
    def exec_MMove(self,player_index,mmove:MMove):
        self.players[player_index].register_MMove(mmove)
        return

    # TODO: test 
    def exec_NMove(self,player_index,nmove:NMove):
        px = self.idn_to_player(nmove.destination_player)
        if self.verbose: print("-- executing NMove by {} on {}".format(self.players[player_index].idn,nmove.destination_player))
        self.players[player_index].register_NMove(nmove,px) 
        return

    def post_round_calculations(self):
        # update the hit survival rate for the player
        for i in range(len(self.players)):
            self.players[i].postmove_update()

    def remove_deceased(self):
        # idn's for deceased players
        deceased_players = set()
        deceased_player_indices = set()

        # collect deceased players
        for (i,x) in enumerate(self.players):
            if x.is_deceased():
                deceased_players |= {x.idn}
                deceased_player_indices |= {i}

        # remove deceased players
        pxs = [p for (i,p) in enumerate(self.players) if i not in deceased_player_indices]
        self.players = pxs

        # remove information for all deceased players in
        # each surviving player's infobase
        for p in self.players:
            for x in deceased_players:
                p.remove_deceased_player(x)
        return

    """
    clears timestamp data for all remaining players,
    primarily the PContext structure. 
    """
    def clear_timestamp_data(self):
        for p in self.players:
            p.pdec.pcontext = None 
        return

    def save_state(self,fp,write_mode="wb"):
        q = DEFAULT_TRAINING_FOLDER + fp
        f = open(q,write_mode)
        pickle.dump(self,f)

    @staticmethod
    def open_state(fp):
        f = open(DEFAULT_TRAINING_FOLDER + fp,"rb")
        q = pickle.load(f)
        assert type(q) == TMEnv, "type {} is not <TMEnv>".format(type(q))
        return q
