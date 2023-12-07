# TODO: careful with <MoveAssemblerType1>; may result in infinite search for improvement (most likely from
#       partial add). 
# file contains classes responsible for calculating moves of player
from player import *


## ARCHITECTURE OF FILE:
## MoveAssemblerType1
##      MicroGraphAssemblySolution: finalized solution to assembly
##          AssemblyInstruction: unit used to describe how a graph is assembled into the solution

"""
tmg := PMove, wanted move
smg := MicroGraph, add move to it so that it equals `tmg`
mv := PMove, move to add to `smg` to equal `tmg`'s target or anti-target
n2tmap := node-to-target node map
target := bool, specifies whether the focus is the target or anti-target of move 
"""
def brute_force_search_node_assignment__full_add(tmg,smg,mv,target:bool,ve_fitscore=ve_fitscore_type1):#n2tmap,target:bool):
    # assign variables based on target/anti-target
    mvt,mvtw = None,None
    node_idn_ctr = None 
    
    if target:
        mvt = MicroGraph.from_ResourceGraph(mv.payoff_target)
        mvtw = MicroGraph.from_ResourceGraph(tmg.payoff_target)

        all_nodes = set(smg.dg.keys()) | set(mvtw.dg.keys())
        greatest_node = max([int(i) for i in all_nodes])
        node_idn_ctr = DefaultNodeIdnCounter(str(greatest_node)) 
    else: 
        mvt = MicroGraph.from_ResourceGraph(mv.antipayoff_target)
        mvtw = MicroGraph.from_ResourceGraph(tmg.antipayoff_target)

        all_nodes = set(smg.dg.keys()) | set(mvtw.dg.keys())
        greatest_node = max([int(i) for i in all_nodes])
        node_idn_ctr = DefaultNodeIdnCounter(str(greatest_node)) 

    # rank the nodes of `mv` target or anti-target by degree 
    nct = mvt.neighbor_count()
    nctvec = sorted([(k,v) for (k,v) in nct.items()],key=lambda x:x[1]) 
    nctvec = [nv[0] for nv in nctvec]

    # get the possible nodes of wm for assignment
    ##wmt = set([k for k in mvtw.dg.keys()]) - set(smg.dg.keys()) 
    wmt = set()
    for k in mvtw.dg.keys():
        s1 = smg.dg[k]
        s2 = mvtw.dg[k]
        s3 = s1 & s2

        if s2 != s3:
            wmt = wmt | {k}

    # every element in deque is of the form:
    ## running MicroGraph solution
    ## node assignment
    ## head nodes to search for next neighbor
    ## remaining nodes of nctvec for assignment
    ## remaining nodes of wanted move for assignment
    ## node idn counter
    search_list = deque() 
    search_list.append([deepcopy(smg),defaultdict(str),[],nctvec,wmt,node_idn_ctr])
    best_soln,best_assignment,best_score = None,None,float('inf')

    while len(search_list) > 0:
        ##print("length of search list: ",len(search_list))
        # get the next reference node (of move) to assign
        q = search_list.popleft()
        ##print("Q3: ", q[3])
        # case: no head, assign head or terminate
        if len(q[2]) == 0: 
            # case: terminate
            if len(q[3]) == 0:
                """
                print("checking solution: ")
                if best_soln != None: print(best_soln.dg)
                print("score")
                print(best_score)
                print("best assignment")
                print(best_assignment)
                """ 
                
                # TODO: determine the best soln here 
                score = ve_fitscore(smg + q[0],mvtw)
                if score < best_score:
                    best_soln = q[0] 
                    best_assignment = q[1] 
                    best_score = score 
                continue

            # case: add a head
            h = q[3].pop(0)
            q[2].append(h) 
 
        # get the neighbors of a head
        h = q[2].pop(0)
        neighbors_h = mvt.dg[h]

        # determine the neighbors that have already been assigned
        assigned_neighbors_h = neighbors_h & set(q[1].keys())
        assigned_neighbors_reftarget = [q[1][a] for a in assigned_neighbors_h]

        # case: head does not have an assignment, assign head to a node, and add
        #       the search candidate back into cache. 
        if h not in q[1]:

            # subcase: no assigned neighbors, any of q[4] would work
            if len(assigned_neighbors_reftarget) == 0:
                for x in q[4]:
                    e0 = deepcopy(q[0])
                    e0.dg[x] = set()
                    e1 = deepcopy(q[1])
                    e1[h] = x
                    e2 = deepcopy(q[2]) 
                    ##print("E2: ",h)
                    ##print(e2)
                    e2.insert(0,h)
                    e3 = deepcopy(q[3])
                    if h in e3: e3.remove(h)
                    e4 = deepcopy(q[4])
                    e4 = e4 - {x}
                    e5 = deepcopy(q[5])
                    search_list.appendleft([e0,e1,e2,e3,e4,e5])

            # determine the subset of nodes of q[4] that intersect
            else: 
                # get the first set of possible neighbors
                fx = assigned_neighbors_reftarget[0]
                qn = mvtw.dg[fx] & q[4]

                # iterate through the rest for intersection op.
                for fx_ in assigned_neighbors_reftarget[1:]:
                    qn = qn & (mvtw.dg[fx] & q[4])

                # make a search candidate for each element in qn
                for qn_ in qn:
                    q2 = deepcopy(q)
                    q2[0].dg[qn_] = deepcopy(assigned_neighbors_reftarget)
                    for anr in assigned_neighbors_reftarget:
                        q2[0].dg[anr] = q2[0].dg[anr] | {qn_}
                    q2[1][h] = qn_
                    q2[2].insert(h,0)
                    q2[4] = q2[4] - {qn_}
                    search_list.appendleft(q2)
            continue

        # case: head already has an assignment, move on to assigning its neighbors
        hwn = q[1][h]

        # iterate through the assigned neighbors and add edges to hwn
        for anh in assigned_neighbors_reftarget:
            q[0].dg[anh] = q[0].dg[anh] | {hwn}
            q[0].dg[hwn] = q[0].dg[hwn] | {anh} 

        neighbors_h_to_assign = list(neighbors_h - assigned_neighbors_h)
        search_list2 = deque([[deepcopy(q), deepcopy(neighbors_h_to_assign)]])
        ##print("TYPE: ", type(search_list2[0][1]))
        while len(search_list2) > 0:
            sl2 = search_list2.popleft()
            ##print("W: ", sl2[1])

            # case: done, add to search candidates
            if len(sl2[1]) == 0:
                search_list.appendleft(sl2[0])
                continue

            # case: not done, assign one neighbor to all possibilities, and add those
            #           candidates back into search_list2
            ns = sl2[1].pop(0)
            ##available = mvtw.dg[hwn] - set(sl2[0[0].dg.keys()) - set(sl2[0][1].values()) 
            available = mvtw.dg[hwn] - set(sl2[0][1].values()) 

            # subcase: none available, make extraneous
            if len(available) == 0:
                nn = next(sl2[0][5])
                sl2[0][0].dg[nn] = {hwn}
                sl2[0][0].dg[hwn] = sl2[0][0].dg[hwn] | {nn}
                sl2[0][1][ns] = nn
                sl2[0][3].remove(ns)
                search_list2.appendleft(sl2) 
                continue

            for a in available:
                # make a copy of candidate and update
                sl3 = deepcopy(sl2)
                sl3[0][0].dg[a] = {hwn}
                sl3[0][0].dg[hwn] = sl3[0][0].dg[hwn] | {a}
                sl3[0][1][ns] = a
                sl3[0][2].append(ns)
                ##print("SL0,3: ", sl3[0][3])
                if ns in sl3[0][3]: sl3[0][3].remove(ns)
                if a in sl3[0][4]: sl3[0][4].remove(a)
                search_list2.appendleft(sl3)
    """    
    print("BEST SOLN")
    print(best_soln.dg)
    print("BEST ASSIGNMENT")
    print(best_assignment)
    print("BEST SCORE")
    print(best_score)
    """
    return best_soln,best_assignment,best_score

    
############################################################################################################

'''
used as a descriptor for assembling a subgraph to a graph
'''
class AssemblyInstruction:

    def __init__(self,ml=[],ai=[],ai2=[]):
        assert len(ml) == len(ai)
        assert len(ai) == len(ai2)

        # move list
        self.ml = ml

        # corresponding assembly instructions for each move
        self.ai_target = ai
        self.ai_antitarget = ai2

    def __str__(self):
        s = "**\tASSEMBLY SOLUTION\n"
        mg1 = MicroGraph(defaultdict(set))
        mg2 = MicroGraph(defaultdict(set))

        for i in range(len(self.ai_target)):
            s += "\t\t[] move {}\n".format(i)
            q1 = self.ai_target[i]
            if type(q1) == type(None):
                s += "\t\t[T]: --\n"
            else:
                mg1 = mg1 + q1[0]
                s += "\t\t[T]:\n\t{}\n".format(mg1.dg)

            q2 = self.ai_antitarget[i]
            if type(q2) == type(None):
                s += "\t\t[A]: --\n"
            else:
                mg2 = mg2 + q2[0]
                s += "\t\t[A]:\n\t{}\n".format(mg2.dg)

        return s 

    '''
    mv := PMove
    ainst := 
            [0] MicroGraph, subgraph to be added 
            [1] defaultdict, key is node identifier of move's resource graph,
            value is node identifier of target resource graph, used for target
    ainst2 := 
            [0] MicroGraph, subgraph to be added
            [1] defaultdict, same format as `ainst`, used for anti-target
    '''
    def add_instruction(self,mv,ainst,ainst2):
        assert type(mv) == PMove,"got {}, want {}".format(type(mv),PMove)
        if type(ainst) != type(None):
            assert type(ainst[0]) == MicroGraph
            assert type(ainst[1]) == defaultdict
        if type(ainst2) != type(None):
            assert type(ainst2[0]) == MicroGraph
            assert type(ainst2[1]) == defaultdict

        self.ml.append(mv)
        self.ai_target.append(ainst)
        self.ai_antitarget.append(ainst2) 
        return
    
    # TODO: caution
    """
    collates all assembly instructions `ai_target` or `ai_antitarget` by +
    """
    def node_to_target_map(self,target:bool): 
        m = defaultdict(str)
        q = self.ai_target if target else self.ai_antitarget
        for q_ in q:
            for (k,v) in q_.items():
                if v[1] == "+" and v[0] != "?":
                    m[k] = v[0]
        return m

"""
<MicroGraphAssemblySolution> represents the highest-scoring assembly procedure found.

Variable <ai> is an <AssemblyInstruction> instance that 
"""
class MicroGraphAssemblySolution:

    """
    wm := wanted move
    """
    def __init__(self,wm):##,mg = None,ai=None):
        assert type(wm) == PMove
        self.wm = wm

        # the MicroGraph that is supposed to be equal to `wm.payoff_target`
        self.mg1 = MicroGraph()
        # the MicroGraph that is supposed to be equal to `wm.antipayoff_target`
        self.mg2 = MicroGraph()
        # assembly instructions
        self.ai = AssemblyInstruction()##if type(ai) == type(None) else ai        
        return

    def solution_fit_score(self,is_target:bool):
        if is_target: 
            wmmg = MicroGraph.from_ResourceGraph(self.wm.payoff_target)
            return ve_fitscore_type1(self.mg1,wmmg)
        wmmg = MicroGraph.from_ResourceGraph(self.wm.antipayoff_target)
        return ve_fitscore_type1(self.mg2,wmmg)

    def update__add(self,mv,target_data,antitarget_data):
        # add to assembly instruction
        self.ai.add_instruction(mv,target_data,antitarget_data)

        # update MicroGraphs
        if type(target_data) != type(None):
            ##print("before")
            ##print(self.mg1.dg)
            self.mg1 = self.mg1 + target_data[0]
            ##
            """
            print("TARGET") 
            print("add")
            print(target_data[0].dg) 
            print("dict")
            print(target_data[1])
            print("after")
            print(self.mg1.dg)
            """
            ## 

        if type(antitarget_data) != type(None):
            self.mg2 = self.mg2 + antitarget_data[0]

            ##
            """
            print("ANTI-TARGET") 
            print("add")
            print(antitarget_data[0].dg) 
            print("dict")
            print(antitarget_data[1])
            print("after")
            print(self.mg2.dg) 
            """
            ## 

        return

    def assemble(self,target:bool,excess_index:int):
        q = self.ai.ai_target if target else self.ai.ai_antitarget
        x = MicroGraph(defaultdict(set))
        if type(excess_index) == type(None): excess_index = len(q) 
        for q_ in q[:excess_index]:
            if type(q_) != type(None): x = x + q_[0]
        return x

"""
class used to construct a `wanted move` by a `player` given the player's
available moves.

Procedure for assembly:
1. full-move assembly (until wanted move's target or anti-target is a proper subgraph of 
    its running solution)
2. partial-move assembly (to "trim" excess nodes and edges from a supergraph, or to add subgraphs of
    resource graphs belonging to the player's available moves.)
3. inverted move (to "trim" excess nodes and edges from a supergraph) 
NOTE: sole focus on undirected graphs. 
"""
class MoveAssemblerType1:

    def __init__(self,player,wanted_move) -> None:
        self.player = player
        self.wanted_move = wanted_move
        self.mas = MicroGraphAssemblySolution(deepcopy(wanted_move))
        self.excess_index = None 
        self.success = True
        self.counts = [0,0,0,0] 

    def assembly_status(self):
        q1 = self.mas.assemble(True,self.excess_index)
        q2 = self.mas.assemble(False,self.excess_index) 

        stat1 = MicroGraph.from_ResourceGraph(self.wanted_move.payoff_target).sub_ve_score(q1)
        stat2 = MicroGraph.from_ResourceGraph(self.wanted_move.antipayoff_target).sub_ve_score(q2)
        ##print("stat score: ", stat1,stat2)
        return stat1[0] + stat1[1] != 0,stat2[0] + stat2[1] != 0 

    def assemble(self):
        self.full_move_assembly()
        s1,s2 = self.assembly_status()
        s3 = s1
        self.partial_move_assembly(s3)
        self.fix_excess()

    ## NOTE: CAREFUL
    def full_move_assembly(self):
        def output_stat():
            s1,s2 = self.assembly_status()
            ##print("SX ",s1,s2) 
            return s1 and s2

        print("-- full move assembly")
        c = 0
        stat = output_stat()
        while stat:
            q = self.full_move_add()
            ##print("-----//-----//-----//-----//")
            stat = output_stat()
            stat = False if q == -1 else stat 
            c += 1
        print("# moves: ", c)
        ##self.counts[0] = deepcopy(c)
        return

    def full_move_add(self):
        # iterate through the moves of the players, and determine the move 
        # with the best fit
        best_move = None
            # each element is of the form: running solution,node assignment,fit score
        best_move_soln_target = [None,None,self.mas.solution_fit_score(True)]
        best_move_soln_antitarget = [None,None,self.mas.solution_fit_score(False)]

        for m in self.player.ms:
            results1 = brute_force_search_node_assignment__full_add(deepcopy(self.wanted_move),\
                deepcopy(self.mas.mg1),deepcopy(m),True,ve_fitscore_type1)
            results2 = brute_force_search_node_assignment__full_add(deepcopy(self.wanted_move),\
                deepcopy(self.mas.mg2),deepcopy(m),False,ve_fitscore_type1)
            
            score = results1[2] + results2[2]
            if best_move_soln_target[2] + best_move_soln_antitarget[2] > score:
                best_move = deepcopy(m)
                best_move_soln_target = results1
                best_move_soln_antitarget = results2

        # update with the best add
        if type(best_move) == type(None): 
            return -1
        ##print("best soln: {}|{}".format(best_move_soln_target[2], best_move_soln_antitarget[2]))
        self.mas.update__add(best_move,[best_move_soln_target[0],best_move_soln_target[1]],\
            [best_move_soln_antitarget[0],best_move_soln_antitarget[1]])
        self.counts[0] += 1
        return

    # NOTE: can refactor!
    def partial_move_assembly(self,is_target=False):
        def output_stat():
            s1,s2 = self.assembly_status()
            return s1 if is_target else s2
        
        print("-- partial move assembly for ", int(is_target)) 
        c = 0
        stat = output_stat()
        while stat:
            q = self.partial_move_add(is_target)
            stat = output_stat()
            if q == -1: 
                stat = False
                self.success = False
            c += 1
            ##print("c: ", c)
        print("# moves: ",c)
        ##self.counts[1] = deepcopy(c)
        return

    # NOTE: can refactor!
    def partial_move_add(self,is_target=False):
        assert type(is_target) == bool 

        # iterate through the moves of the players, and determine the move 
        # with the best fit
        best_move = None
            # each element is of the form: running solution,node assignment,fit score
        best_move_soln = [None,None,self.mas.solution_fit_score(is_target)]
        ##print("best score: ", best_move_soln[2])
        for m in self.player.ms:
            mgx = deepcopy(self.mas.mg1) if is_target else deepcopy(self.mas.mg2) 
            results1 = brute_force_search_node_assignment__full_add(deepcopy(self.wanted_move),\
                mgx,deepcopy(m),is_target)
            score = results1[2]
            if best_move_soln[2] > score:
                best_move = deepcopy(m)
                best_move_soln = results1

        # update with the best add
        if type(best_move) == type(None): 
            print("NO MOVE") 
            return -1

        ##print("best move: ", best_move_soln[0])
        ##print("best move soln: ", best_move_soln[1])
        qx1 = [best_move_soln[0],best_move_soln[1]] if is_target else None
        qx2 = None if is_target else [best_move_soln[0],best_move_soln[1]]
        self.mas.update__add(best_move,qx1,qx2)
        self.counts[1] += 1 
        return 

    def fix_excess(self):

        def fix_excess_(is_target:bool,end_index:int):
            print("excess target: ", is_target)
            # iterate through the assembly moves,
            # and for each move, repeat it if it produced
            # excess on the assembly
            s = MicroGraph(defaultdict(set))

            node_excesses = set()
            edge_excesses = set()
            c = 0
            for i in range(end_index):
                
                mvx = deepcopy(self.mas.ai.ai_target[i]) if is_target\
                    else deepcopy(self.mas.ai.ai_antitarget[i])
                if type(mvx) == type(None): continue
                mvx = mvx[0] 
                s = s + mvx

                wmpt = MicroGraph.from_ResourceGraph(self.wanted_move.payoff_target if\
                    is_target else self.wanted_move.antipayoff_target)
                q = s - wmpt 
                nc = 0
                ec = 0

                # check for excesses
                stat = False
                for (k,v) in q.dg.items():
                    ln,le = len(node_excesses), len(edge_excesses)
                    node_excesses = node_excesses | {k}

                    for v_ in v:
                        qs = sorted([k,v_])
                        es = qs[0] + "," + qs[1]
                        edge_excesses = edge_excesses | {es}
                    stat = len(node_excesses) > ln or len(edge_excesses) > le
                    if stat:
                        break 
                if stat: 
                    print("excess at index {}".format(i)) 
                    qx1 = [mvx,defaultdict(str)] if is_target else None
                    qx2 = [mvx,defaultdict(str)] if not is_target else None
                    self.mas.update__add(self.mas.ai.ml[i],qx1,qx2)
                    ##c += 1

                    if is_target:
                        self.counts[2] += 1
                    else:
                        self.counts[3] += 1

            if is_target:
                self.counts[2] = deepcopy(c)
            else:
                self.counts[3] = deepcopy(c)

        print("FIXING EXCESS")
        self.excess_index = len(self.mas.ai.ml)
        fix_excess_(True,self.excess_index)
        fix_excess_(False,self.excess_index)
        return 