from trapmatch_env import * 
import unittest

def TMEnv_sample_1(verbose_mode=False):
    i = 81#4
    num_players = 3
    num_moves_range = (4,7)
    drange = (16,30)
    connectivity_range = (0.25,0.6)
    excess_range = [10**4,20**6]
    game_modes = ["noneg","public"]
    farse_mach = None
    tme = TMEnv.generate__type_dumb(i,num_players,num_moves_range,\
        drange,connectivity_range,excess_range,game_modes,verbose_mode)
    return tme

def check_defint_for_TMEnv__exact_MG_match(tme:TMEnv):
    s = set([p.idn for p in tme.players])
    stat = True
    for p in tme.players:
        idns = s - {p.idn}
        for i in idns:
            # compare the two MicroGraphs 
            j = tme.idn_to_index(i)
            if j == -1:
                print("no player exists")
                continue
            mg = MicroGraph.from_ResourceGraph(tme.players[j].rg)
            mg2 = p.pdec.pkdb.other_mg[i]
            if type(mg2) == int:
                stat = False
            else:
                stat = (mg == mg2)
            if not stat: break 
    return stat

def check_defint_delta_size(p:Player,wanted_size:int,is_node_delta:bool):
    q = p.pdec.def_int.node_delta if is_node_delta else \
        p.pdec.def_int.edge_delta
    for (k,v) in q.items():
        for (k2,v2) in v.items():
            if len(v2) != wanted_size:
                return False
    return True

class TMEnvClass(unittest.TestCase):

    """
    Uses <TMEnv> sample #1. 
    Moves 3 timestamps; checks for thorough execution.
    """
    def test__TMEnv__move_one_timestamp__case_1(self):
        tme = TMEnv_sample_1()
        for i in range(3):
            print("moving one")
            tme.move_one_timestamp()
        assert True
        return

    """
    Executes one timestamp for <TMEnv> with mode set
    to `preferred_move = PInfo-2`. 

    Checks each player's <PKDB> instance for the correct
    knowledge of other players' graphs. 
    """
    def test__TMEnv__move_one_timestamp__case_2(self):
        tme = TMEnv_sample_1()
        tme.preferred_move = "PInfo-2"
        tme.move_one_timestamp()
        stat = check_defint_for_TMEnv__exact_MG_match(tme)
        assert stat
        return

    """
    Executes two timestamps for <TMEnv> with mode set
    to `preferred_move = PInfo-2`. 

    The first timestamp executes by `preferred_move=PInfo-2`.
    The second timestamp executes by `move_type_deterministic_assignment=[NInfo]`.

    Checks that player 2 has a non-empty container of NegoChips.
    """
    def test__TMEnv__move_one_timestamp__case_3(self):
        tme = TMEnv_sample_1()
        tme.preferred_move = "PInfo-2"
        tme.move_one_timestamp()
        stat = check_defint_for_TMEnv__exact_MG_match(tme)

        ## 
        tme.move_type_deterministic_assignment(["NInfo"])
        tme.preferred_move = None
        tme.move_one_timestamp()

        p2 = tme.idn_to_player("2")
        assert len(p2.pdec.nc.container) > 0
        p1 = tme.idn_to_player("1")
        assert len(p1.pdec.nc.container) > 0
        p0 = tme.idn_to_player("0")
        assert len(p0.pdec.nc.container) == 0

    """
    tests that expected and actual PMove isomorphic attack
    scores are different after running one timestamp with 
    `move_type_deterministic_assignment=NInfo`. 
    """
    def test__TMEnv__move_one_timestamp__case_4(self):

        tme = TMEnv_sample_1()
        tme.preferred_move = "PInfo-2"
        tme.move_one_timestamp()

        ## 
        tme.move_type_deterministic_assignment(["NInfo"])
        tme.preferred_move = None
        tme.move_one_timestamp()
        tme.move_one_timestamp()

        tme.set_player_verbosity(True)
        tme.move_type_deterministic_assignment(None)
        tme.preferred_move = "PInfo-3"
        tme.move_one_timestamp()

        # assert the expected actual differences 
        for p in tme.players:
            exp,act = p.pdec.def_int.cumulative_expected_actual_of_move(3,False)
            assert exp != act

    """
    checks for <DefInt> instances' correct delta sizes
    of nodes+edges. 
    """
    def test__TMEnv__move_one_timestamp__case_5(self):

        tme = TMEnv_sample_1()
        tme.preferred_move = "PInfo-2"

        for i in range(2):
            tme.move_one_timestamp()

            ##
        tme.preferred_move = "PInfo-3"
        for i in range(3):
            tme.move_one_timestamp() 
            ##

        for p in tme.players:
            stat1 = check_defint_delta_size(p,3,True)
            stat2 = check_defint_delta_size(p,2,False)
            assert stat1 and stat2
        return

if __name__ == '__main__':
    unittest.main()