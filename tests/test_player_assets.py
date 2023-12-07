from player_assets import *
import unittest

####################################################

def testing_move_X7():
    mgmt1 = MicroGraph(defaultdict(set,\
            {"0":{"1","2","3","4"},\
            "1":{"0"},\
            "2":{"0"},\
            "3":{"0","4"},\
            "4":{"0","3"}}))
    rgx = ResourceGraph.from_MicroGraph(mgmt1)
    return PMove(30,-30,rgx,ResourceGraph(),True)

def test_resource_graph_11():

    mgrg = MicroGraph(defaultdict(set,\
            {"5":{"6","7","8","9","10","11"},\
            "6":{"5","7"},\
            "7":{"5","6"},\
            "8":{"5","9"},\
            "9":{"5","8"},\
            "10":{"5","11"},\
            "11":{"10","5"}}))

    rg = ResourceGraph.from_MicroGraph(mgrg)
    fs(rg,sort_cst1,node_health_score_assignment_type_1,\
        [170,200],es_func_1)
    return rg 

class PlayerAssetsFileClass(unittest.TestCase):

    ### TODO: add assertions here.
    def test__Move__gauge_payoff_seq_on_RG__case1(self):
        mv = testing_move_X7()
        rg = test_resource_graph_11() 
        print("**\t PMove -- gauge_payoff_seq_on_RG")
        q = mv.gauge_payoff_seq_on_RG(rg,True)
        
        ##print(q[2][0])
        return