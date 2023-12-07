import logging
##from t1 import *
import numpy as np
from resource_defaults import *
from pmove import *
from trapmatch_env import *
import time

# TODO: 
"""
Remember to assign privacy mode from TMEnv to each player's
ResourceGraph.
"""
##################################################################################################################################

def sample_move_1():
    return -1

def sample_move_2():
    return -1

def sample_move_3():
    return -1

def sample_move_4():
    return -1

def sample_antimove_generator(mv,inverse_function):
    return -1

# TODO: requires a class to assign the values
def assign_random_values_to_RG_ve(rg):
    return -1

#################################################
"""
def testing_move_X7():
    mgmt1 = MicroGraph(defaultdict(set,\
            {"0":{"1","2","3","4"},\
            "1":{"0"},\
            "2":{"0"},\
            "3":{"0","4"},\
            "4":{"0","3"}}))
    return PMove(30,-30,mgmt1,MicroGraph(defaultdict(set)),True)

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

mv = testing_move_X7()
rg = test_resource_graph_11() 
print("**\t PMove -- gauge_payoff_seq_on_RG")
q = mv.gauge_payoff_seq_on_RG(rg,True)##,1)
"""

###########################     
"""
def test_player3():

    mg1 = MicroGraph(dgraph=defaultdict(set,\
        {"0":{"1","2","3"},\
        "1":{"0","6"},\
        "2":{"0","5","6","8","9","10"},\
        "3":{"0","4","5"},\
        "4":{"3","10"},\
        "5":{"2","3","10"},\
        "6":{"1","2","7","8"},\
        "7":{"6","8"},\
        "8":{"2","6","7","9"},\
        "9":{"2","8","10","11"},\
        "10":{"2","4","5","9"},\
        "11":{"9"}}))
        
    rg = ResourceGraph.from_MicroGraph(mg1)
    default_rg_value_assignment(rg,[10**6,2* 10 **6])

    mv1 = PMove(4, 5,sample_resource_graph_1(),\
        sample_resource_graph_5(),True)
    mv2 = PMove(3, 6,sample_resource_graph_7(),\
        sample_resource_graph_7(),True)
    ms = [mv1,mv2]
    return Player(rg,ms)

def test_player4():

    mg1 = MicroGraph(dgraph=defaultdict(set,\
        {"0":{"1","2"},\
        "1":{"4","6"},\
        "2":{"0","3","7"},\
        "3":{"2","5","7"},\
        "4":{"1","5","6"},\
        "5":{"3","4","6","7"},\
        "6":{"1","4","5"},\
        "7":{"2","3","5"}}))
        
    rg = ResourceGraph.from_MicroGraph(mg1)
    default_rg_value_assignment(rg,[10**6,2* 10 **6])

    mv1 = PMove(20, 200,sample_resource_graph_1(),\
        sample_resource_graph_2(),True)
    mv2 = PMove(20, 200,sample_resource_graph_8(),\
        sample_resource_graph_8(),True)    
    ms = [mv1,mv2]
    return Player(rg,ms)

def test_player5():

    mg1 = MicroGraph(dgraph=defaultdict(set,\
        {"0":{"1","5"},\
        "1":{"0","2","3","5"},\
        "2":{"1","3","4"},\
        "3":{"1","2","4"},\
        "4":{"2","3","5"},\
        "5":{"0","4","6"},\
        "6":{"5","7"},\
        "7":{"6","8"},\
        "8":{"5","7"}}))

    rg = ResourceGraph.from_MicroGraph(mg1)
    default_rg_value_assignment(rg,[10**6,2* 10 **6])

    mv1 = PMove(20, 200,sample_resource_graph_1(),\
        sample_resource_graph_1(),True)
    mv2 = PMove(20, 200,sample_resource_graph_7(),\
        sample_resource_graph_6(),True)
    mv3 = PMove(20, 200,sample_resource_graph_6(),\
        sample_resource_graph_5(),True)
    
    ms = [mv1,mv2,mv3]
    return Player(rg,ms)

p1 = test_player3()
p2 = test_player4()
p3 = test_player5()
tme = TMEnv([p1,p2,p3])
tme.move_one_timestamp()
"""
######################################################

"""
def test_resource_graph_9():
    mg1 = MicroGraph(dgraph=defaultdict(set,\
        {"0":{"1","2","3","4"},\
        "1":{"0","4"},\
        "2":{"0","3"},\
        "3":{"0","2"},\
        "4":{"0","1","7","5"},\
        "5":{"4","6"},\
        "6":{"5","7"},\
        "7":{"4","6","8"},\
        "8":{"7","9","11"},\
        "9":{"8","10"},\
        "10":{"9","11"},\
        "11":{"8","10"},\
        "12":{"0","13","14","15","16"},\
        "13":{"12"},\
        "14":{"12"},\
        "15":{"12"},\
        "16":{"12"}}))
    rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
    return rg1 

def test_resource_graph_10():
    mg1 = MicroGraph(dgraph=defaultdict(set,\
        {"0":{"1"},\
        "1":{"0","2"},\
        "2":{"1","3","4"},\
        "3":{"2","4"},\
        "4":{"2","3","5"},\
        "5":{"4","6","7"},\
        "6":{"5","7"},\
        "7":{"5","6","8","9","10"},\
        "8":{"7","11","12"},\
        "9":{"7"},\
        "10":{"7","11"},\
        "11":{"8","10","12"},\
        "12":{"8","11","13"},\
        "13":{"12"}}))
    rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
    return rg1 

def testing_move_X6():
    return PMove(20, 200,test_resource_graph_9(),\
        test_resource_graph_10(),True)


def test_player1():
    rg = ResourceGraph({},{})

    mv1 = PMove(20, 200,sample_resource_graph_1(),\
        sample_resource_graph_1(),True)
    mv2 = PMove(20, 200,sample_resource_graph_2(),\
        sample_resource_graph_2(),True)
    mv3 = PMove(20, 200,sample_resource_graph_3(),\
        sample_resource_graph_3(),True)
    mv4 = PMove(20, 200,sample_resource_graph_4(),\
        sample_resource_graph_4(),True)
    mv5 = PMove(20, 200,sample_resource_graph_5(),\
        sample_resource_graph_5(),True)
    
    ms = [mv1,mv2,mv3,mv4,mv5]
    return Player(rg,ms)

def test_player2():
    rg = ResourceGraph({},{})
    mv1 = PMove(20, 200,sample_resource_graph_6(),\
        sample_resource_graph_6(),True)

    ms = [mv1]
    return Player(rg,ms)

p1 = test_player1()
p2 = test_player2() 
"""

#########################################################
"""
def greatest_common_subgraph_case_1():
    mg4 = MicroGraph(defaultdict(set,\
            {"10":{"11","21"},\
            "11":{"10","21"},\
            "21":{"10","11"}}))

    mg3 = MicroGraph(defaultdict(set,\
            {"22":{"10","21","33","41"},\
            "10":{"21","22"},\
            "21":{"10","22"},\
            "33":{"22"},\
            "41":{"22"}}))

    mg2 = MicroGraph(defaultdict(set,\
            {"10":{"12"},\
            "12":{"10","22"},\
            "22":{"12","13","14"},\
            "13":{"22"},\
            "14":{"22"}}))

    mg1 = MicroGraph(defaultdict(set,\
            {"0":{"1","2"},\
            "1":{"0","2"},\
            "2":{"0","1"}}))
    
    return GCSContainer([mg1,mg2,mg3,mg4]) 

gcg1 = greatest_common_subgraph_case_1()
gcg1.search_type = "matching neighbor fit"
gcg1.initialize_cache() 
s1,s2 = gcg1.search() 
"""
#########################################################

##rg = ResourceGraph.generate__type_stdrand(5,12,[0.2,0.5],[2*10**6,3*10**7])

### TEST CASE 1
i = 74#4
num_players = 3
num_moves_range = (4,7)
drange = (16,30)
connectivity_range = (0.25,0.6)
excess_range = [10**4,20**6]
game_modes = ["noneg","public"]
farse_mach = None

tme = TMEnv.generate__type_dumb(i,num_players,num_moves_range,\
    drange,connectivity_range,excess_range,game_modes,farse_mach)

#p = tme.players[0]
t = time.time()
tme.move_one_timestamp()
t2 = time.time()
print("time: ", t2 - t) 

###---###---###---###---###---###---###

# display tests
'''
tme.players[0].display_context(["PMove","AMove","MMove"])
tme.players[0].display_PMoves()
'''


# declare all players
##p1 = Player()

##def __init__(self,rg,ms,idn = None,excess=1000,pcontext_mapper=None):


# declare the TrapEnv instance
##def __init__(self,players,game_mode_1,game_mode_2,farse_mach=None,):





if __name__ == "__main__":
    print("Hello")