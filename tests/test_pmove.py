from resource_defaults import *
from pmove import *
import unittest
import time

def test_resource_graph_X1T():
    mg1 = MicroGraph(dgraph=defaultdict(set,\
                    {"0":{"1","3"},"1":{"0","2"},"2":{"1","3"},\
                    "3":{"0","2","4","6"}, "4":{"3","7"},"5":{"6"},\
                    "6":{"3","5","7","8","10"}, "7":{"4","6","8"},\
                    "8":{"6","7","9"}, "9":{"8","11"},"10":{"6","11"},\
                    "11":{"9","10"}}))
    
    rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
    return rg1 

def test_resource_graph_X1A():
    mg1 = MicroGraph(dgraph=defaultdict(set,\
                {"0":{"1"},"1":{"0","3","6","7"},"2":{"3"},"3":{"1","2","4","5"},\
                 "4":{"3","5"},"5":{"3","4","9"},"6":{"1"},"7":{"1","8"},\
                "8":{"7","9"},"9":{"5","8","10"},"10":{"9"}}))
    rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
    return rg1 

"""
target for MoveAssembler
"""
def testing_move_X1():
    return PMove(20, 200,test_resource_graph_X1T(),\
        test_resource_graph_X1A(),True)

"""
constructor for MoveAssembler
"""
def testing_move_X2():
    return PMove(20, 200,sample_resource_graph_5(),\
        sample_resource_graph_7(),True) # 6 

"""
constructor for MoveAssembler
"""
def testing_move_X3():
    return PMove(20, 200,sample_resource_graph_3(),\
        sample_resource_graph_4(),True)

#########

def test_resource_graph_X2T():
    mg1 = MicroGraph(dgraph=defaultdict(set,\
            {"0":{"1","2","3","4"},\
            "1":{"0"},
            "2":{"0"},
            "3":{"0"},
            "4":{"0"}}))
    rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
    return rg1

def test_resource_graph_X2A():
    mg1 = MicroGraph(dgraph=defaultdict(set,\
            {"0":{"1","2","3","4"},\
            "1":{"0","2"},
            "2":{"0","1"},
            "3":{"0"},
            "4":{"0"}}))
    rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
    return rg1

def testing_move_X4():
    return PMove(20, 200,test_resource_graph_X2T(),\
        test_resource_graph_X2A(),True)

def testing_move_X5():
    return PMove(20, 200,sample_resource_graph_6(),\
        sample_resource_graph_6(),True)


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

####################################################

class PMoveClass(unittest.TestCase):

    def test__brute_force_search_node_assignment__full_add_case1(self):
        tmg = testing_move_X1()
        smg = MicroGraph(defaultdict(set))
        #mv = PMove(20,200,sample_resource_graph_2(),sample_resource_graph_3(),True)

        mv = testing_move_X3()
        q = brute_force_search_node_assignment__full_add(tmg,smg,mv,True)
        assert q[2] <= 19, "got {}, want {}".format(q[2],18)

    def test__brute_force_search_node_assignment__full_add_case2(self):
        
        tmg = testing_move_X4() 
        smg = MicroGraph(defaultdict(set))
        mv = testing_move_X5()

        q = brute_force_search_node_assignment__full_add(tmg,smg,mv,True)
        smg = smg + deepcopy(q[0])
        assert q[2] == 3

        q = brute_force_search_node_assignment__full_add(tmg,smg,mv,True)
        smg = smg + q[0]
        assert q[2] == 2

        q = brute_force_search_node_assignment__full_add(tmg,smg,mv,True)
        smg = smg + q[0]
        assert q[2] == 1

        q = brute_force_search_node_assignment__full_add(tmg,smg,mv,True)
        smg = smg + q[0]
        assert q[2] == 0
        sol = defaultdict(set, \
            {'0': {'4', '1', '2', '3'},\
            '1': {'0'},\
            '2': {'0'},\
            '3': {'0'},\
            '4': {'0'}})

        assert smg.dg == sol

## NOTE: only run one of the below assembly tests due to potential
##       shared resources bug. 
class MoveAssemblerType1Class(unittest.TestCase):
    """
    def test__MoveAssemblerType1__assemble_case1(self):
        p = test_player1()
        mat1 = MoveAssemblerType1(p,testing_move_X6())
        mat1.assemble()
        print("\t** MoveAssemblerType1-- assemble case 1")
        print(mat1.mas) 
        print("---")
        print(mat1.counts)
        print("###################")
        del p
        del mat1
        time.sleep(10)
    """

    def test__MoveAssemblerType1__assemblxe_case2(self):
        print("\t** MoveAssemblerType1-- assemble case 2")

        p2 = test_player2()
        matx = MoveAssemblerType1(p2,testing_move_X6())
        matx.assemble() 
        print("counts: ", matx.counts)
        print("targets\n",len(matx.mas.ai.ai_target))
        print("\nanti-targets\n",len(matx.mas.ai.ai_antitarget))
        ##time.sleep(3)
        assert matx.counts == [17,3,0,0], "got {}".format(matx.counts)
        return 


if __name__ == '__main__':
    unittest.main()