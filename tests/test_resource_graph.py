from resource_defaults import *
from player import *
import unittest

def unordered_setseq__equals(s1,s2):
    if len(s1) != len(s2): return False

    for s in s1:
        stat = False
        for q in s2:
            if q == s:
                stat = True
                break
        if not stat: return False
    return True

class MicroGraphClass(unittest.TestCase):

    def test__MicroGraph___sub__(self):
        rg4 = sample_resource_graph_4()
        rg6 = sample_resource_graph_6()

        mg4 = MicroGraph.from_ResourceGraph(rg4) 
        mg6 = MicroGraph.from_ResourceGraph(rg6)
        mgx = mg4 - mg6

        dg = defaultdict(set, {'2': {'1', '3'},\
            '3': {'0', '2'},\
            '4': {'0'},\
            '5': {'0'}})
        assert mgx.dg == dg
        return

    def test__MicroGraph_alt_subtract(self):
        mgx1 = MicroGraph(defaultdict(set,{"0":set(),"1":set()}))
        mgx2 = MicroGraph(defaultdict(set,{"0":{"1"},"1":{"0"}}))
        mgx = mgx2.alt_subtract(mgx1) 

        sol10 = set()
        sol11 = {'0,1', '1,0'}
        assert mgx[0] == sol10 and mgx[1] == sol11
        return

    def test__MicroGraph_component_seq(self):
        # case 1
        dg = {"0":{"1","2","3"},\
            "1":{"0","3"},\
            "2":{"0","3","4","5"},\
            "3":{"0","1","2","5"},\
            "4":{"2"},\
            "5":{"2","3"},\
            "6":{"7","8","9"},\
            "7":{"6","8"},\
            "8":{"6","7"},\
            "9":{"6"},\
            "10":{"11"},\
            "11":{"10"},\
            "12":{"13","14"},\
            "13":{"12"},\
            "14":{"12"}}
        mg = MicroGraph(defaultdict(set,dg)) 
        cs = mg.component_seq() 
        sol = [{'7', '6', '9', '8'},\
            {'5', '0', '3', '1', '2', '4'},\
            {'14', '13', '12'}, {'11', '10'}]

        assert unordered_setseq__equals(cs,sol)

        # case 2
        dg2 = {"0":{"1"},\
                "1":{"0"},\
                "2":{"3","4"},\
                "3":{"2"},\
                "4":{"2"},\
                "5":set(),\
                "6":{"7","8"},\
                "7":{"6"},\
                "8":{"6"}}
        mg2 = MicroGraph(defaultdict(set,dg2)) 
        cs2 = mg2.component_seq() 

        sol2 = [{'5'}, {'0', '1'}, {'2', '3', '4'}, {'7', '6', '8'}]
        assert unordered_setseq__equals(cs2,sol2) 

        return 

class ResourceGraphClass(unittest.TestCase):

    def test__ResourceGraph_subgraph_isomorphism__case_1(self):

        # case 1: ResourceGraph
        nhm = {"0":20,"1":40,"2":32,"3":71}
        ehm = {"0,1": 14,"0,2":41,"1,3":42,"2,3":27}
        rg = ResourceGraph(nhm,ehm)
        print(MicroGraph.from_ResourceGraph(rg).dg)
        mg1 = MicroGraph(dgraph=defaultdict(set,{"0":{"1"},"1":{"0"}}))
        ql = rg.subgraph_isomorphism(mg1,True)

        # case 2: 
        mg2 = MicroGraph(dgraph=defaultdict(str,{"10":{"20","30"},"20":{"10","40"},"30":{"10","40"},"40":{"20","30"}}))
        ql2 = rg.subgraph_isomorphism(mg2,True)

        # case 3:
        mg3 = MicroGraph(dgraph=defaultdict(str,{"10":{"20","30"},"20":{"10"},"30":{"10"}}))
        ql3 = rg.subgraph_isomorphism(mg3,True)

        assert len(ql) == len(ql2) == len(ql3)
        assert len(ql) == 8

        print(ql) 
        return
    
    def test__assign_health_scores_to_ResourceGraph__case_1(self):

        rg = sample_resource_graph_2()
        sort_function = sort_cst1
        map_function = node_health_score_assignment_type_1
        scale_range = [10,200]
        edge_function = es_func_1

        results = fs(rg,sort_function,map_function,scale_range,edge_function)

        node_health_solution = {'0': 10.0, '1': 10.0, '2': 200.0, '3': 200.0, '4': 10.0, '5': 10.0}
        edge_health_solution = {'0,1': 10.0, '0,2': 105.0, '1,3': 105.0, '1,0': 10.0, '2,3': 200.0,\
        '2,0': 105.0, '2,5': 105.0, '3,4': 105.0, '3,1': 105.0, '3,2': 200.0,\
            '4,3': 105.0, '4,5': 10.0, '5,4': 10.0, '5,2': 105.0}

        assert rg.node_health_map == node_health_solution
        assert rg.edges_health_map == edge_health_solution

    def test__ve_fitscore__case_1(self):

        rg7 = sample_resource_graph_7() 
        rg8 = sample_resource_graph_8()

        mg7 = MicroGraph.from_ResourceGraph(rg7)
        mg8 = MicroGraph.from_ResourceGraph(rg8) 

        # fit-score type 2
        assert ve_fitscore_type1(mg8,mg7) == 4
        assert ve_fitscore_type1(mg7,mg8) == 4

        # fit-score type 2
        assert ve_fitscore_type2(mg8,mg7) == 4
        assert ve_fitscore_type2(mg7,mg8) == 0

    # print-tests
    def test__ResourceGraph_isomap_to_isograph__case_1(self):
        # sample 1
        rg5 = sample_resource_graph_5()
        rg6 = sample_resource_graph_6()
        mg6 = MicroGraph.from_ResourceGraph(rg6)
        si = rg5.subgraph_isomorphism(mg6,True)
        assert len(si) == 6

        for si_ in si:
            si2 = pairseq_to_dict(si_)
            g = rg5.isomap_to_isograph(mg6,si2)
            print(g)

        # sample 2
        rg3 = sample_resource_graph_3()
        rg4 = sample_resource_graph_4()
        mg3 = MicroGraph.from_ResourceGraph(rg3)
        si = rg4.subgraph_isomorphism(mg3,True)
        assert len(si) == 0, "want {}, got {}".format(0,len(si))

        for si_ in si:
            si2 = pairseq_to_dict(si_)
            g = rg4.isomap_to_isograph(mg3,si2)
            print(g)
        return

    def test__ResourceGraph_nearest_partial_n2n_map__case_1(self):

        # case 1
        mgx = MicroGraph(dgraph=defaultdict(set,{"0":{"1","2"},\
                                    "1":{"0","3"},\
                                    "2":{"0","3"},\
                                    "3":{"1","2"}}))
        pm = PMove(30,-30,MicroGraph(defaultdict(set)),mgx,True)
        mg1 = MicroGraph(dgraph=defaultdict(set,{"0":{"1"},"1":{"0"}}))
        rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)

        nmap = rg1.nearest_partial_n2n_map(mgx,reversos = False)
        lnm = list(nmap.values())
        assert lnm.count("?") == 2

        # case 2
        mg2 = MicroGraph(dgraph=defaultdict(set,{"0":{"1"},"1":{"0"},\
                                            "2":set(),"3":set()}))
        rg2 = ResourceGraph.from_MicroGraph(mg2,default_health=1)
        nmap = rg2.nearest_partial_n2n_map(mgx,reversos = False)
        lnm = list(nmap.values())
        assert lnm.count("?") == 0
    
    def test__ResourceGraph_additions_from_partial_n2n_map__case_1(self):
        mgx = MicroGraph(dgraph=defaultdict(set,{"0":{"1","2"},\
                                    "1":{"0","3"},\
                                    "2":{"0","3"},\
                                    "3":{"1","2"}}))
        pm = PMove(30,-30,MicroGraph(defaultdict(set)),mgx,True)
        mg1 = MicroGraph(dgraph=defaultdict(set,{"0":{"1"},"1":{"0"}}))
        rg1 = ResourceGraph.from_MicroGraph(mg1,default_health=1)
        addit = rg1.additions_from_partial_n2n_map(mgx,reversos=False)
        xs = ({'2', '3'}, {'3,1', '1,3', '3,2', '2,0', '2,3', '0,2'})
        assert xs == addit
        
if __name__ == '__main__':
    unittest.main()