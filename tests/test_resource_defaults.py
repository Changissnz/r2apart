from resource_defaults import *
import unittest

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

def greatest_common_subgraph_case_2():

    mg1 = MicroGraph(defaultdict(set,\
            {"1":{"2","3"},\
            "2":{"1","3"},\
            "3":{"1","2","4"},\
            "4":{"3"},\
            "6":set()})) 

    mg2 = MicroGraph(defaultdict(set,\
            {"1":{"2","3"},\
            "2":{"1"},\
            "3":{"1","4","5"},\
            "4":{"3","5"},\
            "5":{"3","4"}}))

    mg3 = MicroGraph(defaultdict(set,\
            {"1":{"2","3"},\
            "2":{"1","3","4"},\
            "3":{"1","2","4"},\
            "4":{"2","3","6"},\
            "5":{"3"},\
            "6":{"4"}}))

    mg4 = MicroGraph(defaultdict(set,\
            {"1":{"2","4"},\
            "2":{"1","3"},\
            "3":{"2","4"},\
            "4":{"1","3"}})) 

    return GCSContainer([mg1,mg2,mg3,mg4])  

def greatest_common_subgraph_case_3():

    mg1 = MicroGraph(defaultdict(set,\
            {"1":{"2","3"},\
            "2":{"1","4"},\
            "3":{"1","4"},\
            "4":{"2","3","5"},\
            "5":{"3","4"}}))

    mg2 = MicroGraph(defaultdict(set,\
            {"1":{"2"},\
            "2":{"1","3","5","6"},\
            "3":{"2","4"},\
            "4":{"3"},\
            "5":{"2"},\
            "6":{"2"}}))

    mg3 = MicroGraph(defaultdict(set,\
            {"1":{"2"},\
            "2":{"1","3","4"},\
            "3":{"2"},\
            "4":{"2","5","6"},\
            "5":{"4","6"},\
            "6":{"4","5"}}))

    mg4 = MicroGraph(defaultdict(set,\
            {"1":set(),\
            "2":{"3"},\
            "3":{"2"},\
            "4":{"5","6"},\
            "5":{"4","6"},\
            "6":{"4","5"}}))

    return GCSContainer([mg1,mg2,mg3,mg4])  

class GCSContainerClass(unittest.TestCase):

    def test__GCSContainer_search__FullNeighborFitT2__case_1(self):
        gcg3 = greatest_common_subgraph_case_3()

        gcg3.search_type = "full neighbor fit- type 2"
        gcg3.initialize_cache() 
        s1,s2 = gcg3.search() 
        assert s2 == 6
        return

    def test__GCSContainer_search__FullNeighborFitT2__case_1(self):
        gcg2 = greatest_common_subgraph_case_2()
        gcg2.search_type = "full neighbor fit- type 1"
        gcg2.initialize_cache() 
        s1,s2 = gcg2.search() 
        assert s2 == 7
        return

    def test__GCSContainer_search__MatchingNeighborFit__case_1(self):
        gcg1 = greatest_common_subgraph_case_1()
        gcg1.search_type = "matching neighbor fit"
        gcg1.initialize_cache() 
        s1,s2 = gcg1.search() 
        assert s2 == 7
        return

if __name__ == '__main__':
    unittest.main()