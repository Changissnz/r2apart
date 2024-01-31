from sample_classes import *
import unittest

"""
wanted_values := [reference timestamp, reference timestamp #2,\
                parent index, current timestamp, target hop seq, \
                length of pcontext seq]
"""
def check_fi(fi,wanted_values):
    assert len(wanted_values) == 6 

    if fi.rt != wanted_values[0]: return False
    if fi.rt2 != wanted_values[1]: return False
    if fi.pmi != wanted_values[2]: return False
    if fi.ct != wanted_values[3]: return False
    if len(fi.th) != wanted_values[4]: return False
    if len(fi.pcontext_seq) != wanted_values[5]: return False
    return True

class FARSEClass(unittest.TestCase):

    def test__FARSE__trial_move_one_timestamp(self):
        tme = TMEnv_sample_1()
        tme.set_player_verbosity(False)
        fm = FARSE(tme,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,perf_func = basic_performance_function)
        fm.mark_training_player("0")
        fm.initialize_FI_cache()
        fm.trial_move_one_timestamp()

        ans = [0,0,"1",1,4,1]
        for x in fm.dec_cache:
            assert check_fi(x.fi,ans)
        assert len(fm.dec_cache) == 11

    def test__FARSE__run_one_hop_round__case_1(self):
        tme = TMEnv_sample_1()
        tme.set_player_verbosity(False)
        fm = FARSE(tme,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,perf_func = basic_performance_function)
        fm.mark_training_player("0")
        fm.initialize_FI_cache()

        sol_hopsearch = [0,1,"1",1,4,1]
        sol_tmp = [0,0,"1",1,3,1]
        fm.run_one_hop_round()

        for x in fm.tmp_cache:
            #print(str(x.fi))
            assert check_fi(x.fi,sol_tmp)

        for x in fm.hopsearch_cache:
            assert check_fi(x.fi,sol_hopsearch)

        assert len(fm.hopsearch_cache) == 1
        assert len(fm.tmp_cache) == 10
        assert len(fm.dec_cache) == 0
        return

    def test__FARSE__run_one_hop_round__case_2(self):
        tme = TMEnv_sample_1()
        tme.set_player_verbosity(False)
        ths = [2,4,5]

        fm = FARSE(tme,timestamp_hop_seq = ths,perf_func = basic_performance_function)
        fm.mark_training_player("0")
        fm.initialize_FI_cache()
        fm.run_one_hop_round()

        assert len(fm.hopsearch_cache) == 1
        assert len(fm.tmp_cache) == 135
        assert len(fm.dec_cache) == 0

 

if __name__ == '__main__':
    unittest.main()