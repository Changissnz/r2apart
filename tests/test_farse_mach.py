from sample_classes import *
import unittest

"""
wanted_values := [reference timestamp, reference timestamp #2,\
                current timestamp, target hop seq, length of pcontext seq]
"""
def check_fi(fi,wanted_values):
    assert len(wanted_values) == 5 

    if fi.rt != wanted_values[0]: return False
    if fi.rt2 != wanted_values[1]: return False
    if fi.ct != wanted_values[2]: return False
    if len(fi.th) != wanted_values[3]: return False
    if len(fi.pcontext_seq) != wanted_values[4]: return False
    return True

class FARSEClass(unittest.TestCase):

    def test__FARSE__trial_move_one_timestamp(self):
        tme = TMEnv_sample_1()
        tme.set_player_verbosity(False)
        fm = FARSE(tme,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,perf_func = basic_performance_function)
        fm.mark_training_player("0")
        fm.initialize_FI_cache()
        fm.trial_move_one_timestamp()

        ans = [0,0,1,4,1]
        for x in fm.tmp_cache:
            assert check_fi(x.fi,ans)

if __name__ == '__main__':
    unittest.main()