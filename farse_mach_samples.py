from sample_classes import * 

def farse_mach_case_1():
    # run FARSE instance
    tme = TMEnv_sample_1()
    tme.set_player_verbosity(False)

    ## case 1: need to alter for assertion
    fm = FARSE(tme,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,\
        perf_func = basic_performance_function,filepaths=["fx","pcontekt","vec"])
    fm.preprocess("0")

    for i in range(15):
        fm.run_one_hop_round()

def farse_mach_case_2():
    tme = TMEnv_sample_1(i=343)
    tme.set_player_verbosity(False) 
    fm = FARSE(tme,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,\
        perf_func = basic_performance_function,filepaths=["fx2","pcontekt","vec"])
    fm.preprocess("0")

    for i in range(35):
        fm.run_one_hop_round()