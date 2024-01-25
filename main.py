from farse_mach import *
import time

def TMEnv_sample_1():
    i = 81#4
    num_players = 3
    num_moves_range = (4,7)
    drange = (16,30)
    connectivity_range = (0.25,0.6)
    excess_range = [10**4,20**6]
    game_modes = ["noneg","public"]
    farse_mach = None
    tme = TMEnv.generate__type_dumb(i,num_players,num_moves_range,\
        drange,connectivity_range,excess_range,game_modes,False)
    return tme

tme = TMEnv_sample_1()

tme.set_player_verbosity(False)
fm = FARSE(tme,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,perf_func = basic_performance_function)
fm.mark_training_player("0")
fm.trial_move_one_timestamp()

if __name__ == "__main__":
    print("Hello")


