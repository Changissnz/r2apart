# file contains classes used for tests
from farse_mach import *

def TMEnv_sample_1(verbose_mode=False,i=81):
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