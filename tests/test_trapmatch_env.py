from trapmatch_env import * 
import unittest

def TMEnv_sample_1(verbose_mode=False):
    i = 81#4
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

class TMEnvClass(unittest.TestCase):

    def test__TMEnv__move_one_timestamp__case_1(self):
        tme = TMEnv_sample_1()
        for i in range(3):
            print("moving one")
            tme.move_one_timestamp()
        return

if __name__ == '__main__':
    unittest.main()