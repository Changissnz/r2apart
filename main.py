####
"""
def fu_func():
    print("F.U.")

if __name__ == "__main__":
    fu_func()
"""
####
import logging
import numpy as np
from resource_defaults import *
from pmove import *
from trapmatch_env import *
import time

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
t = time.time()
tme.move_one_timestamp()
t2 = time.time()
print("time: ", t2 - t) 

if __name__ == "__main__":
    print("Hello")