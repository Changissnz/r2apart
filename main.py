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

### TEST CASE 1: <TMEnv>

i = 81#4
num_players = 3
num_moves_range = (4,7)
drange = (16,30)
connectivity_range = (0.25,0.6)
excess_range = [10**4,20**6]
game_modes = ["noneg","public"]
farse_mach = None

tme = TMEnv.generate__type_dumb(i,num_players,num_moves_range,\
    drange,connectivity_range,excess_range,game_modes,farse_mach)

### demonstrate moving one timestamp
t = time.time()
    ##

tme.move_one_timestamp()
tme.save_state(fp = "pickled_tme_state")

### demonstrating executing a PMove by player 0
###############
"""
tme.feed_moving_player_info(0)

# execute a PMove
q = tme.players[0].pdec.pcontext.pmove_prediction
qm = set(q.keys())
mv = qm.pop()
print("executing move {} for player 0".format(mv))
index = tme.players[0].pmove_idn_to_index(mv)
tme.exec_PMove(0,index)
"""
################
#\"""
t2 = time.time()
print("time: ", t2 - t) 
#\"""
######################################################

# loading a saved TMEnv
"""
tme = TMEnv.open_state("pickled_tme_state_2")
tme.move_one_timestamp()
tme.save_state(fp = "pickled_tme_state_3")
"""
####
##############################
"""
###############################
# focus on player 1, PMove 3.
px = tme.idn_to_player("1")
q = px.pmove_idn_to_index("2")
mv = px.ms[q]

print("TARGET")
print(mv.payoff_target)
print("ANTITARGET")
print(mv.antipayoff_target)
    # try calculating the number of isomorphisms for 
    # PMove 3, antitarget
##mgpt = MicroGraph.from_ResourceGraph(mv.payoff_target)
##mgat = MicroGraph.from_ResourceGraph(mv.antipayoff_target)
##seq = px.rg.subgraph_isomorphism(mgpt,all_iso=True)
##seq2 = px.rg.subgraph_isomorphism(mgat,all_iso=True)

    # PMove 2, antitarget
##mgpt = MicroGraph.from_ResourceGraph(mv.payoff_target)
##mgat = MicroGraph.from_ResourceGraph(mv.antipayoff_target)
##seq = px.rg.subgraph_isomorphism(mgpt,all_iso=True)
##seq2 = px.rg.subgraph_isomorphism(mgat,all_iso=True)
"""

if __name__ == "__main__":
    print("Hello")