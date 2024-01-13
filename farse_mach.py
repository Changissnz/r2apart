# the FARSE machine, used to train players to make better moves. 
from trapmatch_env import *
from morebs2 import ball_comp_test_cases,ball_comp,violation_handler

DEFAULT_TS_HOP_SEQ = [2,3]

"""
used for training purposes
"""
class WantedDec:

      def __init__(self):

            return -1

## STEPS
# implement the below two functions
# extend BallComp into CategoricalBallComp
      # class stores the weights necessary
      # for each context to output a specified
      # decision. 

"""
return:
- float value, larger values correspond to better performance
"""
def basic_performance_function(player):
      return -1

def normalized_context_function(player):

      return -1

"""
decision-tree learning system that applies trial-and-error 
principles alongside metrological functions.

FARSE is an acronym for an abstract machine,
and stands for: 
* F ix all players except for player P_t to train so that only P_t's moves can be
      alternated at a particular timestamp. All of the other players will act entirely
      by their pre-programmed functionalities.
* A lternate the possible moves of P_t at a selected timestamp t_i in order to observe for
      moves that lead to improvements over pre-programmed functionalities.
* R ank the moves conducted by P_t at time t_i.
* S ave the results of the trial-and-error process into a format such as a decision
      function.
* E xecute all of P_t's future moves using the knowledge (such as improved decision functions)
      produced through the previous steps. 
"""
class FARSE:

      def __init__(self,tmenv,ballcomp_instance,\
            timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,perf_func = basic_performance_function):
            self.tme = tmenv
            self.tme_pre = None
            self.ballcomp_instance = ballcomp_instance
            self.pf = perf_func

            # initial index in tme, idn of player
            self.training_player = None
            self.training_player_active = None  
            self.context_move_index = [None,None]
            return

      def mark_training_player(self,idn):
            q = self.tme.idn_to_index(idn)
            assert q != -1
            self.training_player = (q,idn)
            self.training_player_active = (True,False) 

      """
      only the training player is in verbose mode
      """
      def set_verbosity(self):
            return -1

      def trial_move_one_timestamp(self):
            # set the ordering
            self.tme.set_ts_ordering()

            # make a copy of the environment
            self.tme_pre = deepcopy(self.tme)

            # move each player initially
            self.cycle_timestamp(False)
            
            # TODO: register info here

            self.restore_TMEnv_back_to_timestamp()

            # cycle until training player done
            while self.training_player_active[0] and \
                  not self.training_player_active[1]:
                  self.cycle_timestamp(True)
                  self.restore_TMEnv_back_to_timestamp()
            return

      def restore_TMEnv_back_to_timestamp(self):
            self.tme = deepcopy(self.tme_pre) 
            return

      """
      method to repeat activity for a timestamp 
      """
      def cycle_timestamp(self,next_iter:bool):
            assert type(self.training_player) != type(None)

            # convert the ordering to identifiers
            idns = []
            for tso in self.ts_ordering:
                  idns.append(self.players[tso].idn) 

            for idn in idns:
                  stat1,stat2 = self.trial_move_one_player(idn,next_iter)
                  if idn == self.training_player[1]: 
                        self.training_player_active = (stat1,stat2) 
            return 

      """
      return:
      - player active, False or ?if training player?finished?
      """
      def trial_move_one_player(self,p_idn:str,next_iter:bool):
            assert type(self.training_player) != type(None)

            q = self.tme.idn_to_index(p_idn)
            
            # player has been terminated
            if q == -1:
                  return False, False 

            # case: training player that has already been
            #       initialized, move on to the next
            if p_idn == self.training_player[1] and next_iter:
                  m = self.select_next_move_for_training_player(q)
                  if type(m) == type(None): return True,True
                  self.tme.exec_player_choice(q,m)
                  return True,False

            # case: run basic <TMEnv.move_one_player>
            self.tme.move_one_player(q)

            # set the index if training player
            if p_idn == self.training_player[1]:
                  q2 = self.tme.idn_to_index(p_idn)
                  l = len(self.tme.players[q2].pdec.pcontext.pcd)
                  self.context_move_index[0] = 1
                  self.context_move_index[1] = l - 1
            return True,False 

      """
      """
      def select_next_move_for_training_player(self,i:int):
            if self.context_move_index[0] > self.context_move_index[1]:
                  return None
            nm = self.tme.players[i].pcontext.pcd[self.context_move_index[0]]
            self.context_move_index[0] = self.context_move_index[0] + 1
            return nm

      """
      call this at the end of each 
      """
      def fetch_relevant_info_on_timestamp(self):
            return -1

      # compares two timestamps, r (reference) and 
      # p (post) to determine how the player has fared
      # due in part to its decisions. 
      def compare_timestamps(self,r,p):
            return -1

      def select_best_decision_at_timestamp(self):
            return -1

      #############################################

      def process_timestamps(self):

            return -1

      def score_training_player_performance(self):
            return -1 

      """
      record context of training player into BallComp instance
      """
      def log_context(self):
            return -1