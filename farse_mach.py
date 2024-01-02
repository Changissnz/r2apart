# the FARSE machine, used to train players to make better moves. 
from trapmatch_env import *

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

      def __init__(self,tmenv):
            self.tme = tmenv
            return

      def mark_training_player(self,idn):
            return -1

      def trial_move_one_timestamp(self):
            return -1


      def move_one_timestamp(self):
            return -1

      def full_alternate_one_timestamp(self):
            return -1

      def alternate_one_timestamp(self):
            return -1

      # compares two timestamps, r (reference) and 
      # p (post) to determine how the player has fared
      # due in part to its decisions. 
      def compare_timestamps(self,r,p):
            return -1

