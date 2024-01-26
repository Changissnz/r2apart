# the FARSE machine, used to train players to make better moves. 
from trapmatch_env import *
from morebs2 import ball_comp_test_cases,ball_comp,violation_handler

DEFAULT_TS_HOP_SEQ = [2,3,6]

# TODO: this is a basic function
"""
return:
- float value, larger values correspond to better performance
"""
def basic_performance_function(player_pre,player_post):
      assert type(player_pre) == type(player_post)
      assert type(player_pre) == Player

      h1 = player_post.cumulative_health()
      h0 = player_pre.cumulative_health() 
      
      # calculate difference in health
      diff = h1 - h0 

      # add difference to post-health
      return h1 + diff

def std_dev(seq,meen):
      if len(seq) == 0:
            return 0
      diff = [(s - mean) ** 2 for s in seq] 
      return mean_safe_division(diff) 

"""
return:
- all control point values are normalized to the smallest non-zero number.
"""
def normalized_context_function(quad):
      assert len(quad) == 4
      q2 = [q for q in quad if abs(q) > 0]
      q2 = min(q2)
      return [q / q2 for q in quad]

def is_pertinent_timestamp(ts,hop_seq):
      # case: timestamp an element of hop_seq
      if ts in hop_seq:
            return True
      #h = 
      return -1

"""
control points are: mean, min, max, and std. dev. 

For every one of the types found in STD_DEC_WEIGHT_SEQLABELS,
calculates control points.

Control point captures are used for <BallComp> classifications.

return: 
- sequence of float values
"""
def control_point_capture_on_PContextDecision(pcd:PContextDecision):

      """
      """
      def calculate_control_points(subseq):
            subseq2 = [s[1] for s in subseq]
            if len(subseq2) == 0:
                  return 0,0,0,0
            q = [mean_safe_division(subseq2)]
            q.append(min(subseq2))
            q.append(max(subseq2))
            q.append(std_dev(subseq2,q[0]))
            return q

      cxs = []
      for x in STD_DEC_WEIGHT_SEQLABELS:
            indices = pcd.indices_for_possible_move_type(x)
            q = [pcd.ranking[i] for i in indices]
            cx = calculate_control_points(subseq)
            cx = normalized_context_function(cx)
            cxs.extend(cx) 
      return cxs

"""
class stores an element of <FARSE>'s trials to be written to the data file
"""
class HopInfo:

      def __init__(self,tmenv_idn:int,next_tmenv_idn:int,hop:int,stdcontext_vec_seq,best_decision_index_seq):
            self.tmenv_idn = tmenv_idn
            self.next_tmenv_idn = next_tmenv_idn
            self.hop = hop
            self.stdcontext_vec_seq = stdcontext_vec_seq
            self.best_decision_index_seq = best_decision_index_seq
            return

"""
the data structure used to store hop sequence information for a
<FARSE>'s `training_player`.
"""
class HopStamp:

      def __init__(self,reference_timestamp,hop_seq):
            self.reference_timestamp = reference_timestamp
            self.hop_seq = hop_seq
            self.hop_infos = defaultdict(None)
            return 

      def add_one_hop_info(self,tmenv_idn:int,next_tmenv_idn:int,hop:int,stdcontext_vec_seq,best_decision_index_seq):
            hi = HopInfo(tmenv_idn,next_tmenv_idn,hop,stdcontext_vec_seq,best_decision_index_seq)
            self.hop_infos[hop] = hi
            return

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

      def __init__(self,tmenv,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,perf_func = basic_performance_function):
            self.tme = tmenv
            self.tme_pre = None
            self.ths = timestamp_hop_seq
            self.pf = perf_func

            # [index of player @ start, idn of player]
            self.training_player = None

            # [terminated status, status of cycle being finished for the timestamp]
            self.training_player_active = None  
            
            # initial index in tme, idn of player
            self.context_move_index = [None,None] 
            # value from 1-MAX(`self.ths`)
            self.hopsearch_index = 0
            self.tmenv_counter = DefaultNodeIdnCounter("0")
            self.set_TMEnv_idn()

            self.timestamp_counter = 0
            self.reference_timestamp = None 

            self.hopsearch_cache = []
            self.tmp_cache = []    
            return

      def initialize_FI_cache(self):
            self.load_training_info_into_TMEnv()
            self.hopsearch_cache.append(deepcopy(self.tme))

      def mark_training_player(self,idn):
            q = self.tme.idn_to_index(idn)
            assert q != -1
            self.training_player = (q,idn)
            self.training_player_active = (True,False) 

      def fetch_training_player(self):
            assert type(self.training_player) != type(None)
            index = self.tme.idn_to_index(self.training_player[1])
            return self.tme.players[index]

      """
      only the training player is in verbose mode
      """
      def set_verbosity(self):
            return -1

      """
      runs one hop round, recording all the contexts 
      of the `training_player` into file `fp1`, and
      the best decision for each of the hops in `ths`. 
      """
      def run_one_hop_round(self):
            # set reference timestamp here.
            return -1

      def trial_move_one_timestamp(self):
            
            # set the ordering
            self.tme.set_ts_ordering()

            # make a copy of the environment
            self.tme_pre = deepcopy(self.tme)

            # move each player initially
            print("initial analysis & move")
            self.cycle_timestamp(False)
            
            # TODO: register info here
            self.restore_TMEnv_back_to_timestamp()

            # cycle until training player done
            while self.training_player_active[0] and \
                  not self.training_player_active[1]:
                  self.cycle_timestamp(True)
                  self.restore_TMEnv_back_to_timestamp()

            self.timestamp_counter += 1
            return

      def restore_TMEnv_back_to_timestamp(self):
            tme2 = self.tme
            self.tme = deepcopy(self.tme_pre)

            # get the training PContext for `tme2`
            index = tme2.idn_to_index(self.training_player[1])
            q = tme2.players[index].pdec.pcontext

            # assign it to `self.tme`
            index2 = self.tme.idn_to_index(self.training_player[1])
            self.tme.players[index2].pdec.pcontext = q

            return

      """
      method to repeat activity for a timestamp 
      """
      def cycle_timestamp(self,next_iter:bool):
            assert type(self.training_player) != type(None)
            parent_idn = deepcopy(self.tme.idn)
            self.set_TMEnv_idn()
            self.load_training_info_into_TMEnv()

                  ###
            print("** context move index")
            print(self.context_move_index)
            print("** tmp cache len")
            print(len(self.tmp_cache))
                  ###

            # convert the ordering to identifiers
            idns = []
            for tso in self.tme.ts_ordering:
                  idns.append(self.tme.players[tso].idn) 

            for idn in idns:
                  stat1,stat2 = self.trial_move_one_player(idn,next_iter)
                  # case: is training_player
                  if idn == self.training_player[1]: 
                        self.training_player_active = (stat1,stat2)
                        dec_index = self.context_move_index[0] -1 

                        if self.training_player_active[0] and not \
                              self.training_player_active[1]:
                              self.add_training_cycle_to_tmpcache()##parent_idn,dec_index)
            return 

      def load_training_info_into_TMEnv(self):
            p = self.fetch_training_player()
            hp = p.hollow_player()

            # case: FARSE has not started running
            if type(self.tme_pre) == None:
                  self.tme.fi = FARSEInfo(None,self.timestamp_counter,self.timestamp_counter,\
                        self.timestamp_counter,hp,None,deepcopy(self.ths))
                  return 


            # case: FARSE already started running 
            self.tme.fi.ct = self.timestamp_counter
            self.tme.fi.pti = self.tme_pre.idn
            if type(self.context_move_index[0]) == None:
                  self.tme.fi.tpdi = 0
            else:
                  self.tme.fi.tpdi = self.context_move_index[0]
            return

      def set_TMEnv_idn(self):
            self.tme.assign_idn(next(self.tmenv_counter))
            return

      """
      adds the TMEnv to cache and determines if it is the best solution
      """
      def add_training_cycle_to_tmpcache(self):
            # 
            self.tmp_cache.append(deepcopy(self.tme))
            return

      def review_tmp_cache(self):
            scores = []
            # score the performance
            for (i,x) in enumerate(self.tmpcache):
                  score = self.review_tme(x)
                  scores.append((i,score))

            ## NOTE: determine the best move based on
            ##       references. 
            """
            x = max(scores,key=lambda y: y[1])
            """
            return -1

      """
      outputs the score for the tme
      """
      def review_tme(self,tme):

            # get the score
            px = tme.idn_to_player(self.training_player[1])
            assert type(px) != type(None)
            return self.pf(tme.fi.hpr,px)
            
      """
      return:
      - player active, False or ?if training player?finished?
      """
      def trial_move_one_player(self,p_idn:str,next_iter:bool):
            print("moving player: ", p_idn, " | next iter: ", next_iter) 
            print("")
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
                  l = len(self.tme.players[q2].pdec.pcontext.pcd.ranking)
                  self.context_move_index[0] = 1
                  self.context_move_index[1] = l - 1
            return True,False 

      """
      """
      def select_next_move_for_training_player(self,i:int):
            if self.context_move_index[0] > self.context_move_index[1]:
                  return None
            nm = self.tme.players[i].pdec.pcontext.pcd.ranking[self.context_move_index[0]]
            self.context_move_index[0] = self.context_move_index[0] + 1
            return nm

      """
      call this at the end of each 
      """
      def clear_TMEnv_data(self):
            assert type(self.training_player) != type(None)
            # store the pcontext of training player into variable
            i = self.tme.idn_to_index(self.training_player[1])
            pc = self.tme.players[i].pdec.pcontext

            # clear all timestamp data
            self.tme.clear_timestamp_data()

            # upload PContext back into training player
            self.tme.players[i].pdec.pcontext = pc
            return

      # compares two timestamps, r (reference) and 
      # p (post) to determine how the player has fared
      # due in part to its decisions. 
      def compare_timestamps(self,r,p):
            return -1

      def select_best_decision_at_timestamp(self):
            return -1

      """

      """
      def cache_candidate_selection(self,size=1):

            return -1

      #############################################

      def process_timestamps(self):
            return -1

      """
      record context of training player into BallComp instance
      """
      def log_context(self):
            return -1
