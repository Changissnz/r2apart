# the FARSE machine, used to train players to make better moves. 
from farse_mach_writer import *

DEFAULT_TS_HOP_SEQ = [1,2,3,5]

class FARSESearchBestSolutions:

      """
      map is
            TMEnv parent identifier -> hop length -> (best score, <PContext seq.>)
      """
      def __init__(self,defdict=defaultdict(defaultdict)):
            self.defdict = defdict

      """
      return:
      - is best solution, valid next hop
      """
      def process_tmenv(self,tmenv,score):
            assert len(tmenv.fi.th) > 0

            next_hop = tmenv.fi.th[0]
            max_hop = tmenv.fi.th[-1]

            # check if the timestamp differences match
            diff = tmenv.fi.ct - tmenv.fi.rt2
            q = next_hop - diff
            assert q == 0

            if q == 0:
                  ##print("PMI: ", tmenv.fi.pmi)
                  if tmenv.fi.pmi not in self.defdict:
                        self.defdict[tmenv.fi.pmi] = defaultdict(None)
                  if next_hop not in self.defdict[tmenv.fi.pmi]:
                        self.defdict[tmenv.fi.pmi][next_hop] = - float('inf')

                  # case: better solution
                  if score > self.defdict[tmenv.fi.pmi][next_hop]:
                        self.defdict[tmenv.fi.pmi][next_hop] = score
                        return True, True
                  return False,True
            else:
                  if q + tmenv.fi.ct > tmenv.fi.rt + max_hop:
                        return False,False
                  return False,True

      """
      
      """
      def write_PContexts_out_to_file(self,tmenv,hop,fp):
            return -1

      def write_context_out_to_file(self,tmenv,hop,fp):

            return -1

## have to process by (TMEnv parent, hop)
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

      def __init__(self,tmenv,timestamp_hop_seq = DEFAULT_TS_HOP_SEQ,\
            perf_func = basic_performance_function,filepaths=None):
            self.tme = tmenv
            self.tme_pre = None
            self.ths = timestamp_hop_seq
            self.pf = perf_func
            self.filepaths = filepaths
            self.fwriter = None

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
            self.dec_cache = []

            self.fsbs = FARSESearchBestSolutions()

            # TMEnv idn, hop
            self.cache_target = [None,None]
            return

      def preprocess(self,training_player_idn):
            self.mark_training_player(training_player_idn)
            self.initialize_FI_cache()
            self.initialize_writer()
            return -1

      def initialize_writer(self):
            if type(self.filepaths) == type(None):
                  print("no filepaths for writer")
                  return
            self.fwriter = FARSEWriter(self.filepaths[0],\
                  self.filepaths[1],self.filepaths[2])
            self.fwriter.preprocess()

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
            if len(self.hopsearch_cache) == 0:
                  return False

            self.tme = deepcopy(self.hopsearch_cache.pop(0))
            self.trial_move_one_timestamp()

            # variable to reset timestamp
            timestamp_marker = len(self.tmp_cache)
            c = 0
            while len(self.tmp_cache) > 0:
                  self.tme = deepcopy(self.tmp_cache.pop(0))
                  self.trial_move_one_timestamp()
                  c += 1

                  if c != timestamp_marker:
                        self.timestamp_counter -= 1
                  else:
                        print("next timestamp: ",self.timestamp_counter)
                        c = 0
                        timestamp_marker = len(self.tmp_cache)
            self.review_dec_cache()
            return

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
            """
            print("** context move index")
            print(self.context_move_index)
            print("** tmp cache len")
            print(len(self.tmp_cache))
            """
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
                        ##dec_index = self.context_move_index[0] -1 

            if self.training_player_active[0] and not \
                  self.training_player_active[1]:
                  self.add_training_cycle_to_tmpcache()##parent_idn,dec_index)

            return 

      def load_training_info_into_TMEnv(self):
            p = self.fetch_training_player()
            hp = p.hollow_player()

            # case: FARSE has not started running
            if type(self.tme_pre) == type(None):
                  self.tme.fi = FARSEInfo(None,self.timestamp_counter,self.timestamp_counter,\
                        self.timestamp_counter,hp,None,deepcopy(self.ths))
                  return 

            # case: FARSE already started running 
            self.tme.fi.ct = self.timestamp_counter + 1
            self.tme.fi.pmi = self.tme_pre.idn
            """
            index = 0 if type(self.context_move_index[0]) == type(None) else \
                  self.context_move_index[0]
            p.pdec.pcontext.set_selection_descriptor(index)
            self.tme.fi.pcontext_seq.append(deepcopy(p.pcontext))
            """

            return

      def set_TMEnv_idn(self):
            self.tme.assign_idn(next(self.tmenv_counter))
            return

      """
      adds the TMEnv instance loaded into `self.tme` and checks if
      dec_cache or tmp_cache. 
      """
      def add_training_cycle_to_tmpcache(self):
            #
            p = self.fetch_training_player()
            index = 0 if type(self.context_move_index[0]) == type(None) else \
                  self.context_move_index[0]
            try:
                  p.pdec.pcontext.set_selection_descriptor(index)
            except:
                  print("index {} out of bounds".format(index))
                  return

            assert len(self.tme.fi.th) > 0

            # update the PContext seq
            self.tme.fi.pcontext_seq.append(deepcopy(p.pdec.pcontext))

            # check if length of next hop
            if self.tme.fi.ct - self.tme.fi.rt2 == self.tme.fi.th[0]:
                  self.dec_cache.append(deepcopy(self.tme))
            else:
                  self.tmp_cache.append(deepcopy(self.tme))
            return

      def review_dec_cache(self):
            ##print("REVIEWING")
            best_tme_index = None
            # score the performance
            for (i,x) in enumerate(self.dec_cache):
                  score = self.review_tme(x)
                  ##print("score: ",score)
                  stat1,stat2 = self.fsbs.process_tmenv(x,score)

                  # invalid next hop
                  if not stat2:
                        continue

                  # best solution so far
                  if stat1: 
                        best_tme_index = i

            # add the best solution back to `hopsearch_cache`
            # as two copies, one with the previous hop and one
            # without 
            if type(best_tme_index) == type(None):
                  print("NO BEST SOLUTION")
                  return

                  # update the hollow player
            tme = self.dec_cache.pop(best_tme_index)
            p = self.fetch_training_player()
            tme.fi.hpr = p.hollow_player()
            tme.fi.rt2 = tme.fi.ct
            print("BEST SOLUTION")
            print(str(tme.fi))
            print()

            q = tme.fi.th[0]
            self.hopsearch_cache.append(tme)
            self.write_to_file(tme,q)
            
            tme2 = deepcopy(tme)
            tme2.fi.th = tme2.fi.th[1:]
            self.hopsearch_cache.append(tme2)

            # for the remaining solutions, add them back to tmp_cache
            l = len(self.dec_cache)
            self.tmp_cache = self.dec_cache
            self.dec_cache = [] 

            for x in self.tmp_cache:
                  x.fi.th = x.fi.th[1:]

            ## ?? REMOVE
            assert len(self.tmp_cache) == l

      def write_to_file(self,tme,hop_length):
            if type(self.fwriter) == type(None):
                  print("no file-write")
                  return
            self.fwriter.write_TME_to_file(tme,hop_length)
            return

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
