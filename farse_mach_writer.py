from trapmatch_env import *
from morebs2 import ball_comp_test_cases,ball_comp,violation_handler
import os
import pandas as pd


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


class FARSEWriter:

    def __init__(self,folder_path,fp1,fp2):
        self.folder_path = folder_path
        # filename prefix for PContext
        self.fp1 = fp1
        # filename for vectorized form of PContext 
        self.fp2 = fp2

        self.file_idn_counter = DefaultNodeIdnCounter("0")
        self.context_idn_counter = DefaultNodeIdnCounter("0")
        self.bookmark_file = None

    def preprocess(self,overwrite=True):
        self.make_dir(overwrite)
        self.declare_bookmark_file()
        with open(self.folder_path + "/" + self.fp2,"w") as fi:
            return 
        return

    def make_dir(self,overwrite=True):
        stat = os.path.exists(self.folder_path)
        if not stat and not overwrite:
            raise ValueError("folder already exists!")
        os.makedirs(self.folder_path)
        return

    def declare_bookmark_file(self):
        q = self.folder_path + "/" + "bookmark.md"
        self.bookmark_file = open(q,"w")
        return 

    def write_TME_to_file(self,tme,hop_length:int):
        # write out PContext sequence and capture vectors
        # into list
        assert len(tme.fi.pcontext_seq) >= hop_length
        q = deepcopy(tme.fi.pcontext_seq[-hop_length:])
        fi_idn = next(self.file_idn_counter)
        idns = []

        vecs = []
        for q_ in q:
            idn = next(self.context_idn_counter)
            q_.idn = idn
            idns.append(idn)

            v = control_point_capture_on_PContextDecision(q_)
            v.insert(0,idn)
            vecs.append(v) 

        f = open(self.folder_path + "/" + self.fp1,\
                "wb")
        pickle.dump(q,f)

        # log it into bookmark
        self.write_to_bookmark(fi_idn,idns)    

        # write out vectorized forms
        df = pd.DataFrame(vecs)
        df.to_csv(path_or_bufstr = self.folder_path + "/" + self.fp2,\
                mode="a")
        return

    def write_to_bookmark(self,file_idn,context_idn_seq):
        cis = "-".join(context_idn_seq)
        q = file_idn + "," + cis + "\n"
        self.bookmark_file.write(q)
        return

    def end_write(self):
        self.bookmark_file.close()

class FARSEReader:

    def __init__(self,folder_path,fp1,fp2):
        self.folder_path = folder_path
        # filename prefix for PContext
        self.fp1 = fp1
        # filename for vectorized form of PContext 
        self.fp2 = fp2