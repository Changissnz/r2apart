from trapmatch_env import *
from morebs2 import ball_comp_test_cases,ball_comp,violation_handler
import os
import csv 
import shutil

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
      diff = [(s - meen) ** 2 for s in seq] 
      return mean_safe_division(diff) 

"""
return:
- all control point values are normalized to the smallest non-zero number.
"""
def normalized_context_function(quad):
    assert len(quad) == 4
    q2 = [q for q in quad if abs(q) > 0]
    q2 = min(q2) if len(q2) > 0 else 0
    if q2 == 0:
        return [0,0,0,0]
    return [q / q2 for q in quad]

"""
control points are: mean, min, max, and std. dev. 

For every one of the types found in STD_DEC_WEIGHT_SEQLABELS,
calculates control points.

Control point captures are used for <BallComp> classifications.

return: 
- sequence of float values
"""
def control_point_capture_on_PContextDecision(pc:PContext):

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
      print("PCD LENGTH: {}".format(len(pc.pcd.ranking)))
      for x in STD_DEC_WEIGHT_SEQLABELS:
            indices = pc.pcd.indices_for_possible_move_type(x)
            subseq = [pc.pcd.ranking[i] for i in indices]
            print("control points for {}: {}".format(x,indices))
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
        self.vfile_writer = None
        self.vfile = None

    def preprocess(self,overwrite=True):
        self.make_dir(overwrite)
        self.declare_bookmark_file()
        q = self.folder_path + "/" + self.fp2
        self.vfile = open(q,"w",newline="")
        self.vfile_writer = csv.writer(self.vfile,delimiter=",")
        return

    def make_dir(self,overwrite=True):
        stat = os.path.exists(self.folder_path)
        if not stat and not overwrite:
            raise ValueError("folder already exists!")
        print("making dir")
        if stat and overwrite:
            if os.path.exists(self.folder_path):
                shutil.rmtree(self.folder_path)
        os.makedirs(self.folder_path)
        return

    def declare_bookmark_file(self):
        q = self.folder_path + "/" + "bookmark.md"
        self.bookmark_file = open(q,"w")
        return 

    def write_TME_to_file(self,tme,hop_length:int):
        # write out PContext sequence and capture vectors
        # into list
        """
        print("write hop length: ", hop_length)
        """

        assert len(tme.fi.pcontext_seq) >= hop_length
        q = deepcopy(tme.fi.pcontext_seq[-hop_length:])
        fi_idn = next(self.file_idn_counter)
        idns = []

        vecs = []
        for q_ in q:
            """
            print("-- context")
            print(q_.condensed_form(False))
            """
            print("-- context seq")
            print(q_.pcd.to_one_vec())
            print()

            idn = next(self.context_idn_counter)
            q_.idn = idn
            idns.append(idn)

            v = control_point_capture_on_PContextDecision(q_)
            v.insert(0,idn)
            vecs.append(v) 

        f = open(self.folder_path + "/" + self.fp1 + "-" + fi_idn,\
                "wb")
        pickle.dump(q,f)

        # log it into bookmark
        self.write_to_bookmark(fi_idn,idns)    

        # write out vectorized forms
        """
        print("writing vecs")
        print(vecs)
        print()
        """
        for v in vecs:
            self.vfile_writer.writerow(v)
        self.flush_data()

        return

    def write_to_bookmark(self,file_idn,context_idn_seq):
        cis = "-".join(context_idn_seq)
        q = file_idn + "," + cis + "\n"
        self.bookmark_file.write(q)
        return

    def flush_data(self):
        self.bookmark_file.flush()
        self.vfile.flush()

    def end_write(self):
        self.bookmark_file.close()
        self.vfile.close()

class FARSEReader:

    def __init__(self,folder_path,fp1,fp2):
        self.folder_path = folder_path
        assert "." not in fp1
        # filename prefix for PContext
        self.fp1 = fp1

        # filename for vectorized form of PContext 
        self.fp2 = fp2
        assert os.path.exists(self.folder_path)

        # bookmark file 
        self.d = None 
        self.num_contexts = None
        self.preprocess()

    def preprocess(self):
        self.count_number_of_contexts()
        self.load_bookmark_file()

    """
    list of vector samples, corresponding list of <PContext> samples. 
    """
    def fetch_by_key(self,k:int):
        v = self.d[k]
        q = self.folder_path + "/" + self.fp1 + "-" + str(k)
        xs0,xs1 = None,None
        if os.path.isfile(q):
            f = open(q,"rb")
            xs1 = pickle.load(f)
            xs0 = self.fetch_vector_samples(v) 
            return xs0,xs1 
        else:
            print("[!] NOT A FILE [!!]")
        return None

    def fetch_vector_samples(self,vector_idns):
        vector_idns = sorted(vector_idns)
        fx = open(self.folder_path + "/" + self.fp2,"r")
        samples = {} 
        for line in fx:
            if len(vector_idns) == 0:
                break 

            x = line.split(",")
            assert len(x) > 0
            if int(x[0]) in vector_idns:
                q = x[1:]
                q = [float(q_) for q_ in q]
                samples[int(x[0])] = q
                vector_idns.remove(int(x[0]))
        fx.close() 
        return samples 

    """
    """
    def count_number_of_contexts(self):
        folder_elements = os.listdir(self.folder_path)
        contexts = 0
        for f in folder_elements:
            stat = os.path.isfile(self.folder_path + "/" + f) 
            if stat:
                q = f.split("-")
                if q[0] == self.fp1:
                    contexts += 1
        self.num_contexts = contexts 
        return contexts

    def load_bookmark_file(self):
        q = self.folder_path + "/" + "bookmark.md"
        assert os.path.isfile(q)
        self.d = defaultdict(list)
        with open(q,"r") as fx:
            filelines = fx.readlines() 
            for fx in filelines:
                fx_ = fx.rstrip()
                qx = fx_.split(",")
                q0 = qx[0]
                q1 = qx[1].split("-")
                q1 = [int(q1_) for q1_ in q1]
                self.d[int(q0)] = q1
        return