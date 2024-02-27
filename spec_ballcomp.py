"""
file contains a specialized <BallComp> class, for use with 
<ZugdeinlichtMach> 
"""

from morebs2 import ball_comp_test_cases,ball_comp,violation_handler
from collections import defaultdict
from copy import deepcopy 
import numpy as np

# default value that a maximum decision is to exceed
# non-maximum decisions.
DEFAULT_FITTING_EQ_CONSTANT = 5.0

"""
ex1 := np.array, 1-dimensional, same length as `ex2`;
       supposed to output max.
ex2 := np.array, 1-dimensional, same length as `ex1`.
"""
def solve_equation_type__even_dist_AND_single_varseq(ex1,ex2,c):
    assert len(ex1) == len(ex2)
    ex1,ex2 = np.array(ex1),np.array(ex2)
    q = ex1 - ex2
    x = np.sum(np.abs(q))
    if x == 0:
        return None
    c2 = c / x
    ans = np.zeros(len(ex1))
    for i in range(len(q)):
        s = -1 if q[i] < 0.0 else 1
        ans[i] = s * q[i]
    ans = np.append(ans,0)
    return ans 

"""
uses frequency collection to obtain the best fit 
with the aid of a search queue

Every weight-fitter instance is to be used to solve
the most frequent class of one <Ball> instance pertaining
to a <SpecializedBallComp>. 
"""
class WeightFitterType1:

    def __init__(self):
        self.reference = None
        self.other_samples = None
        self.search_queue = []
        self.tally = []
        return

    def set_sample(self,reference,other_samples):
        self.set_reference(reference)
        self.set_other_samples(other_samples)
        return

    def set_reference(self,reference):
        self.reference = reference 
        return

    def set_other_samples(self,other_samples):
        self.other_samples = other_samples
        return

    def fit_one(self,reference,other_samples,label:int,is_max_label:bool):\
        
        self.set_sample(reference,other_samples)

        # case: not max label
        if not is_max_label:
            print("NOT MAX LABEL")
            return 
        
        l = len(self.reference) 

        # case: 0 other samples
        if len(self.other_samples) == 0: 
            x = [1 for i in range(0,l)]
            x.append(0)
            self.search_queue.append(x)
            return 

        # NOTE: not optimal
        for o in self.other_samples:
            q = solve_equation_type__even_dist_AND_single_varseq(\
                deepcopy(self.reference),deepcopy(o),DEFAULT_FITTING_EQ_CONSTANT)
            self.search_queue.append(q)
        return

class SpecializedBallComp:

    def __init__(self,maxBalls,maxRadius,weight_fitter_type=WeightFitterType1):
        vh = violation_handler.ViolationHandler1(maxBalls,maxRadius) 
        self.bc = ball_comp.BallComp(maxBalls,maxRadius,vh,0)
        self.wft = weight_fitter_type

        # ball idn -> class idn -> frequency
        self.ball2class_freq = defaultdict(None)
        self.ball2weightfitter = defaultdict(None)
        self.ball2maxclass = defaultdict(None)

    def initialize_weightfitter(self):
        for k in self.ball2class_freq.keys():
            self.ball2weightfitter[k] = self.wft()
        return

    """
    BallComp construction
    """
    def fit_proc1(self,vec):
        if self.bc.terminateDelta:
            return
        return self.bc.conduct_decision(vec)

    """
    Ball2Class frequency collector
    """
    def fit_proc2(self,vec,pcontext):
        label = self.bc.ball_label_for_point(vec)
        class_idn = pcontext.selection_descriptor
        class_idn = class_idn.split("-")[0]

        if label not in self.ball2class_freq:
            self.ball2class_freq[label] = defaultdict(int)
        self.ball2class_freq[label][class_idn] += 1
        return

    """
    method called after fit_proc(1&2) and before fit_proc3
    """
    def prefit_process(self):
        # get the max class for each ball
        for (k,v) in self.ball2class_freq.items():
            q = self.ranked_class_frequency_for_ball(k)
            if len(q) == 0: continue
            q1 = q[0][0].split("-")
            self.ball2maxclass[k] = q1[0] 

        # TODO:
        # empty each Ball's data to save memory

        # initialize a weight-fitter class for each Ball
        self.initialize_weightfitter()
        return

    def ranked_class_frequency_for_ball(self,ball_idn):
        q = []
        for k,v in self.ball2class_freq[ball_idn].items():
            q.append([k,v])
        return sorted(q,key=lambda x: x[1],reverse=True)
 
    """
    weight fitting
    """
    def fit_proc3(self,vec,pcontext):
        label = self.bc.ball_label_for_point(vec)
        class_idn = deepcopy(pcontext.selection_descriptor)
        q = class_idn.split("-")
        opt_arg = "" 
        if len(q) > 1:
            opt_arg = q[1]

        # retrieve the info from the PContext:
        vecs = pcontext.selection_to_sample_vectors(opt_arg)

        # case: no samples
        if len(vecs) == 0:
            print("NO SAMPLES")
            return
     
        max_class = self.ball2maxclass[label] 
        wf = self.ball2weightfitter[label]

        reference = vecs.pop(0)
        wf.fit_one(reference,vecs,label,max_class == q[0])

    def bc_search_queue_size_display(self):
        for (k,v) in self.ball2weightfitter.items():
            print("ball idn: ",k)
            print("weight size: ",len(v.search_queue))
        return

"""
hop-separated specialized ball comp; 
contains h number of <SpecializedBallComp> instances, 
each corresponding to a unique hop length.
"""
class HopSeparatedSpecBallComp:

    def __init__(self,hop_ball_radiii):
        self.d = defaultdict(None)
        self.initialize_sbc(hop_ball_radiii)
        return

    def initialize_sbc(self,hbr):
        for x in hbr:
            self.d[x[0]] = SpecializedBallComp(x[1],x[2])

    def fit_proc1(self,hl,vec):
        assert hl in self.d
        self.d[hl].fit_proc1(vec)
        return

    def fit_proc2(self,hl,vec,pcontext):
        assert hl in self.d
        self.d[hl].fit_proc2(vec,pcontext)
        return

    def run_prefit_proc(self):
        for v in self.d.values():
            v.prefit_process()

    def fit_proc3(self,hl,vec,pcontext):
        assert hl in self.d
        self.d[hl].fit_proc3(vec,pcontext)
        return

    def basic_display(self):
        for (k,v) in self.d.items():
            print("SBC key: {}".format(k))
            v.bc_search_queue_size_display()
        return -1