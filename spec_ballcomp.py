"""
file contains a specialized <BallComp> class, for use with 
<ZugdeinlichtMach> 
"""

from morebs2 import ball_comp_test_cases,ball_comp,violation_handler

"""
uses frequency collection to obtain the best fit 
with the aid of a search queue

Every weight-fitter instance is to be used with a 


"""
class WeightFitter:

    """
    is_single_varseq denotes 
    """
    def __init__(self,is_single_varseq:bool):
        self.is_single_varseq = is_single_varseq
        self.reference = None
        self.other_samples = None
        # other sample index
        self.osi = None
        self.search_queue = []
        return

    def set_reference(self,reference):
        return -1

    def set_other_samples(self,other_samples):
        return -1 

    def fit_next(self):
        return -1

    def calculate_dirvec(self):
        return -1

    """
    calculates the possible candidates
    """
    def dirvec_single_varseq(self):
        q = self.other_samples[osi]
        z0 = self.reference > q
        z1 = self.reference - q
        self.reference - 

    def possible_dirvec1(self): 
        return -1 


class SpecializedBallComp:

    def __init__(self,maxBalls,maxRadius):
        vh1 = violation_handler.ViolationHandler1(maxBalls,maxRadius)
        self.bc = ball_comp.BallComp(maxBalls,maxRadius,vh1,verbose=0)
        self.ball2class_freq = defaultdict(None)
        self.ball2decfunc = defaultdict(None) 

    """
    BallComp construction
    """
    def fit_sample_round_1(self,vec):
        return -1

    """
    Ball2Class frequency collector
    """
    def fit_sample_round_2(self,vec,pcontext):
        return -1

    """
    weight fitting
    """
    def fit_sample_round_3(self,vec,pcontext):
        return -1


