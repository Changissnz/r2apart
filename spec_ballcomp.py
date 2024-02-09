"""
file contains a specialized <BallComp> class, for use with 
<ZugdeinlichtMach> 
"""

from morebs2 import ball_comp_test_cases,ball_comp,violation_handler

class SpecializedBallComp:

    def __init__(self,maxBalls,maxRadius):
        vh1 = violation_handler.ViolationHandler1(maxBalls,maxRadius)
        self.bc = ball_comp.BallComp(maxBalls,maxRadius,vh1,verbose=0)
        self.ball2class_freq = defaultdict(None)

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


