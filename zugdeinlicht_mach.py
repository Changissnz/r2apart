from farse_mach import * 
from spec_ballcomp import *

# NOTENCHKO (notierro)
## uses 3rd party library, numpy

def default_ballcomp_radius(minimum,maximum,num_balls,ratio_of_max):
    diff = maximum - minimum
    max_radius = max(diff) / (num_balls * 2)
    return max_radius * ratio_of_max 

"""
Algorithm that uses stochastic ordering of points to 
construct a BallComp instance, or add onto it.
"""
class ZugdeinlichtMach:

    def __init__(self,ballcomp_arg,training_data_folders,rdm_seed:int,\
        timestamp_hop_seq = deepcopy(DEFAULT_TS_HOP_SEQ)):
        assert type(training_data_folders) == list 
        self.training_data_folders = training_data_folders
        self.target_folder = None
        self.tf_file_ordering = None
        self.farse_reader = None        
        if type(rdm_seed) != type(None):
            random.seed(rdm_seed)
        self.timestamp_hop_seq = timestamp_hop_seq
        self.ballcomp = None

        # case: pickled instance of BallComp stored in file open it
        if type(ballcomp_arg) == str:
            f = open(ballcomp_arg,"rb")
            self.ballcomp = pickle.load(f)
        # case: wanted number of Balls,
        elif type(ballcomp_arg) == list:
            self.initialize_ballcomp(ballcomp_arg)
        elif type(ballcomp_arg) == SpecializedBallComp: 
            self.ballcomp = ballcomp_arg
        else: 
            assert False
        return 

    def main_proc(self):
        print("BC PROC #1")
        self.bc_proc1()
        print("BC PROC #2")
        self.bc_proc_2_OR_3(2)
        print("\t-- PREFIT")
        self.ball_comp.run_prefit_proc()
        print("BC PROC #3")
        self.bc_proc_2_OR_3(3)
        return

    ####################### data folder operations

    def set_target_folder(self):
        if len(self.training_data_folders) == 0:
            return False 

        # set target folder
        i = random.randint(0,len(self.training_data_folders) - 1) 
        self.set_target_folder_(i)
        return True

    def set_target_folder_(self,i):
        self.target_folder = self.training_data_folders.pop(i)
        assert len(self.target_folder) == 3
        assert type(self.target_folder) == tuple

        # set FARSE reader
        self.farse_reader = FARSEReader(self.target_folder[0],\
            self.target_folder[1],self.target_folder[2])

        # set the ordering of keys
        qk = list(self.farse_reader.d.keys())
        ordering = random_ordering(len(qk))
        self.tf_file_ordering = [qk[i] for i in ordering]

    ##############################################

    """
    ballcomp_arg := list, each element is [hop length,number of balls,ratio of max]
    """
    def initialize_ballcomp(self,ballcomp_arg):
        hssbc_args = []
        min_max_dict = self.prelim_analysis()
        for x in ballcomp_arg:
            if x[0] not in min_max_dict: 
                continue 

            minimum = min_max_dict[x[0]][0]
            maximum = min_max_dict[x[0]][1]


            radius = default_ballcomp_radius(minimum,maximum,x[1],x[2])
            hssbc_args.append([x[0],x[1],radius])
        self.ball_comp = HopSeparatedSpecBallComp(hssbc_args)

    """
    preliminary analysis collects the range of 
    values in the 24-vec for the all folders 
    according to the order specified by the random seed.  

    * method called after `set_target_folder`. 
    """
    def prelim_analysis(self):
        l = len(STD_DEC_WEIGHT_SEQLABELS) * 4 
        minimum = np.ones(l) * float("inf")
        maximum = np.ones(l) * -float("inf")

        min_max_dict = defaultdict(None)
        for x in self.timestamp_hop_seq:
            min_max_dict[x] = [deepcopy(minimum),deepcopy(maximum)]

        tdf_copy = deepcopy(self.training_data_folders)
        stat = self.set_target_folder()
        while stat:
            min_max_dict = self.range_analysis_on_folder(min_max_dict)
            stat = self.set_target_folder()

        self.training_data_folders = tdf_copy
        self.target_folder = None
        self.farse_reader = None 
        self.tf_file_ordering = None

        null_keys = []
        for (k,v) in min_max_dict.items():
            if (v[0] == minimum).any():
                null_keys.append(k)
                continue
            if (v[1] == maximum).any():
                null_keys.append(k)
            
        ##print("NULL HOP KEYS ARE: ",null_keys)
        for k in null_keys:
            del min_max_dict[k]

        return min_max_dict

    def range_analysis_on_folder(self,min_max_dict):
        for k in self.tf_file_ordering:
            v = self.farse_reader.d[k]
            q = self.farse_reader.fetch_vector_samples(v)
            minimum = min_max_dict[len(q)][0]
            maximum = min_max_dict[len(q)][1]
            for v_ in q.values():
                v2 = np.array(v_) 
                
                v3 = np.array([minimum,v2])
                minimum = np.min(v3,axis=0)

                v3 = np.array([maximum,v2])
                maximum = np.max(v3,axis=0)
            min_max_dict[len(q)] = [minimum,maximum]
        return min_max_dict

    ##############################################
    ################### BallComp phase 1: construction

    def bc_proc1(self):
        ##print("#1 number of training folders: {}".format(len(self.training_data_folders)))
        tdf_copy = deepcopy(self.training_data_folders)
        stat = self.set_target_folder()
        while stat:
            self.bc_proc1_next_folder()
            stat = self.set_target_folder()
        self.training_data_folders = tdf_copy
        return
    
    """
    """
    def bc_proc1_next_folder(self):
        for k in self.tf_file_ordering:
            v = self.farse_reader.d[k]
            q = self.farse_reader.fetch_vector_samples(v)
            l = len(q) 
            bc = self.ball_comp.d[l]
            if bc.bc.terminateDelta: 
                continue

            for v_ in q.values():
                y = self.ball_comp.fit_proc1(l,np.array(v_))
        return 

    ############################################## 
    ################## BallComp phase 2: frequency analysis on each Ball
    ##################      and phase 3: weight-fitting

    def bc_proc_2_OR_3(self,phase_number):
        ##print("#{} number of training folders: {}".format(phase_number,len(self.training_data_folders)))
        tdf_copy = deepcopy(self.training_data_folders)
        stat = self.set_target_folder()
        while stat:
            self.bc_proc_2_OR_3_(phase_number)
            stat = self.set_target_folder()
        self.training_data_folders = tdf_copy
        return 

    def bc_proc_2_OR_3_(self,phase_number):
        assert phase_number in [2,3]
        f = self.ball_comp.fit_proc2 if phase_number == 2 \
                else self.ball_comp.fit_proc3
        for k in self.tf_file_ordering:
            q = self.farse_reader.fetch_by_key(k)
            assert type(q) != type(None) 
            vecs,contexts = q[0],q[1]
            l = len(contexts)
            for (v,c) in zip(vecs,contexts):
                f(l,np.array(v),c)
        return