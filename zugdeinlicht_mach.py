from farse_mach import * 

class WeightFitter:

    def __init__(self):
        return

    def fit_context(self):
        return -1

"""
Algorithm that uses stochastic ordering of points to 
construct a BallComp instance, or add onto it.
"""
class ZugdeinlichtMach:

    def __init__(self,ballcomp_arg,training_data_folders,rdm_seed:int):
        # case: file, open it
        if type(ballcomp_arg) == str:
            f = open(ballcomp_arg,"rb")
            self.ballcomp = pickle.load(f)
        elif type(ballcomp_arg) == BallComp:
            self.ballcomp = ballcomp_arg
        else: 
            assert False
        assert type(training_data_folders) == list 
        self.training_data_folders = training_data_folders
        if type(rdm_seed) == type(None):
            random.seed(rdm_seed)
        self.target_folder = None
        self.tf_file_ordering = None
        self.farse_reader = None 
        return 
        
    def set_target_folder(self):
        # set target folder
        i = random.randint(0,len(self.training_data_folders) - 1) 
        self.target_folder = self.training_data_folders.pop(i)
        assert len(self.target_folder) == 3
        assert type(self.target_folder) == tuple

        # set FARSE reader
        self.farse_reader = FARSEReader(self.target_folder[0],\
            self.target_folder[1],self.target_folder[2])

        # set the ordering of keys
        qk = list(self.d.keys())
        ordering = random_ordering(len(qk))
        self.tf_file_ordering = [qk[i] for i in ordering]

    ##############################################

    def folder_prelim_analysis(self):
        return -1

    ##############################################

    def process_next_folder(self):
        return -1

    def process_next_file(self):
        if len(self.tf_file_ordering) == 0:
            return False 
        q = self.tf_file_ordering.pop(0)
        x = self.farse_reader.fetch_by_key(q)
        assert len(x) == 2
        vec_samples,pcontexts = x[0],x[1]
        return True

    def fit_sample(self,vector_sample,pcontext:PContext): 
        return -1

    ############################################## 