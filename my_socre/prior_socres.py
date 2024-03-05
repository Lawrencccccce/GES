


from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen
from .prior_knowledge import PriorKnowledge









class PriorScore(GaussObsL0Pen):


    def __init__(self, data, lmbda=None, method='scatter', cache=True, debug=0):
        super().__init__(data = data, lmbda = lmbda, method = method, cache = cache, debug = debug)
        self.prior = PriorKnowledge()

    # full score with prior
    def full_score(self, A):
        return super().full_score(A)
    





