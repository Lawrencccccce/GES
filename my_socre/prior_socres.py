


from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen
from .prior_knowledge import PriorKnowledge
import numpy as np







# BIC lower the better

class PriorScore(GaussObsL0Pen):


    def __init__(self, data, dataset, lmbda=None, method='scatter', cache=True, debug=0):
        super().__init__(data = data, lmbda = lmbda, method = method, cache = cache, debug = debug)
        self.data = data
        self.prior = PriorKnowledge()
        self.dataset = dataset
        self.LLM_scores = {}
        self._get_LLM_scores()
        print(self.LLM_scores)

    # full score with prior
    def full_score(self, A):
        return super().full_score(A)
    
    def _get_LLM_scores(self):
        for dataset in self.prior.prior_knowledge:
            if dataset == self.dataset:
                for model in self.prior.prior_knowledge[dataset]:
                    if model not in self.LLM_scores:
                        self.LLM_scores[model] = {}
                    LLM_result = self.prior.prior_knowledge[dataset][model]
                    LLM_result[LLM_result == 2] = 0

                    self.LLM_scores[model] = super().full_score(LLM_result)
        
        self._apply_transformation_to_prior_result()


    def _apply_transformation_to_prior_result(self):

        if len(self.LLM_scores) == 0:
            return

        result = np.array([])
        for model in self.LLM_scores:
            result = np.append(result, -self.LLM_scores[model])

        z_score = (result - np.mean(result)) / np.std(result)
        softmax_result = np.exp(z_score) / np.sum(np.exp(z_score))

        for i, model in enumerate(self.LLM_scores):
            self.LLM_scores[model] = softmax_result[i]          
    





