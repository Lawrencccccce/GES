


from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen
from .prior_knowledge import PriorKnowledge
import numpy as np







# BIC lower the better

class PriorScore(GaussObsL0Pen):


    def __init__(self, data, dataset, lmbda=None, method='scatter', cache=True, debug=0, prior_weight = 1):
        super().__init__(data = data, lmbda = lmbda, method = method, cache = cache, debug = debug)
        self.n, self.p = data.shape
        self.data = data
        self.prior = PriorKnowledge(dataset)
        self.dataset = dataset
        self.LLM_weights = {}
        self.prior_weight = prior_weight
        self._get_LLM_scores()

    # full score with prior
    def full_score(self, A):

        prior_penalty = 0
        for dataset in self.prior.prior_knowledge:
            if dataset == self.dataset:
                for model in self.prior.prior_knowledge[dataset]:
                    prior_penalty_model = 0
                    LLM_result = self.prior.prior_knowledge[dataset][model]
                    for i in range(self.p):
                        for j in range(self.p):
                            if LLM_result[i][j] != 2 and A[i][j] != LLM_result[i][j]:
                                prior_penalty_model += 1
                    prior_penalty += prior_penalty_model * self.LLM_weights[model]
    
        return super().full_score(A) + prior_penalty * self.prior_weight
    
    def local_score(self, x, pa):
        """
        Return the local score of a given node and a set of
        parents. If self.cache=True, will use previously computed
        score if possible.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        if self._cache is None:
            return self._compute_local_score(x, pa)
        else:
            key = (x, tuple(sorted(pa)))
            try:
                score = self._cache[key]
                print("score%s: using cached value %0.2f" %
                      (key, score)) if self._debug >= 2 else None
            except KeyError:
                score = self._compute_local_score(x, pa)
                self._cache[key] = score
                print("score%s = %0.2f" % (key, score)) if self._debug >= 2 else None

            prior_penalty = 0
            for dataset in self.prior.prior_knowledge:
                if dataset == self.dataset:
                    for model in self.prior.prior_knowledge[dataset]:
                        prior_penalty_model = 0
                        LLM_result = self.prior.prior_knowledge[dataset][model]
                        for p in pa:
                            if LLM_result[p][x] != 1:
                                prior_penalty_model += 1
                        prior_penalty += prior_penalty_model * self.LLM_weights[model]
            return score + prior_penalty * self.prior_weight

    
    def _get_LLM_scores(self):
        for dataset in self.prior.prior_knowledge:
            if dataset == self.dataset:
                for model in self.prior.prior_knowledge[dataset]:
                    if model not in self.LLM_weights:
                        self.LLM_weights[model] = {}
                    LLM_result = self.prior.prior_knowledge[dataset][model]
                    LLM_result[LLM_result == 2] = 0

                    self.LLM_weights[model] = super().full_score(LLM_result)
        
        self.transfer_LLM_score_to_weights()


    def transfer_LLM_score_to_weights(self):

        if len(self.LLM_weights) == 0:
            return

        result = np.array([])
        for model in self.LLM_weights:
            result = np.append(result, -self.LLM_weights[model])

        z_score = (result - np.mean(result)) / np.std(result)
        softmax_result = np.exp(z_score) / np.sum(np.exp(z_score))

        for i, model in enumerate(self.LLM_weights):
            self.LLM_weights[model] = softmax_result[i]          
    





