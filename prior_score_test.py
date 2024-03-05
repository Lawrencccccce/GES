import numpy as np
import sempler
import ges
from my_socre import PriorScore





if __name__ == "__main__":
    # Generate observational data from a Gaussian SCM using sempler
    A = np.array([[0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]])
    W = A * np.random.uniform(1, 2, A.shape) # sample weights
    data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)

    score_class = PriorScore(data)
    # Run GES with the gaussian BIC score
    estimate, score = ges.fit(score_class)

    print(estimate, score)