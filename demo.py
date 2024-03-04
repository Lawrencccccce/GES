from ges.main import fit_bic
import io
import numpy as np
import os
import matplotlib.pyplot as plt




if __name__ == "__main__":

    # Load the data
    current_dir = os.getcwd()
    datapath = os.path.join('datasets', 'LUCAS.npy')
    datapath = os.path.join(current_dir, datapath)

    data = np.load(datapath).astype(np.float32)
    print(data)


    estimate_g, score = fit_bic(data)
    print(estimate_g)
    print(score)















