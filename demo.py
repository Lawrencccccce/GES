from ges.main import fit_bic, fit
from datetime import datetime
from pytz import timezone
from my_socre import PriorScore, count_accuracy, PriorKnowledge
import io
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import sys
import time



def create_dir(dir_path):
    """
    dir_path - A path of directory to create if it is not found
    :param dir:
    :return exit_code: 0:success -1:failed
    """
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        sys.exit(-1)

def plot_result(estimate_g, ground_truth, plot_dir, dataset, title, accuracy = None):

    fig = plt.figure(3)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(title + f" SHD: {accuracy['shd']}")
    ax.imshow(np.around(estimate_g).astype(int),cmap=plt.cm.binary)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('true_graph')
    ax.imshow(ground_truth, cmap=plt.cm.binary)
    plt.savefig('{}/{}_estimated_graph_{}.png'.format(plot_dir, dataset, datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]))
    plt.close()
    print(f"{title} \n {accuracy}")

def generate_data_path(dataset):
    # Load the data
    current_dir = os.getcwd()
    datapath = os.path.join('datasets', dataset + '.npy')
    datapath = os.path.join(current_dir, datapath)

    sol_path = os.path.join('datasets', dataset + '_sol.npy')
    sol_path = os.path.join(current_dir, sol_path)

    plot_dir = os.path.join(current_dir, 'plots')
    create_dir(plot_dir)

    return datapath, sol_path, plot_dir


def accuracy_test(dataset):

    # Load the data
    datapath, sol_path, plot_dir = generate_data_path(dataset)

    data = np.load(datapath).astype(np.float32)
    ground_truth = np.load(sol_path).astype(np.float32)

    score_class = PriorScore(data, dataset = dataset, prior_weight= 0.29)

    estimate_g_bic, score_bic = fit_bic(data)
    estimate_g_prior, score_prior = fit(score_class)

    plot_result(estimate_g_bic, ground_truth, plot_dir, dataset, f"BIC score: {score_bic:.2f}", count_accuracy(ground_truth, estimate_g_bic))
    plot_result(estimate_g_prior, ground_truth, plot_dir, dataset, f"BIC with prior: {score_prior:.2f}", count_accuracy(ground_truth, estimate_g_prior))

def time_test(dataset):
    # Load the data
    datapath, sol_path, plot_dir = generate_data_path(dataset)

    data = np.load(datapath).astype(np.float32)
    ground_truth = np.load(sol_path).astype(np.float32)

    prior_knowledge = PriorKnowledge(dataset)

    start_time = time.time()
    estimate_g, score = fit_bic(data)
    end_time = time.time()
    BIC_time = end_time - start_time

    start_time = time.time()
    estimate_g_prior, score_prior = fit_bic(data, A0 = prior_knowledge.intersection_result[dataset])
    end_time = time.time()
    BIC_prior_time = end_time - start_time

    plot_result(estimate_g, ground_truth, plot_dir, dataset, f"BIC time: {BIC_time:.2f}", count_accuracy(ground_truth, estimate_g))
    plot_result(estimate_g_prior, ground_truth, plot_dir, dataset, f"BIC with prior: {BIC_prior_time:.2f}", count_accuracy(ground_truth, estimate_g_prior))

if __name__ == "__main__":

    datasets = ['LUCAS', "Asia", "SACHS"]

    for dataset in datasets:
        accuracy_test(dataset)

    

    















