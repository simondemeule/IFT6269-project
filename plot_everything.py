from utils import find_last_run_index
from plotting import *

datasets_all = ['MNIST', 'SVHN', 'CIFAR', 'Gaussian']
divergences_all = ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']

for dataset in datasets_all:
    # Plots that comprise of all divergences for a dataset
    runs = []
    divergences = []
    for divergence in divergences_all:
        run = find_last_run_index(dataset, divergence)
        if run != -1:
            # There exists at least a run for this divergence
            # WARNING: this will not plot the latest run if the increasing numbering sequence is broken. 
            # It will start at zero and go up from there until no further run is found.
            divergences.append(divergence)
            runs.append(run)
    plot_divergence_all(dataset, divergences, runs, show_plot=False)
    plot_real_fake_all(dataset, divergences, runs, show_plot=False)

    # Plots that are specific to each divergence
    for (divergence, run) in zip(divergences, runs):
        plot_divergence_training(dataset, divergence, run, show_plot=False)
        plot_divergence_other(dataset, divergence, run, show_plot=False)
        plot_real_fake_training(dataset, divergence, run, show_plot=False)