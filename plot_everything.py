from utils import find_last_run_index
from plotting import *

datasets_all = ['MNIST', 'SVHN', 'CIFAR', 'Gaussian']
divergences_all = ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon', 'piecewise']

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
    """
    # For training runs with different divergences, plot all losses
    plot_divergence_all(dataset, divergences, runs, show_plot=False)
    # For training runs with different divergences, plot all real / fake statistics
    plot_real_fake_all(dataset, divergences, runs, show_plot=False)
    # For training runs with different divergences, plot all parameter step sizes
    plot_walk_all(dataset, divergences, runs, show_plot=False)
    """
    # For training runs with different divergences, plot all murals
    plot_mural(dataset, divergences, runs, 50, 5, epoch_shape_out=(5, 1), column=True)

    """
    # Plots that are specific to each divergence
    for (divergence, run) in zip(divergences, runs):
        # For a single training run with a specific divergence, plot losses
        plot_divergence_training(dataset, divergence, run, show_plot=False)
        # For a single training run with a specific divergence, plot losses for all other divergences
        plot_divergence_other(dataset, divergence, run, show_plot=False)
        # For a single training run with a specific divergence, plot real / fake statistics
        plot_real_fake_training(dataset, divergence, run, show_plot=False)
        # For a single training run with a specific divergence, plot parameter step sizes
        plot_walk_training(dataset, divergence, run, show_plot=False)
    """