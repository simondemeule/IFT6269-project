import plotting as p

datasets = ['MNIST', 'svhn', 'CIFAR', 'Gaussian']
for dataset in datasets:
    p.plot_divergence_all(dataset, show_plot=False)