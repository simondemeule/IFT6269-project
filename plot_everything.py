import plotting as p

datasets = ['MNIST', 'svhn', 'CIFAR', 'Gaussian']
divergence_names = ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']

for dataset in datasets:
    p.plot_divergence_all(dataset, show_plot=False)
    p.plot_real_fake_all(dataset, show_plot=False)
    for divergence in divergence_names:
        p.plot_divergence_training(dataset, divergence, show_plot=False)
        p.plot_divergence_other(dataset, divergence, show_plot=False)
        p.plot_real_fake_training(dataset, divergence, show_plot=False)