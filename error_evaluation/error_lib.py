from pydoc import cram
import numpy as np
import matplotlib.pyplot as plt


def compute_error_binned(model, dataset, number_samples, number_bins, error_sample_size, weights=None):

    bin_edges_qkl = [
        np.linspace(0.5, 2, number_bins+1),           # q2
        np.linspace(0, np.pi, number_bins+1),         # k
        np.linspace(0, np.pi, number_bins+1),         # l
    ]

    predictions = []

    for _ in range(number_samples):
        error_sample = dataset.sample(error_sample_size)

        if weights is not None:
            error_sample_weights = weights.loc[error_sample.index]
        else:
            error_sample_weights = None
        error_sample = error_sample.values

        error_sample_hist, _ = np.histogramdd(
            error_sample,
            bins=bin_edges_qkl,
            weights=error_sample_weights,
            density=True
        )
        data_to_eval = error_sample_hist.reshape(
            number_bins, number_bins, number_bins, 1)
        error_prediction = model.predict(np.array([data_to_eval]))

        predictions.append(error_prediction[0])

    return np.array(predictions)


def plot_2d_error(x_predictions, y_predictions, bins=(30,30), color_map=plt.cm.Reds, cmin=1, c_range=(-3,3), show=True):
    bins_x = np.linspace(c_range[0], c_range[1], bins[0] + 1)
    bins_y = np.linspace(c_range[0], c_range[1], bins[1] + 1)
    plt.hist2d(x_predictions, y_predictions, bins=(bins_x, bins_y), cmap=color_map, cmin=cmin)
    plt.xlabel('c9')
    plt.ylabel('c10')
    plt.ylim(c_range)
    plt.xlim(c_range)

    if show:
        plt.show()