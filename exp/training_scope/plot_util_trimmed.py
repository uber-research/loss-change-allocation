'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import h5py
import os
import math
import time
from scipy import stats
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d
from scipy.stats import dweibull
from ast import literal_eval
from IPython import embed
import pdb

######################################## handling large matrices ########################################

'''
Loading entire matrix takes too much memory: var = np.array(h5pyfile['var'])
Reading one line at a time takes too much time: var = h5pyfile['var']
So instead, use this class as a wrapper to read in chunks at a time.

Usage:
    var = util.streamds(h5pyfile['var'])
    for i in range(var.shape[0]):
        do_something(var[i])
Note that you cannot close the h5py file until after

Limitations for now:
    - must be a 2D h5py dataset
    - only useful for indexing by the first axis (time aka iteration)
    - only useful for going forward in time, not backwards
'''
class streamds():
    def __init__(self, ds, chunk_size):
        self.ds = ds
        self.chunk_size = chunk_size
        self.shape = ds.shape
        self.startind = ds.shape[0]
        self.buffer = np.zeros((chunk_size, ds.shape[1]))

    def __getitem__(self, i):
        if isinstance(i, int):
            i = i % self.shape[0]
            if i < self.startind or i >= self.startind + self.chunk_size:
                self.update_buffer(i)
            return self.buffer[i - self.startind]
        else:
            return self.ds[i]

    def update_buffer(self, i):
        self.startind = i
        self.buffer = np.array(self.ds[self.startind:self.startind + self.chunk_size])

    def sum(self, axis=-1):
        if axis == 0:
            ret = np.zeros(self.ds.shape[1])
            for start in range(0, self.ds.shape[0], self.chunk_size):
                ret += self.ds[start : start + self.chunk_size].sum(axis=0)
            return ret
        elif axis == 1:
            ret = np.zeros(self.ds.shape[0])
            for start in range(0, self.ds.shape[0], self.chunk_size):
                ret[start : start + self.chunk_size] += self.ds[start : start + self.chunk_size].sum(axis=1)
            return ret
        elif axis == -1 and len(self.shape) == 2:
            return self.sum(axis=0).sum()
        else:
            print('sorry no can do')

    def __len__(self):
        return len(self.ds)


######################################## helpers ########################################

# concatenate arrays from multiple gpus (different keys in the same file)
def concat_arrays(hf, keyroot):
    grads = np.array(hf['{}_0'.format(keyroot)])
    for i in range(1, 4):
        nextkey = '{}_{}'.format(keyroot, i)
        if nextkey in hf.keys():
            grads = np.concatenate((grads, hf[nextkey][1:]))
    return grads

# return loss curve based on starting loss and the LC totals
def get_approx_loss(loss, helped):
    return loss[0] + np.concatenate((np.array([0]), np.cumsum(helped.sum(axis=1))))

# e.g. [1, 4, 2, 4, 1] for num_splits = 4
def get_rk_coeffs(num_splits):
    coeffs = np.ones(num_splits + 1)
    coeffs[1::2] = 4
    coeffs[2:-1:2] = 2
    return coeffs

def get_helped_rk_adaptive(weights, gradients, splits_per_iter):
    helped = np.zeros((weights.shape[0] - 1, weights.shape[1]))
    coeffs = {}
    for i in [2, 4, 8, 16, 32]:
        coeffs[i] = get_rk_coeffs(i)

    grads_ind = 0
    for ts in range(weights.shape[0] - 1):
        num_splits = splits_per_iter[ts]
        split_gradients = gradients[grads_ind:grads_ind + num_splits + 1]
        grads_ind += num_splits
        k = np.matmul(coeffs[num_splits], split_gradients) / coeffs[num_splits].sum()
        helped[ts] = np.multiply(k, weights[ts + 1] - weights[ts])
    return helped

def get_helped_fo(weights, gradients, splits_per_iter): # first order
    helped = np.zeros((weights.shape[0] - 1, weights.shape[1]))
    whole_indices = np.append(0, np.cumsum(splits_per_iter))

    for ts in range(weights.shape[0] - 1):
        k = gradients[whole_indices[ts]]
        helped[ts] = np.multiply(k, weights[ts + 1] - weights[ts])
    return helped

def get_helped_adaptive_straight_path(weights, gradients, splits_per_iter):
    weights_delta = (weights[-1] - weights[0]) / (weights.shape[0] - 1)
    helped = np.zeros((len(splits_per_iter), weights.shape[1]))
    coeffs = {}
    for i in [2, 4, 8, 16, 32]:
        coeffs[i] = get_rk_coeffs(i)

    grads_ind = 0
    for ts in range(len(splits_per_iter)):
        num_splits = splits_per_iter[ts]
        split_gradients = gradients[grads_ind:grads_ind + num_splits + 1]
        grads_ind += num_splits
        k = np.matmul(coeffs[num_splits], split_gradients) / coeffs[num_splits].sum()
        helped[ts] = np.multiply(k, weights_delta)
    return helped

# run plots on abs(weight movements) instead of helped
def get_helped_weight_movements(weights):
    helped = np.zeros((weights.shape[0] - 1, weights.shape[1]))
    for ts in range(weights.shape[0] - 1):
        helped[ts] = (weights[ts + 1] - weights[ts])
    return helped

# returns list of variables as np arrays in their original shape
def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

# same as split_and_shape except ignore batch_norm layers, and make conv weights 2d
def split_and_shape_plotting(one_time_slice, shapes, names):
    variables = []
    plot_shapes = []
    plot_names = []
    offset = 0
    for i, shape in enumerate(shapes):
        num_params = np.prod(shape)
        if 'batch_norm' not in names[i]:
            plot_names.append(names[i])
            plot_shapes.append(shape)  # for drawing lines separating filters
            layer = one_time_slice[offset : offset + num_params].reshape(shape)
            if len(shape) == 4:  # for conv2d layers
                # new shape: (filter_rows * inputs, filter_cols * outputs)
                newshape = (shape[0] * shape[2], shape[1] * shape[3])
                # magic axes swapping to get the dimension order correct
                # TODO temporary transpose to make plots horizontal
                variables.append(layer.transpose(2, 0, 3, 1).reshape(newshape).T)
            elif len(shape) == 1 or len(shape) == 2:
                variables.append(layer.reshape(shape))
            else:
                raise Exception('Cannont handle tensor that isnt 1, 2, or 4 dimensions')
        offset += num_params
    return variables, plot_shapes, plot_names

# for smoothing plots. Only use odd window sizes for now
def smooth_plot(arr, window_size):
    if window_size <= 1:
        return arr
    cumul_vec = np.cumsum(np.insert(arr, 0, 0))
    middle = (cumul_vec[window_size:] - cumul_vec[:-window_size]) / window_size
    ends = int(window_size / 2)
    return np.concatenate((arr[:ends], middle, arr[-ends:]))

# get neurons, given either a 2d fc layer or a 4d conv layer
def get_neurons(layer):
    return np.sum(layer, axis=tuple(range(len(layer.shape) - 1)))

# TODO this assumes all layers have a bias layer immediately after it
# returns list. each element is a vector
def get_neurons_plus_biases(split_vars):  # flattened and joined
    layers = []
    for i in range(int(len(split_vars)/2)):
        layers.append(get_neurons(split_vars[2*i]) + split_vars[2*i + 1].flatten())
    return layers

# given index in flattened array, return which layer it is
def get_layer(shapes, ind):
    sizes = [np.product(s) for s in shapes]
    for i, size in enumerate(sizes):
        if ind < size:
            return i
        ind -= size
    return -1

# like split_and_shape but returns matrices of shape ((original shape of layer), T)
def get_layer_trajectories(all_trajs, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(all_trajs[offset : offset + num_params].reshape(shape + (-1,)))
        offset += num_params
    return variables

# number of local optima aka number of times the derivative changes signs
def num_local_optima(values):
    return num_cross_zero(np.diff(values, axis=0))

# number of times the sequence switches signs
def num_cross_zero(values):
    return (values[:-1] * values[1:] < 0).sum(axis=0)

# shift histogram bins left until 0 is a boundary
# then add an extra bin to the right
def align_bins_zero(bins):
    if np.max(bins) < 0 or np.min(bins) > 0 or 0 in bins:
        return bins
    shift = 0
    for i in range(len(bins)):
        if bins[i] < 0 and bins[i+1] > 0:
            shift = bins[i+1]
    ret = np.subtract(bins, shift)
    return np.append(ret, ret[-1] + ret[1] - ret[0])

def gini(array):
    if len(array) == 0:
        return 0
    # Copied from https://github.com/oliviaguest/gini/blob/master/gini.py
    # based on http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # Values cannot be negative:
    # instead of shifting them with array -= np.amin(array),
    # just set negative values to 0
    array = array.flatten().clip(min=0)

    array = np.sort(array)
    index = np.arange(array.shape[0]) + 1
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def negative_gini(array):
    return gini(-array[np.where(array < 0)])

def positive_gini(array):
    return gini(array[np.where(array > 0)])

# generate for each layer (in a list) a N x T matrix where T = total iterations, N = number of neurons
def get_neuron_trajs_all_layers(helped, shapes):
    neuron_trajs = get_neurons_plus_biases(split_and_shape(helped[0], shapes))
    for i, layer in enumerate(neuron_trajs):
        neuron_trajs[i] = layer.reshape(-1, 1)
    for ts in range(1, helped.shape[0]):
        curneurons = get_neurons_plus_biases(split_and_shape(helped[ts], shapes))
        for i in range(len(neuron_trajs)):
            neuron_trajs[i] = np.concatenate((neuron_trajs[i], curneurons[i].reshape(-1, 1)), axis=1)
    return neuron_trajs

def save_or_show(plot_dest):
    '''Save figure or show it. Allowable destinations and actions:
    Instance of PdfPages: write page to pdf
    String like "/home/user/results/foo": write /home/user/results/foo.pdf and /home/user/results/foo.png
    None: display instead of writing.
    '''
    if isinstance(plot_dest, PdfPages):
        plot_dest.savefig()
        plt.close()
    elif isinstance(plot_dest, str):
        plt.savefig('%s.png' % plot_dest)
        plt.savefig('%s.pdf' % plot_dest)
        plt.close()
    elif plot_dest is None:
        plt.show()
    else:
        raise Exception('Not sure what to do with this plot destination: %s' % repr(plot_dest))

######################################## manual plotting for paper ########################################
# more customizeable filenames and titles
# less for running scripts over a bunch of runs

# same as percent_helped_per_iter(), but only plots the main one
def phpi_manual_plot(helped, trainloss, sigma=1, title=None, save_dest=None):
    positives, negatives, zeros, weighted_pos, weighted_neg = [], [], [], [], []
    for ts in range(helped.shape[0]):
        cur_ts = helped[ts] # need to do this to convert streamds to np array
        positives.append((cur_ts > 0).sum())
        negatives.append((cur_ts < 0).sum())
        zeros.append((cur_ts == 0).sum())
        weighted_pos.append(cur_ts[cur_ts > 0].sum())
        weighted_neg.append(cur_ts[cur_ts < 0].sum())

    # (hurtcm, zerocm, helpcm) = ((.95, .45, .45), 'w', (.55, .95, .55))
    (hurtcm, zerocm, helpcm) = ((.9, .25, .25), 'w', (.3, .75, .3))
    x = np.arange(helped.shape[0])
    y1 = 100 * gaussian_filter1d(np.array(negatives) / helped.shape[1], sigma)
    y2 = 100 * gaussian_filter1d(np.array(zeros) / helped.shape[1], sigma)
    y3 = 100 * gaussian_filter1d(np.array(positives) / helped.shape[1], sigma)
    fig, ax1 = plt.subplots()
    ax1.fill_between(x, y1 + y2, y1 + y2 + y3, label='hurt', color=hurtcm)
    ax1.fill_between(x, y1, y1 + y2, label='zero', color=zerocm)
    ax1.fill_between(x, y1, label='helped', color=helpcm)
    ax1.axhline(50, color='k', alpha=0.5, linestyle=':')
    ax1.plot([], 'ko-', alpha=0.8, label='loss')
    ax1.legend()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('percentage')
    # for spacing: 3 for mnist, 15 for resnet
    offset = -15 if helped.shape[0] > 2000 else -3
    ax1.set_xlim(offset, helped.shape[0])
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(trainloss, 'k', alpha=0.7)
    ax2.plot(trainloss[0], 'ko')
    ax2.plot(len(trainloss), trainloss[-1], 'ko')
    ax2.set_ylabel('loss')
    if title:
        plt.title(title)
    else:
        plt.title('Average percent helped per iter = {:.5f}'.format(y1.mean()))
    print('Helped: {:.5f}, zero: {:.5f}, hurt: {:.5f}'.format(y1.mean(), y2.mean(), y3.mean()))
    print('Helped / (helped + hurt): {:.5f}'.format((y1 / (y1 + y3)).mean()))
    if save_dest:
        plt.savefig(save_dest)
    plt.show()

# histogram of all TxD elements of H
# plot_both means plot log scale and non log scale side by side
def hist_entire_matrix_manual_plot(helped, log_scale=True, cutoff_perc=0, plot_both=False,
                                   title=None, save_dest=None, plot_weibull=False):
    # (hurtcm, zerocm, helpcm) = ((.95, .45, .45), 'w', (.55, .95, .55))
    (hurtcm, zerocm, helpcm) = ((.9, .25, .25), 'w', (.3, .75, .3))
    tstart = time.time()
    if cutoff_perc == 0:
        _, bins = np.histogram(helped[helped != 0], bins=39)
    else:
        lower, upper = np.percentile(helped, cutoff_perc), np.percentile(helped, 100 - cutoff_perc)
        bins = np.arange(lower, upper, (upper - lower) / 39)
    aligned_bins = align_bins_zero(bins)
    zero_bin = np.where(aligned_bins == 0)[0][0]
    print(time.time() - tstart)

    if plot_weibull:
        # these hardcoded numbers work on jan30_fixactivations/190206_000228_6bc76f8_janice_wholearns_cifar_resnet_sgd0.1
        tt = np.linspace(-9.125406551714053e-07, 9.125406551714053e-07, 151)
        tt[75] = tt[74]
        dd = dweibull(0.53, 0, scale=5.1e-8)

    if plot_both:
        plt.figure(figsize=(12, 4), dpi=120)
        plt.subplot(1, 2, 1)
        counts, _, _ = plt.hist(helped[helped < 0], bins=aligned_bins[:zero_bin + 1], log=True, color=helpcm)
        plt.hist(helped[helped > 0], bins=aligned_bins[zero_bin:], log=True, color=hurtcm)
        plt.axvline(x=0, color='k', alpha=0.5, linestyle=':')
        plt.plot([helped.mean(), helped.mean()], [0, np.sqrt(np.max(counts))], color='y')
        plt.xlim(aligned_bins[0], aligned_bins[-1])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
        plt.xlabel('LC')
        plt.ylabel('number of parameters')
        plt.title('Log scale')
        if plot_weibull:
            plt.plot(tt - (2e-6 * 0.1 / 200), 63*dd.pdf(tt), 'k--', linewidth=2)
        print(time.time() - tstart)
        plt.subplot(1, 2, 2)
        plt.hist(helped[helped < 0], bins=aligned_bins[:zero_bin + 1], log=False, color=helpcm)
        plt.hist(helped[helped > 0], bins=aligned_bins[zero_bin:], log=False, color=hurtcm)
        plt.axvline(x=0, color='k', alpha=0.5, linestyle=':')
        # plt.plot([helped.mean(), helped.mean()], [0, np.max(counts)/3], color='m')
        plt.xlim(aligned_bins[0], aligned_bins[-1])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
        plt.xlabel('LC')
        plt.ylabel('number of parameters')
        plt.title('Regular scale')
        if plot_weibull:
            plt.plot(tt - (2e-6 * 0.1 / 200), 63*dd.pdf(tt), 'k--', linewidth=2)
        # if title:
        #     plt.suptitle(title)
        # else:
        #     plt.suptitle('Histogram of all LC elements')
    else:
        plt.hist(helped[helped < 0], bins=aligned_bins[:zero_bin + 1], log=log_scale, color=helpcm)
        print(time.time() - tstart)
        plt.hist(helped[helped > 0], bins=aligned_bins[zero_bin:], log=log_scale, color=hurtcm)
        plt.axvline(x=0, color='k', alpha=0.5, linestyle=':')
        plt.xlabel('LC')
        plt.ylabel('number of parameters')
        if title:
            plt.title(title)
        else:
            plt.title('Histogram of all LC elements')
    print(time.time() - tstart)

    if save_dest:
        plt.savefig(save_dest)
    plt.show()

def iterations_helped_per_param(helped, title=None, save_dest=None):
    iters_per_param = (helped < 0).sum(axis=0) / helped.shape[0] * 100
    plt.hist(iters_per_param, bins=50)
    plt.axvline(50, c='k', linestyle=':', alpha=0.5)
    plt.xlabel('percent of iterations helped')
    plt.ylabel('number of parameters')
    if title:
        plt.title(title)
    else:
        plt.title('Iterations helped per parameter')
    if save_dest:
        plt.savefig(save_dest)
    plt.show()


# same as plot_trajectories_per_layer() but only hte random one
def random_trajectories_manual_plot(helped, shapes, names, neuron_trajs=None, color_percentile=99, title=None, save_dest=None):
    # custom color map matching the green and red used in other plots
    # red is negative and green is positive because default colormaps are like that
    cdict = {'red':   [[0.0,  0.4, 0.4],
                       [0.3, 0.9, 0.9],
                       [0.5,  1.0, 1.0],
                       [0.7, 0.3, 0.3],
                       [1.0,  0.0, 0.0]],
             'green': [[0.0,  0.0, 0.0],
                       [0.3, 0.25, 0.25],
                       [0.5,  1.0, 1.0],
                       [0.7, 0.75, 0.75],
                       [1.0,  0.3, 0.3]],
             'blue':  [[0.0,  0.0, 0.0],
                       [0.3, 0.25, 0.25],
                       [0.5,  1.0, 1.0],
                       [0.7, 0.3, 0.3],
                       [1.0,  0.0, 0.0]]}
    custom_cmap = LinearSegmentedColormap('lc', segmentdata=cdict) # previously used PiYG
    num_sample = 50

    offset = 0
    for i, shape in enumerate(shapes):
        # layer shape is (iteration, all params in layer flattened)
        num_params = np.prod(shape)
        layer = helped[:, offset : offset + num_params].cumsum(axis=0)
        offset += num_params
        randinds = np.random.choice(layer.shape[1], size=min(num_sample, layer.shape[1]), replace=False)
        rand_trajs = layer[:, randinds].T

        if 'kernel' in names[i] and neuron_trajs:
            fig, axs = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 4]})
            cmap_range = np.percentile(np.absolute(neuron_trajs[i//2]), color_percentile)
            axs[0].imshow(-neuron_trajs[i//2], cmap=custom_cmap, vmin=-cmap_range, vmax=cmap_range, aspect='auto')
            axs[0].set_yticks(np.arange(0, neuron_trajs[i//2].shape[0], int(neuron_trajs[i//2].shape[0] / 5)))
            axs[0].set_ylabel('channels')

            for j, traj in enumerate(rand_trajs):
                axs[1].plot(np.concatenate((np.array([0]), traj)), alpha=0.5)
            axs[1].axhline(0, color='k', alpha=0.5, linestyle=':')
            axs[1].set_xlim(-1, helped.shape[0] + 1)
            axs[1].set_xlabel('iteration')
            axs[1].set_ylabel('per-parameter cumulative LC')
            if title:
                axs[0].set_title('{}, {} channels over time'.format(title, names[i].replace(': kernel', '')))
                axs[1].set_title('{}, {} {} random parameters'.format(title, names[i], num_sample))
            else:
                axs[0].set_title('{} channels over time'.format(names[i].replace(': kernel', '')))
                axs[1].set_title('{} {} random parameters'.format(names[i], num_sample))
        else:
            # just plot the trajectories
            plt.figure(figsize=(12, 5))
            for j, traj in enumerate(rand_trajs):
                plt.plot(np.concatenate((np.array([0]), traj)), alpha=0.5)
            plt.axhline(0, color='k', alpha=0.5, linestyle=':')
            plt.xlim(-1, helped.shape[0] + 1)
            plt.xlabel('iteration')
            plt.ylabel('per-parameter cumulative LC')
            if title:
                plt.title('{}, {} random parameters'.format(title, names[i]))
            else:
                plt.title(names[i] + ' random parameters')

        if save_dest:
            plt.savefig(save_dest)
        plt.show()

######################################## plotters for notebooks ########################################

def plot_loss_train_test(trainloss, testloss, plot_dest=None):
    plt.plot(trainloss, label='train loss')
    plt.plot(testloss, label='test loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('train and test loss')
    plt.legend()
    save_or_show(plot_dest)

# plots approx loss based on rk4, returns diff per iter
def plot_approx_loss(loss, helped, plot_dest=None):
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax1.plot(np.array(loss), label='actual loss')
    approx_loss = np.array(get_approx_loss(loss, helped))
    ax1.plot(approx_loss, label='sum approx loss')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss')
    ax1.legend()

    relative_err = abs(((approx_loss[-1] - approx_loss[0]) - (loss[-1] - loss[0])) / (loss[-1] - loss[0]))

    diff = np.array(loss[:len(approx_loss)]) - approx_loss
    diff_per_iter = [diff[i+1] - diff[i] for i in range(len(diff) - 1)]
    ax2 = ax1.twinx()
    ax2.plot(diff_per_iter, 'C1', alpha=0.5, linewidth=0.5, marker='.', markersize=1)
    ax2.set_ylabel('diff per iteration')
    plt.title('true and approximated loss (err %.2f%%)' % (relative_err * 100))
    save_or_show(plot_dest)
    if not plot_dest:
        plt.hist(diff_per_iter, bins=50)
        plt.show()
        print(np.std(diff_per_iter))

    return np.array(diff_per_iter)

# kinda a subset of save_layer_train_test below...
# plots total helped per kernel layer and all layers
def plot_total_layers(total_helped, names, shapes, plot_dest=None):
    layer_indices = np.append(0, np.cumsum([np.prod(shape) for shape in shapes]))
    total_kernels = []
    total_all = []
    for i, name in enumerate(names):
        startind, endind = layer_indices[i], layer_indices[i+1]
        layer_total = total_helped[startind:endind].sum()
        total_all.append(layer_total)
        if not ('batch_norm' in name or 'bias' in name):
            total_kernels.append(layer_total)
    print('Bias and batch norm layers helped {} percent of total'.format(
        100 * (1 - np.sum(total_kernels) / np.sum(total_all))))

    # TODO make nice stacked chart over time
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(total_kernels)), total_kernels)
    plt.axhline(0, c='k', linestyle=':')
    plt.title('Kernel layers')
    plt.xlabel('Layer')
    plt.ylabel('IOL per layer')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(total_all)), total_all)
    plt.axhline(0, c='k', linestyle=':')
    plt.title('All layers')
    plt.xlabel('Layer')
    plt.ylabel('IOL per layer')
    save_or_show(plot_dest)

def get_percent_helped_no_plots(helped):
    negatives = (helped < 0).sum(axis=1)
    return (negatives / helped.shape[1]).mean()

def percent_helped_per_iter_all_layers(helped, trainloss, names, shapes, skip_plots=True, plot_dest=None):
    layer_indices = np.append(0, np.cumsum([np.prod(shape) for shape in shapes]))
    kernel_inds = np.array(['kernel' in name for name in names])
    layer_mean_percs = []
    layer_max_percs = []
    for i in range(len(names)):
        startind, endind = layer_indices[i], layer_indices[i+1]
        if not skip_plots:
            print(names[i])
        means, maxes = percent_helped_per_iter(np.array(helped[:, startind:endind]), trainloss, skip_plots=skip_plots)
        layer_mean_percs.append(means)
        layer_max_percs.append(maxes)
    plt.plot(layer_mean_percs)
    plt.plot(np.arange(len(names))[kernel_inds], np.array(layer_mean_percs)[kernel_inds], '.', c='C0', label='kernels')
    plt.xlabel('layer')
    plt.ylabel('percent helped')
    plt.title('Average percent of weights helping each iter')
    plt.legend()
    save_or_show(plot_dest)

    plt.plot(layer_max_percs)
    plt.plot(np.arange(len(names))[kernel_inds], np.array(layer_max_percs)[kernel_inds], '.', c='C0', label='kernels')
    plt.xlabel('layer')
    plt.ylabel('max percent helped')
    plt.title('Max percent of weights helping at any iter')
    plt.legend()

def percent_helped_histograms(helped):
    # histogram of how many params helped, over each iteration
    negatives = helped < 0
    params_per_iter = negatives.sum(axis=1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(params_per_iter, bins=50)
    plt.xlabel('params helped per iteration')
    plt.ylabel('number of iterations')
    plt.title('Mean = {:.2f}, stdev = {:.2f}'.format(params_per_iter.mean(), params_per_iter.std()))

    # histogram of how many iters helped, over each parameter
    iters_per_param = negatives.sum(axis=0)
    plt.subplot(1, 2, 2)
    plt.hist(iters_per_param, bins=50)
    plt.xlabel('iterations helped per param')
    plt.ylabel('number of params')
    plt.title('Mean = {:.2f}, stdev = {:.2f}'.format(iters_per_param.mean(), iters_per_param.std()))
    plt.show()

# only prints out counts. no plots.
def count_oscillations(weights, grads_train, helped, splits_per_iter, names, shapes):
    layer_indices = np.append(0, np.cumsum([np.prod(shape) for shape in shapes]))
    whole_indices = np.append(0, np.cumsum(splits_per_iter))
    grads_whole_step = grads_train[whole_indices]
    grad_zeros, grad_turns, weight_turns, helped_turns = 0, 0, 0, 0
    l1, l2, l3, l4 = [], [], [], []
    print('grad cross zeros, grad turns, weight turns, helped turns:')
    for i, name in enumerate(names):
        startind, endind = layer_indices[i], layer_indices[i+1]
        ret1 = num_cross_zero(grads_whole_step[:, startind:endind])
        ret2 = num_local_optima(grads_whole_step[:, startind:endind])
        ret3 = num_local_optima(weights[:, startind:endind])
        ret4 = num_local_optima(helped[:, startind:endind])
        if 'kernel' in name:
            print('{}: {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(i, ret1.mean(), ret2.mean(), ret3.mean(), ret4.mean()))
        l1.append(ret1.mean())
        l2.append(ret2.mean())
        l3.append(ret3.mean())
        l4.append(ret4.mean())
        grad_zeros += ret1.sum()
        grad_turns += ret2.sum()
        weight_turns += ret3.sum()
        helped_turns += ret4.sum()
    c = weights.shape[1]
    print('All: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(grad_zeros / c, grad_turns / c, weight_turns / c, helped_turns / c))

    itp = np.array(['kernel' in name for name in names])
    plt.plot(np.array(l1)[itp], '.-')
    plt.xlabel('kernel layer')
    plt.ylabel('count')
    plt.title('Number of times gradient crosses 0')
    plt.show()
    plt.plot(np.array(l3)[itp], '.-')
    plt.xlabel('kernel layer')
    plt.ylabel('count')
    plt.title('Number of times weight switches directions')
    plt.show()

# helped train and test trajectories (cumulative) for each layer
def plot_trajectory_per_layer(helped, names, shapes, helped_test=None, layer_inds=[], iters=-1, plot_dest=None):
    if iters == -1:
        iters = helped.shape[0]
    if len(layer_inds) == 0:
        layer_inds = range(len(names))
    layer_indices = np.append(0, np.cumsum([np.prod(shape) for shape in shapes]))
    for i, name in enumerate(names):
        if i not in layer_inds:
            continue
        startind, endind = layer_indices[i], layer_indices[i+1]
        traj = np.cumsum(helped[:, startind:endind].sum(axis=1))
        plt.plot(traj[:iters])

        print(name, np.argsort(traj)[:10])
        if helped_test:
            plt.plot(np.cumsum(helped_test[:iters, startind:endind].sum(axis=1))) # can also plot overfit here instead

        plt.axhline(0, c='k', linestyle=':')
        plt.xlabel('iterations')
        plt.ylabel('IOL, summed over parameters')
        plt.title(name)
        if isinstance(plot_dest, str):
            save_or_show('%s_%02d' % (plot_dest, i))
        else:
            save_or_show(plot_dest)

# top `percent` percent helped X% of total helped
def top_percent_helped(values, percent):
    # only considers negative values
    negatives = values[np.where(values < 0)]
    if len(negatives) == 0:
        return -1 # no params helped
    ordered = sorted(-negatives, reverse=True) # make positive, largest first
    top = np.sum(ordered[:int(math.ceil(percent / 100. * len(ordered)))]) / np.sum(ordered)
    return top

# returns fraction of helping params needed to make up a given percentage of helped
def frac_params_needed(values, percent):
    # only considers negative values
    ordered_cumul = np.sort(values[np.where(values < 0)]).cumsum()
    if len(ordered_cumul) == 0:
        return -1 # no params helped
    percentile = percent * ordered_cumul[-1]
    return np.argmax(ordered_cumul <= percentile) / len(ordered_cumul)

# plot helped distributions
def plot_distributions_gini(values, shapes, names, top_perc=1, plot_dest=None):
    num_bins = 29 # one is added later

    # mostly redundant code as what's in the loop
    _, bins = np.histogram(values, bins=num_bins)
    aligned_bins = align_bins_zero(bins)
    plt.hist(values, bins=aligned_bins, log=True)
    plt.xlabel('total helped')
    plt.ylabel('number of params (log scale)')
    perc_hurt = 100 * np.count_nonzero(values > 0) / values.size
    plt.title('{}: {:.2f} percent hurt\nhelped gini = {:.4f}, hurt gini = {:.4f}'.format(
        'All layers', perc_hurt, negative_gini(values), positive_gini(values)))
    plt.axvline(x=0, color='k', alpha=0.5, linestyle=':')
    if isinstance(plot_dest, str):
        save_or_show('%s_%s' % (plot_dest, 'all_layers'))
    else:
        save_or_show(plot_dest)

    print('All layers together:')
    print('Helped gini = {:.5f}, hurt gini = {:.5f}, {:.5f} percent hurt'.format(
        negative_gini(values), positive_gini(values), perc_hurt))
    print('90 percent of learning done by {:.3f} percent of parameters'.format(100 * frac_params_needed(values, 0.9)))
    print('Top {} percent helped {:.5f} percent'.format(top_perc, top_percent_helped(values, top_perc) * 100))

    for i, layer in enumerate(split_and_shape(values, shapes)):
        num_bins = 29 if layer.size > 50 else 19
        _, bins = np.histogram(layer.flatten(), bins=num_bins)
        aligned_bins = align_bins_zero(bins)
        plt.hist(layer.flatten(), bins=aligned_bins)
        plt.xlabel('total helped')
        plt.ylabel('number of params')
        perc_hurt = 100 * np.count_nonzero(layer > 0) / layer.size
        plt.title('{}: {:.2f} percent hurt\nhelped gini = {:.4f}, hurt gini = {:.4f}'.format(
            names[i], perc_hurt, negative_gini(layer), positive_gini(layer)))
        plt.axvline(x=0, color='k', alpha=0.5, linestyle=':')
        if isinstance(plot_dest, str):
            save_or_show('%s_%02d' % (plot_dest, i))
        else:
            save_or_show(plot_dest)

        print('{}:'.format(names[i]))
        print('Helped gini = {:.5f}, hurt gini = {:.5f}, {:.5f} percent hurt'.format(
            negative_gini(layer), positive_gini(layer), perc_hurt))
        print('90 percent of learning done by {:.3f} percent of parameters'.format(100 * frac_params_needed(layer, 0.9)))
        print('Top {} percent helped {:.5f} percent'.format(top_perc, top_percent_helped(layer, top_perc) * 100))

# amount each layer helped - summed and averaged
def helped_by_layer(helped, shapes, names, smoothing=5, ignore_batchnorm=False, plot_dest=None):
    helped_sum = [[] for _ in range(len(shapes))] # helped in one iteration, separated by layer
    helped_mean = [[] for _ in range(len(shapes))]
    for ts in range(helped.shape[0]):
        variables = split_and_shape(helped[ts], shapes)
        for v, var in enumerate(variables):
            helped_sum[v].append(np.sum(var))
            helped_mean[v].append(np.mean(var))

    # averaged
    for i, var in enumerate(helped_mean):
        if ignore_batchnorm and 'batch_norm' in names[i]:
            continue
        plt.plot(smooth_plot(var, smoothing), label=names[i])
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('helped each iteration')
    plt.title('total helped each iteration, by layer, averaged')
    if isinstance(plot_dest, str):
        save_or_show(plot_dest + '_averaged')
    else:
        save_or_show(plot_dest)

    # summed
    for i, var in enumerate(helped_sum):
        if ignore_batchnorm and 'batch_norm' in names[i]:
            continue
        plt.plot(smooth_plot(var, smoothing), label=names[i])
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('helped each iteration')
    plt.title('total helped each iteration, by layer, sum')
    if isinstance(plot_dest, str):
        save_or_show(plot_dest + '_sum')
    else:
        save_or_show(plot_dest)

    # summed but stretched
    for i, var in enumerate(helped_sum):
        if ignore_batchnorm and 'batch_norm' in names[i]:
            continue
        plt.plot(smooth_plot(np.array(var) / -min(var), smoothing), label=names[i])
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('helped each iteration')
    plt.title('total helped each iteration, by layer, normalized by max')
    if isinstance(plot_dest, str):
        save_or_show(plot_dest + '_normalized')
    else:
        save_or_show(plot_dest)

# plot how much each neuron helped, each iteration. Large 2d plot for each layer
def plot_neurons_all_time(neuron_trajs, names, color_percentile=99, plot_dest=None):
    # custom color map matching the green and red used in other plots
    # red is negative and green is positive because default colormaps are like that
    cdict = {'red':   [[0.0,  0.4, 0.4],
                       [0.4,  1.0, 1.0],
                       [0.5,  1.0, 1.0],
                       [1.0,  0.0, 0.0]],
             'green': [[0.0,  0.0, 0.0],
                       [0.5,  1.0, 1.0],
                       [0.6,  1.0, 1.0],
                       [1.0,  0.3, 0.3]],
             'blue':  [[0.0,  0.0, 0.0],
                       [0.5,  1.0, 1.0],
                       [1.0,  0.0, 0.0]]}
    custom_cmap = LinearSegmentedColormap('lc', segmentdata=cdict) # previously used PiYG

    for i, layer in enumerate(neuron_trajs):
        #if 'batch_norm' in names[2 * i]:
        #    continue
        fig_height = min(3 + int(layer.shape[0] / 20), 15)
        plt.figure(figsize=(15, fig_height))
        cmap_range = np.percentile(np.absolute(layer), color_percentile)
        # sorted_layer = sorted(list(layer.T), key=lambda x: np.average(np.arange(len(x)), weights=np.abs(x)))
        plt.imshow(-layer, cmap=custom_cmap, vmin=-cmap_range, vmax=cmap_range, aspect='auto')
        plt.xlabel('iterations')
        plt.ylabel('neurons')
        plt.title(names[2 * i])
        if isinstance(plot_dest, str):
            save_or_show('%s_%02d' % (plot_dest, i))
        else:
            save_or_show(plot_dest)

def plot_trajectories_per_layer(helped, shapes, names, plot_dest=None):
    offset = 0
    for i, shape in enumerate(shapes):
        # layer shape is (iteration, all params in layer flattened)
        num_params = np.prod(shape)
        layer = helped[:, offset : offset + num_params].cumsum(axis=0)
        offset += num_params

        bestinds = np.argsort(layer[-1])[:50]
        randinds = np.random.choice(layer.shape[1], size=min(50, layer.shape[1]), replace=False)
        best_trajs = layer[:, bestinds].T
        rand_trajs = layer[:, randinds].T
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        for j, traj in enumerate(best_trajs):
            plt.plot(np.concatenate((np.array([0]), traj)), alpha=0.5) # label=bestinds[j])
        plt.xlabel('iterations')
        plt.ylabel('cumulative helped')
        plt.subplot(1, 2, 2)
        for j, traj in enumerate(rand_trajs):
            plt.plot(np.concatenate((np.array([0]), traj)), alpha=0.5)
        plt.xlabel('iterations')
        plt.ylabel('cumulative helped')
        plt.suptitle(names[i])
        #plt.tight_layout()
        if isinstance(plot_dest, str):
            save_or_show('%s_%02d' % (plot_dest, i))
        else:
            save_or_show(plot_dest)
