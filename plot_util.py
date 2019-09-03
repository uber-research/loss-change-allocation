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

# return loss curve based on starting loss and the LCA totals
def get_approx_loss(loss, helped):
    return loss[0] + np.concatenate((np.array([0]), np.cumsum(helped.sum(axis=1))))

# e.g. [1, 4, 2, 4, 1] for num_splits = 4
def get_rk_coeffs(num_splits):
    coeffs = np.ones(num_splits + 1)
    coeffs[1::2] = 4
    coeffs[2:-1:2] = 2
    return coeffs

def get_lca_rk_adaptive(weights, gradients, splits_per_iter):
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

# returns list of variables as np arrays in their original shape
def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

# get neurons, given either a 2d fc layer or a 4d conv layer
def get_neurons(layer):
    return np.sum(layer, axis=tuple(range(len(layer.shape) - 1)))

# This assumes all layers have a bias layer immediately after it
# returns list. each element is a vector
def get_neurons_plus_biases(split_vars):  # flattened and joined
    layers = []
    for i in range(int(len(split_vars)/2)):
        layers.append(get_neurons(split_vars[2*i]) + split_vars[2*i + 1].flatten())
    return layers

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

######################################## plotters for notebooks ########################################

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
    plt.ylabel('LCA per layer')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(total_all)), total_all)
    plt.axhline(0, c='k', linestyle=':')
    plt.title('All layers')
    plt.xlabel('Layer')
    plt.ylabel('LCA per layer')
    save_or_show(plot_dest)

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
        plt.ylabel('LCA, summed over parameters')
        plt.title(name)
        if isinstance(plot_dest, str):
            save_or_show('%s_%02d' % (plot_dest, i))
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
