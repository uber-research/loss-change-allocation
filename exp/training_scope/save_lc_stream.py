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

import numpy as np
import argparse
import h5py
import os
import time
from ast import literal_eval
import sys

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)

from general.util import mkdir_p
import matplotlib
matplotlib.use('Agg')
import plot_util as util

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('resdir', type=str, help='Path to dir containing weights and gradients file.')
    parser.add_argument('--custom_filename', type=str, default='', help='Filename if not gradients_adaptive')
    parser.add_argument('--chunk_size', type=int, default=200, help='chunk size (iterations)')
    parser.add_argument('--max_iters', type=int, default=9999999999, help='only do first n iterations')
    parser.add_argument('--first_order', action='store_true', help='calculate with first order instead of rk4')
    return parser

def stream_helped_rk_adaptive(weights, splits_per_iter, grads_train_list, grads_test_list, helped_train, helped_test, args):
    coeffs = {}
    for i in [2, 4, 8, 16, 32]:
        coeffs[i] = util.get_rk_coeffs(i)

    iters_to_calc = min(args.max_iters, weights.shape[0] - 1)
    weights_prev = weights[0]
    timerstart = time.time()
    list_ind = 0 # which np array in grads_train_list to use, from 0 to 3
    grads_ind = 0 # which index in grads_train_list[list_ind] we're currently at

    for chunk in range(0, iters_to_calc, args.chunk_size):
        buffer_helped_train = np.zeros((args.chunk_size, weights.shape[1]))
        buffer_helped_test = np.zeros((args.chunk_size, weights.shape[1]))
        for i in range(args.chunk_size):
            ts = chunk + i

            num_splits = splits_per_iter[ts]
            grads_train_iter = grads_train_list[list_ind][grads_ind:grads_ind + num_splits + 1]
            grads_test_iter = grads_test_list[list_ind][grads_ind:grads_ind + num_splits + 1]
            k_train = np.matmul(coeffs[num_splits], grads_train_iter) / coeffs[num_splits].sum()
            k_test = np.matmul(coeffs[num_splits], grads_test_iter) / coeffs[num_splits].sum()

            weights_next = weights[ts + 1]
            buffer_helped_train[i] = np.multiply(k_train, weights_next - weights_prev)
            buffer_helped_test[i] = np.multiply(k_test, weights_next - weights_prev)
            grads_ind += num_splits
            if grads_ind + 1 >= grads_train_list[list_ind].shape[0]:
                list_ind += 1
                grads_ind = 0
            weights_prev = weights_next
        # write every chunk_size
        helped_train[chunk : chunk + args.chunk_size] = buffer_helped_train
        helped_test[chunk : chunk + args.chunk_size] = buffer_helped_test
        print('Completed {}/{} iterations ({:.2f} s)'.format(
            ts + 1, iters_to_calc, time.time() - timerstart))

# used for side experiments
def stream_helped_first_order(weights, splits_per_iter, grads_train_list, grads_test_list, helped_train, helped_test, args):
    iters_to_calc = min(args.max_iters, weights.shape[0] - 1)
    weights_prev = weights[0]
    timerstart = time.time()
    list_ind = 0 # which np array in grads_train_list to use, from 0 to 3
    grads_ind = 0 # which index in grads_train_list[list_ind] we're currently at

    for chunk in range(0, iters_to_calc, args.chunk_size):
        buffer_helped_train = np.zeros((args.chunk_size, weights.shape[1]))
        buffer_helped_test = np.zeros((args.chunk_size, weights.shape[1]))
        for i in range(args.chunk_size):
            ts = chunk + i

            num_splits = splits_per_iter[ts]
            k_train = grads_train_list[list_ind][grads_ind]
            k_test = grads_test_list[list_ind][grads_ind]

            weights_next = weights[ts + 1]
            buffer_helped_train[i] = np.multiply(k_train, weights_next - weights_prev)
            buffer_helped_test[i] = np.multiply(k_test, weights_next - weights_prev)
            grads_ind += num_splits
            if grads_ind + 1 >= grads_train_list[list_ind].shape[0]:
                list_ind += 1
                grads_ind = 0
            weights_prev = weights_next
        # write every chunk_size
        helped_train[chunk : chunk + args.chunk_size] = buffer_helped_train
        helped_test[chunk : chunk + args.chunk_size] = buffer_helped_test
        print('Completed {}/{} iterations ({:.2f} s)'.format(
            ts + 1, iters_to_calc, time.time() - timerstart))

def get_streamds_list(hf, keyroot, chunk_size):
    grads_list = [util.streamds(hf['{}_0'.format(keyroot)], chunk_size)]
    for i in range(1, 4):
        nextkey = '{}_{}'.format(keyroot, i)
        if nextkey in hf.keys():
            grads_list.append(util.streamds(hf[nextkey], chunk_size))
    return grads_list

def main():
    parser = make_parser()
    args = parser.parse_args()

    # read weights
    hf_weights = h5py.File(args.resdir + '/weights', 'r')
    weights = util.streamds(hf_weights['all_weights'], args.chunk_size)
    assert min(args.max_iters, weights.shape[0] - 1) % args.chunk_size == 0, 'make chunk_size divide evenly into num iters'

    # read gradients
    gradients_filename = '/gradients_adaptive'
    if args.custom_filename:
        gradients_filename = '/' + args.custom_filename
    hf_gradients = h5py.File(args.resdir + gradients_filename, 'r')
    grads_train_list = get_streamds_list(hf_gradients, 'grads_train', args.chunk_size)
    grads_test_list = get_streamds_list(hf_gradients, 'grads_test', args.chunk_size)
    splits_per_iter = np.array(hf_gradients['num_splits'])

    # set up output
    suffix = ''
    if args.max_iters < weights.shape[0] - 1:
        suffix = '_{}iters'.format(args.max_iters)
    filename = '/helped_first_order' if args.first_order else '/helped'
    hf_helped = h5py.File(args.resdir + filename + suffix, 'w-')
    helped_train = hf_helped.create_dataset('helped', (min(args.max_iters, weights.shape[0] - 1),weights.shape[1]),
        dtype='f4', compression='gzip')
    helped_test = hf_helped.create_dataset('helped_test', (min(args.max_iters, weights.shape[0] - 1), weights.shape[1]),
        dtype='f4', compression='gzip')

    # calculate helped and write to file
    if args.first_order:
        stream_helped_first_order(weights, splits_per_iter, grads_train_list, grads_test_list, helped_train, helped_test, args)
    else:
        stream_helped_rk_adaptive(weights, splits_per_iter, grads_train_list, grads_test_list, helped_train, helped_test, args)

    hf_helped.close()
    hf_weights.close()
    hf_gradients.close()

if __name__ == '__main__':
    main()
