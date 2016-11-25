#!/usr/bin/env python2.7

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter
import sys


def plot_losses(Ltrain, train_samples_number, Ltest, test_samples_number, root_dir, timestamp, alternate_name=''):
    if not alternate_name:
        np.save(root_dir + 'logfiles/' + timestamp + '/Ltrain.npy', Ltrain)
        np.save(root_dir + 'logfiles/' + timestamp + '/train_samples_number.npy', train_samples_number)
        np.save(root_dir + 'logfiles/' + timestamp + '/Ltest.npy', Ltest)
        np.save(root_dir + 'logfiles/' + timestamp + '/test_samples_number.npy', test_samples_number)

    if min(Ltrain) > 0 and min(Ltest) > 0:
        print('plot losses')
        Ltrain = np.array(Ltrain) / train_samples_number
        Ltest = np.array(Ltest) / test_samples_number

        # standard loss plot
        plt.figure()
        plt.plot(Ltrain[1:], 'b', label='training set loss')
        plt.plot(Ltest[:-1], 'r', label='test set loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('mean loss per sample')
        plt.savefig(root_dir + 'logfiles/' + timestamp + '/losses' + alternate_name + '.png')
        plt.close()

        # log scale
        plt.figure()
        plt.plot(Ltrain[1:], 'b', label='training set loss')
        plt.plot(Ltest[:-1], 'r', label='test set loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('mean loss per sample (log scale)')
        plt.savefig(root_dir + 'logfiles/' + timestamp + '/losses_log_scale' + alternate_name + '.png')
        plt.close()

        # leave out the first 10 iterations
        plt.figure()
        plt.plot(Ltrain[1+10:], 'b', label='training set loss')
        plt.plot(Ltest[10:-1], 'r', label='test set loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('mean loss per sample')
        plt.savefig(root_dir + 'logfiles/' + timestamp + '/losses_starting_at_epoch_10' + alternate_name + '.png')
        plt.close()

        # leave out the first 10 iterations, log scale
        plt.figure()
        plt.plot(Ltrain[1+10:], 'b', label='training set loss')
        plt.plot(Ltest[10:-1], 'r', label='test set loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('mean loss per sample')
        plt.savefig(root_dir + 'logfiles/' + timestamp + '/losses_starting_at_epoch_10_log_scale' + alternate_name + '.png')
        plt.close()
