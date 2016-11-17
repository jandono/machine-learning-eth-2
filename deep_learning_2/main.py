#!/usr/bin/env python2.7

import os
os.environ["THEANO_FLAGS"] = "device=gpu,lib.cnmem=0.9"#,compute_test_value = raise"#,allow_gc=False"
import numpy as np
import random
import time
import sys
import Queue as Q
import itertools
import atexit
import shutil
import re
import math
from collections import Counter

import theano
import theano.tensor as T
import lasagne
import theano.sandbox.cuda.basic_ops as sbcuda
from theano.compile.nanguardmode import NanGuardMode


import print_module
import load_multiple_slices
import plot_module
####################### CONSTANTS #######################

# A trained network is only loaded if CONTINUE_TRAINING=1
TRAINED_NETWORK_TIMESTAMP = '2016-10-23-18:17:29'
#TRAINED_NETWORK_TIMESTAMP = '2016-10-09-19:10:23'
#TRAINED_NETWORK_TIMESTAMP = '2016-10-09-13:55:57'

# priority queue: linear
CONTINUE_TRAINING = 0 #load an already trained network and continue the training
TEST_ONLY = 0 # load the network and run test set
LOAD_EPOCH = 50 # network parameters of all trainings epochs are saved. Specify which ones shoud be used

# Multi-level loss
# All lines where you have to change code to configure different multi-level loss functions are marked with MULTI_LEVEL

DATA_DIR = '/data/gallussb/2_mlproject/' # has to end with a /

# Local root directory with code and logfiles
ROOT_DIR = '/home/gallussb/Documents/2_mlproject/'  # has to end with a /

MB_SIZE = 1
TEST_MB_SIZE = 1

LEARNING_RATE = 0.000001

# MIN_AGE = 1.0
# MAX_AGE = 100.0

# Create log files:
LOGFILES = DATA_DIR + 'logfiles/'
TIMESTAMP = time.strftime("%Y-%m-%d-%H:%M:%S")
os.mkdir(LOGFILES + TIMESTAMP)
print((LOGFILES + TIMESTAMP))
DEBUG_FILE = open(LOGFILES + TIMESTAMP + '/debug.txt', 'w')
INFO_FILE = open(LOGFILES + TIMESTAMP + '/info.txt', 'w')
os.mkdir(LOGFILES + TIMESTAMP + '/test_outputs/')


# write error messages to file stderr.txt and terminal
class ErrorLogger(object):
    def __init__(self):
        self.terminal = sys.stderr
        self.log = open(LOGFILES + TIMESTAMP + '/stderr.txt', 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stderr = ErrorLogger()

# Theano Debugging for NaNs
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break

########################################################


def buildMB(list_of_np_arrays, indices):
    """
    Build a minibatch from samples of variable size into a numpy.ndarray.
    Array shape convention: (samples, channels, rows, columns).
        list_of_np_arrays : list of numpy arrays (either inputs or targets of entire training data set).
        indices : indices to select from entire data for the minibatch.
    """

    #Debug
    #print("data_size {}".format(list_of_np_arrays[0].shape))

    # Input of multiple size can also be processed, they dimensions of the
    # biggest element in the minibatch are selected
    list_shape = [max([list_of_np_arrays[i].shape[j] for i in indices])
                  for j in range(list_of_np_arrays[0].ndim)]
    #Debug
    #print("list_shape {}".format(list_shape))

    if not list_shape: # targets can be just single numbers, i.e. doubles
        batch_shape = [len(indices), 1]
        #print("These are the targets, batch shape {}".format(batch_shape))
    elif len(list_shape) == 1: # targets for classification are sparse vectors
        batch_shape = [len(indices), list_shape[0]]
    else: # training input, only one input channel. Change this for multiple input channels
        batch_shape = [len(indices), 1, list_shape[0], list_shape[1]]

    #print("batch_shape {}".format(batch_shape))
    batch = np.zeros(batch_shape, dtype=list_of_np_arrays[0].dtype)
    for i in range(len(indices)):
        current = list_of_np_arrays[indices[i]]
        current_shape = current.shape
        if not current_shape:
            # targets are single numbers
            batch[i, 0] = current
        elif len(list_shape) == 1:
            batch[i, 0:current_shape[0]] = current
        else:
            batch[i, :, 0:current_shape[0], 0:current_shape[1]] = current
    return batch

def runMB(XYlist, indices, eval_fn, vis_fn=None):
    """
    Build a minibatch from full data (using the buildMB function) and
    apply any evaluation (e.g. training) and visualization functions.
        XYlist :
            List consisting of all inputs, and optionally all targets, etc.
            This is done for versatility. Particularly:
            Training case: XYlist consists of inputs and targets.
            Testing case: XYlist can consist of inputs only, or also include
            targets (for error calculation) if targets are available.
    """
    MBlist = [buildMB(X, indices) for X in XYlist]
    results = eval_fn(MBlist)
    # I don't use vis_fn at the moment
    if vis_fn is not None:
        print("vis_fn is used")
        vis_fn(MBlist, results)
    # # for the position histogram
    # results.append(MBlist[3])
    return results


def printMB(train_targets, YhatMB, idcMB):
    for i in range(YhatMB.shape[0]):
        index = idcMB[i]
        DEBUG_FILE.write("{0}:\t -> {1}\n".format(train_targets[index], YhatMB[i]))
    DEBUG_FILE.flush()

def define_network(X):
    # Neural network configuration
    pad = 'valid' #full, same or valid
    nonlinearity = lasagne.nonlinearities.leaky_rectify
    # Consider Normal, Glorot and He initialization as well
    #W = lasagne.init.Orthogonal(gain='relu')
    W = lasagne.init.Normal()
    pool_pad = (0,0)
    # if padding in MaxPool2DLayer is used, ignore_border has to be set to True

    # VGG 11 layers
    # 176 x 208 x 176
    # input image size has to be explicitly specified
    network = lasagne.layers.InputLayer(
        shape=(None, 1, 176, 208), input_var=X)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, pad=pool_pad)

    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, pad=pool_pad)

    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, pad=pool_pad)

    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, pad=pool_pad)

    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=3, pad=pad,
                                         nonlinearity=nonlinearity, W=W)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, pad=pool_pad)

    # Global Pooling for each channel
    #If we don't have variable input size, we don't need that. But a fully connected network on too many neurons isn't a good idea
    # network = lasagne.layers.GlobalPoolLayer(network, pool_function=T.max)


    # DEBUG: Don't use inspect
    # "Inspect" layer
    #inspect = lasagne.layers.get_output(network)


    denselayer_nonlinearity = lasagne.nonlinearities.leaky_rectify
    # output_nonlinearity = lasagne.nonlinearities.softmax # for classification
    output_nonlinearity = lasagne.nonlinearities.sigmoid

    #network = lasagne.layers.DenseLayer(
        #network, 4096, nonlinearity=denselayer_nonlinearity, W=W)
    #network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, 2048, nonlinearity=denselayer_nonlinearity, W=W)
    network = lasagne.layers.dropout(network, p=0.5)
    network = lasagne.layers.DenseLayer(
        network, 1024, nonlinearity=denselayer_nonlinearity, W=W)
    network = lasagne.layers.dropout(network, p=0.5)

    # Output layer. One neuron for regression
    output = lasagne.layers.DenseLayer(network, 1, nonlinearity=output_nonlinearity)

    # Debug
    #print("output shape {}".format(lasagne.layers.get_output_shape(output)))
    ############## LOAD exisiting network if desired #####################
    if CONTINUE_TRAINING or TEST_ONLY:
        saved_params_npz = np.load(LOGFILES + TRAINED_NETWORK_TIMESTAMP + '/network_parameters_epoch_' + str(LOAD_EPOCH) + '.npz')
        saved_params = [x[1] for x in np.lib.npyio.NpzFile.items(saved_params_npz)]
        saved_params_singles = saved_params[0]
        np.lib.npyio.NpzFile.close(saved_params_npz)
        lasagne.layers.set_all_param_values(output, saved_params_singles, trainable=True)


    params = lasagne.layers.get_all_params(output, trainable=True)
    INFO_FILE.write('Number of parameters - {:d}\n'.format(
        lasagne.layers.count_params(output)))
    INFO_FILE.write(print_module.lasagne2str(output) + '\n')

    print('Defined Network')
    #return {'output': output, 'inspect': inspect, 'params': params}
    return {'output': output, 'params': params}

# def normalize_targets(data):
#     """
#     data: ndarray
#     """
#     normalized = np.zeros(data.shape, dtype = np.float32)
#     for i in range(data.shape[0]):
#         normalized[i] = (data[i] - MIN_AGE) / (MAX_AGE - MIN_AGE)
#     return normalized

# def denormalize_outputs(data):
#     denormalized = np.zeros(data.shape, dtype = np.float32)
#     for i in range(data.shape[0]):
#         denormalized[i] = data[i] * (MAX_AGE - MIN_AGE) + MIN_AGE
#     return denormalized

def print_outputs(data, epoch=0):
    output_file = open(LOGFILES + TIMESTAMP + '/test_outputs/outputs_epoch_' + str(epoch) + '.csv', 'w')
    output_file.write("ID,Prediction\n")
    for i in range(data.shape[0]):
        output_file.write("{},{} \n".format(i+1,int(data[i])))

def train_model(learning_rate, data, num_epochs=100):
    """
    Training function.
    """
    train_images = data['X']
    print("train_images {}".format(train_images.shape))
    train_targets = data['y']
    print("train_targets {}".format(train_targets.shape))
    # print("Target dimension {}".format(train_targets.shape))
    test_images = data['Z']
    print("test_images {}".format(test_images.shape))

    slices_limit = data['SLICES_LIMIT'] * 2 + 1

    #print(train_targets)


    # Theano variables
    # Float (ftensor4, fmatrix) for CUDA (32 bit)
    X = T.ftensor4('inputs')

    network_interface = define_network(X)
    #output, inspect, params = network_interface['output'], network_interface['inspect'], network_interface['params']
    output, params = network_interface['output'], network_interface['params']

    # Save network when there is an exception
    atexit.register(save_network, output) #, filename_prolongation='aborted')

    # Target outputs, doubles. Not a matrix any more, because we do regression
    Y = T.fmatrix('targets')
    #Y = T.fvector('targets') # or use T.frow
    # Y.tag.test_value = np.random.rand(10).astype(np.float32)


    # Neural network output

    Yhat_train = lasagne.layers.get_output(output)
    #print("Yhat_train type {}".format(type(Yhat_train)))
    # Thought this would help, but it doesn't
    # Yhat_train = T.extra_ops.squeeze(Yhat_train)
    Yhat_test = lasagne.layers.get_output(output, deterministic=True)


    # Loss functions
    def loss_fn_train(Yhat, Y_X):
        L = lasagne.objectives.binary_crossentropy(Yhat, Y_X) #loss function was categorical_crossentropy
        # SUM = L.mean()
        SUM = L.sum()
        #print("SUM type {}".format(type(SUM)))
        return SUM
    # def loss_fn_test(Yhat, Y_X):
    #     L = lasagne.objectives.squared_error(Yhat, Y_X)
    #     SUM = L.sum()
    #     return SUM


    Ltrain = loss_fn_train(Yhat_train, Y)
    #print("Ltrain type {}".format(type(Ltrain)))
    # print("Ltrain_singles type {}".format(type(Ltrain_singles)))

    # Ltest = loss_fn_test(Yhat_test, Y)

    print('Defined loss calculation')

    # Training function
    updates = lasagne.updates.adam(Ltrain, params, learning_rate=learning_rate)
    INFO_FILE.write("Adam {:g}\n".format(learning_rate))

    print('Defined updates')

    # train_fn = theano.function([X, Y], [Yhat_train, Ltrain, Ltrain_singles, inspect], updates=updates)
    train_fn = theano.function([X, Y], [Yhat_train, Ltrain], updates=updates)
                                                    #mode=theano.compile.MonitorMode(post_func=detect_nan))
                                                    #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)) #, on_unused_input='warn'), mode='DebugMode')

    # test_fn = theano.function([X], [Yhat_test, inspect]) #, on_unused_input='warn')
    test_fn = theano.function([X], [Yhat_test]) #, on_unused_input='warn')

    print('Defined train and test function')

    indices_train = range(train_images.shape[0])
    indices_test  = range(test_images.shape[0])
    INFO_FILE.write("Training samples: {:d}\nTest samples: {:d}\n".format(
                len(indices_train), len(indices_test)))

    print('Defined training and test set')


    LossTrain = []
    LossTest = []

    train_minibatches_per_epoch = int(len(indices_train) / MB_SIZE)
    train_samples_per_epoch = train_minibatches_per_epoch * MB_SIZE

    # Measure runtime per sample
    total_training_time = 0.0

    for e in xrange(num_epochs):
        t = time.time()
        print "Epoch {:03d}/{:03d}:".format(e + 1, num_epochs)

        # shuffle the training indices
        random.shuffle(indices_train) # in-place


        INFO_FILE.write("Epoch {:03d}/{:03d}:".format(e + 1, num_epochs))
        INFO_FILE.flush()
        DEBUG_FILE.write("Epoch {:03d}/{:03d}\n".format(e + 1, num_epochs))



        ##### TRAINING WITH MINIBATCHES ###########
        LossTrain.append(.0)
        train_start = time.time()

        #train_single_outputs = np.zeros(shape=(len(train_images),1), dtype = np.float32)

        for b in range(0, len(indices_train), MB_SIZE):

            # take indices for the minibatch
            indicesMB = indices_train[b: b + MB_SIZE] # careful with incomplete minibatches
            #print("Debug4 {}".format(b))
            #DEBUG to see the last sample that was trained before crashing
            # debug_samples = [train_targets[x] for x in indicesMB]
            # for d in debug_samples:
            #     DEBUG_FILE.write('{}\n'.format(d))

            #print('train function')
            # YhattrainMB, LtrainMB, Ltrain_singlesMB, inspectMB = runMB(
            #     [train_images, train_targets], indicesMB, eval_fn=(lambda XYlist: train_fn(XYlist[0], XYlist[1])))
            YhattrainMB, LtrainMB = runMB(
                [train_images, train_targets], indicesMB, eval_fn=(lambda XYlist: train_fn(XYlist[0], XYlist[1])))
            #print('append loss')
            LossTrain[-1] += LtrainMB
            #train_single_outputs[b * MB_SIZE : (b+1) * MB_SIZE, 0] = YhattrainMB

            # if MB_SIZE == 1:
            #     Yhat_denormalized = np.rint(denormalize_outputs(np.array(YhattrainMB)))
            # else:
            #     Yhat_denormalized = np.squeeze(np.rint(denormalize_outputs(np.array(YhattrainMB))))

            printMB(train_targets, YhattrainMB, indicesMB)


        total_training_time += time.time() - train_start
        save_network(output, filename_prolongation='_epoch_' + str(e))


        ############# RUN THE TEST DATA ################

        outputs_test = []

        DEBUG_FILE.write('Test set\n')
        for b in range(0, len(indices_test), slices_limit):
            indicesMB = indices_test[b: b + slices_limit]

            # #DEBUG
            # debug_samples = [sample_names[x] for x in indicesMB]
            # for d in debug_samples:
            #     DEBUG_FILE.write('{}\n'.format(d))

            # YhattestMB, inspectMB = runMB([test_images], indicesMB, eval_fn=(lambda XYlist: test_fn(XYlist[0])))
            YhattestMB = runMB([test_images], indicesMB, eval_fn=(lambda XYlist: test_fn(XYlist[0])))
            #YhattestMB, LtestMB, inspectMB = runMB([test_images], indicesMB, eval_fn=(lambda XYlist: test_fn(XYlist[0])))
            #LossTest[-1] += LtestMB
            #print(type(YhattestMB))

            avg = np.average(YhattestMB)

            outputs_test.append(avg)


        # Save the outputs
        # printable_outputs = np.squeeze(np.rint(denormalize_outputs(np.array(outputs_test)))) # round to the nearest integer
        printable_outputs = np.squeeze(np.array(outputs_test)) # round to the nearest integer
        #print(type(printable_outputs))
        #print(printable_outputs.shape)
        print_outputs(printable_outputs, e)


        ########### WRITE TO LOGFILES####################

        print("Time elapsed before writing to logfiles: {:.2f} seconds".format(time.time() - t))
        INFO_FILE.write("\tTotal train set error: {:11.5f}\n".format(LossTrain[-1]))

        INFO_FILE.write('\n')
        INFO_FILE.flush()
        #CONFUSION_MATRIX_FILE.flush()

        print("Time elapsed total: {:.2f} seconds".format(time.time() - t))



    # TODO adapt loss display
    LossTest = [1] * len(LossTrain)
    plot_module.plot_losses(LossTrain, train_samples_per_epoch, LossTest, len(indices_test), DATA_DIR, TIMESTAMP)

    INFO_FILE.write('\nMinibatch size: {:d}\n'.format(MB_SIZE))
    training_time_per_sample = float(total_training_time) / (num_epochs * train_samples_per_epoch)
    INFO_FILE.write('Average training time per sample: {:f} s\n'.format(training_time_per_sample))

    DEBUG_FILE.close()
    INFO_FILE.close()
    return {"params": params, "test_fn": test_fn}

def test_trained_model(data):
    test_images = data['Z']
    print("test_images {}".format(test_images.shape))

    slices_limit = data['SLICES_LIMIT'] * 2 + 1

    # Theano variables
    # Float (ftensor4, fmatrix) for CUDA (32 bit)
    X = T.ftensor4('inputs')

    network_interface = define_network(X)
    #output, inspect, params = network_interface['output'], network_interface['inspect'], network_interface['params']
    output, params = network_interface['output'], network_interface['params']

    # Neural network output

    Yhat_test = lasagne.layers.get_output(output, deterministic=True)

    test_fn = theano.function([X], [Yhat_test]) #, on_unused_input='warn')

    print('Defined test function')

    indices_test  = range(test_images.shape[0])
    INFO_FILE.write("Test samples: {:d}\n".format(len(indices_test)))

    print('Defined test set')

    outputs_test = []

    DEBUG_FILE.write('Test set\n')
    for b in range(0, len(indices_test), slices_limit):
        indicesMB = indices_test[b: b + slices_limit]

        # #DEBUG
        # debug_samples = [sample_names[x] for x in indicesMB]
        # for d in debug_samples:
        #     DEBUG_FILE.write('{}\n'.format(d))

        # YhattestMB, inspectMB = runMB([test_images], indicesMB, eval_fn=(lambda XYlist: test_fn(XYlist[0])))
        YhattestMB = runMB([test_images], indicesMB, eval_fn=(lambda XYlist: test_fn(XYlist[0])))
        #YhattestMB, LtestMB, inspectMB = runMB([test_images], indicesMB, eval_fn=(lambda XYlist: test_fn(XYlist[0])))

        avg = np.average(YhattestMB)

        outputs_test.append(avg)

    # Save the outputs
    #printable_outputs = np.squeeze(np.rint(denormalize_outputs(np.array(outputs_test)))) # round to the nearest integer
    printable_outputs = np.squeeze(np.array(outputs_test)) # round to the nearest integer
    print_outputs(printable_outputs, LOAD_EPOCH)

    DEBUG_FILE.close()
    INFO_FILE.close()
    return {"params": params, "test_fn": test_fn}



def save_network(output, filename_prolongation=''):
    final_params = lasagne.layers.get_all_param_values(output, trainable=True)
    np.savez(LOGFILES + TIMESTAMP + '/network_parameters' + filename_prolongation + '.npz', final_params)

def system_config():
    np.set_printoptions(threshold='nan')
    # raise all errors to find the NaN in the new loss calculation
    np.seterr(all='raise')


def main(epochs=1, max_samples=sys.maxint, tag=''):
    system_config()

    # GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
    # freeGPUMemInGBs = GPUFreeMemoryInBytes/1024./1024/1024
    # INFO_FILE.write('GPU free memory in GB: {}\n'.format(freeGPUMemInGBs))

    if tag:
        INFO_FILE.write(tag + '\n')

    mri_data = load_multiple_slices.load_data(max_samples)

    if TEST_ONLY:
        test_trained_model(mri_data)
    else:
        # Train the neural network
        learning_rate = LEARNING_RATE
        result = None
        while result == None:
            # while loop in order to catch NaN of weights
            result = train_model(learning_rate, mri_data, num_epochs=epochs)
            learning_rate = 0.95 * learning_rate #if an error arises


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Regression of age from MRI data using Lasagne.")
        print("Usage: %s [EPOCHS [MAX_SAMPLES [TAG]]]]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
        print()
        print("MAX_SAMPLES: maximum number of data samples to read (default: no limit)")
        print()
        print("TAG: Describe the current setup (network params etc.)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['epochs'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['max_samples'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['tag'] = str(sys.argv[3])
        main(**kwargs)
