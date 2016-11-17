#!/usr/bin/env python2.7
import numpy as np
import lasagne

def lasagne2str(network):
    layers = lasagne.layers.get_all_layers(network)
    result = ''
    for layer in layers:
        t = type(layer)
        if t is lasagne.layers.input.InputLayer:
            pass
        elif t is lasagne.layers.conv.Conv2DLayer:
            result += ' {}[{}]'.format(layer.num_filters,
                                       'x'.join([str(fs) for fs in layer.filter_size]))
        elif t is lasagne.layers.pool.MaxPool2DLayer:
            result += ' max[{}]'.format('x'.join([str(fs)
                                                  for fs in layer.pool_size]))
        elif t is lasagne.layers.conv.Conv1DLayer:
            result += ' {}[{}]'.format(layer.num_filters, layer.filter_size)
        elif t is lasagne.layers.pool.MaxPool1DLayer:
            result += ' max[{}]'.format(layer.pool_size)
        elif t is lasagne.layers.DropoutLayer:
            result += ' d{:g}'.format(layer.p)
        elif t is lasagne.layers.DenseLayer:
            result += ' fc[{}]'.format(layer.num_units)
        else:
            result += ' ' + t.__name__
        #result += str(lasagne.layers.get_output_shape(layer, input_shapes=None))+' '
    return result.strip()
