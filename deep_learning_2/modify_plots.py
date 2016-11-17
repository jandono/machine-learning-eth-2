# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:14:24 2016

@author: gallusse
"""

import p3a_plot
import numpy as np

ROOT = '/usr/stud/gallusse/Documents/'
LOGFILES = 'logfiles/'
#RUN = 'playing_with_figures'
#RUN = '2016-08-29-16:14:59_consistency_test_1'
#RUN = '2016-09-01-10:49:40_with_gt_consistency'

#for PQ validation
#RUN = '2016-09-07-20:36:05_vgg_short_no_1d_pooling'
#RUN = '2016-09-06-00:32:16_without_pq_vgg_plus1_short'

#architectures
#RUN = '2016-09-04-14:05:37_CAT_single_losses_networkA'
#RUN = '2016-09-06-00:22:46_vgg_plus1_long_networkB'
#RUN = '2016-09-04-21:23:09_vgg_plus2_networkC'

RUN = '2016-09-09-19:03:19_lower_learning_rate_after_20_epochs'


PROTEIN_LENGTHS = 0
CLASS_HISTOGRAM = 0
POSITION_TRAIN = 0
POSITION_TEST = 0
PRIORITY = 0
LOSS = 1
CONSISTENCY = 0



if PROTEIN_LENGTHS:
    path = ROOT + LOGFILES + RUN + '/' + 'protein_lengths.npy'
    protein_lengths = np.load(path)
    p3a_plot.plot_protein_lengths(protein_lengths, ROOT, RUN, alternate_name='_1000limit5')
    
if CLASS_HISTOGRAM:
    path = ROOT + LOGFILES + RUN + '/' + 'raw_classes_for_class_histogram.npy'
    classes = np.load(path)
    p3a_plot.plot_class_size_distribution_histogram(classes, ROOT, RUN, alternate_name='_hist0')

if POSITION_TRAIN:
    path = ROOT + LOGFILES + RUN + '/position_histograms/' + 'position_histogram_train_epoch_100.npy'
    position_list = np.load(path)
    p3a_plot.plot_position_histogram(position_list, 100, ROOT, RUN, 'train', 10, alternate_name='_new')
    
if POSITION_TEST:
    path = ROOT + LOGFILES + RUN + '/position_histograms/' + 'position_histogram_test_epoch_100.npy'
    position_list = np.load(path)
    p3a_plot.plot_position_histogram(position_list, 100, ROOT, RUN, 'test', 10, alternate_name='_new')

if PRIORITY:
    # special for the case 2016-09-06-00:32:16_without_pq_vgg_plus1_short
#    train_iter_per_sample = np.full(63122, 20,dtype=np.int)
#    path = ROOT + LOGFILES + RUN + '/' + 'raw_classes_for_class_histogram.npy'
#    t_cath_classes = np.load(path)

    path = ROOT + LOGFILES + RUN + '/' + 'train_iterations_per_sample.npy'
    train_iter_per_sample = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 't_cath_classes_for_pq_validation.npy'
    t_cath_classes = np.load(path)
    p3a_plot.pq_validation(t_cath_classes, train_iter_per_sample, ROOT, RUN, point_size = 2, alternate_name='_200')
    
if LOSS:
    path = ROOT + LOGFILES + RUN + '/' + 'Ltrain.npy'
    Ltrain = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 'train_samples_number.npy'
    train_samples_number = np.load(path)        
    path = ROOT + LOGFILES + RUN + '/' + 'Ltest.npy'
    Ltest = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 'test_samples_number.npy' 
    test_samples_number = np.load(path)

    p3a_plot.plot_losses (Ltrain, train_samples_number, Ltest, test_samples_number, ROOT, RUN, alternate_name='_shifted')         
    
    
if CONSISTENCY:
    # load softmax outputs
    #train
    path = ROOT + LOGFILES + RUN + '/' + 'c_softmax_outputs_train.npy'
    c_softmax_outputs_train = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 'a_softmax_outputs_train.npy'
    a_softmax_outputs_train = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 't_softmax_outputs_train.npy'
    t_softmax_outputs_train = np.load(path)
    
    #test
    path = ROOT + LOGFILES + RUN + '/' + 'c_softmax_outputs_test.npy'
    c_softmax_outputs_test = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 'a_softmax_outputs_test.npy'
    a_softmax_outputs_test = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 't_softmax_outputs_test.npy'
    t_softmax_outputs_test = np.load(path)
    
    #load dictionairies
    path = ROOT + LOGFILES + RUN + '/' + 'class_dict_level_1.npy'
    c_class_dict = np.load(path).item()
#    c_class_dict = c_class_dict_0d.item()
#    print(c_class_dict)
#    print(type(c_class_dict))
    path = ROOT + LOGFILES + RUN + '/' + 'class_dict_level_2.npy'
    a_class_dict = np.load(path).item()
    path = ROOT + LOGFILES + RUN + '/' + 'class_dict_level_3.npy'
    t_class_dict = np.load(path).item()
    
    #ground truth data train
    path = ROOT + LOGFILES + RUN + '/' + 'c_ground_truth_for_consistency_train.npy'
    c_ground_truth_train = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 'a_ground_truth_for_consistency_train.npy'
    a_ground_truth_train = np.load(path)
    
    #ground truth data test
    path = ROOT + LOGFILES + RUN + '/' + 'c_ground_truth_for_consistency_test.npy'
    c_ground_truth_test = np.load(path)
    path = ROOT + LOGFILES + RUN + '/' + 'a_ground_truth_for_consistency_test.npy'
    a_ground_truth_test = np.load(path)
    
#    #train
#    p3a_plot.three_level_consistency(c_softmax_outputs_train, a_softmax_outputs_train, t_softmax_outputs_train,
#                                     c_class_dict, a_class_dict, t_class_dict, ROOT, RUN, train_or_test='train', alternate_name='_hist2d_gnuplot_2')     
    #test                                 
    p3a_plot.three_level_consistency(c_softmax_outputs_test, a_softmax_outputs_test, t_softmax_outputs_test,
                                     c_class_dict, a_class_dict, t_class_dict, ROOT, RUN, train_or_test='test', alternate_name='_hist2d_gnuplot_2')
                                     
                                     
    # GROUND TRUTH                                 
                                      #train
#    p3a_plot.gt_consistency(c_softmax_outputs_train, a_softmax_outputs_train, t_softmax_outputs_train,
#                                     c_class_dict, a_class_dict, t_class_dict, ROOT, RUN, c_ground_truth_train, a_ground_truth_train,
#                                     train_or_test='train', alternate_name='_gnuplot')     
#    #test                                 
#    p3a_plot.gt_consistency(c_softmax_outputs_test, a_softmax_outputs_test, t_softmax_outputs_test,
#                                     c_class_dict, a_class_dict, t_class_dict, ROOT, RUN, c_ground_truth_test, a_ground_truth_test,
#                                     train_or_test='test', alternate_name='_gnuplot')
                                                                 

    