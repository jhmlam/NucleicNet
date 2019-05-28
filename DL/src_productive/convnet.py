import numpy as np
import pandas as pd
from numpy.random import randint
import tflearn
import tensorflow as tf
from evaluate_model import *
from tf_model_component import *
from sklearn.model_selection import train_test_split
import math
#import cPickle
import sys
import os
import gc

log_file_name = '5_classes_classification'
summary_dir = '../log'

def generate_batch(feature, label, batch_size):
    batch_index = randint(0,len(feature),batch_size)
    feature_batch = batch_process(feature, batch_index)
    label_batch = batch_process(label, batch_index)
    return (feature_batch, label_batch)

def deep_learning_model(n_classes, test_feature,
	model_name, load_model = True):
	#Parameters
	batch_size = 1024
	check_step = 10000


	#Network parameter
	n_input = len(test_feature[0])

	#Define the fully connected layers
	n_fc = 2
	n_nodes = [2048, 1024]
	# n_nodes = [256, 64]

	#tf graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	kr = tf.placeholder(tf.float32)


	#Define convolutional layers weight and bias
	#The input would be a vector. Rember to reshape the input
	# to [-1, 1, length, 1]
	conv_weights = {
		'wc1': weight_variable([1, 5, 1, 16]),
		'wc2': weight_variable([1, 5, 16, 32]),
		'wc3': weight_variable([1, 5, 32, 64]),
		'wc4': weight_variable([1, 5, 64, 128]),
		'w_out': weight_variable([n_nodes[-1], n_classes])
	}

	conv_bias = {
		'b1': bias_variable([16]),
		'b2': bias_variable([32]),
		'b3': bias_variable([64]),
		'b4': bias_variable([128]),
		'b_out': bias_variable([n_classes])
		
	}

	# Model graph
	x_reshape = tf.reshape(x, shape=[-1, 1, n_input, 1])
	x_reshape = tflearn.batch_normalization(x_reshape)

	conv1 = selu(tflearn.batch_normalization(
		conv2d(x_reshape, conv_weights['wc1']) + conv_bias['b1']))
	conv1 = max_pool2d(conv1, 1, 3)

	conv2 = selu(tflearn.batch_normalization(
		conv2d(conv1, conv_weights['wc2']) + conv_bias['b2']))
	conv2 = max_pool2d(conv2, 1, 3)

	conv3 = selu(tflearn.batch_normalization(
		conv2d(conv2, conv_weights['wc3']) + conv_bias['b3']))
	conv3 = max_pool2d(conv3, 1, 3)

	conv4 = selu(tflearn.batch_normalization(
		conv2d(conv3, conv_weights['wc4']) + conv_bias['b4']))
	conv4 = max_pool2d(conv4, 1, 3)

	fc_input = tf.reshape(conv4, [-1,np.prod(conv4.get_shape().as_list()[1:])])
	fc_output = fully_connected(n_fc, n_nodes, fc_input)
	fc_output = tf.nn.dropout(fc_output, kr)

	y_logit = tf.matmul(fc_output, conv_weights['w_out']) + conv_bias['b_out']
	y = tf.nn.softmax(y_logit)

	# Evaluate model
	predicted_label = tf.argmax(y, 1)

	# Start the session
	config = tf.ConfigProto()
	config.log_device_placement=False
	config.allow_soft_placement=True
	config.gpu_options.allow_growth=True
	sess = tf.InteractiveSession(config=config)
	# Initializing the variables
	sess.run(tf.global_variables_initializer())

	# Load and save model
	saver = tf.train.Saver()
	if load_model==True:
	    saver.restore(sess,model_name)
		
	def whole_set_check():
	    pred_test_feature = []
	    predict_test_label=[]
	    prob_test = []
	    number_of_full_batch=int(math.floor(len(test_feature)/batch_size))
	    for i in range(number_of_full_batch):
	        prob_out, predicted_label_out,feature_out = sess.run(
	        	[y, predicted_label, fc_output],
	            feed_dict={x: test_feature[i*batch_size:(i+1)*batch_size],
	            kr: 1})
	        prob_test += list(prob_out)
	        predict_test_label+=list(predicted_label_out)
	        pred_test_feature+=list(feature_out)
	    
	    prob_out, predicted_label_out,feature_out = sess.run([
	    	y, predicted_label,fc_output],
	        feed_dict={x: test_feature[number_of_full_batch*batch_size:], 
	        kr: 1})
	    

	    prob_test += list(prob_out)
	    predict_test_label+=list(predicted_label_out)
	    pred_test_feature+=list(feature_out)

	    return (prob_test,predict_test_label, pred_test_feature)	


	prob_test, test_pred_label, pred_test_feature =whole_set_check()
	#put back the log file redirection
	return prob_test, test_pred_label, np.array(pred_test_feature)
