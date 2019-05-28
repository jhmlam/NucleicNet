
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.conv import global_max_pool
import numpy as np
import pdb

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Building Residual Network
def resnext_model(X, Y, testX, testY, n_epoch):
	n_input = len(X[0])
	n_classes = len(Y[0])
	X = np.reshape(X, (-1, 1, n_input, 1))
	testX = np.reshape(testX, (-1, 1, n_input, 1))
	net = tflearn.input_data(shape=[None, 1, n_input, 1])
	net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001, activation='selu')
	net = tflearn.resnext_block(net, n, 16, 32, activation='selu')
	net = tflearn.resnext_block(net, 1, 32, 32, downsample=True, activation='selu')
	net = tflearn.resnext_block(net, n-1, 32, 32, activation='selu')
	net = tflearn.resnext_block(net, 1, 64, 32, downsample=True, activation='selu')
	net = tflearn.resnext_block(net, n-1, 64, 32, activation='selu')
	net = tflearn.batch_normalization(net)
	net = tflearn.activation(net, 'selu')
	# net_p = tflearn.global_avg_pool(net, name='net_p')
	net_p = global_max_pool(net, name='net_p')
	# Regression
	net = tflearn.fully_connected(net_p, 2048, activation='selu')
	net = tflearn.dropout(net, 0.5)
	net = tflearn.fully_connected(net, 256, activation='selu')
	net = tflearn.dropout(net, 0.7)
	net = tflearn.fully_connected(net, n_classes, activation='softmax')
	# opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
	net = tflearn.regression(net, optimizer='adam',
	                         loss='categorical_crossentropy')
	# Training
	model = tflearn.DNN(net, checkpoint_path='../model/model_resnext_grid_adam',
	                    max_checkpoints=10, tensorboard_verbose=0,
	                    clip_gradients=0.)
	# model.load('../model/model_resnext_grid_adam-70000')
	model.fit(X, Y, n_epoch=n_epoch, validation_set=(testX[:], testY),
	          snapshot_epoch=False, snapshot_step=50000,
	          show_metric=True, batch_size=128, shuffle=True,
	          run_id='resnext')
	prob = model.predict(testX)
	# pdb.set_trace()

	# get the hidden layer value after global average pooling
	m2 = tflearn.DNN(net_p, session = model.session)

	feature_train = list()
	for i in range(4):
		feature_train_temp = m2.predict(X[i*4000:(i+1)*4000])
		feature_train += list(feature_train_temp)
	feature_train_temp = m2.predict(X[16000:])
	feature_train =  np.array(feature_train+list(feature_train_temp))

	feature_test = m2.predict(testX)
	
	# pdb.set_trace()
	return prob, feature_train, feature_test
