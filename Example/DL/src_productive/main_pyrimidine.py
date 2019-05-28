#!/usr/bin/env python

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
import pickle
import sys
from resnext_model import resnext_model
from convnet import *
import pdb
from multiprocessing import Pool
import os
import tqdm
import argparse

# We do not want the program occupy two gpus while only use one
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
load_model = True
train_steps = 0


# # This is the model for 5 class prediction
# model_name = '../model/conv_model.ckpt'
# # This is for the 5 class classification
# class2label = {'NONSITE':0, 'RADE': 1, 'RGUA': 1,
# 	'RURA':2, 'RCYT':2, 'RPHO':3, 'RRIB': 4}

# This is the model for the pyrimidine prediction
model_name = '../model/grid_pyrimidine.ckpt'
# This is for the pyrimidine classification
class2label = {'NONSITE':-1, 'RADE': -1, 'RGUA': -1,
	'RURA':0, 'RCYT':1, 'RPHO':-1, 'RRIB': -1}
label2class = {0: 'RURA', 1: 'RCYT'}

class2label_full = {'NONSITE':0, 'RADE': 1, 'RGUA': 2,
	'RURA':3, 'RCYT':4, 'RPHO':5, 'RRIB': 6}

# # This is the model for the purine prediction
# model_name = '../model/grid_purine.ckpt'
# # This is for the pyrimidine classification
# class2label = {'NONSITE':-1, 'RADE': 0, 'RGUA': 1,
# 	'RURA':-1, 'RCYT':-1, 'RPHO':-1, 'RRIB': -1}

def import_data(file_path, label):
	df = pd.read_pickle(file_path)
	df['label'] = pd.Series(np.ones(df.shape[0])*label)
	return df


def svm_classifier(feature_train, label_train, feature_test):
	from sklearn import svm
	clf = svm.SVC()
	clf.fit(feature_train, label_train)
	result = clf.predict(feature_test)
	# pdb.set_trace()
	return result

def load_pickle(file):
	with open(file, 'r') as f:
		l = pd.read_pickle(f)
	return l

def load_feature_label(path, file_name):
	feature_file = path+file_name+'.reducedpkl2'
	lable_file = path+file_name+'.reducedlabel2'
	# feature_file = path+file_name+'.pkl'
	# lable_file = path+file_name+'.label'
	feature = load_pickle(feature_file)
	label = load_pickle(lable_file)
	return feature, label
# [1, 799] are features, 799 is the label
def load_single_rna_grid(file_name, path='../dataset/GridCentroidMidline/'):
	df, label = load_feature_label(path, file_name)
	label = np.array(map(lambda x: class2label[x], label))
	df['label']= pd.Series(label)
	df = df[df.label != -1]
	# print('Finish load {}'.format(file_name))
	return df

def load_single_rna_grid_full(file_name, path='../dataset/GridCentroidMidline/'):
	df, label = load_feature_label(path, file_name)
	label = np.array(map(lambda x: class2label_full[x], label))
	df['label']= pd.Series(label)
	# print('Finish load {}'.format(file_name))
	return df
	
def load_all_rna_grid(name_list):
	p = Pool()
	df_list = list(tqdm.tqdm(p.imap(load_single_rna_grid, name_list[:700]),
		total=len(name_list[:700])))
	# df_list = list(tqdm.tqdm(p.imap(load_single_rna_grid, name_list[700:]),
	# 	total=len(name_list[700:])))
	return df_list

def drop_most_zeros(df):
	df = df.reset_index(drop=True)
	zeros_index = df['label']==0
	zeros_index = np.where(zeros_index.values)[0]
	# pdb.set_trace()
	drop_index = np.random.choice(zeros_index,
		size=int(float(len(zeros_index))*0.97), replace=False)
	df_out = df.drop(df.index[drop_index])
	return df_out

def extract_annotation(df, test_index, pred_label, true_label, p_name):
	test_df = df.iloc[test_index]
	test_df = test_df.reset_index(drop=True)
	ann_all = test_df['Annotation'].values
	p_dict =dict()
	for i in range(len(ann_all)):
		if p_name in ann_all[i]:
			p_dict[ann_all[i]] = (pred_label[i], true_label[i])
	return p_dict

def save_pickle(name, data):
	with open('{}.pickle'.format(name),'wb') as f:
		pickle.dump(data, f)

def save_all(name_list, df, test_index, pred_label, true_label):
	for i in range(len(name_list)):
		d = extract_annotation(df, test_index, pred_label, 
			true_label,name_list[i])
		save_pickle(name_list[i], d)
		print('{} finished'.format(name_list[i]))

def svm_classifier(feature_train, label_train, feature_test):
	from sklearn.ensemble import BaggingClassifier
	from sklearn.svm import SVC
	n_estimators = 20
	clf = BaggingClassifier(SVC(verbose=1, cache_size=100000),
		n_estimators=n_estimators,
		max_samples=1.0/n_estimators)
	clf.fit(feature_train, label_train)
	result = clf.predict(feature_test)
	# pdb.set_trace()
	return result

if __name__ == '__main__':
	# name_list = load_pickle('../dataset/grid_list.pickle')

	# df_list = load_all_rna_grid(name_list)
	# df = pd.concat(df_list)
	# del df_list
	parse = argparse.ArgumentParser(description='Arguments')
	parse.add_argument('-i', action='store', dest='input', help='the input file')
	parse.add_argument('-o', action='store', dest='output', help='output prefix')
	arg = parse.parse_args()

	# df = pd.read_pickle('../blind_test/20180404_output/0000_result.pickle')
	df = pd.read_pickle(arg.input)

	coarse_label = df['coarse_label'].values

	# be careful about the index
	grid_feature = df.iloc[:, 1:-1].values

	# find the place that need to be refined
	tbd_index = np.where(coarse_label=='pyrimidine')[0]
	tbd_index_list = list(tbd_index)


	prob_test, test_pred_label, deep_feature = deep_learning_model(2,
		grid_feature[tbd_index],model_name)
	test_pred_label = np.array(test_pred_label)

	for i,ind in enumerate(tbd_index_list):
		coarse_label[ind] = label2class[test_pred_label[i]]
	final_label_encoding = map(lambda x: class2label_full[x], coarse_label)

	with open(arg.output+'_pyrimidine_score.pickle','wb') as f:
		pickle.dump(prob_test, f)

	df.iloc[:,[0,-1]].to_pickle(arg.output+'_result.pickle')

