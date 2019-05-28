import argparse
import pandas as pd
import numpy as np
import pickle


if __name__ == '__main__':
	parse = argparse.ArgumentParser(description='Arguments')
	parse.add_argument('-i', action='store', dest='input', help='the input file')
	parse.add_argument('-o', action='store', dest='output', help='output prefix')
	arg = parse.parse_args()

	df = pd.read_pickle(arg.input)
	with open(arg.output+'_pyrimidine_score.pickle','rb') as f:
		score_pyrimidine = pickle.load(f)
	with open(arg.output+'_purine_score.pickle','rb') as f:
		score_purine = pickle.load(f)
	with open(arg.output+'_5_class_score.pickle','rb') as f:
		score_5 = pickle.load(f)	

	coarse_label = df['coarse_label'].values

	final_result = list()


	# X, A, U, P, R, G, C
	for i in range(len(coarse_label)):
		temp = score_5[i]
		temp[1] = temp[1]*0.5
		temp[2] = temp[2]*0.5
		temp = np.append(temp, temp[1])
		temp = np.append(temp, temp[2])
		final_result.append(temp)

	# deal with purine
	purine_index = np.where((coarse_label=='RADE') | (coarse_label=='RGUA'))[0]
	purine_index_list = list(purine_index)
	for i,ind in enumerate(purine_index_list):
		final_result[ind][1] = final_result[ind][1]*2*score_purine[i][0]
		final_result[ind][5] = final_result[ind][5]*2*score_purine[i][1]

	# deal with pyrimidine
	pyrimidine_index = np.where((coarse_label=='RURA') | (coarse_label=='RCYT'))[0]
	pyrimidine_index_list = list(pyrimidine_index)
	for i,ind in enumerate(pyrimidine_index_list):
		final_result[ind][2] = final_result[ind][2]*2*score_pyrimidine[i][0]
		final_result[ind][6] = final_result[ind][6]*2*score_pyrimidine[i][1]

	# Add the result into the dataframe
	final_result = np.array(final_result)
	df['NONSITE'] = pd.Series(final_result[:,0], index=df.index)
	df['P'] = pd.Series(final_result[:,3], index=df.index)
	df['R'] = pd.Series(final_result[:,4], index=df.index)
	df['A'] = pd.Series(final_result[:,1], index=df.index)
	df['U'] = pd.Series(final_result[:,2], index=df.index)
	df['C'] = pd.Series(final_result[:,6], index=df.index)
	df['G'] = pd.Series(final_result[:,5], index=df.index)

	with open(arg.output+'_result_reformat.pickle','wb') as f:
		pickle.dump(df, f)
