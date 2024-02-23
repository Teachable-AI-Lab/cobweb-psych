from multinomial_cobweb import MultinomialCobwebTree
from multinomial_cobweb.visualize import visualize
from random import shuffle, seed, sample
import time
import csv
import numpy as np
import pandas as pd
from copy import copy, deepcopy

# Stimuli encodings:
# [stimulus_id, letter1, letter2, letter3, letter4, letter5, letter6, category]
stimuli_A = [
[1, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0], [3, 0, 1, 0, 0, 0, 0, 0],
[4, 0, 0, 1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 1, 0, 0], [6, 0, 0, 0, 0, 0, 1, 0],
[7, 1, 1, 1, 1, 0, 1, 0]]
stimuli_B = [
[8, 1, 1, 1, 1, 1, 1, 1], [9, 0, 1, 1, 1, 1, 1, 1], [10, 1, 0, 1, 1, 1, 1, 1],
[11, 1, 1, 0, 1, 1, 1, 1], [12, 1, 1, 1, 0, 1, 1, 1], [13, 1, 1, 1, 1, 1, 0, 1],
[14, 0, 0, 0, 1, 0, 0, 1]]
stimuli = stimuli_A + stimuli_B

attribute_values = {
	'letter1': ['g', 'w'],
	'letter2': ['a', 'y'],
	'letter3': ['f', 's'],
	'letter4': ['u', 'e'],
	'letter5': ['z', 'r'],
	'letter6': ['i', 'o'],
	'category': ['A', 'B'],
}


def encoding2stimuli(encoding):
	return {'stimulus': {str(encoding[0]): 1},
			'letter1': {attribute_values['letter1'][encoding[1]]: 1},
			'letter2': {attribute_values['letter2'][encoding[2]]: 1},
			'letter3': {attribute_values['letter3'][encoding[3]]: 1},
			'letter4': {attribute_values['letter4'][encoding[4]]: 1},
			'letter5': {attribute_values['letter5'][encoding[5]]: 1},
			'letter6': {attribute_values['letter6'][encoding[6]]: 1},
			'category': {attribute_values['category'][encoding[7]]: 1}
			}


def make_prediction(prob_dict):
	options = list(prob_dict.keys())
	# print(options)
	if len(options) < 2:
		return options[0]
	else:
		return 'A' if prob_dict['A'] >= prob_dict['B'] else 'B'


def predicted_probability(prob_dict, category):
	options = list(prob_dict.keys())
	if len(options) < 2:
		if options[0] == category:
			return 1.
		else:
			return 0.
	else:
		return prob_dict[category]



def stimulus_prediction_dataframe(stimulus_te_index, pred_dict_leaf, pred_dict_basic, pred_dict_best,
	random_seed, block, epoch):
	stimulus_te = stimuli_te[stimulus_te_index]
	row =  {
		"stimulus": str(stimulus_te_index + 1),
		# "letter1": stimulus_te['letter1'],
		# "letter2": stimulus_te['letter2'],
		# "letter3": stimulus_te['letter3'],
		# "letter4": stimulus_te['letter4'],
		# "letter5": stimulus_te['letter5'],
		# "letter6": stimulus_te['letter6'],
		# "category": list(stimuli_tr[stimulus_te_index]['category'].keys())[0],
		"category": 'A' if (stimulus_te_index + 1 <= 7) else 'B',
		"seed": random_seed,
		"block": block,
		"epoch": epoch,
		"leaf_A": predicted_probability(pred_dict_leaf, 'A'),
		"leaf_B": predicted_probability(pred_dict_leaf, 'B'),
		"basic_A": predicted_probability(pred_dict_basic, 'A'),
		"basic_B": predicted_probability(pred_dict_basic, 'B'),
		"best_A": predicted_probability(pred_dict_best, 'A'),
		"best_B": predicted_probability(pred_dict_best, 'B'),
	}
	return pd.DataFrame([row])


random_seeds = [1, 32, 64, 128, 356]
blocks = 10
epochs = 5
stimuli = [encoding2stimuli(ls) for ls in stimuli]
stimuli_tr = [{k: v for k, v in stimulus.items() if k != 'stimulus'} for stimulus in stimuli]
stimuli_te = [{k: v for k, v in stimulus.items() if k != 'category'} for stimulus in stimuli_tr]

# list of dataframes (rows):
dfs = []

for random_seed in random_seeds:

	# Preprocess the stimuli data
	seed(random_seed)
	stimuli_tr_shuffled = sample(stimuli_tr, len(stimuli_tr))

	# Each random seed executes multiple iterations.
	for epoch in range(1, epochs + 1):

		model = MultinomialCobwebTree()

		for block in range(1, blocks + 1):

			# stimuli_tr_shuffled = sample(stimuli_tr_shuffled, len(stimuli_tr_shuffled))
			shuffle(stimuli_tr_shuffled)

			# Train the model:
			model.fit(stimuli_tr_shuffled)

			# Predict:
			for i in range(len(stimuli_te)):
				pred_dict_leaf = model.categorize(stimuli_te[i]).predict()['category']
				pred_dict_basic = model.categorize(stimuli_te[i]).predict_basic()['category']
				pred_dict_best = model.categorize(stimuli_te[i]).predict_best(stimuli_te[i])['category']
				df_stimulus_te = stimulus_prediction_dataframe(i, pred_dict_leaf, pred_dict_basic, pred_dict_best,
					random_seed, block, epoch)
				dfs.append(df_stimulus_te)

visualize(model)
df = pd.concat(dfs, ignore_index=True)
df.to_csv(f"exp_smith-minda_blocks{int(blocks)}_nseeds{int(len(random_seeds))}_epoch{epochs}.csv", index=False)




