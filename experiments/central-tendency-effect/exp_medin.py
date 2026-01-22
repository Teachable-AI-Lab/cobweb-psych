from multinomial_cobweb import MultinomialCobwebTree
from concept_formation.visualize import visualize
from random import shuffle, seed
import time
import csv
import numpy as np
import pandas as pd
import sys
from scipy.stats import rankdata
from pathlib import Path

class MedinDataset(object):

	def __init__(self, random_seed=32, stimulus_type='geometric'):

		if stimulus_type == 'geometric':
			self.attribute_values = {
			'color': ['red', 'blue'],
			'form': ['triangle', 'circle'],
			'size': ['large', 'small'],
			'number': [1, 2],
			}
		elif stimulus_type == 'face':
			self.attribute_values = {
			'eye height': [2.5, 5],  # either 2.5 or 5 cm from the top of the face outline
			'eye separation': [1.5, 3.5],
			'nose length': [1.5, 3.0],
			'mouth height': [1.5, 3.0],  # either 1.5 or 3.0 cm from the chin line
			}
		self.stimulus_type = stimulus_type


		training_stimuli_ls = [[4, 1, 1, 1, 0, 'A'], [7, 1, 0, 1, 0, 'A'],
							   [15, 1, 0, 1, 1, 'A'], [13, 1, 1, 0, 1, 'A'],
							   [5, 0, 1, 1, 1, 'A'], [12, 1, 1, 0, 0, 'B'],
							   [2, 0, 1, 1, 0, 'B'], [14, 0, 0, 0, 1, 'B'],
							   [10, 0, 0, 0, 0, 'B']]
		transfer_stimuli_ls = [[1, 1, 0, 0, 1, 'A'], [3, 1, 0, 0, 0, 'B'],
							   [6, 1, 1, 1, 1, 'A'], [8, 0, 0, 1, 0, 'B'],
							   [9, 0, 1, 0, 1, 'A'], [11, 0, 0, 1, 1, 'A'],
							   [16, 0, 1, 0, 0, 'B']]
		test_stimuli_ls = training_stimuli_ls + transfer_stimuli_ls

		# All stimuli contain the No. of stimulus and its classification
		self.train_stimuli_complete = self.list2dict(training_stimuli_ls)
		self.test_stimuli_complete = self.list2dict(test_stimuli_ls)

		seed(random_seed)
		shuffle(self.train_stimuli_complete)
		shuffle(self.test_stimuli_complete)

		# Stimuli used in experiments
		self.train_stimuli = [{k: v for k, v in stimulus.items() if k != "stimulus"} for stimulus in self.train_stimuli_complete]
		self.test_stimuli = [{k: v for k, v in stimulus.items() if k not in ["stimulus", "classification"]} for stimulus in self.test_stimuli_complete]


	def list2dict(self, lss):
		stimuli = []
		if self.stimulus_type == 'geometric':
			for ls in lss:
				new_stimulus = {}
				new_stimulus['stimulus'] = {ls[0]: 1}
				new_stimulus['color'] = {self.attribute_values['color'][ls[1]]: 1}
				new_stimulus['form'] = {self.attribute_values['form'][ls[2]]: 1}
				new_stimulus['size'] = {self.attribute_values['size'][ls[3]]: 1}
				new_stimulus['number'] = {self.attribute_values['color'][ls[4]]: 1}
				new_stimulus['classification'] = {ls[-1]: 1}
				stimuli.append(new_stimulus)
		elif self.stimulus_type == 'face':
			for ls in lss:
				new_stimulus = {}
				new_stimulus['stimulus'] = {ls[0]: 1}
				new_stimulus['eye height'] = {self.attribute_values['eye height'][ls[1]]: 1}
				new_stimulus['eye separation'] = {self.attribute_values['eye separation'][ls[2]]: 1}
				new_stimulus['nose length'] = {self.attribute_values['nose length'][ls[3]]: 1}
				new_stimulus['mouth height'] = {self.attribute_values['mouth height'][ls[4]]: 1}
				new_stimulus['classification'] = {ls[-1]: 1}
				stimuli.append(new_stimulus)
		return stimuli

	

class Experiment(object):

	def __init__(self, random_seeds=[1, 32, 64, 128, 356], epochs=5, stimulus_type='geometric', rank_method='ordinal'):
		"""
		- random seeds: the random seeds used in an experiment. 
			The experiment will be conducted for every random seed with a given nbr of `epochs`
		- epochs: the nbr of times conducting the experiment for each random seed.

		So an experiment will be conducted len(random_seeds) * epochs times.

		- stimulus_type: ['geometric' or 'face'].
			stimulus can be of geometric form (Experiment 2) or schematic faces (Experiment 3).

		- rank_method: ['ordinal', 'min', 'max', 'average', 'dense']
			the parameter for scipy.stats.rankdata().
		"""

		if stimulus_type == 'geometric':
			self.observed_probs = [0.59, 0.84, 0.69, 0.78, 0.81,
							   	   0.94, 0.88, 0.66, 0.5, 0.97, 0.62,
							   	   0.84, 0.88, 0.88, 0.81, 0.84]
		elif stimulus_type == 'face':
			self.observed_probs = [0.72, 0.72, 0.44, 0.97, 0.72,
								   0.98, 0.97, 0.77, 0.27, 0.95, 0.39,
								   0.67, 0.81, 0.97, 0.92, 0.91]
		self.stimulus_type = stimulus_type

		self.rank_method = rank_method
		self.observed_ranking = rankdata(self.observed_probs, method=rank_method)

		self.random_seeds = random_seeds
		self.epochs = epochs

		self.df = self.batch_experiments()

	def iteration(self, random_seed, reaction_time=True):
		"""
		A single train-and-test process.
		"""
		tree = MultinomialCobwebTree()

		# prepare for dataset
		dataset = MedinDataset(random_seed)
		train_stimuli = dataset.train_stimuli
		test_stimuli = dataset.test_stimuli
		# train_stimuli_complete = dataset.train_stimuli_complete
		test_stimuli_complete = dataset.test_stimuli_complete

		# Train:
		tree.fit(train_stimuli)

		# Test:
		predicted_probs_leaf = [0] * len(test_stimuli)
		predicted_probs_basic = [0] * len(test_stimuli)
		predicted_probs_best = [0] * len(test_stimuli)
		if reaction_time:
			process_times = [0] * len(test_stimuli)

		results = []

		for i in range(len(test_stimuli)):

			if reaction_time:
				start = time.time()
			node = tree.categorize(test_stimuli[i])
			prediction_leaf = node.predict()
			prediction_basic = node.predict_basic()
			prediction_best = node.predict_best(test_stimuli[i])
			if reaction_time:
				end = time.time()

			ground_truth = list(test_stimuli_complete[i]['classification'].keys())[0]
			stimulus = list(test_stimuli_complete[i]['stimulus'].keys())[0]

			predicted_probs_leaf[stimulus - 1] = prediction_leaf['classification'][ground_truth]
			predicted_probs_basic[stimulus - 1] = prediction_basic['classification'][ground_truth]
			predicted_probs_best[stimulus - 1] = prediction_best['classification'][ground_truth]
			if reaction_time:
				process_times[stimulus - 1] = (end - start) * 100000

		return predicted_probs_leaf, predicted_probs_basic, predicted_probs_best, process_times


	def correlation2observed(self, pred_probs, compared='prob'):
		if compared == 'prob':
			correlation = np.corrcoef(self.observed_probs, pred_probs)
		elif compared == 'rank':
			correlation = np.corrcoef(self.observed_ranking, rankdata(pred_probs, method=self.rank_method))
		return correlation[0, 1]


	def experiment_dataframe(self, random_seed, n_iteration, reaction_time=True):
		stimulus_index = list(range(1, len(self.observed_probs) + 1))
		predicted_probs_leaf, predicted_probs_basic, predicted_probs_best, process_times = self.iteration(random_seed, reaction_time=True)
		dataframe = {
		"stimulus": stimulus_index,
		"seed": [random_seed] * len(stimulus_index),
		"iteration": [n_iteration] * len(stimulus_index),
		"observed_probs": self.observed_probs,
		"observed_ranking": self.observed_ranking,
		"predicted_probs_leaf": predicted_probs_leaf,
		"predicted_probs_basic": predicted_probs_basic,
		"predicted_probs_best": predicted_probs_best,
		"reaction_time": process_times,
		"correlation_probs_leaf": [self.correlation2observed(predicted_probs_leaf)] * len(stimulus_index),
		"correlation_probs_basic": [self.correlation2observed(predicted_probs_basic)] * len(stimulus_index),
		"correlation_probs_best": [self.correlation2observed(predicted_probs_best)] * len(stimulus_index),
		"correlation_rank_leaf": [self.correlation2observed(predicted_probs_leaf, compared='rank')] * len(stimulus_index),
		"correlation_rank_basic": [self.correlation2observed(predicted_probs_basic, compared='rank')] * len(stimulus_index),
		"correlation_rank_best": [self.correlation2observed(predicted_probs_best, compared='rank')] * len(stimulus_index),
		}
		return pd.DataFrame(dataframe)


	def batch_experiments(self):
		for i in range(len(self.random_seeds)):
			for j in range(self.epochs):
				df_exp = self.experiment_dataframe(self.random_seeds[i], j + 1)
				if i == 0 and j == 0:
					df = df_exp
				else:
					# print(df['correlation_probs_leaf'].isnull().all())
					# if df['correlation_probs_leaf'].isnull().all():
					# 	print("Random seed {} is invalid.".format(self.random_seeds[i]))
					# 	break
					df = pd.concat([df, df_exp], ignore_index=True)
		return df


	def df2csv(self):
		results_dir = Path(__file__).resolve().parent / "results" / self.stimulus_type
		results_dir.mkdir(parents=True, exist_ok=True)
		fname = f"exp_medin_type-{self.stimulus_type}_nseeds{int(len(self.random_seeds))}_epoch{self.epochs}.csv"
		self.df.to_csv(str(results_dir / fname), index=False)


	def avg_correlations(self):
		df_filtered = self.df[self.df['correlation_probs_leaf'].notnull()]
		print(sum(self.df['correlation_probs_leaf'].isnull()))
		output = {
		"correlation_probs_leaf": df_filtered['correlation_probs_leaf'].mean(),
		"correlation_probs_basic": df_filtered['correlation_probs_basic'].mean(),
		"correlation_probs_best": df_filtered['correlation_probs_best'].mean(),
		"correlation_rank_leaf": df_filtered['correlation_rank_leaf'].mean(),
		"correlation_rank_basic": df_filtered['correlation_rank_basic'].mean(),
		"correlation_rank_best": df_filtered['correlation_rank_best'].mean()
		}
		return output


	def quantile_correlations(self):
		df_filtered = self.df[self.df['correlation_probs_leaf'].notnull()]
		print(sum(self.df['correlation_probs_leaf'].isnull()))
		output = {
		"correlation_probs_leaf": df_filtered['correlation_probs_leaf'].quantile(0.75),
		"correlation_probs_basic": df_filtered['correlation_probs_basic'].quantile(0.75),
		"correlation_probs_best": df_filtered['correlation_probs_best'].quantile(0.75),
		"correlation_rank_leaf": df_filtered['correlation_rank_leaf'].quantile(0.75),
		"correlation_rank_basic": df_filtered['correlation_rank_basic'].quantile(0.75),
		"correlation_rank_best": df_filtered['correlation_rank_best'].quantile(0.75)
		}
		return output


	def compare4n7(self):
		"""
		In the original paper:

		Under prototype theory, stimulus 4 (1110) should be at least as close as 7 (1010) is to the prototype (1111).
		More generally, all independent cue models will predict that Stimulus 4 will be easier to learn than 7,
		because for the only dimension where the two stimuli differ, 4 will have a positive weight and 7 a negative weight.

		In contrast, in their context model, stimulus 7 should be easier to learn than 4,
		because the effect of number of highly similar patterns is the most important factor in performance.
		S7 is highly similar to two other Category A patterns (4 and 15), but is not highly similar to any Category B patterns.
		S4 is highly similar to only one Category A pattern (7), and to two Category B patterns (2 and 12), 
		and hence should be more difficult to learn.
		"""

		df_4 = self.df[self.df['stimulus'] == 4]
		df_7 = self.df[self.df['stimulus'] == 7]
		output_4 = {
		"stimulus": 4,
		"predicted_probs_leaf": df_4['predicted_probs_leaf'].mean(),
		"predicted_probs_basic": df_4['predicted_probs_basic'].mean(),
		"predicted_probs_best": df_4['predicted_probs_best'].mean()
		}
		output_7 = {
		"stimulus": 7,
		"predicted_probs_leaf": df_7['predicted_probs_leaf'].mean(),
		"predicted_probs_basic": df_7['predicted_probs_basic'].mean(),
		"predicted_probs_best": df_7['predicted_probs_best'].mean()
		}
		return output_4, output_7



# ===============================================

seeds = [int(i) for i in np.arange(0, 2000, 50)]
# print(seeds)
# seeds = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]

# You should set either 'stimulus_type'='geometric' or 'face'!
experiments = Experiment(random_seeds=seeds, epochs=5, rank_method='ordinal', stimulus_type='face')
experiments.df2csv()
print(experiments.avg_correlations())
print(experiments.quantile_correlations())
output_4, output_7 = experiments.compare4n7()
print(output_4)
print(output_7)

