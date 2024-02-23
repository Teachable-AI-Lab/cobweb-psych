from multinomial_cobweb import MultinomialCobwebTree
from multinomial_cobweb.visualize import visualize
from random import shuffle, seed, sample
import time
import csv
import numpy as np
import pandas as pd
from copy import copy, deepcopy


# The binary encodings of all stimuli in a task: [stimulus_index, dim1, dim2, dim3]
stimuli_encoding = [
[1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 1, 0], [4, 0, 1, 1],
[5, 1, 0, 0], [6, 1, 0, 1], [7, 1, 1, 0], [8, 1, 1, 1]
]

attribute_values = {
	'size': ['small', 'large'],
	'color': ['white', 'black'],
	'form': ['square', 'triangle'],
}

def encoding2stimuli(encoding, classification):
	return {'stimulus': {str(encoding[0]): 1}, 
			'size': {attribute_values['size'][encoding[1]]: 1},
			'color': {attribute_values['color'][encoding[2]]: 1},
			'form': {attribute_values['form'][encoding[3]]: 1},
			'classification': {classification: 1}}


def make_prediction(prob_dict):
	options = list(prob_dict.keys())
	# print(options)
	if len(options) < 2:
		return options[0]
	else:
		return 'A' if prob_dict['A'] >= prob_dict['B'] else 'B'


class Task(object):
	"""
	A single task (1, 2, ..., 6) in a single implementation (given a specific random seed).
	"""

	def __init__(self, task_type, task_index, stimuli_indices_A, stimuli_indices_B, random_seed=32):
		self.type = task_type
		self.task_index = task_index
		self.stimuli_indices_A = stimuli_indices_A
		self.stimuli_indices_B = stimuli_indices_B
		self.random_seed = random_seed

		stimuli_encodings_A = [stimuli_encoding[i-1] for i in stimuli_indices_A]
		stimuli_encodings_B = [stimuli_encoding[i-1] for i in stimuli_indices_B]
		self.stimuli_A = [encoding2stimuli(encoding, 'A') for encoding in stimuli_encodings_A]
		self.stimuli_B = [encoding2stimuli(encoding, 'B') for encoding in stimuli_encodings_B]

		seed(random_seed)
		self.stimuli = self.stimuli_A + self.stimuli_B
		shuffle(self.stimuli)
		self.stimuli_trained = [{k: v for k, v in stimulus.items() if k != 'stimulus'} for stimulus in self.stimuli]
		self.stimuli_tested = [{k: v for k, v in stimulus.items() if k not in ['classification', 'stimulus']} for stimulus in self.stimuli]
		self.model = MultinomialCobwebTree()


	def incremental_tr_te(self):
		"""
		Performing the incremental train-test task.
		Test the model after a new instance is trained.
		"""
		nbr_correct = {'leaf': [], 'basic': [], 'best': []}

		model = MultinomialCobwebTree()
		# test:
		for stimulus_cat in self.stimuli_tested:
			model.ifit(stimulus_cat)

		for i in range(len(self.stimuli_trained)):
			# model = MultinomialCobwebTree()

			# Train:
			# for stimulus_tr in self.stimuli_trained[:i+1]:
			# 	model.ifit(stimulus_tr)
			model.ifit(self.stimuli_trained[i])

			# Predict and check performance:
			correct_leaf = 0
			correct_basic = 0
			correct_best = 0
			for j in range(len(self.stimuli_tested)):
				stimulus_cat = self.stimuli_tested[j]
				prediction_leaf = make_prediction(model.categorize(stimulus_cat).predict()['classification'])
				prediction_basic = make_prediction(model.categorize(stimulus_cat).predict_basic()['classification'])
				prediction_best = make_prediction(model.categorize(stimulus_cat).predict_best(stimulus_cat)['classification'])
				ground_truth = list(self.stimuli_trained[j]['classification'].keys())[0]
				# print(prediction_leaf, prediction_basic, prediction_best, ground_truth)
				if prediction_leaf == ground_truth:
					correct_leaf += 1
				if prediction_basic == ground_truth:
					correct_basic += 1
				if prediction_best == ground_truth:
					correct_best += 1

				# print(correct_leaf, correct_basic, correct_best)

			nbr_correct['leaf'].append(correct_leaf)
			nbr_correct['basic'].append(correct_basic)
			nbr_correct['best'].append(correct_best)

			# print(correct_leaf, correct_basic, correct_best)

		self.nbr_correct_incremental = nbr_correct


	def test_model(self, trained=True):
		"""
		Test the model after training all stimuli.
		"""
		if not trained:
			self.train_model()

		correct_leaf = 0
		correct_basic = 0
		correct_best = 0

		for i in range(len(self.stimuli_tested)):
			stimulus_cat = self.stimuli_tested[i]
			prediction_leaf = make_prediction(self.model.categorize(stimulus_cat).predict()['classification'])
			prediction_basic = make_prediction(self.model.categorize(stimulus_cat).predict_basic()['classification'])
			prediction_best = make_prediction(self.model.categorize(stimulus_cat).predict_best(stimulus_cat)['classification'])
			ground_truth = list(self.stimuli_trained[i]['classification'].keys())[0]
			if prediction_leaf == ground_truth:
				correct_leaf += 1
			if prediction_basic == ground_truth:
				correct_basic += 1
			if prediction_best == ground_truth:
				correct_best += 1

		self.nbr_correct = {'leaf': correct_leaf, 'basic': correct_basic, 'best': correct_best}


	def train_model(self, shuffle=False):
		"""
		train the model stored in the class with all training stimuli.
		"""
		if shuffle:
			stimuli_trained = self.stimuli_tested.copy()
			shuffle(stimuli_trained)
			for stimulus in self.stimuli_trained:
				self.model.ifit(stimulus)
		else:
			for stimulus in self.stimuli_trained:
				self.model.ifit(stimulus)


	def tr_te_multple_blocks(self, n_blocks):
		"""
		First train all stimuli, then test all stimuli.
		Do with multiple iterations and see how the performance evolve.
		"""
		nbr_correct = {'leaf': [], 'basic': [], 'best': []}
		model = MultinomialCobwebTree()
		stimuli_tr_shuffled = self.stimuli_trained.copy()

		for block in range(n_blocks):
			# Train:
			shuffle(stimuli_tr_shuffled)
			# for i in range(len(self.stimuli_trained)):
			# 	model.ifit(self.stimuli_trained[i])
			model.fit(stimuli_tr_shuffled)
			# Predict:
			correct_leaf = 0
			correct_basic = 0
			correct_best = 0
			for j in range(len(self.stimuli_tested)):
				stimulus_cat = self.stimuli_tested[j]
				prediction_leaf = make_prediction(model.categorize(stimulus_cat).predict()['classification'])
				prediction_basic = make_prediction(model.categorize(stimulus_cat).predict_basic()['classification'])
				prediction_best = make_prediction(model.categorize(stimulus_cat).predict_best(stimulus_cat)['classification'])
				ground_truth = list(self.stimuli_trained[j]['classification'].keys())[0]
				# print(prediction_leaf, prediction_basic, prediction_best, ground_truth)
				if prediction_leaf == ground_truth:
					correct_leaf += 1
				if prediction_basic == ground_truth:
					correct_basic += 1
				if prediction_best == ground_truth:
					correct_best += 1

			# Update the nbr of corrects for each approach for this block.
			nbr_correct['leaf'].append(correct_leaf)
			nbr_correct['basic'].append(correct_basic)
			nbr_correct['best'].append(correct_best)
			# print(model.root.count)

		self.nbr_correct_multiple = nbr_correct
		# visualize(model)



	def compute_accuracy(self, execute_type="one"):
		"""
		Compute the accuracy for either an incremental case or the usual case (test after training all)
		"""
		if execute_type == "incremental":
			accuracy = {'leaf': [], 'basic': [], 'best': []}
			for i in range(len(self.nbr_correct_incremental['leaf'])):
				accuracy['leaf'].append(self.nbr_correct_incremental['leaf'][i] / len(self.stimuli_tested))
				accuracy['basic'].append(self.nbr_correct_incremental['basic'][i] / len(self.stimuli_tested))
				accuracy['best'].append(self.nbr_correct_incremental['best'][i] / len(self.stimuli_tested))
			return accuracy
		elif execute_type == "one":
			return {'leaf': self.nbr_correct['leaf'] / len(self.stimuli_tested),
					'basic': self.nbr_correct['basic'] / len(self.stimuli_tested),
					'best': self.nbr_correct['best'] / len(self.stimuli_tested)}
		elif execute_type == "multiple":
			accuracy = {'leaf': [], 'basic': [], 'best': []}
			for i in range(len(self.nbr_correct_multiple['leaf'])):
				accuracy['leaf'].append(self.nbr_correct_multiple['leaf'][i] / len(self.stimuli_tested))
				accuracy['basic'].append(self.nbr_correct_multiple['basic'][i] / len(self.stimuli_tested))
				accuracy['best'].append(self.nbr_correct_multiple['best'][i] / len(self.stimuli_tested))
			return accuracy



	def test_summary(self, execute_type="one"):
		"""
		execute_type: "incremental", "one", "multiple"
		"""
		if execute_type == "incremental":
			accuracy = self.compute_accuracy(execute_type="incremental")
			dataframe = {
				"task_type": [self.type] * len(accuracy['leaf']),
				"seed": [self.random_seed] * len(accuracy['leaf']),
				"stimuli_learned": list(range(1, len(accuracy['leaf']) + 1)),
				"nbr_correct_leaf": self.nbr_correct_incremental['leaf'],
				"nbr_correct_basic": self.nbr_correct_incremental['basic'],
				"nbr_correct_best": self.nbr_correct_incremental['best'],
				"accuracy_leaf": accuracy['leaf'],
				"accuracy_basic": accuracy['basic'],
				"accuracy_best": accuracy['best'],
				}
		elif execute_type == "one":
			accuracy = self.compute_accuracy(execute_type="one")
			dataframe = {
				"task_type": self.type,
				"seed": self.random_seed,
				"nbr_correct_leaf": self.nbr_correct['leaf'],
				"nbr_correct_basic": self.nbr_correct['basic'],
				"nbr_correct_best": self.nbr_correct['best'],
				"accuracy_leaf": accuracy['leaf'],
				"accuracy_basic": accuracy['basic'],
				"accuracy_best": accuracy['best'],
				}
		elif execute_type == "multiple":
			accuracy = self.compute_accuracy(execute_type="multiple")
			dataframe = {
				"task_type": [self.type] * len(accuracy['leaf']),
				"seed": [self.random_seed] * len(accuracy['leaf']),
				"block": list(range(1, len(accuracy['leaf']) + 1)),
				"nbr_correct_leaf": self.nbr_correct_multiple['leaf'],
				"nbr_correct_basic": self.nbr_correct_multiple['basic'],
				"nbr_correct_best": self.nbr_correct_multiple['best'],
				"accuracy_leaf": accuracy['leaf'],
				"accuracy_basic": accuracy['basic'],
				"accuracy_best": accuracy['best'],
				}
		return dataframe




class TaskSet(object):
	"""
	The collection of experiments specified by the task type (1, 2, ..., 6).
	In each TaskSet, it is initialized with task blocks:
	each block is 
	The object of a 'task type', 
	in general, a batch of the task of the same type with different random seeds.

	- task_type: 1, ..., 6
	- random_seeds: random seeds used in the TaskSet, or the block of experiments.
	- epochs: the repeating times of an experiment given a specified task.
	- task_index: a list of indices of tasks that you want to execute specificially.

	So each task type has 6 tasks (or len(task_index) if want to implement specific tasks under the type),
	each task have len(random_seeds), each with `epochs` repeating times.
	Therefore, each task will be implemented len(random_seeds) * epochs times,
	and each task type has 6 (or len(task_index)) * len(random_seeds) * epochs experiments to be implemented.
	"""

	def __init__(self, task_type, random_seeds=[1, 32, 64, 128, 256], epochs=5, task_index=None):
		self.type = task_type
		self.tasks = []
		self.random_seeds = random_seeds
		self.epochs = epochs

		# Initialization of the dataset:
		with open('tasks_smith.csv', 'r', encoding='utf-8-sig') as file:
			for row in csv.DictReader(file):
				if int(row['type']) == task_type:
					if task_index is None:
						self.read_row(row)
					elif int(row['task']) in task_index:
						self.read_row(row)


	def read_row(self, row):
		"""
		The iteration process of initialization of tasks.
		A row:
		task | type | stimuliA | stimuliB
		6 | 1 | 2,4,6,8 | 1,3,5,7
		"""
		task_index = int(row['task'])
		stimuli_indices_A = [int(index) for index in row['stimuliA'].split(',')]
		stimuli_indices_B = [int(index) for index in row['stimuliB'].split(',')]

		task = {'task': task_index, 'experiments': []}
		for random_seed in self.random_seeds:
			task['experiments'].append(Task(self.type, task_index, stimuli_indices_A, stimuli_indices_B, random_seed))
		self.tasks.append(task)


	def blocks_incremental(self, return_data=False):
		"""
		Study 1: Incremental (test once after training a new stimulus)
		"""
		# trials = list(range(1, self.epochs + 1))
		first_pass = 0
		for task in self.tasks:
			for experiment in task['experiments']:
				for epoch in range(1, self.epochs + 1):
					experiment.incremental_tr_te()
					summary = experiment.test_summary(execute_type="incremental")
					summary['epoch'] = [epoch] * len(experiment.nbr_correct_incremental['leaf'])
					if first_pass == 0:
						df_task = pd.DataFrame(summary)
						first_pass = 1
					else:
						df_task = pd.concat([df_task, pd.DataFrame(summary)], ignore_index=True)
					accuracy_exp = experiment.compute_accuracy(execute_type="incremental")
		self.dataframe_incremental = df_task
		if return_data:
			return df_task


	def blocks_test(self, return_data=False):
		"""
		Study 2: After training all stimuli
		"""
		first_pass = 0
		rows = []
		for task in self.tasks:
			for experiment in task['experiments']:
				for epoch in range(1, self.epochs + 1):
					experiment.test_model(trained=False)
					summary = experiment.test_summary(execute_type="one")
					summary['epoch'] = epoch
					rows.append(summary)
		self.dataframe_test = pd.DataFrame(rows)
		if return_data:
			return pd.DataFrame(rows)


	def blocks_test_multiple(self, n_blocks, return_data=False):
		"""
		Study 3: Training and testing all stimuli multiple times (blocks).
		"""
		first_pass = 0
		for task in self.tasks:
			for experiment in task['experiments']:
				for epoch in range(1, self.epochs + 1):
					experiment.tr_te_multple_blocks(n_blocks=n_blocks)
					summary = experiment.test_summary(execute_type="multiple")
					summary['epoch'] = [epoch] * len(experiment.nbr_correct_multiple['leaf'])
					if first_pass == 0:
						df_task = pd.DataFrame(summary)
						first_pass = 1
					else:
						df_task = pd.concat([df_task, pd.DataFrame(summary)], ignore_index=True)
						accuracy_exp = experiment.compute_accuracy(execute_type="multiple")

		self.dataframe_multiple = df_task
		if return_data:
			return df_task

	
	def summary_incremental(self, dataframe=True, quantile=False):

		if not dataframe:
			df = self.blocks_incremental(return_data=True)
		else:
			df = self.dataframe_incremental

		if quantile:
			summary_accuracy = {
			'leaf': df['accuracy_leaf'].quantile(0.75),
			'basic': df['accuracy_basic'].quantile(0.75),
			'best': df['accuracy_best'].quantile(0.75)
			}
			summary_nbr_correct = {
			'leaf': df['nbr_correct_leaf'].quantile(0.75),
			'basic': df['nbr_correct_basic'].quantile(0.75),
			'best': df['nbr_correct_best'].quantile(0.75)
			}
		else:  # mean
			summary_accuracy = {
			'leaf': df['accuracy_leaf'].mean(),
			'basic': df['accuracy_basic'].mean(),
			'best': df['accuracy_best'].mean()
			}
			summary_nbr_correct = {
			'leaf': df['nbr_correct_leaf'].mean(),
			'basic': df['nbr_correct_basic'].mean(),
			'best': df['nbr_correct_best'].mean()
			}
		summary_accuracy_incremental = {'leaf': [], 'basic': [], 'best': []}
		summary_nbr_correct_incremental = {'leaf': [], 'basic': [], 'best': []}

		for i in range(1, 9):
			df_subset = df[df['stimuli_learned'] == i]
			if quantile:
				summary_accuracy_incremental['leaf'].append(df_subset['accuracy_leaf'].quantile(0.75))
				summary_accuracy_incremental['basic'].append(df_subset['accuracy_basic'].quantile(0.75))
				summary_accuracy_incremental['best'].append(df_subset['accuracy_best'].quantile(0.75))
				summary_nbr_correct_incremental['leaf'].append(df_subset['nbr_correct_leaf'].quantile(0.75))
				summary_nbr_correct_incremental['basic'].append(df_subset['nbr_correct_basic'].quantile(0.75))
				summary_nbr_correct_incremental['best'].append(df_subset['nbr_correct_best'].quantile(0.75))
			else:
				# print(df_subset['accuracy_leaf'].mean())
				summary_accuracy_incremental['leaf'].append(df_subset['accuracy_leaf'].mean())
				summary_accuracy_incremental['basic'].append(df_subset['accuracy_basic'].mean())
				summary_accuracy_incremental['best'].append(df_subset['accuracy_best'].mean())
				summary_nbr_correct_incremental['leaf'].append(df_subset['nbr_correct_leaf'].mean())
				summary_nbr_correct_incremental['basic'].append(df_subset['nbr_correct_basic'].mean())
				summary_nbr_correct_incremental['best'].append(df_subset['nbr_correct_best'].mean())

		return summary_accuracy, summary_nbr_correct, summary_accuracy_incremental, summary_nbr_correct_incremental


	def summary_test(self, dataframe=True, quantile=False):

		if not dataframe:
			df = self.blocks_test(return_data=True)
		else:
			df = self.dataframe_test

		if quantile:
			summary_accuracy = {
			'leaf': df['accuracy_leaf'].quantile(0.75),
			'basic': df['accuracy_basic'].quantile(0.75),
			'best': df['accuracy_best'].quantile(0.75)
			}
			summary_nbr_correct = {
			'leaf': df['nbr_correct_leaf'].quantile(0.75),
			'basic': df['nbr_correct_basic'].quantile(0.75),
			'best': df['nbr_correct_best'].quantile(0.75)
			}
		else:  # mean
			summary_accuracy = {
			'leaf': df['accuracy_leaf'].mean(),
			'basic': df['accuracy_basic'].mean(),
			'best': df['accuracy_best'].mean()
			}
			summary_nbr_correct = {
			'leaf': df['nbr_correct_leaf'].mean(),
			'basic': df['nbr_correct_basic'].mean(),
			'best': df['nbr_correct_best'].mean()
			}

		return summary_accuracy, summary_nbr_correct


	def df2csv(self, incremental=False):
		if incremental:
			self.dataframe_incremental.to_csv(f"exp_shepard_type{self.type}_incre_nseeds{int(len(self.random_seeds))}_epoch{self.epochs}.csv")
		else:
			self.dataframe_test.to_csv(f"exp_shepard_type{self.type}_test_nseeds{int(len(self.random_seeds))}_epoch{self.epochs}.csv")


# ===============================================

def compute_correlations(probs_leaf, probs_basic, probs_best):
	"""
	Compute the correlations:
	Each list of given predicted probabilities (probs_leaf, probs_basic, probs_best)
	is compared to every block of observations (23 + All -> 24).
	"""
	correlations_leaf = []
	correlations_basic = []
	correlations_best = []

	with open("human_results_smith.csv", 'r', encoding='utf-8-sig') as file:
		reader = csv.reader(file)
		next(reader)
		for row in reader:
			# print(row)
			row = [eval(entry) if entry != 'all' else 'all' for entry in row]
			coefficient_leaf = np.corrcoef(row[1:-1], probs_leaf)
			coefficient_basic = np.corrcoef(row[1:-1], probs_basic)
			coefficient_best = np.corrcoef(row[1:-1], probs_best)
			correlations_leaf.append(coefficient_leaf[0, 1])
			correlations_basic.append(coefficient_basic[0, 1])
			correlations_best.append(coefficient_best[0, 1])
			# print(f"Block {row[0]}:")
			# print(f"leaf: {coefficient_leaf[0, 1]}, basic: {coefficient_basic[0, 1]}, best: {coefficient_best[0, 1]}")
	
	dataframe = {
	'block': list(range(1, 24, 2)) + ['all'],
	'leaf': correlations_leaf,
	'basic': correlations_basic,
	'best': correlations_best
	}
	# print(dataframe)
	return pd.DataFrame(dataframe)


# Common parameters:
task_types = [1, 2, 3, 4, 5, 6]
random_seeds = [1, 32, 64, 128, 256]
epochs = 5
n_stimuli = 8

"""
-------------------------------------------------
Experiment 1: test on all types.
"""
probs_leaf = []
probs_basic = []
probs_best = []
for task_type in task_types:
	task_set = TaskSet(task_type=task_type, random_seeds=random_seeds, epochs=epochs)
	if task_type == 1:
		df = task_set.blocks_test(return_data=True)
	else:
		df = pd.concat([df, task_set.blocks_test(return_data=True)], ignore_index=True)
	summary_accuracy, summary_nbr_correct = task_set.summary_test(dataframe=True)
	probs_leaf.append(summary_accuracy['leaf'])
	probs_basic.append(summary_accuracy['basic'])
	probs_best.append(summary_accuracy['best'])

# Output the csv data:
df.to_csv(f"exp_shepard_test_nseeds{int(len(random_seeds))}_epoch{epochs}.csv", index=False)

# Compute the correlations:
# print(probs_leaf, probs_basic, probs_best)
df_corr_test = compute_correlations(probs_leaf, probs_basic, probs_best)
df_corr_test.to_csv(f"exp_shepard_test_correlation_nseeds{int(len(random_seeds))}_epoch{epochs}.csv", index=False)



"""
-------------------------------------------------
Experiment 2: train the model in blocks. So the training stimuli is used for multiple times.
"""
n_blocks = 23
probs_leaf = []
probs_basic = []
probs_best = []
for task_type in task_types:
	task_set = TaskSet(task_type=task_type, random_seeds=random_seeds, epochs=epochs)
	if task_type == 1:
		df = task_set.blocks_test_multiple(n_blocks=n_blocks, return_data=True)
	else:
		df = pd.concat([df, task_set.blocks_test_multiple(n_blocks=n_blocks, return_data=True)], ignore_index=True)
df.to_csv(f"exp_shepard_multiple_blocks{int(n_blocks)}_nseeds{int(len(random_seeds))}_epoch{epochs}.csv", index=False)


