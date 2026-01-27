# from multinomial_cobweb import MultinomialCobwebTree
from cobweb.cobweb_discrete import CobwebDiscreteTree
# from multinomial_cobweb.visualize import visualize
from random import shuffle, seed, sample, choice
import time
import csv
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from pprint import pprint

# Stimuli encodings:
# [stimulus_id, letter1, letter2, letter3, letter4, letter5, letter6, category]

# Using the coding of Murphy and Smith 1982 provided in Corter and Gluck 1992.
stimuli = [
        {'Superordinate': 'Pounder', 'Basic': 'Hammer', 'Subordinate': 'Hammer 1',
         'Handle': '1', 'Shaft': '1', 'Head': '1', 'Size': '1'},
        {'Superordinate': 'Pounder', 'Basic': 'Hammer', 'Subordinate': 'Hammer 1',
         'Handle': '1', 'Shaft': '1', 'Head': '1', 'Size': '2'},
        {'Superordinate': 'Pounder', 'Basic': 'Hammer', 'Subordinate': 'Hammer 2',
         'Handle': '1', 'Shaft': '1', 'Head': '2', 'Size': '1'},
        {'Superordinate': 'Pounder', 'Basic': 'Hammer', 'Subordinate': 'Hammer 2',
         'Handle': '1', 'Shaft': '1', 'Head': '2', 'Size': '2'},
        {'Superordinate': 'Pounder', 'Basic': 'Brick', 'Subordinate': 'Brick 1',
         'Handle': '2', 'Shaft': '2', 'Head': '3', 'Size': '1'},
        {'Superordinate': 'Pounder', 'Basic': 'Brick', 'Subordinate': 'Brick 1',
         'Handle': '2', 'Shaft': '2', 'Head': '3', 'Size': '2'},
        {'Superordinate': 'Pounder', 'Basic': 'Brick', 'Subordinate': 'Brick 2',
         'Handle': '3', 'Shaft': '2', 'Head': '3', 'Size': '1'},
        {'Superordinate': 'Pounder', 'Basic': 'Brick', 'Subordinate': 'Brick 2',
         'Handle': '3', 'Shaft': '2', 'Head': '3', 'Size': '2'},
        {'Superordinate': 'Cutter', 'Basic': 'Knife', 'Subordinate': 'Knife 1',
         'Handle': '4', 'Shaft': '3', 'Head': '4', 'Size': '1'},
        {'Superordinate': 'Cutter', 'Basic': 'Knife', 'Subordinate': 'Knife 1',
         'Handle': '4', 'Shaft': '3', 'Head': '4', 'Size': '2'},
        {'Superordinate': 'Cutter', 'Basic': 'Knife', 'Subordinate': 'Knife 2',
         'Handle': '4', 'Shaft': '3', 'Head': '5', 'Size': '1'},
        {'Superordinate': 'Cutter', 'Basic': 'Knife', 'Subordinate': 'Knife 2',
         'Handle': '4', 'Shaft': '3', 'Head': '5', 'Size': '2'},
        {'Superordinate': 'Cutter', 'Basic': 'Pizza cutter', 'Subordinate': 'Pizza 1',
         'Handle': '5', 'Shaft': '4', 'Head': '6', 'Size': '1'},
        {'Superordinate': 'Cutter', 'Basic': 'Pizza cutter', 'Subordinate': 'Pizza 1',
         'Handle': '5', 'Shaft': '4', 'Head': '6', 'Size': '2'},
        {'Superordinate': 'Cutter', 'Basic': 'Pizza cutter', 'Subordinate': 'Pizza 2',
         'Handle': '5', 'Shaft': '5', 'Head': '6', 'Size': '1'},
        {'Superordinate': 'Cutter', 'Basic': 'Pizza cutter', 'Subordinate': 'Pizza 2',
         'Handle': '5', 'Shaft': '5', 'Head': '6', 'Size': '2'}
        ]

conditions = {'basic-first': ['Basic', 'Subordinate', 'Superordinate'],
              'subordinate-first': ['Subordinate', 'Superordinate', 'Basic'],
              'superordinate-first': ['Superordinate', 'Subordinate', 'Basic']}


# training phase (done in block level by level)
    # initial exposure to each example + label (without any feedback specified)
    # category name + features -> true/false with feedback
    # this was done for 2 *blocks* of 16, each example was presented twice once
        # with a correct and once with an incorrect label; feedback was given
    # Trials that led to errors were repeated at the end and if participant
        # reptead an erro or made more than 4 errors on a block they were tested
        # on an additional block (most finished in 2 blocks and none need >4).
        # note a block is a repeat of the 16 examples - not sure on correctness of extra blocks...
# testing phase
    # same as training, but presented with categories at all three levels
    # used 10 blocks of 28 trials each, with the first 2 blocks being practice
    # for all blocks at each category level, half trials required a true response and half a false.
    # over the 8 experimental blocks (excluding first 2), each category name was presented equally on true trials, but some random variation on false trials.
    # each example served in a total of 8 subordinate trials, four basic, and 2 superordinate (half true at each level)
    # the order of the blocks and trials in each block was randomized per participant (36 participants).


participants = [i+1*36 for i in range(10)]

keys = {}
values = {}
dfs = []

for participant in participants:

    if participant % 3 == 0:
        condition = 'basic-first'
    elif participant % 3 == 1:
        condition = 'subordinate-first'
    elif participant % 3 == 2:
        condition = 'superordinate-first'

    seed(participant)
    model = CobwebDiscreteTree(alpha=0.1)

    # TRAINING
    for batch in conditions[condition]:

        # # base training - reviewing correct categories
        # for i in range(1):
        #     stimuli_tr = []
        #     labels = set([example[batch] for example in stimuli])
        #     for example in stimuli:
        #         true_x = {a: example[a] for a in example if a != 'Basic' and a != 'Superordinate' and a != 'Subordinate'}
        #         true_x['label'] = example[batch]
        #         true_x['match'] = "true"
        #         stimuli_tr.append(true_x)

        #     for stimulus in stimuli_tr:
        #         for k, v in stimulus.items():
        #             if k not in keys:
        #                 keys[k] = len(keys)
        #             if v not in values:
        #                 values[v] = len(values)

        #     pprint(stimuli_tr)
        #     stimuli_tr = [{keys[a]: {values[s[a]]: 1.0} for a in s} for s in stimuli_tr]
        #     shuffle(stimuli_tr)

        #     for s in stimuli_tr:
        #         model.ifit(s)

        for i in range(20):
            stimuli_tr = []
            labels = set([example[batch] for example in stimuli])
            for example in stimuli:
                true_x = {a: example[a] for a in example if a != 'Basic' and a != 'Superordinate' and a != 'Subordinate'}
                false_x = {a: true_x[a] for a in true_x}

                true_x['label'] = example[batch]

                foils = list(labels - set([example[batch]]))
                false_x['label'] = choice(foils)

                true_x['match'] = "true"
                false_x['match'] = "false"

                stimuli_tr.append(true_x)
                stimuli_tr.append(false_x)

            for stimulus in stimuli_tr:
                for k, v in stimulus.items():
                    if k not in keys:
                        keys[k] = len(keys)
                    if v not in values:
                        values[v] = len(values)

            stimuli_tr = [{keys[a]: {values[s[a]]: 1.0} for a in s} for s in stimuli_tr]
            shuffle(stimuli_tr)

            keys_reverse = {v: k for k,v in keys.items()}
            values_reverse = {v: k for k,v in values.items()}

            while len(stimuli_tr) > 0:
                print(len(stimuli_tr))
                retrain = []
                for s in stimuli_tr:
                    match = list(s[keys['match']].keys())[0]
                    x = {a: {v: s[a][v] for v in s[a]} for a in s if a != keys['match']}
                    pred_dict = model.predict(x, 30)
                    # pred_dict = model.predict_pmi(x, keys['match'], 30)
                    pred_dict = {keys_reverse[k]: {values_reverse[v]: pred_dict[k][v] for v in pred_dict[k]} for k in pred_dict}

                    if 'match' not in pred_dict or match not in values_reverse or values_reverse[match] not in pred_dict['match'] or pred_dict['match'][values_reverse[match]] < 0.5:
                        # if match in values_reverse and values_reverse[match] in pred_dict['match']:
                        #     p = pred_dict['match'][values_reverse[match]]
                        #     print(p)
                        retrain.append(s)
                    # else:
                    #     p = pred_dict['match'][values_reverse[match]]
                    #     print(p)
                    model.ifit(s)

                stimuli_tr = retrain

    print("DONE TRAINING")
    # model.dump_json('tree.json')

    # TESTING

    # TODO NEED TO HANDLE FALSE LABELS ACCORDING TO EXPERIMENTAL DESIGN

    labels_super = set([example["Superordinate"] for example in stimuli])
    labels_basic = set([example["Basic"] for example in stimuli])
    labels_sub = set([example["Subordinate"] for example in stimuli])
    # labels = labels_super + labels_basic + labels_sub

    stimuli_te = []
    for example in stimuli:
        base_x = {a: example[a] for a in example if a != 'Basic' and a != 'Superordinate' and a != 'Subordinate'}

        # super
        true_x = {a: base_x[a] for a in base_x}
        true_x['label'] = example['Superordinate']
        true_x['label-type'] = 'Superordinate'
        true_x['match'] = "true"
        stimuli_te.append(true_x)

        for label in list(labels_super - set([true_x['label']])):
            false_x = {a: base_x[a] for a in base_x}
            false_x['label'] = label
            false_x['label-type'] = 'Superordinate'
            false_x['match'] = "false"
            stimuli_te.append(false_x)

        # basic
        true_x = {a: base_x[a] for a in base_x}
        true_x['label'] = example['Basic']
        true_x['label-type'] = 'Basic'
        true_x['match'] = "true"
        stimuli_te.append(true_x)

        for label in list(labels_basic - set([true_x['label']])):
            false_x = {a: base_x[a] for a in base_x}
            false_x['label'] = label
            false_x['label-type'] = 'Basic'
            false_x['match'] = "false"
            stimuli_te.append(false_x)

        # sub
        true_x = {a: base_x[a] for a in base_x}
        true_x['label'] = example['Subordinate']
        true_x['label-type'] = 'Subordinate'
        true_x['match'] = "true"
        stimuli_te.append(true_x)

        for label in list(labels_sub - set([true_x['label']])):
            false_x = {a: base_x[a] for a in base_x}
            false_x['label'] = label
            false_x['label-type'] = 'Subordinate'
            false_x['match'] = "false"
            stimuli_te.append(false_x)

    for stimulus in stimuli_te:
        for k, v in stimulus.items():
            if k not in keys:
                keys[k] = len(keys)
            if v not in values:
                values[v] = len(values)

    #stimuli_XX = [(s['match'], s['label-type'], {a: {s[a]: 1.0} for a in s if a != 'match' and a != 'label-type'}) for s in stimuli_te]
    # print(stimuli_XX)

    print(len(stimuli_te))
    stimuli_te = [(s['match'], s['label-type'], {keys[a]: {values[s[a]]: 1.0} for a in s if a != 'match' and a != 'label-type'}) for s in stimuli_te]
    shuffle(stimuli_te)

    keys_reverse = {v: k for k,v in keys.items()}
    values_reverse = {v: k for k,v in values.items()}

    for i in range(len(stimuli_te)):

        # # print()
        # for j in range(0, 20, 1):
        #     pred_dict = model.predict(stimuli_te[i][2], j)
        #     pred_dict = {keys_reverse[k]: {values_reverse[v]: pred_dict[k][v] for v in pred_dict[k]} for k in pred_dict}['match']
        #     p = pred_dict[stimuli_te[i][0]]
        #     # print(j, p)
        #     data_row = {
        #             'participant seed': participant,
        #             'nodes_expanded': j,
        #             'condition': condition,
        #             'category_level': stimuli_te[i][1],
        #             'probability': p}

        #     dfs.append(pd.DataFrame([data_row]))

        # pred_dict = model.categorize(stimuli_te[i][2]).get_best(stimuli_te[i][2]).predict_probs()
        # pred_dict = model.categorize(stimuli_te[i][2]).get_basic(30, 30).predict_probs()
        pred_dict = model.predict(stimuli_te[i][2], 30)

        # pred_dict = model.predict_pmi(stimuli_te[i][2], keys['match'], 100)

        pred_dict = {keys_reverse[k]: {values_reverse[v]: pred_dict[k][v] for v in pred_dict[k]} for k in pred_dict}['match']

        p = pred_dict[stimuli_te[i][0]]


        data_row = {
                'participant seed': participant,
                'condition': condition,
                'category_level': stimuli_te[i][1],
                'probability': p}

        dfs.append(pd.DataFrame([data_row]))

    # raise Exception('one participant')

df = pd.concat(dfs, ignore_index=True)
df.to_csv(f"results/smith_and_murphy_exp1.csv", index=False)


pprint(keys)
pprint(values)
model.dump_json('tree.json')
