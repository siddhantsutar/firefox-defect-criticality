#!/usr/bin/env python3
from collections import Counter
import json
import numpy as np
import os
import pathlib
from shutil import copyfile

DATA_FILE = "../data/firefox-defect-criticality_set.json"
MAX_LAST_SAVED = 10


def dimensionality_reduction(train_data, test_data, new_dimensions):
    feature_count = Counter()
    index_feature_train = {}
    index_feature_test = {}
    train_data = transpose_list(train_data)
    test_data = transpose_list(test_data)
    for i in range(len(train_data)):
        feature_count[i] = train_data[i].count(1)
        index_feature_train[i] = train_data[i]
        index_feature_test[i] = test_data[i]
    train_data = []
    test_data = []
    for each in feature_count.most_common(new_dimensions):
        train_data.append(index_feature_train[each[0]])
        test_data.append(index_feature_test[each[0]])
    train_data = transpose_list(train_data)
    test_data = transpose_list(test_data)
    return train_data, test_data


def duplicate_model(model_path):
	existGDBPath = pathlib.Path(model_path)
	wkspFldr = existGDBPath.parent
	for file in os.listdir(wkspFldr):
		from_model = os.path.join(wkspFldr, file)
		to_model = from_model.replace(".model", "_winner.model")
		if model_path in from_model:
			copyfile(from_model, to_model)


def early_stopping(train, max_epochs, model_path):
    epoch = 1
    epochs_last_saved = 0
    winner_accuracy = 0
    winner_epoch = 0
    while epochs_last_saved < MAX_LAST_SAVED and epoch < max_epochs:
        cursor_accuracy = train(epoch)
        if cursor_accuracy >= winner_accuracy:
            winner_accuracy = cursor_accuracy
            winner_epoch = epoch
            epochs_last_saved = 0
            duplicate_model(model_path)
        else:
            epochs_last_saved += 1
        epoch += 1
    print("-----")
    print("Best accuracy: ", winner_accuracy, ", total epochs: ", winner_epoch, sep='')


def get_next_batch(start, batch_size, x, y):
	end = start + batch_size
	return np.array(x[start:end]), np.array(y[start:end])


def load_feature_sets(train=True):
    with open(DATA_FILE) as infile:
        data = json.load(infile)
    if train:
        return data[0], data[1], data[2], data[3]
    else:
        return data[2], data[3]


def transpose_list(data):
    return list(map(list, zip(*data)))