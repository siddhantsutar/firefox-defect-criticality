import json
import numpy as np
import os
import pathlib
from shutil import copyfile

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
    max_last_saved = 10
    winner_accuracy = 0
    winner_epoch = 0
    while epochs_last_saved < max_last_saved and epoch < max_epochs:
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
    with open("../data/firefox-defect-criticality_set.json") as infile:
        data = json.load(infile)
    if train:
        return data[0], data[1], data[2], data[3]
    else:
        return data[2], data[3]