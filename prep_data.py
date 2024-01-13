import torch
import numpy as np
import os
import re
import random

############################################## HELPER FUNCTIONS ##############################################
# This function returns the index number of a mmWave sample file
def get_file_num(filename):
		return int(re.findall(r'\d+', filename)[0])
# This function loads the mmWave samples from a set of text files and stacks them into an array
def load_samples_from_files(data_dir):
	samples = []
	files = os.listdir(data_dir)
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		samples.append(data)
	return np.row_stack(samples)
# This function removes the 10 beams with the highest power (called the center beams) and returns the reduced data array
def remove_center_beams(data, scenario_num=0):
	if scenario_num == 17:
		new_data = np.delete(data, list(range(27,37)), axis=1)
	elif scenario_num in [18, 19]:
		new_data = np.delete(data, list(range(24,34)), axis=1)
	elif scenario_num in [20, 21]:
		new_data = np.delete(data, list(range(25,35)), axis=1)
	else:
		raise Exception("Invalid scenario number")
	return new_data
# This function returns the part of the data array corresponding to the 10 center beams
def get_center_beams(data, scenario_num=0):
	if scenario_num == 17:
		new_data = data[:,27:37]
	elif scenario_num in [18, 19]:
		new_data = data[:,24:34]
	elif scenario_num in [20, 21]:
		new_data = data[:,25:35]
	else:
		raise Exception("Invalid scenario number")
	return new_data
# This function returns the set of indices for the 10 beams with the highest power
def get_center_beams_indx(scenario_num=0):
	if scenario_num == 17:
		return list(range(27,37))
	elif scenario_num in [18, 19]:
		return list(range(24,34))
	elif scenario_num in [20, 21]:
		return list(range(25,35))
	else:
		raise Exception("Invalid scenario number")
# This function computes the mean and population standard deviation of the input array
def get_statistics(data):
	num_entries = data.shape[0] * data.shape[1]
	total_mean = np.sum(data)/num_entries
	total_stdev = np.linalg.norm(data - total_mean)/np.sqrt(num_entries)
	return total_mean, total_stdev
# This function creates candidate sequences of observations and their corresponding labels to be used as training data
	# or test data. The sequences are found differently depending on whether or not they contain blockages. For the
	# sequences with blockages, we first find the start of the blockage, then choose the delay between the last observation
	# and the blockage, and then extract the sequence of samples corresponding to the observation period. For the sequences
	# without blockages, we simply extract a sequence of samples for which no blockage occurs within those samples or
	# within the prediction period following those samples.
def split_data(samples, labels, Tobs, Tp):
	blockage_points = np.where(labels == 1)[0]
	blockage_array = np.array(blockage_points, dtype=np.int32)
	# Determine the spacing between blockage points
	blockage_spacings = blockage_array[1:] - blockage_array[:-1]
	# Determine the blockage points which mark the start of a blockage
	blockage_start_points = np.array([blockage_points[0]], dtype=np.int32)
	blockage_start_points = np.concatenate([blockage_start_points, (blockage_points[1:])[blockage_spacings > 1]])
	# Generate a list of <observation sequence, label> pairings with blockages
	blk_obs_target_pairs = []
	for bsp in blockage_start_points.tolist():
		# Make sure there is at least a full Tobs observations available, or else we cannot use the blockage
		if Tobs > bsp:
			continue
		for j in range(Tp):
			# Make sure there are enough observation samples available 
			if Tobs + j > bsp:
				continue
			# Make sure the observation samples do not contain blockages
			if np.sum(labels[bsp - (Tobs + j):bsp]) > 0:
				continue
			obs = samples[bsp - (Tobs + j):bsp - j]
			blk_obs_target_pairs.append([obs, 1])
	
	# Generate a list of <observation sequence, label> pairings without blockages
	no_blk_candidate_pairs = []
	# On each iteration, the index i points to the sample after the last observation sample
	for i in range(Tobs,samples.shape[0] - Tp):
		if np.sum(labels[i-Tobs:i+Tp]) == 0:
			obs = samples[i-Tobs:i]
			no_blk_candidate_pairs.append([obs, 0])

	# Uniformly sample the pairings without blockages to get a subset matching the number of pairings with blockages
	# This is done because there are many more possible unique sequences without blockages than with blockages
	choices = np.random.choice(len(no_blk_candidate_pairs), len(blk_obs_target_pairs), replace=False)
	no_blk_obs_target_pairs = [no_blk_candidate_pairs[i] for i in choices]

	return [blk_obs_target_pairs, no_blk_obs_target_pairs]
# This function generates the observation sequences (using the above function) and divides them into train and test sets
def train_test_group(data, labels, train_ratio, obs_length, pred_length):
	# Get the split data
	blk, no_blk = split_data(data, labels, obs_length, pred_length)
	# Divide into train and test data
	test_start_blk = (int)(np.floor(len(blk)*train_ratio))
	blk_train = blk[:test_start_blk]
	blk_test = blk[test_start_blk:]
	test_start_no_blk = (int)(np.floor(len(no_blk)*train_ratio))
	no_blk_train = no_blk[:test_start_no_blk]
	no_blk_test = no_blk[test_start_no_blk:]
	return blk_train + no_blk_train, blk_test + no_blk_test
# This function takes a set of sequences and groups them into batches, where each batch is a tensor with the sequences
	# stacked along a new dimension
def batch_data(data, batch_size):
	num_batches = len(data)//batch_size
	batches = []
	for i in range(num_batches):
		batch_data = data[i*batch_size:i*batch_size + batch_size]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		batches.append([batch_observations, batch_labels])
	if len(data) % batch_size != 0:
		batch_data = data[num_batches*batch_size:]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		batches.append([batch_observations, batch_labels])
	return batches
# This function augments sequences of observations by reversing each of the mmwave data vectors. The augmented sequences
	# are then added to the original sequences to increase the number of samples
def augment_data(data):
	augmented_data = []
	for sample in data:
		augmented_data.append([sample[0], sample[1]])
		augmented_data.append([np.flip(sample[0], axis=0).copy(), sample[1]])
	return augmented_data

############################################## MAIN DATA PREPARATION FUNCTIONS ##############################################
# This function prepares the data from a specific scenario
def preprocess_scen_data(scenario_num, pred_length, obs_length, train_ratio, augment=False, remove_center=False, joint_normalize=True):
	# Get the appropriate directory to pull data from
	data_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
	files = os.listdir(data_dir)

	# Load the mmwave samples
	full_data = []
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		full_data.append(data)
	full_data = np.row_stack(full_data)

	# If requested, remove the 10 beams with the highest power (this varies by scenario)
	if remove_center:
		full_data = remove_center_beams(full_data, scenario_num=scenario_num)

	# Normalize the data
	if joint_normalize and not remove_center:
		center_indx = get_center_beams_indx(scenario_num=scenario_num)
		# First normalize the center beams to have mean 0 and standard deviation 1
		center = get_center_beams(full_data, scenario_num=scenario_num)
		center_mean, center_stdev = get_statistics(center)
		full_data[:,center_indx] = (center - center_mean)/center_stdev
		# Then normalize the non-center beams to have mean 0 and standard deviation 1
		center_removed = remove_center_beams(full_data, scenario_num=scenario_num)
		center_removed_mean, center_removed_stdev = get_statistics(center_removed)
		full_data[:,list(set(range(full_data.shape[1])) - set(center_indx))] = (center_removed - center_removed_mean)/center_removed_stdev
	else:
		data_mean, data_stdev = get_statistics(full_data)
		full_data = (full_data - data_mean)/data_stdev
	
	# Get the labels
	labels_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\label_data".format(scenario_num)
	files = os.listdir(labels_dir)

	labels = []
	for file in sorted(files, key=get_file_num):
		label = np.loadtxt(labels_dir + '\\' + file)
		labels.append(label)
	labels = np.expand_dims(np.array(labels), 1)

	# Get the training and test sequences from the full normalized data array, along with the corresponding labels
	train_data, test_data = train_test_group(full_data, labels, train_ratio, obs_length, pred_length)

	# If requested, perform data augmentation to increase the number of samples
	if augment:
		train_data = augment_data(train_data)

	return [train_data, test_data]

# This is the function now used for getting the train and test data
def preprocess_data_main(pred_length, obs_length, train_ratio, batch_size, augment=False, remove_center=False, joint_normalize=True, shuffle=True):
	scenarios = [17, 18, 19, 20, 21]
	full_train_data = []
	full_test_data = []
	for scenario_num in scenarios:
		train_data, test_data = preprocess_scen_data(scenario_num, pred_length, obs_length, train_ratio, augment=augment, remove_center=remove_center, joint_normalize=joint_normalize)
		full_train_data = full_train_data + train_data
		full_test_data = full_test_data + test_data
	# If requested, shuffle the data samples before grouping them into batches
	if shuffle:
		random.shuffle(full_train_data)
		random.shuffle(full_test_data)

	train_batches = batch_data(full_train_data, batch_size)
	test_batches = batch_data(full_test_data, batch_size)

	return [train_batches, test_batches, len(full_train_data), len(full_test_data)]
