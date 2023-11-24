import torch
import numpy as np
import os
import re
import random

def get_file_num(filename):
		return int(re.findall(r'\d+', filename)[0])

# Old
def preprocess_data(scenario_num, pred_length):
	# Number of samples in the train and test datasets for each of the scenarios
	scen_20_train_len = 44255 #64254
	scen_20_test_len = 64254 - 44255 - 1 #70837 - 64254
	scen_17_train_len = 65000
	scen_17_test_len = 5000

	scenario_map = {17:[scen_17_train_len, scen_17_test_len], 20:[scen_20_train_len, scen_20_test_len]}

	# Get the appropriate directory to pull data from
	data_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Prototyping\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
	files = os.listdir(data_dir)

	# Load the mmwave samples
	full_data = []
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		full_data.append(data)
	full_data = np.row_stack(full_data)
	# print(full_data.shape)

	observation_size = full_data.shape[1]
	full_len = full_data.shape[0]

	# Normalize the data
	data_mean = np.sum(full_data)/(observation_size * full_len)
	data_stdev = np.linalg.norm(full_data)/np.sqrt(observation_size * full_len)
	full_data = (full_data - data_mean)/data_stdev

	# Calculate the number of samples available for the train and test sets, adjusting based on the prediction length
	total_train_observations = scenario_map[scenario_num][0]
	total_test_observations = scenario_map[scenario_num][1] - pred_length

	# Split into train and test data
	train_data = np.zeros((1, total_train_observations, observation_size), dtype=np.float32)
	test_data = np.zeros((1, total_test_observations, observation_size), dtype=np.float32)
	train_data[0,:,:] = full_data[:total_train_observations,:]
	test_data[0,:,:] = full_data[total_train_observations:-(pred_length+1),:]

	train_len = total_train_observations
	test_len = total_test_observations - pred_length

	# Get the target predictions
	labels_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Prototyping\\scenario{}\\unit1\\label_data".format(scenario_num)
	files = os.listdir(labels_dir)

	labels = []
	for file in sorted(files, key=get_file_num):
		label = np.loadtxt(labels_dir + '\\' + file)
		labels.append(label)
	targets = np.expand_dims(np.array(labels), 1)

	train_targets = targets[pred_length:train_len+pred_length]
	test_targets = targets[train_len+pred_length:]

	# print(total_train_observations, train_targets.shape)
	# print(total_test_observations, test_targets.shape)

	train_data = torch.from_numpy(train_data)
	train_targets = torch.from_numpy(train_targets)
	test_data = torch.from_numpy(test_data)
	test_targets = torch.from_numpy(test_targets)

	return [train_data, train_targets, test_data, test_targets]
# Old
def preprocess_data_2(scenario_num, pred_length):
	# Number of samples in the train and test datasets for each of the scenarios
	scen_20_train_len = 44255 #64254
	scen_20_test_len = 64254 - 44255 - 1 #70837 - 64254
	scen_17_train_len = 65000
	scen_17_test_len = 5000

	scenario_map = {17:[scen_17_train_len, scen_17_test_len], 20:[scen_20_train_len, scen_20_test_len]}

	# Get the appropriate directory to pull data from
	data_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Prototyping\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
	files = os.listdir(data_dir)

	# Load the mmwave samples
	full_data = []
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		full_data.append(data)
	full_data = np.row_stack(full_data)
	# print(full_data.shape)

	observation_size = full_data.shape[1]
	full_len = full_data.shape[0]

	# Normalize the data
	data_mean = np.sum(full_data)/(observation_size * full_len)
	data_stdev = np.linalg.norm(full_data)/np.sqrt(observation_size * full_len)
	full_data = (full_data - data_mean)/data_stdev

	# Calculate the number of samples available for the train and test sets, adjusting based on the prediction length
	total_train_observations = scenario_map[scenario_num][0]
	total_test_observations = scenario_map[scenario_num][1] - pred_length

	# Split into train and test data
	train_data = np.zeros((1, total_train_observations, observation_size), dtype=np.float32)
	test_data = np.zeros((1, total_test_observations, observation_size), dtype=np.float32)
	train_data[0,:,:] = full_data[:total_train_observations,:]
	test_data[0,:,:] = full_data[total_train_observations:-(pred_length+1),:]

	train_len = total_train_observations
	test_len = total_test_observations - pred_length

	# Get the target predictions
	labels_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Prototyping\\scenario{}\\unit1\\label_data".format(scenario_num)
	files = os.listdir(labels_dir)

	labels = []
	for file in sorted(files, key=get_file_num):
		label = np.loadtxt(labels_dir + '\\' + file)
		labels.append(label)
	labels = np.expand_dims(np.array(labels), 1)

	# Get the actual prediction targets
	targets = []
	for i in range(labels.shape[0]):
		if np.sum(labels[i:i+pred_length+1]) > 0:
			targets.append(1)
		else:
			targets.append(0)
	targets = np.expand_dims(np.array(targets), 1)

	blockage_points = np.where(labels == 1)[0]
	# Discards the data points where a blockage is in effect
	# If we are going to use this, we need to so it before splitting into train and test sets
	# targets = np.delete(targets,blockage_points,0)

	# plt.plot(labels[75:200])
	# plt.plot(0.99*targets[75:200])
	# plt.show()

	train_targets = targets[pred_length:train_len+pred_length]
	test_targets = targets[train_len+pred_length:]

	train_data = torch.from_numpy(train_data)
	train_targets = torch.from_numpy(train_targets)
	test_data = torch.from_numpy(test_data)
	test_targets = torch.from_numpy(test_targets)

	train_blockage_points = blockage_points[blockage_points < train_len]
	test_blockage_points = blockage_points[blockage_points >= train_len]

	return [train_data, train_targets, test_data, test_targets, train_blockage_points, test_blockage_points]

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
			obs = samples[bsp - (Tobs + j):bsp]
			blk_obs_target_pairs.append([obs, 1])
	
	# Generate a list of <observation sequence, label> pairings without blockages
	no_blk_candidate_pairs = []
	# On each iteration, the index i points to the sample after the last observation sample
	for i in range(Tobs,samples.shape[0] - Tp):
		if np.sum(labels[i-Tobs:i+Tp]) == 0:
			obs = samples[i-Tobs:i]
			no_blk_candidate_pairs.append([obs, 0])

	# Uniformly sample the pairings without blockages to get a subset matching the number of pairings with blockages
	choices = np.random.choice(len(no_blk_candidate_pairs), len(blk_obs_target_pairs), replace=False)
	no_blk_obs_target_pairs = [no_blk_candidate_pairs[i] for i in choices]

	return [blk_obs_target_pairs, no_blk_obs_target_pairs]
# This is the version which is currently used
def preprocess_data_3(scenario_num, pred_length, obs_length, train_ratio, shuffle=True, augment=False):
	# Get the appropriate directory to pull data from
	data_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
	files = os.listdir(data_dir)

	# Load the mmwave samples
	full_data = []
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		full_data.append(data)
	full_data = np.row_stack(full_data)
	# print(full_data.shape)

	observation_size = full_data.shape[1]
	full_len = full_data.shape[0]

	# Normalize the data
	data_mean = np.sum(full_data)/(observation_size * full_len)
	data_stdev = np.linalg.norm(full_data)/np.sqrt(observation_size * full_len)
	full_data = (full_data - data_mean)/data_stdev

	# Get the labels
	labels_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\label_data".format(scenario_num)
	files = os.listdir(labels_dir)

	labels = []
	for file in sorted(files, key=get_file_num):
		label = np.loadtxt(labels_dir + '\\' + file)
		labels.append(label)
	labels = np.expand_dims(np.array(labels), 1)

	# Get the split data
	blk, no_blk = split_data(full_data, labels, obs_length, pred_length)

	# Divide into train and test data
	test_start_blk = (int)(np.floor(len(blk)*train_ratio))
	blk_train = blk[:test_start_blk]
	blk_test = blk[test_start_blk:]
	test_start_no_blk = (int)(np.floor(len(no_blk)*train_ratio))
	no_blk_train = no_blk[:test_start_no_blk]
	no_blk_test = no_blk[test_start_no_blk:]

	train_data = blk_train + no_blk_train
	test_data = blk_test + no_blk_test

	# If requested, perform data augmentation to increase the number of samples
	if augment:
		train_data = augment_data(train_data)
	
	# If requested, randomize the order of the data before returning it
	if shuffle:
		random.shuffle(train_data)
		random.shuffle(test_data)

	return [train_data, test_data]

def get_all_outdoor_data(pred_length, obs_length, train_ratio, augment=False):
	scenarios = [17, 18, 20, 21]
	full_train_data = []
	full_test_data = []
	for scenario_num in scenarios:
		train_data, test_data = preprocess_data_3(scenario_num, pred_length, obs_length, train_ratio, shuffle=False, augment=augment)
		full_train_data = full_train_data + train_data
		full_test_data = full_test_data + test_data
	random.shuffle(full_train_data)
	random.shuffle(full_test_data)
	return [full_train_data, full_test_data]

def get_all_outdoor_data_batched(pred_length, obs_length, train_ratio, batch_size, augment=False):
	full_train_data, full_test_data = get_all_outdoor_data(pred_length, obs_length, train_ratio, augment=augment)

	num_full_train_batches = len(full_train_data)//batch_size
	num_full_test_batches = len(full_test_data)//batch_size

	train_batches = []
	for i in range(num_full_train_batches):
		batch_data = full_train_data[i*batch_size:i*batch_size + batch_size]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		train_batches.append([batch_observations, batch_labels])
	if len(full_train_data) % batch_size != 0:
		batch_data = full_train_data[num_full_train_batches*batch_size:]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		train_batches.append([batch_observations, batch_labels])

	test_batches = []
	for i in range(num_full_test_batches):
		batch_data = full_test_data[i*batch_size:i*batch_size + batch_size]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		test_batches.append([batch_observations, batch_labels])
	if len(full_test_data) % batch_size != 0:
		batch_data = full_test_data[num_full_test_batches*batch_size:]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		test_batches.append([batch_observations, batch_labels])

	return [train_batches, test_batches]


# Augment data by reversing each of the mmwave data vectors
def augment_data(data):
	augmented_data = []
	for sample in data:
		augmented_data.append([sample[0], sample[1]])
		# I need to use .copy() here to avoid an error about negative strides
		augmented_data.append([np.flip(sample[0], axis=0).copy(), sample[1]])
	return augmented_data