import torch
import numpy as np
import os
import re
import random

def get_file_num(filename):
		return int(re.findall(r'\d+', filename)[0])

def load_samples_from_files(data_dir):
	samples = []
	files = os.listdir(data_dir)
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		samples.append(data)
	return np.row_stack(samples)

def remove_center_beams(data, scenario_num=0):
	# If requested, remove the 10 center beams
	# if remove_center:
	# 	new_data = np.delete(data, list(range(28,38)), axis=1)

	if scenario_num == 17:
		new_data = np.delete(data, list(range(27,37)), axis=1)
	elif scenario_num in [18, 19]:
		new_data = np.delete(data, list(range(24,34)), axis=1)
	elif scenario_num in [20, 21]:
		new_data = np.delete(data, list(range(25,35)), axis=1)
	return new_data

def get_center_beams(data, scenario_num=0):
	if scenario_num == 17:
		new_data = data[:,27:37]
	elif scenario_num in [18, 19]:
		new_data = data[:,24:34]
	elif scenario_num in [20, 21]:
		new_data = data[:,25:35]
	return new_data

def get_center_beams_indx(scenario_num=0):
	if scenario_num == 17:
		return list(range(27,37))
	elif scenario_num in [18, 19]:
		return list(range(24,34))
	elif scenario_num in [20, 21]:
		return list(range(25,35))
	else:
		raise Exception("Invalid scenario number")

def get_statistics_from_matrix(data):
	total_mean = np.sum(data)/(data.shape[0] * data.shape[1])
	# This is the population standard deviation
	total_stdev = np.linalg.norm(data - total_mean)/np.sqrt(data.shape[0] * data.shape[1])
	return total_mean, total_stdev

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
	choices = np.random.choice(len(no_blk_candidate_pairs), len(blk_obs_target_pairs), replace=False)
	no_blk_obs_target_pairs = [no_blk_candidate_pairs[i] for i in choices]

	return [blk_obs_target_pairs, no_blk_obs_target_pairs]

# This is the function now used for getting the train and test data
def preprocess_data_main(pred_length, obs_length, train_ratio, batch_size, augment=False, remove_center=False, normalize_center_separately=True, shuffle=False):
	# For each scenario:
	# 1. Load the mmwave samples into a matrix (note the number of samples)
	# 2. Remove the center if desired
	# 3. Get the labels
	# Then with all the data:
	# 1. Normalize the data jointly (possibly concatenate into a single matrix)
	# 2. Split the data
	# 3. Divide into train and test data
	# 4. If requested, perform data augmentation to increase the number of samples
	# 5. If requested, randomize the order of the data before returning it
	# 6. Batch the data
	scenarios = [17, 18, 19, 20, 21]
	all_data = []
	all_center_data = []
	all_labels = []
	scenario_num_samples = []
	for scenario_num in scenarios:
		# Get the appropriate directory to pull data from
		data_dir = os.getcwd() + "\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
		# Load the mmwave samples as vectors
		scenario_data = load_samples_from_files(data_dir)
		# Concatenate the vectors into a matrix
		scenario_data = np.row_stack(scenario_data)
		# If requested, remove the 10 beams with the highest power
		if remove_center:
			scenario_data = remove_center_beams(scenario_data, scenario_num=scenario_num)
		elif normalize_center_separately:
			all_center_data.append(get_center_beams(scenario_data, scenario_num=scenario_num))
		all_data.append(scenario_data)
		# Keep track of the number of samples in each scenario
		scenario_num_samples.append(scenario_data.shape[0])
		#######################################################################################
		# Load the target labels
		labels_dir = os.getcwd() + "\\scenario{}\\unit1\\label_data".format(scenario_num)
		scenario_labels = load_samples_from_files(labels_dir)
		# Convert the labels to a column vector
		scenario_labels = np.expand_dims(np.array(scenario_labels), 1)
		all_labels.append(scenario_labels)
	
	if not remove_center and normalize_center_separately:
		# Find the overall statistics for the center beams and the non-center beams separately
		center_mean, center_stdev = get_statistics_from_matrix(np.column_stack(all_center_data))
		non_center_mean, non_center_stdev = get_statistics_from_matrix(remove_center_beams(scenario_data, scenario_num=scenario_num))
	else:
		# Find the overall statistics
		mean, stdev = get_statistics_from_matrix(np.row_stack(all_data))

	# Perform the normalization
	for i, ns in enumerate(scenario_num_samples):
		

###########################################################################################################################################
# The functions below this section are no longer used
###########################################################################################################################################
def preprocess_data_3(scenario_num, pred_length, obs_length, train_ratio, shuffle=True, augment=False, remove_center=False, joint_normalize=True):
	# Get the appropriate directory to pull data from
	data_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
	files = os.listdir(data_dir)

	# Load the mmwave samples
	full_data = []
	for file in sorted(files, key=get_file_num):
		data = np.loadtxt(data_dir + '\\' + file)
		full_data.append(data)
	full_data = np.row_stack(full_data)

	if remove_center:
		full_data = remove_center_beams(full_data, scenario_num=scenario_num)

	observation_size = full_data.shape[1]
	full_len = full_data.shape[0]

	# Normalize the data
	if joint_normalize:
		if remove_center:
			# This is with the old method where we remove beams 28 to 37
			# This will not work if the seed changes
			data_mean = 0.017562789572373498
			data_stdev = 0.016296896041679495
			full_data = (full_data - data_mean)/data_stdev
		else:
			# First normalize the center beams to have mean 0 and standard deviation 1
			center = get_center_beams(full_data, scenario_num=scenario_num)
			center_mean, center_stdev = get_full_data_statistics(center, 10)
			full_data[:,get_center_beams_indx(scenario_num=scenario_num)] = (center - center_mean)/center_stdev
			# Then normalize the non-center beams to have mean 0 and standard deviation 1
			center_removed = remove_center_beams(full_data, scenario_num=scenario_num)
			center_removed_mean, center_removed_stdev = get_full_data_statistics(center_removed, observation_size - 10)
			full_data[:,list(set(range(observation_size)) - set(get_center_beams_indx(scenario_num=scenario_num)))] = (center_removed - center_removed_mean)/center_removed_stdev
	else :
		data_mean = np.sum(full_data)/(observation_size * full_len)
		# data_stdev = np.linalg.norm(full_data)/np.sqrt(observation_size * full_len)
		data_stdev = np.linalg.norm(full_data - data_mean)/np.sqrt(observation_size * full_len)
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

def preprocess_data_batched(scenario_num, pred_length, obs_length, train_ratio, batch_size, shuffle=True, augment=False, remove_center=False, joint_normalize=True):
	train_data, test_data = preprocess_data_3(scenario_num, pred_length, obs_length, train_ratio, shuffle=shuffle, augment=augment, remove_center=remove_center, joint_normalize=joint_normalize)

	num_train_batches = len(train_data)//batch_size
	num_test_batches = len(test_data)//batch_size

	train_batches = []
	for i in range(num_train_batches):
		batch_data = train_data[i*batch_size:i*batch_size + batch_size]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		train_batches.append([batch_observations, batch_labels])
	if len(train_data) % batch_size != 0:
		batch_data = train_data[num_train_batches*batch_size:]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		train_batches.append([batch_observations, batch_labels])

	test_batches = []
	for i in range(num_test_batches):
		batch_data = test_data[i*batch_size:i*batch_size + batch_size]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		test_batches.append([batch_observations, batch_labels])
	if len(test_data) % batch_size != 0:
		batch_data = test_data[num_test_batches*batch_size:]
		batch_observations = torch.stack([torch.from_numpy(data[0]) for data in batch_data])
		batch_labels = torch.stack([torch.tensor(data[1]) for data in batch_data])
		test_batches.append([batch_observations, batch_labels])

	return [train_batches, test_batches, len(train_data), len(test_data)]

def get_all_outdoor_data(pred_length, obs_length, train_ratio, augment=False, remove_center=False, joint_normalize=True, repeat_samples=False):
	scenarios = [17, 18, 19, 20, 21]
	full_train_data = []
	full_test_data = []
	for scenario_num in scenarios:
		train_data, test_data = preprocess_data_3(scenario_num, pred_length, obs_length, train_ratio, shuffle=False, augment=augment, remove_center=remove_center, joint_normalize=joint_normalize)
		full_train_data = full_train_data + train_data
		# Add some of the data an extra time for certain scenarios so that they are less underrepresented
		if repeat_samples and (scenario_num == 20 or scenario_num == 21):
			print("Repeating samples for scenario {}".format(scenario_num))
			full_train_data = full_train_data + train_data
		full_test_data = full_test_data + test_data
	random.shuffle(full_train_data)
	random.shuffle(full_test_data)
	return [full_train_data, full_test_data]

def get_all_outdoor_data_batched(pred_length, obs_length, train_ratio, batch_size, augment=False, remove_center=False, joint_normalize=True, repeat_samples=False):
	full_train_data, full_test_data = get_all_outdoor_data(pred_length, obs_length, train_ratio, augment=augment, remove_center=remove_center, joint_normalize=joint_normalize, repeat_samples=repeat_samples)

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

	return [train_batches, test_batches, len(full_train_data), len(full_test_data)]

# Augment data by reversing each of the mmwave data vectors
def augment_data(data):
	augmented_data = []
	for sample in data:
		augmented_data.append([sample[0], sample[1]])
		# I need to use .copy() here to avoid an error about negative strides
		augmented_data.append([np.flip(sample[0], axis=0).copy(), sample[1]])
	return augmented_data

# all_data is an array containing all the data concatenated together
def get_full_data_statistics(all_data, observation_size):
	total_mean = np.sum(all_data)/(observation_size * all_data.shape[0])
	# This is the population standard deviation
	total_stdev = np.linalg.norm(all_data - total_mean)/np.sqrt(observation_size * all_data.shape[0])
	return total_mean, total_stdev

# This is incorrect; the data is not shuffled before being split into train and test sets
def get_all_data_joint_normalized(pred_length, obs_length, train_ratio, augment=False, remove_center=False):
	scenarios = [17, 18, 19, 20, 21]
	all_data = []
	for scenario_num in scenarios:
		# Get the appropriate directory to pull data from
		data_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\mmWave_data".format(scenario_num)
		files = os.listdir(data_dir)
		# Load the mmwave samples
		full_data = []
		for file in sorted(files, key=get_file_num):
			data = np.loadtxt(data_dir + '\\' + file)
			full_data.append(data)
		full_data = np.row_stack(full_data)
		full_data = remove_center_beams(full_data, scenario_num=scenario_num)
		all_data.append(full_data)
	observation_size = all_data[0].shape[1]
	all_data_matrix = np.row_stack(all_data)
	data_mean, data_stdev = get_full_data_statistics(all_data_matrix, observation_size)
	# print("Data mean: {}".format(data_mean))
	# print("Data stdev: {}".format(data_stdev))
	# Normalize the separate data sets using the statistics from the combined data set
	all_data = [(data - data_mean)/data_stdev for data in all_data]

	# Get the labels
	all_labels = []
	for scenario_num in scenarios:
		labels_dir = "C:\\Users\\jcarl\\Desktop\\ECEN 683\\Project Repo\\ECEN683-Blockage-Prediction\\scenario{}\\unit1\\label_data".format(scenario_num)
		files = os.listdir(labels_dir)
		labels = []
		for file in sorted(files, key=get_file_num):
			label = np.loadtxt(labels_dir + '\\' + file)
			labels.append(label)
		labels = np.expand_dims(np.array(labels), 1)
		all_labels.append(labels)

	# Get the split data
	all_train_data = []
	all_test_data = []
	for i in range(len(scenarios)):
		# Get the split data
		blk, no_blk = split_data(all_data[i], all_labels[i], obs_length, pred_length)
		# Divide into train and test data
		test_start_blk = (int)(np.floor(len(blk)*train_ratio))
		blk_train = blk[:test_start_blk]
		blk_test = blk[test_start_blk:]
		test_start_no_blk = (int)(np.floor(len(no_blk)*train_ratio))
		no_blk_train = no_blk[:test_start_no_blk]
		no_blk_test = no_blk[test_start_no_blk:]
		# Combine the samples with and without blockages
		train_data = blk_train + no_blk_train
		test_data = blk_test + no_blk_test
		# If requested, perform data augmentation to increase the number of samples
		if augment:
			train_data = augment_data(train_data)
		all_train_data += train_data
		all_test_data += test_data
		
	random.shuffle(all_train_data)
	random.shuffle(all_test_data)

	return [all_train_data, all_test_data, data_mean, data_stdev]

def get_batches(full_train_data, full_test_data, batch_size):
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

	return [train_batches, test_batches, len(full_train_data), len(full_test_data)]