# 03/15/2022 edited by Zhiyu Wan
# 05/14/2022 add normalization
import numpy as np
import os.path
import time
import sys

'''
Usage: synthetic_risk_model_vumc_mem.py [model] [exp_id] [theta] [train_filename] [test_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_vumc_mem.py iwae 1 32 train_vumc test_vumc syn_ _vumc _ Results_Synthetic_VUMC/

1. [model]: name of data generation model. Selected from ['iwae', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'iwae'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [theta]: the threshold for the euclidean distance between two records. Default: '32'. Try: '64'.
4. [train_filename]: the filename of the training file. Default: 'train_vumc'.
5. [test_filename]: the filename of the test file. Default: 'test_vumc'.
6. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
7. [suffix_syn]: the suffix of the synthetic filename. Default: '_vumc'.
8. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
9. [output_directory]: output directory. Default: 'Results_Synthetic_VUMC/'.
'''

def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy


def find_replicant(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0],1) + np.sum(real.T ** 2,axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)


def each_group(model):
    distance_train = np.zeros(n_train)
    distance_test = np.zeros(n_test)
    if model != 'real':
        steps = np.ceil(n_train / batchsize)
        for i in range(int(steps)):
            distance_train[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], fake)

    steps = np.ceil(n_test / batchsize)
    for i in range(int(steps)):
        distance_test[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake)

    n_tp = np.sum(distance_train <= theta)  # true positive counts
    n_fn = n_train - n_tp
    n_fp = np.sum(distance_test <= theta)  # false positive counts
    f1 = n_tp / (n_tp + (n_fp + n_fn) / 2)  # F1 score
    return f1


if __name__ == '__main__':
    # Default configuration
    for model in ['iwae', 'medgan', 'medbgan', 'medwgan', 'emrwgan', 'dpgan', 'real']:
        #model = 'medgan'
        exp_id = "1"
        theta = 5
        train_patient_filename = 'train_raw_correct'
        test_patient_filename = 'test_raw_correct'
        prefix_syn = 'correct_round_syn_'
        suffix_syn = ''
        infix_syn = '_'
        Result_folder = "Results_Synthetic_VUMC_V3/"
        if not os.path.exists(Result_folder):
            os.mkdir(Result_folder)
        batchsize = 1000
        exp_name = "Mem_Risk"

        start1 = time.time()
        # Enable the input of parameters
        if len(sys.argv) >= 2:
            model = sys.argv[1]
        if len(sys.argv) >= 3:
            exp_id = sys.argv[2]
        if len(sys.argv) >= 4:
            theta = float(sys.argv[3])
        if len(sys.argv) >= 5:
            train_patient_filename = sys.argv[4]
        if len(sys.argv) >= 6:
            test_patient_filename = sys.argv[5]
        if len(sys.argv) >= 7:
            prefix_syn = sys.argv[6]
        if len(sys.argv) >= 8:
            suffix_syn = sys.argv[7]
        if len(sys.argv) >= 9:
            infix_syn = sys.argv[8]
        if len(sys.argv) >= 10:
            Result_folder = sys.argv[9]
        print("output_directory: " + Result_folder)
        print("train_filename: " + train_patient_filename)
        print("test_filename: " + test_patient_filename)
        print("syn_filename: " + prefix_syn + model + infix_syn + exp_id + suffix_syn)
        print("theta: " + str(theta))
        # load datasets
        train = np.load('data/' + train_patient_filename + '.npy')
        test = np.load('data/' + test_patient_filename + '.npy')
        n_train = np.shape(train)[0]
        n_test = np.shape(test)[0]
        if model == 'real':
            fake = train.copy()
        else:
            fake_filename = prefix_syn + model + infix_syn + exp_id + suffix_syn
            fake = np.load('data/' + fake_filename + '.npy')
        elapsed1 = (time.time() - start1)
        start2 = time.time()
        # normalization
        [n_row, n_col] = fake.shape
        for j in range(n_col-8, n_col):
            normal_max = np.amax(fake[:, j])
            normal_min = np.amin(fake[:, j])
            normal_range = normal_max - normal_min
            fake[:, j] = (fake[:, j] - normal_min) / normal_range
            train[:, j] = (train[:, j] - normal_min) / normal_range
            test[:, j] = (test[:, j] - normal_min) / normal_range

        result = each_group(model)
        elapsed2 = (time.time() - start2)
        print("Risk: " + str(result) + ".")
        print("Time used: " + str(elapsed1 + elapsed2) + " seconds.")
        print("Loading time used: " + str(elapsed1) + " seconds.")
        print("Computing time used: " + str(elapsed2) + " seconds.")
        with open(Result_folder + exp_name + "_" + model + "_" + exp_id + "_theta" + str(theta) + ".txt", 'w') as f:
            f.write(str(result) + "\n")
            f.write("Time used: " + str(elapsed1 + elapsed2) + " seconds.\n")
            f.write("Loading time used: " + str(elapsed1) + " seconds.\n")
            f.write("Computing time used: " + str(elapsed2) + " seconds.\n")
