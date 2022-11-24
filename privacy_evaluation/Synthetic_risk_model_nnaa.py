# 03/15/2022 edited by Zhiyu Wan
# 05/14/2022 add normalization
# 10/03/2022 add nearest neighbor adversarial accuracy
import numpy as np
import os.path
import time
import sys

'''
Usage: synthetic_risk_model_nnaa.py [model] [exp_id] [train_filename] [test_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_nnaa.py iwae 1 train_vumc test_vumc syn_ _vumc _ Results_Synthetic_VUMC/

1. [model]: name of data generation model. Selected from ['iwae', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'iwae'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [train_filename]: the filename of the training file. Default: 'train_vumc'.
4. [test_filename]: the filename of the test file. Default: 'test_vumc'.
5. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
6. [suffix_syn]: the suffix of the synthetic filename. Default: '_vumc'.
7. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
8. [output_directory]: output directory. Default: 'Results_Synthetic_VUMC/'.
'''

def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy


def find_replicant(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)

def find_replicant_self(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    n_col = np.shape(distance_matrix)[1]
    min_distance = np.zeros(n_col)
    for i in range(n_col):
        sorted_column = np.sort(distance_matrix[:, i])
        min_distance[i] = sorted_column[1]
    return min_distance

def each_group(model):
    steps = np.ceil(n_test / batchsize).astype(int)
    if n_train == n_test:
        n_draw = 1
    else:
        n_draw = np.ceil(n_train / n_test).astype(int)
    # training dataset
    distance_train_TS = np.zeros(n_test)
    distance_train_TT = np.zeros(n_test)
    distance_train_ST = np.zeros(n_test)
    distance_train_SS = np.zeros(n_test)
    aa_train = 0

    if model != 'real':
        for ii in range(n_draw):
            np.random.seed(ii)
            train_sample = np.random.permutation(train)[:n_test]
            np.random.seed(ii)
            fake_sample = np.random.permutation(fake)[:n_test]
            for i in range(steps):
                distance_train_TS[i * batchsize:(i + 1) * batchsize] = find_replicant(train_sample[i * batchsize:(i + 1) * batchsize], fake_sample)
                distance_train_ST[i * batchsize:(i + 1) * batchsize] = find_replicant(fake_sample[i * batchsize:(i + 1) * batchsize], train_sample)
                distance_train_TT[i * batchsize:(i + 1) * batchsize] = find_replicant_self(train_sample[i * batchsize:(i + 1) * batchsize], train_sample)
                distance_train_SS[i * batchsize:(i + 1) * batchsize] = find_replicant_self(fake_sample[i * batchsize:(i + 1) * batchsize], fake_sample)
            aa_train += (np.sum(distance_train_TS > distance_train_TT) + np.sum(distance_train_ST > distance_train_SS)) / n_test / 2
        aa_train /= n_draw

    # test dataset
    distance_test_TS = np.zeros(n_test)
    distance_test_TT = np.zeros(n_test)
    distance_test_ST = np.zeros(n_test)
    distance_test_SS = np.zeros(n_test)
    aa_test = 0
    for ii in range(n_draw):
        np.random.seed(ii)
        fake_sample = np.random.permutation(fake)[:n_test]
        for i in range(steps):
            distance_test_TS[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake_sample)
            distance_test_ST[i * batchsize:(i + 1) * batchsize] = find_replicant(fake_sample[i * batchsize:(i + 1) * batchsize], test)
            distance_test_TT[i * batchsize:(i + 1) * batchsize] = find_replicant_self(test[i * batchsize:(i + 1) * batchsize], test)
            distance_test_SS[i * batchsize:(i + 1) * batchsize] = find_replicant_self(fake_sample[i * batchsize:(i + 1) * batchsize], fake_sample)
        aa_test += (np.sum(distance_test_TS > distance_test_TT) + np.sum(distance_test_ST > distance_test_SS)) / n_test / 2
    aa_test /= n_draw

    privacy_loss = aa_test - aa_train
    return privacy_loss


if __name__ == '__main__':
    # Default configuration
    dataset = "vumc"  # or "uw"
    model = 'real'
    exp_id = "1"
    train_patient_filename = 'train_' + dataset
    test_patient_filename = 'test_' + dataset
    prefix_syn = 'syn_'
    suffix_syn = '_' + dataset
    infix_syn = '_'
    if dataset == 'vumc':
        n_cont_col = 8  # number of columns for continuous features from the right
    else:
        n_cont_col = 0
    Result_folder = "Results_Synthetic_" + dataset + "+/"
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)
    batchsize = 1000
    exp_name = "NNAA_Risk"

    start1 = time.time()
    # Enable the input of parameters
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    if len(sys.argv) >= 3:
        exp_id = sys.argv[2]
    if len(sys.argv) >= 4:
        train_patient_filename = sys.argv[3]
    if len(sys.argv) >= 5:
        test_patient_filename = sys.argv[4]
    if len(sys.argv) >= 6:
        prefix_syn = sys.argv[5]
    if len(sys.argv) >= 7:
        suffix_syn = sys.argv[6]
    if len(sys.argv) >= 8:
        infix_syn = sys.argv[7]
    if len(sys.argv) >= 9:
        Result_folder = sys.argv[8]
    print("output_directory: " + Result_folder)
    print("train_filename: " + train_patient_filename)
    print("test_filename: " + test_patient_filename)
    print("syn_filename: " + prefix_syn + model + infix_syn + exp_id + suffix_syn)

    # load datasets
    #train = np.load('data/' + train_patient_filename + '.npy')
    train = np.load('data/train_vumc_fake_1434.npy')
    #test = np.load('data/' + test_patient_filename + '.npy')
    test = np.load('data/test_vumc_fake_615.npy')
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
    for j in range(n_col-n_cont_col, n_col):
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
    with open(Result_folder + exp_name + "_" + model + "_" + exp_id + ".txt", 'w') as f:
        f.write(str(result) + "\n")
        f.write("Time used: " + str(elapsed1 + elapsed2) + " seconds.\n")
        f.write("Loading time used: " + str(elapsed1) + " seconds.\n")
        f.write("Computing time used: " + str(elapsed2) + " seconds.\n")
