# 03/15/2022 edited by Zhiyu Wan
import numpy as np
import os.path
import time


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
    train_patient_filename = 'train_raw_14349'
    train = np.genfromtxt('data/' + train_patient_filename + '.csv', delimiter=',', skip_header=1)
    test_patient_filename = 'test_raw'
    test = np.genfromtxt('data/' + test_patient_filename + '.csv', delimiter=',', skip_header=1)
    if model == 'real':
        fake = train
    else:
        fake_filename = 'round_syn_' + model + '_' + Exp_ID
        fake = np.genfromtxt('data/' + fake_filename + '.csv', delimiter=',', skip_header=1)

    # compute weights
    #entropy = []
    #n_attr = np.shape(train)[1]  # number of attributes
    #for j in range(n_attr):
    #    entropy.append(get_entropy(train[:, j]))
    #weight = np.asarray(entropy) / sum(entropy)

    # normalization
    #
    #normal_range = np.maximum(np.amax(fake, axis=0) - np.amin(fake, axis=0), 1)
    #train = train / normal_range
    #test = test / normal_range
    #fake = fake / normal_range
    #normal_weight = weight / normal_range
    #normal_weight_mat = normal_weight.reshape(-1, 1)

    n_train = np.shape(train)[0]
    n_test = np.shape(test)[0]
    distance_train = np.zeros(n_train)
    distance_test = np.zeros(n_test)
    if model != 'real':
        steps = np.ceil(n_train / batchsize)
        for i in range(int(steps)):
            distance_train[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], fake)

    steps = np.ceil(n_test / batchsize)
    for i in range(int(steps)):
        distance_test[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake)

    n_fp = np.floor(beta * n_test).astype(int)  # tolerable false positive counts
    sorted_distance_test = np.sort(distance_test, axis=None)
    theta = sorted_distance_test[n_fp]  # threshold
    n_tp = np.sum(distance_train <= theta)  # true positive counts
    n_fp = np.sum(distance_test <= theta)  # false positive counts
    precision = n_tp / (n_tp+n_fp)
    recall = n_tp / n_train  # true positive rate
    f1 = 2 * precision * recall / (precision + recall)  # F1 score
    return f1


if __name__ == '__main__':
    batchsize = 1000
    Exp_name = "Mem_Risk_"
    Exp_ID = "3"
    Result_folder = "Results_Synthetic_Mem/"
    beta = 0.05
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)
    print(Result_folder)
    results = []
    start1 = time.time()
    for m in ['iwae', 'medgan', 'medbgan','emrwgan', 'medwgan', 'dpgan', 'real']:
        print("model=" + m)
        results.append(each_group(m))
    elapsed1 = (time.time() - start1)
    print("Time used: " + str(elapsed1) + " seconds.\n")
    np.savetxt(Result_folder + Exp_name + "Ex" + Exp_ID + ".csv", np.array(results), delimiter=",")
