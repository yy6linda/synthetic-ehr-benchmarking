# 02/18/2022 edited by Zhiyu Wan
# 05/13/2022 changed back to one-hot coding
# 6/7/2022 fix bugs
import numpy as np
import time
from scipy import stats
import os.path
from synthetic_risk_model_utils import reorder_data
import sys

'''
Usage: synthetic_risk_model_attr.py [model] [exp_id] [x] [y] [original_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_attr.py baseline 1 0 8 train_uw syn_ _vumc _ Results_Synthetic_VUMC/

1. [model]: name of data generation model. Selected from ['baseline', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'real'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [x]: 10 to x is the number of neighbours. A integer larger than -1. Default: '0'. Try: '1'.
4. [y]: 2 to y is the number of sensitive attributes A integer larger than -1. Default: '8'. Try: '10'.
5. [original_filename]: the filename of the original patient file. Default: 'train_vumc'.
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


def find_neighbour(r, r_, data, data_, k, cont_sense_attr):
    # k: k nearest neighbours

    diff_array = np.abs(data - r)
    diff_array_max = np.amax(diff_array, axis=0)
    diff_array_max2 = np.maximum(diff_array_max, 1)
    diff_array_rate = diff_array/diff_array_max2
    diff = np.sum(diff_array_rate, axis=1)
    thresh = np.sort(diff)[k-1]
    idxs = np.arange(len(data))[diff <= thresh]  # not exactly k neighbours?
    predict = stats.mode(data_[idxs])[0][0]

    if N_cont > 0:
        bin_r_ = r_[np.logical_not(cont_sense_attr)]
        bin_predict = predict[np.logical_not(cont_sense_attr)]
        cont_r_ = r_[cont_sense_attr]
        cont_predict = predict[cont_sense_attr]
        bin_n = len(bin_r_)  # number of binary attributes
        true_pos = ((bin_predict + bin_r_) == 2)
        false_pos = np.array([(bin_r_[i] == 0) and (bin_predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(bin_r_[i] == 1) and (bin_predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = np.logical_and(cont_predict <= cont_r_ * 1.1, cont_predict >= cont_r_ * 0.9)
    else:
        bin_n = len(r_)  # number of binary attributes
        true_pos = ((predict + r_) == 2)
        false_pos = np.array([(r_[i] == 0) and (predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(r_[i] == 1) and (predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = 0
    return true_pos, false_pos, false_neg, correct_cont_predict


class Model(object):
    def __init__(self, fake, n, k, attr_idx):
        self.fake = fake
        self.n = n  # number of attributes used by the attacker
        self.k = k  # k nearest neighbours
        self.true_pos = []
        self.false_pos = []
        self.false_neg = []
        self.attr_idx = attr_idx  # selected attributes' indexes
        self.attr_idx_ = np.array([j for j in range(N_attr) if j not in attr_idx])  # unselected attributes' indexes
        self.data = self.fake[:, self.attr_idx]
        self.data_ = self.fake[:, self.attr_idx_]
        if N_cont > 0:
            self.correct = []
            self.cont_sense_attr = cont_sense[self.attr_idx_]

    def single_r(self, R):
        r = R[self.attr_idx]  # tested record's selected attributes
        r_ = R[self.attr_idx_]  # tested record's unselected attributes
        if N_cont > 0:
            true_pos, false_pos, false_neg, correct = find_neighbour(r, r_, self.data, self.data_, self.k, self.cont_sense_attr)
            self.correct.append(correct)
        else:
            true_pos, false_pos, false_neg, _ = find_neighbour(r, r_, self.data, self.data_, self.k, 0)
        self.true_pos.append(true_pos)
        self.false_pos.append(false_pos)
        self.false_neg.append(false_neg)



def cal_score(n, k):
    # 2^n: the number of attributes used by the attacker
    # 10^k: the number of neighbours

    real_disease = real[:, SENSE_BEGIN:SENSE_END]
    disease_attr_idx = np.flipud(np.argsort(np.mean(real_disease, axis=0)))[:2**n]  # sorted by how common a disease is
    if ordered_data:
        attr_idx = np.concatenate([np.array(range(SENSE_BEGIN)), np.array([N_attr - 1]), disease_attr_idx + SENSE_BEGIN])
    else:
        attr_idx = np.concatenate([np.array(range(SENSE_BEGIN)), disease_attr_idx + SENSE_BEGIN])
    model = Model(fake, 2 ** n, 10 ** k, attr_idx)
    n_rows = np.shape(real)[0]
    for i in range(n_rows):
        if i % 100 == 0:
            print("patient#: " + str(i))
        record = real[i, :]
        model.single_r(record)

    # binary part
    tp_array = np.stack(model.true_pos, axis=0)  # array of true positives
    fp_array = np.stack(model.false_pos, axis=0)  # array of false positives
    fn_array = np.stack(model.false_neg, axis=0)  # array of false negatives
    tpc = np.sum(tp_array, axis=0)  # vector of true positive count
    fpc = np.sum(fp_array, axis=0)  # vector of false positive count
    fnc = np.sum(fn_array, axis=0)  # vector of false negative count
    f1 = np.nan_to_num(tpc / (tpc + 0.5 * (fpc + fnc)))

    # continuous part
    if N_cont > 0:
        correct_array = np.stack(model.correct, axis=0)  # array of correctness
        accuracy = np.mean(correct_array, axis=0)

    # compute weights
    entropy = []
    real_ = real[:, model.attr_idx_]
    n_attr_ = np.shape(real_)[1]  # number of predicted attributes
    for j in range(n_attr_):
        entropy.append(get_entropy(real_[:, j]))
    weight = np.asarray(entropy) / sum(entropy)
    if N_cont > 0:
        bin_weight = weight[np.logical_not(model.cont_sense_attr)]
        cont_weight = weight[model.cont_sense_attr]
        score = np.sum(np.concatenate([f1, accuracy]) * np.concatenate([bin_weight, cont_weight]))
    else:
        score = np.sum(f1 * weight)
    return score


if __name__ == '__main__':
    # Default configuration
    dataset = "vumc"  # or "uw"
    model = 'real'
    exp_id = "1"
    x = 0  # 10 to x is the number of neighbours [0, 1]
    y = 8  # 2 to y is the number of sensitive attributes used by the attacker [0, 11]
    original_patient_filename = 'train_' + dataset
    prefix_syn = 'syn_'
    suffix_syn = '_' + dataset
    infix_syn = '_'
    Result_folder = "Results_Synthetic_" + dataset + "/"

    start1 = time.time()
    # Enable the input of parameters
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    if len(sys.argv) >= 3:
        exp_id = sys.argv[2]
    if len(sys.argv) >= 4:
        x = int(sys.argv[3])
    if len(sys.argv) >= 5:
        y = int(sys.argv[4])
    if len(sys.argv) >= 6:
        original_patient_filename = sys.argv[5]
    if len(sys.argv) >= 7:
        prefix_syn = sys.argv[6]
    if len(sys.argv) >= 8:
        suffix_syn = sys.argv[7]
    if len(sys.argv) >= 9:
        infix_syn = sys.argv[8]
    if len(sys.argv) >= 10:
        Result_folder = sys.argv[9]
    print("output_directory: " + Result_folder)
    print("original_filename: " + original_patient_filename)
    print("syn_filename: " + prefix_syn + model + infix_syn + exp_id + suffix_syn)
    print("x: " + str(x))
    print("y: " + str(y))
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)

    if dataset == 'vumc':
        SENSE_BEGIN = 7  # first 7 attributes are not sensitive
        N_attr = 2596  # number of total attributes
        N_cont = 8  # number of continuous attributes
        ordered_data = True
    else:
        SENSE_BEGIN = 8  # first 8 attributes are not sensitive
        N_attr = 2670  # number of total attributes
        N_cont = 0  # number of continuous attributes
        ordered_data = False
    SENSE_END = N_attr - N_cont
    if N_cont > 0:
        cont_sense = np.array([False for i in range(SENSE_END)] + [True for i in range(N_cont)])
    exp_name = "Attr_Risk"

    # load datasets
    if ordered_data:
        real = np.load('data/' + original_patient_filename + '.npy')
    else:
        real = reorder_data('data/' + original_patient_filename + '.npy')
    if model == 'real':
        fake = real
    else:
        fake_filename = prefix_syn + model + infix_syn + exp_id + suffix_syn
        if ordered_data:
            fake = np.load('data/' + fake_filename + '.npy')
        else:
            fake = reorder_data('data/' + fake_filename + '.npy')
    result = cal_score(y, x)
    elapsed1 = (time.time() - start1)
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1) + " seconds.")
    with open(Result_folder + exp_name + "_" + model + "_" + exp_id + "_x" + str(x) + "_y" + str(y) + ".txt", 'w') as f:
        f.write(str(result) + "\n")
        f.write("Time used: " + str(elapsed1) + " seconds.\n")
