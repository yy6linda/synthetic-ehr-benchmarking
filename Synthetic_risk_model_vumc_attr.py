# 02/18/2022 edited by Zhiyu Wan (in use)
import numpy as np
import time
from scipy import stats
import os.path

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

    bin_r_ = r_[np.logical_not(cont_sense_attr)]
    bin_predict = predict[np.logical_not(cont_sense_attr)]
    cont_r_ = r_[cont_sense_attr]
    cont_predict = predict[cont_sense_attr]
    bin_n = len(bin_r_)  # number of binary attributes
    true_pos = ((bin_predict + bin_r_) == 2)
    false_pos = np.array([(bin_r_[i] == 0) and (bin_predict[i] == 1) for i in range(bin_n)])
    false_neg = np.array([(bin_r_[i] == 1) and (bin_predict[i] == 0) for i in range(bin_n)])
    correct_cont_predict = np.logical_and(cont_predict <= cont_r_ * 1.1, cont_predict >= cont_r_ * 0.9)
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
        self.correct = []
        self.data = self.fake[:, self.attr_idx]
        self.data_ = self.fake[:, self.attr_idx_]
        self.cont_sense_attr = cont_sense[self.attr_idx_]

    def single_r(self, R):
        r = R[self.attr_idx]  # tested record's selected attributes
        r_ = R[self.attr_idx_]  # tested record's unselected attributes
        true_pos, false_pos, false_neg, correct = find_neighbour(r, r_, self.data, self.data_, self.k, self.cont_sense_attr)
        self.true_pos.append(true_pos)
        self.false_pos.append(false_pos)
        self.false_neg.append(false_neg)
        self.correct.append(correct)


def cal_score(n, k, model):
    # 2^n: the number of attributes used by the attacker
    # 10^k: the number of neighbours
    # model: 'medgan', 'medbgan', 'iwae', 'emrwgan'
    original_patient_filename = 'train_raw_14349'
    real = np.genfromtxt('data/' + original_patient_filename + '.csv', delimiter=',', skip_header=1)
    if model == 'real':
        fake = real
    else:
        original_fake_filename = 'round_syn_' + model + '_' + Exp_ID
        fake = np.genfromtxt('data/' + original_fake_filename + '.csv', delimiter=',', skip_header=1)
    #num = int(fake.shape[0] * 0.1)  # 10% sample
    #idx = np.random.choice(real.shape[0], num, replace=False)
    real_disease = real[:, SENSE_BEGIN:sense_end]
    disease_attr_idx = np.flipud(np.argsort(np.mean(real_disease, axis=0)))[:2**n]  # sorted by how common a disease is
    attr_idx = np.concatenate([np.array([0, 1, 2, 3]), disease_attr_idx + 4])
    model = Model(fake, 2 ** n, 10 ** k, attr_idx)
    #for record in real[idx]:
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
    correct_array = np.stack(model.correct, axis=0)  # array of correctness
    accuracy = np.mean(correct_array, axis=0)

    # compute weights
    entropy = []
    real_ = real[:, model.attr_idx_]
    n_attr_ = np.shape(real_)[1]  # number of predicted attributes
    for j in range(n_attr_):
        entropy.append(get_entropy(real_[:, j]))
    weight = np.asarray(entropy) / sum(entropy)
    bin_weight = weight[np.logical_not(model.cont_sense_attr)]
    cont_weight = weight[model.cont_sense_attr]

    #score = np.mean(np.concatenate([f1, accuracy]))
    score = np.sum(np.concatenate([f1, accuracy]) * np.concatenate([bin_weight, cont_weight]))
    return score


if __name__ == '__main__':
    Exp_name = "Attr_Risk_"
    Exp_ID = "3"
    Result_folder = "Results_Synthetic_Attr/"
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)
    print(Result_folder)

    SENSE_BEGIN = 4  # first 4 attributes are not sensitive
    x = 0  # for x in [0]:  # 10 to the number of neighbours [0, 1]
    y = 8  #for y in [0, 2, 4, 6, 8]:  # 2 to the number of sensitive attributes used by the attacker [0, 9, 10, 11]
    start1 = time.time()
    N_attr = 2592  # number of total attributes
    N_cont = 7  # number of continuous attributes
    sense_end = N_attr - N_cont
    cont_sense = np.array([False for i in range(N_attr - N_cont)] + [True for i in range(N_cont)])
    results = []
    for m in ['iwae', 'medgan', 'medbgan','emrwgan', 'medwgan', 'dpgan', 'real']:
        #result = []
        print("y=" + str(y))
        print("model=" + m)
        #result.append(cal_score(y, x, m))
        results.append(cal_score(y, x, m))
    elapsed1 = (time.time() - start1)
    print("Time used: " + str(elapsed1) + " seconds.\n")
    np.savetxt(Result_folder + Exp_name + "Ex" + Exp_ID + "_y" + str(y) + ".csv", np.array(results), delimiter=",")
