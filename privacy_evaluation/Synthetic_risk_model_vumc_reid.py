import numpy as np
import time
from scipy.linalg import cholesky
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os.path
import datetime

'''
Usage: synthetic_risk_model_vumc_reid.py [model] [exp_id] [theta] [original_filename] [pop_filename] [prefix_syn] [infix_syn] [output_directory] [n_phe_qid]

Example: synthetic_risk_model_vumc_reid.py iwae 1 0.05 train_vumc pop_vumc syn_ _vumc _ Results_Synthetic_VUMC/ 7

1. [model]: name of data generation model. Selected from ['iwae', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'iwae'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [theta]: ratio of the correctly inferred attributes in a successful attack. A real number in [0, 1]. Default: '0.05'. Try: '0.001'.
4. [original_filename]: the filename of the original patient file. Default: 'train_vumc'.
5. [pop_filename]: the filename of the population file with demographics (QIDs) only. Default: 'pop_vumc'.
6. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
7. [suffix_syn]: the suffix of the synthetic filename. Default: '_vumc'.
8. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
9. [output_directory]: output directory. Default: 'Results_Synthetic_VUMC/'.
10. [n_phe_qid]: the number of phenotypic qids (most common diseases). An integer in [0, 10]. Default: '7'.
'''


def replace_dataset(pop, level):
    (n_po, m_po) = pop.shape
    final_pop = pop.copy()
    if np.array_equal(level, np.asarray(MAX_LEVELS)):
        return final_pop
    elif np.array_equal(level, np.asarray([0] * n_qid)):
        return np.zeros(pop.shape)
    else:
        for i in range(n_po):
            new_pop_row = pop[i, :].copy()
            for j in range(len(level)):
                attr = j
                attr_value = new_pop_row[j]
                attr_level = level[j]
                tuple_replace = (attr, attr_value, attr_level)
                if tuple_replace in dic_replace:
                    new_pop_row[j] = dic_replace[tuple_replace]
                else:
                    if attr_level == 0:
                        new_pop_row[j] = 0
                        dic_replace[tuple_replace] = 0
                    elif attr_level == MAX_LEVELS[j]:
                        dic_replace[tuple_replace] = attr_value
                    else:
                        list_new_values = generalization_mat[attr][attr_level-1]
                        final_value = 0
                        for i_value in range(len(list_new_values)):
                            if attr_value >= list_new_values[i_value]:
                                final_value = i_value
                            else:
                                break
                        new_pop_row[j] = final_value
                        dic_replace[tuple_replace] = final_value
            final_pop[i, :] = new_pop_row
        return final_pop


def generate_lattice_dfs(gen_levels):  # DFS
    lattice = [gen_levels]
    for i in range(len(gen_levels)):
        if gen_levels[i] != 0:
            gen_levels_new = gen_levels.copy()
            gen_levels_new[i] -= 1
            if not gen_levels_new in lattice:
                lattice.append(gen_levels_new)
            lattice = generate_lattice_dfs(gen_levels_new, lattice)
    return lattice


def generate_lattice_bfs(top_gen_levels):  # BFS
    visited = [top_gen_levels]
    queue = [top_gen_levels]
    lattice = []
    while queue:
        gen_levels = queue.pop(0)
        lattice.append(gen_levels)
        for i in range(len(gen_levels)):
            if gen_levels[i] != 0:
                gen_levels_new = gen_levels.copy()
                gen_levels_new[i] -= 1
                if not gen_levels_new in visited:
                    visited.append(gen_levels_new)
                    queue.append(gen_levels_new)
    return lattice


def rand_lamb(n_simu):
    vr_mode = vr_mean * 3 - vr_min - vr_max
    vr = np.random.triangular(vr_min, vr_mode, vr_max, (n_simu, 1))
    er_mode = er_mean * 3 - er_min - er_max
    er = np.random.triangular(er_min, er_mode, er_max, (n_simu, 1))
    corr_mat = np.array([[1, 0.5], [0.5, 1]])
    upper_chol = cholesky(corr_mat)
    vals = np.hstack((vr, er))
    cor_vals = np.dot(vals, upper_chol)
    cor_vr = cor_vals[:, 0]
    cor_er = cor_vals[:, 1]
    lamb_s = cor_vr * (1 - np.power(1 - cor_er, n_qid))
    lamb_sp = (1 + lamb_s) / 2
    return lamb_sp


if __name__ == '__main__':

    # default configuration
    model = 'real'
    exp_id = "2"
    theta = 0.001  # ratio of the correctly inferred attributed in a successful attack.
    original_patient_filename = 'train_raw_correct'
    pop_filename = 'pop_vumc'
    prefix_syn = 'correct_round_syn_'
    suffix_syn = ''
    infix_syn = '_'
    Result_folder = "Results_Synthetic_VUMC/"
    n_phe_qid = 7

    start1 = time.time()
    # Enable the input of parameters
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    if len(sys.argv) >= 3:
        exp_id = sys.argv[2]
    if len(sys.argv) >= 4:
        theta = float(sys.argv[3])
    if len(sys.argv) >= 5:
        original_patient_filename = sys.argv[4]
    if len(sys.argv) >= 6:
        pop_filename = sys.argv[5]
    if len(sys.argv) >= 7:
        prefix_syn = sys.argv[6]
    if len(sys.argv) >= 8:
        suffix_syn = sys.argv[7]
    if len(sys.argv) >= 9:
        infix_syn = sys.argv[8]
    if len(sys.argv) >= 10:
        Result_folder = sys.argv[9]
    if len(sys.argv) >= 11:
        n_phe_qid = int(sys.argv[10])
    print("output_directory: " + Result_folder)
    print("original_filename: " + original_patient_filename)
    print("pop_filename: " + pop_filename)
    print("syn_filename: " + prefix_syn + model + infix_syn + exp_id + suffix_syn)
    print("theta: " + str(theta))
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)

    randomization = True
    top_phe = [509, 1129, 602, 1185, 654, 1200, 1176, 525, 1183, 698]
    # 401.1, 745, 465, 785, 512.8, 798, 773, 418, 783, 530.11
    qid_index = [0, 1, 2] + [top_phe[i] for i in range(n_phe_qid)]
    n_qid = len(qid_index)
    n_index = 2592
    sense_index = [j for j in range(n_index) if j not in qid_index]  # non-QID attributes' indexes
    # sense_index.reverse()
    n_sense_index = len(sense_index)
    nom_sense = [True for i in range(n_sense_index - 7)] + [False for i in range(7)]
    #nom_sense.reverse()
    theta_distance_pop = 0
    vr_mean = 0.23  # verification rate
    vr_min = 0.1
    vr_max = 0.43  # 0.61
    er_mean = 0.0426  # data error rate
    er_min = 0.0013
    er_max = 0.065  # 0.269
    exp_name = "Reid_Risk"
    pid = os.getpid()

    # input patient dataset
    original_patient_array = np.genfromtxt('data/' + original_patient_filename + '.csv', delimiter=',', skip_header=1)
    (n_patient, _) = original_patient_array.shape

    # input fake patient dataset
    if model == 'real':
        original_fake_array = original_patient_array
    else:
        fake_filename = prefix_syn + model + infix_syn + exp_id + suffix_syn
        original_fake_array = np.genfromtxt('data/' + fake_filename + '.csv', delimiter=',', skip_header=1)
    (n_fake, _) = original_fake_array.shape

    # input pop dataset
    original_pop_array = np.genfromtxt('data/' + pop_filename + '.csv', delimiter=',', skip_header=1)
    (n_pop, _) = original_pop_array.shape

    # preprocess datasets
    original_pop_array_qid = original_pop_array[:, 0:n_qid]
    original_patient_array_qid = original_patient_array[:, qid_index]
    original_fake_array_qid = original_fake_array[:, qid_index]
    patient_array_sense = original_patient_array[:, sense_index]
    fake_array_sense = original_fake_array[:, sense_index]

    MAX_LEVELS = [1, 4, 2] + [1] * n_phe_qid  # maximal generalization level for each QID
    generalization_mat = [[], [[0, 60], [0, 30, 60, 90], [i * 10 for i in range(12)]], [[0, 1, 2]]] + [[]] * n_phe_qid

    dic_replace = {}

    lattice = generate_lattice_bfs(MAX_LEVELS)

    dic_risk = {}
    n_lattice_nodes = len(lattice)
    risk_a = np.zeros((n_patient, n_lattice_nodes))
    risk_b = np.zeros((n_patient, n_lattice_nodes))
    if randomization:
        list_lamb = rand_lamb(n_patient)
    else:
        list_lamb = np.ones(n_patient)

    # clustering
    dic_cluster = {}
    dic_p = {}

    for j in range(n_sense_index):
        if not nom_sense[j]:
            patient_sense_j = patient_array_sense[:, j].reshape(-1, 1)
            range_n_clusters = [i + 2 for i in range(15)]
            list_silhouette = []
            silhouette_max = -1
            for num_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(patient_sense_j)
                cluster_labels = kmeans.labels_
                silhouette = silhouette_score(patient_sense_j, cluster_labels)
                list_silhouette.append(silhouette)
                print("[PID:" + str(pid) + " (" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")] (Attr:"
                      + str(j) + ") n_clusters, silhouette_score: " + str(num_clusters) + ", " + str(silhouette))
                if silhouette > silhouette_max:
                    final_cluster_labels = cluster_labels
                    silhouette_max = silhouette
            dic_cluster[j] = final_cluster_labels

    start2 = time.time()
    for i_lattice in range(n_lattice_nodes):
        levels = lattice[i_lattice]
        if i_lattice == 0:
            print("[PID:" + str(pid) + " (" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")] Levels (" + str(i_lattice) + "): " + str(levels) + " (0% completed)")
        else:
            progress = i_lattice / n_lattice_nodes
            remaining_time = (time.time() - start2) * (1 - progress) / progress
            remaining_mins = remaining_time // 60
            remaining_seconds = remaining_time % 60
            print("[PID:" + str(pid) + " (" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")] Levels (" + str(i_lattice) + "): " + str(levels) + " (" + str(progress * 100) + "% completed; finish in "
                  + str(remaining_mins) + " minutes " + str(remaining_seconds) + " seconds)")
        if sum(levels) == 0:
            pass
        else:
            pop_array_qid = replace_dataset(original_pop_array_qid, levels)
            patient_array_qid = replace_dataset(original_patient_array_qid, levels)
            fake_array_qid = replace_dataset(original_fake_array_qid, levels)
            for i in range(n_patient):
                if i % 1000 == 0:
                    print("[PID:" + str(pid) + "] Patient#: " + str(i) + " - " + str(i + 999))
                record_qid = patient_array_qid[i, :]
                tuple_qid_level = (tuple(record_qid), tuple(levels))
                if tuple_qid_level in dic_risk:
                    (risk_a[i, i_lattice], risk_b[i, i_lattice]) = dic_risk[tuple_qid_level]
                else:
                    group_size_patient = 0
                    group_size_pop = 0
                    match_in_fake = False
                    learn_sth_new = False
                    # compute I
                    distance = np.sum(np.absolute(fake_array_qid - record_qid), axis=1)
                    match_fake = distance == 0
                    match_in_fake = np.count_nonzero(match_fake) > 0
                    if match_in_fake:
                        # compute group size A
                        distance = np.sum(np.absolute(patient_array_qid - record_qid), axis=1)
                        match_patient = distance == 0
                        group_size_patient = np.count_nonzero(match_patient)
                        if group_size_patient > 0:
                            # compute R
                            new_info = 0
                            for j in range(len(sense_index)):
                                record_sense_j = patient_array_sense[i, j]
                                patient_sense_j = patient_array_sense[:, j]
                                fake_match_sense_j = fake_array_sense[match_fake, j]
                                if nom_sense[j]:
                                    if (record_sense_j, j) in dic_p:
                                        p = dic_p[(record_sense_j, j)]
                                    else:
                                        p = np.sum(record_sense_j == patient_sense_j) / n_patient
                                        dic_p[(record_sense_j, j)] = p
                                    d = 1 - p
                                    iverson = record_sense_j in fake_match_sense_j
                                    if d * iverson > np.sqrt(p * d):
                                        new_info += 1
                                else:
                                    final_cluster_labels = dic_cluster[j]
                                    if (final_cluster_labels[i], j) in dic_p:
                                        p = dic_p[(final_cluster_labels[i], j)]
                                    else:
                                        p = np.sum(final_cluster_labels == final_cluster_labels[i]) / n_patient
                                        dic_p[(final_cluster_labels[i], j)] = p
                                    ad = np.min(np.absolute(fake_match_sense_j - record_sense_j))
                                    mad = np.median(np.absolute(patient_sense_j - np.median(patient_sense_j)))
                                    if p * ad < 1.48 * mad:
                                        new_info += 1
                                if new_info >= theta * n_sense_index:
                                    learn_sth_new = True
                                    break
                            risk_a[i, i_lattice] = 1 / group_size_patient * match_in_fake * learn_sth_new * list_lamb[i]

                        else:
                            print("group size in the patient sample is zero!")
                            risk_a[i, i_lattice] = 0

                        # compute group size B
                        distance = np.sum(np.absolute(pop_array_qid - record_qid), axis=1)
                        match_pop = distance <= theta_distance_pop
                        group_size_pop = np.count_nonzero(match_pop)
                        if group_size_pop > 0:
                            risk_b[i, i_lattice] = 1 / group_size_pop * match_in_fake * learn_sth_new * list_lamb[i]
                        else:
                            print("[PID:" + str(pid) + "] patient #" + str(i) + ": " + str(record_qid) + " (level_id: " + str(i_lattice) + ") group size in the population is zero!")
                            risk_b[i, i_lattice] = 0
                        dic_risk[tuple_qid_level] = (risk_a[i, i_lattice], risk_b[i, i_lattice])
                    else:
                        risk_a[i, i_lattice] = 0
                        risk_b[i, i_lattice] = 0
                        dic_risk[tuple_qid_level] = (0, 0)
    risk_a_worse = np.amax(risk_a, axis=1)
    risk_b_worse = np.amax(risk_b, axis=1)
    sum_risk_a_worse = np.sum(risk_a_worse)
    sum_risk_b_worse = np.sum(risk_b_worse)
    result = max(1 / n_pop * sum_risk_a_worse, 1 / n_patient * sum_risk_b_worse)
    elapsed1 = (time.time() - start1)
    with open(Result_folder + exp_name + "_" + model + "_" + exp_id + "_qid" + str(n_qid) + "_theta" + str(theta) + ".txt", 'w') as f:
        f.write(str(result) + "\n")
        f.write("Time used: " + str(elapsed1) + " seconds.\n")
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1) + " seconds.")
