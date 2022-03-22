import numpy as np
import time
from scipy.linalg import cholesky
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configuration

Result_folder = "Results_Synthetic_Reid/"
print(Result_folder)

randomization = True
qid_index = [0, 1, 2]
n_qid = len(qid_index)
n_index = 2592
sense_index = [i for i in range(n_qid, n_index)]
#sense_index.reverse()
n_sense_index = len(sense_index)
theta_L = 0.001
str_theta_L = str(int(theta_L*1000)) + 'e-3'
nom_sense = [True for i in range(n_sense_index - 7)] + [False for i in range(7)]
#nom_sense.reverse()
theta_distance_pop = 0

vr_mean = 0.23  # verification rate
vr_min = 0.1
vr_max = 0.43  # 0.61
er_mean = 0.0426  # data error rate
er_min = 0.0013
er_max = 0.065  # 0.269

# input patient dataset
original_patient_filename = 'train_raw_14349'
original_patient_array = np.genfromtxt('data/' + original_patient_filename + '.csv', delimiter=',', skip_header=1)
(n_r, _) = original_patient_array.shape
n_patient = n_r

# input fake patient dataset
original_fake_filename = 'round_syn_dpgan_3'
original_fake_array = np.genfromtxt('data/' + original_fake_filename + '.csv', delimiter=',', skip_header=1)
(n_r, _) = original_fake_array.shape
n_fake = n_r

# input pop dataset
original_pop_filename = 'VUMC_ALL_01112022_635668_newage'
original_pop_array = np.genfromtxt('data/' + original_pop_filename + '.csv', delimiter=',', skip_header=1)
(n_r, _) = original_pop_array.shape
n_pop = n_r

# preprocess datasets
original_pop_array_qid = original_pop_array[:, qid_index]
original_patient_array_qid = original_patient_array[:, qid_index]
original_fake_array_qid = original_fake_array[:, qid_index]
patient_array_sense = original_patient_array[:, sense_index]
fake_array_sense = original_fake_array[:, sense_index]

MAX_LEVELS = [1, 4, 2]  # maximal generalization level for each QID
generalization_mat = [[], [[0, 60], [0, 30, 60, 90], [i * 10 for i in range(12)]], [[0, 1, 2]]]
#subsets_levels = [1] * n_qid


dic_replace = {}
def replace_dataset(pop, level):
    (n_po, m_po) = pop.shape
    final_pop = pop.copy()
    if np.array_equal(level, np.asarray(MAX_LEVELS)):
        return final_pop
    elif np.array_equal(level, np.asarray([0, 0, 0])):
        return np.zeros(pop.shape)
    else:
        for i in range(n_po):
            new_pop_row = pop[i, :].copy()
            for j in range(len(level)):
                attr = j
                attr_value = new_pop_row[j]  # pop[i, j]
                attr_level = level[j]
                tuple_replace = (attr, attr_value, attr_level)
                if tuple_replace in dic_replace:
                    new_pop_row[j] = dic_replace[tuple_replace]
                    #print("hit dic_replace!")
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

lattice = generate_lattice_bfs(MAX_LEVELS)

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


start1 = time.time()

dic_risk = {}
n_lattice_nodes = len(lattice)
risk_a = np.zeros((n_patient, n_lattice_nodes))
risk_b = np.zeros((n_patient, n_lattice_nodes))
#new_info_tensor = np.zeros((n_patient, n_lattice_nodes, n_sense_index))
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
            #cluster_labels2 = kmeans.predict(patient_sense_j)
            silhouette = silhouette_score(patient_sense_j, cluster_labels)
            list_silhouette.append(silhouette)
            print("n_cluster - silhouette_score: " + str(num_clusters) + ', ' + str(silhouette))
            if silhouette > silhouette_max:
                final_cluster_labels = cluster_labels
                silhouette_max = silhouette
        dic_cluster[j] = final_cluster_labels


for i_lattice in range(n_lattice_nodes):
    levels = lattice[i_lattice]
    print("Levels: " + str(levels))
    if sum(levels) == 0:
        pass
    else:
        pop_array_qid = replace_dataset(original_pop_array_qid, levels)
        patient_array_qid = replace_dataset(original_patient_array_qid, levels)
        fake_array_qid = replace_dataset(original_fake_array_qid, levels)
        for i in range(n_patient):
            #new_info_vector = np.zeros(n_sense_index)
            if i % 1000 == 0:
                print("patient#: " + str(i))
            record_qid = patient_array_qid[i, :]
            tuple_qid_level = (tuple(record_qid), tuple(levels))
            if tuple_qid_level in dic_risk:
                (risk_a[i, i_lattice], risk_b[i, i_lattice]) = dic_risk[tuple_qid_level]
                #print("hit dic_risk!")
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
                                    #new_info_tensor[i, i_lattice, j] = 1
                                    #new_info_vector[j] = 1
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
                                    #new_info_tensor[i, i_lattice, j] = 1
                                    #new_info_vector[j] = 1
                                    new_info += 1
                            if new_info >= theta_L * n_sense_index:
                                learn_sth_new = True
                                break
                        risk_a[i, i_lattice] = 1 / group_size_patient * match_in_fake * learn_sth_new * list_lamb[i]
                        #if i == 3:
                            #np.savetxt(Result_folder + "new_info_vector.csv", new_info_vector, delimiter=",")
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
                        print("group size in the population is zero!")
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
with open(Result_folder + original_fake_filename + "_result_" + str_theta_L + ".txt", 'w') as f:
    f.write("Risk: " + str(result) + ".\n")
    f.write("Time used: " + str(elapsed1) + " seconds.\n")
print("Time used: " + str(elapsed1) + " seconds.\n")
print("Risk: " + str(result) + ".\n")

#np.savetxt(Result_folder + original_fake_filename + "_risk_a_" + str_theta_L + ".csv", risk_a, delimiter=",")
#np.savetxt(Result_folder + original_fake_filename + "_risk_b_" + str_theta_L + ".csv", risk_b, delimiter=",")
