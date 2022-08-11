# 04/01/2022 edited by Zhiyu Wan (in use)
# 06/07/2022 edited by Zhiyu Wan for attribute inference
import numpy as np

def prepare_data(file):
    fake_data = np.load(file)
    print("Loading data from: " + file)
    [n_row, n_col] = fake_data.shape
    fake_data2 = np.zeros((n_row, n_col - 5))
    for i in range(n_row):
        if i % 1000 == 0:
            print("loading patient#: " + str(i))
        for j in range(n_col - 5):
            if j == 0:
                fake_data2[i, j] = fake_data[i, 2599].astype(int)
            elif j == 1:
                if fake_data[i, 2597]:
                    fake_data2[i, j] = 1
                elif fake_data[i, 2598]:
                    fake_data2[i, j] = 2
                elif fake_data[i, 2600]:
                    fake_data2[i, j] = 3
                elif fake_data[i, 2601]:
                    fake_data2[i, j] = 4
                elif fake_data[i, 2602]:
                    fake_data2[i, j] = 5
            elif j == 2:
                fake_data2[i, j] = fake_data[i, n_col - 1].astype(int)
            elif j < 2599:
                fake_data2[i, j] = fake_data[i, j - 3].astype(int)
            else:
                fake_data2[i, j] = fake_data[i, j + 4].astype(int)
    return fake_data2


def prepare_data_vumc(file):
    fake_data = np.load(file)
    print("Loading data from: " + file)
    [n_row, n_col] = fake_data.shape
    fake_data2 = np.zeros((n_row, n_col - 4))
    for i in range(n_row):
        if i % 100 == 0:
            print("patient#: " + str(i))
        for j in range(n_col - 4):
            if j == 0:
                fake_data2[i, j] = np.floor(fake_data[i, 6] + 0.5).astype(int)
            elif j == 1:
                fake_data2[i, j] = np.floor(fake_data[i, n_col - 1]).astype(int)
            elif j == 2:
                if fake_data[i, 0] == 1:
                    fake_data2[i, j] = 1
                elif fake_data[i, 1] == 1:
                    fake_data2[i, j] = 2
                elif fake_data[i, 2] == 1:
                    fake_data2[i, j] = 3
                elif fake_data[i, 3] == 1:
                    fake_data2[i, j] = 4
            elif j == 3:
                fake_data2[i, j] = np.floor(fake_data[i, 5] + 0.5).astype(int)
            elif j >= n_col - 11:
                fake_data2[i, j] = fake_data[i, j + 3]
            else:
                fake_data2[i, j] = np.floor(fake_data[i, j + 3] + 0.5).astype(int)
    return fake_data2


def reorder_data(file):
    fake_data = np.load(file)
    print("Loading data from: " + file)
    [n_row, n_col] = fake_data.shape
    fake_data2 = np.zeros((n_row, n_col))
    for i in range(n_row):
        if i % 1000 == 0:
            print("loading patient#: " + str(i))
        for j in range(n_col):
            if j == 0:  # gender
                fake_data2[i, j] = fake_data[i, 2599].astype(int)
            elif j == 1:  # Asian
                fake_data2[i, j] = fake_data[i, 2596].astype(int)
            elif j == 2:  # Black
                fake_data2[i, j] = fake_data[i, 2597].astype(int)
            elif j == 3:  # White
                fake_data2[i, j] = fake_data[i, 2598].astype(int)
            elif j == 4:  # Unknown
                fake_data2[i, j] = fake_data[i, 2600].astype(int)
            elif j == 5:  # Pacific islander
                fake_data2[i, j] = fake_data[i, 2601].astype(int)
            elif j == 6:  # American Indian
                fake_data2[i, j] = fake_data[i, 2602].astype(int)
            elif j == 7:  # label
                fake_data2[i, j] = fake_data[i, n_col - 1].astype(int)
            elif j < 2604:  # phenotype part 1
                fake_data2[i, j] = fake_data[i, j - 8].astype(int)
            else:  # phenotype part 2
                fake_data2[i, j] = fake_data[i, j - 1].astype(int)
    return fake_data2
