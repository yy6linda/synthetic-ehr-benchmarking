# 04/01/2022 edited by Zhiyu Wan (in use)
# 06/07/2022 edited by Zhiyu Wan for attribute inference
import numpy as np

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