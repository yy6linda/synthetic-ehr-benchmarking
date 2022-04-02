# 04/01/2022 edited by Zhiyu Wan (in use)
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