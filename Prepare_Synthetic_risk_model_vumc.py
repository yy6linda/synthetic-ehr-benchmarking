import numpy as np
import time
# Configuration

filename = "round_syn_dpgan_3"
start1 = time.time()

f = open("data/header.csv", "r")
headline = f.readline().rstrip("\n")
f.close()
# input patient dataset
fake_data = np.load('data/' + filename + '.npy')
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

np.savetxt("data/" + filename + ".csv", fake_data2, header=headline, delimiter=",", comments='')
elapsed1 = (time.time() - start1)
print("Time used: " + str(elapsed1) + " seconds.\n")