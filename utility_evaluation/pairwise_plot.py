import numpy as np
import matplotlib.pyplot as plt

real_pos = np.load('training_pos.npy')
real_neg = np.load('training_neg.npy')
real = np.concatenate([real_pos, real_neg])
real_data = [real_pos,real_neg,real]
syn_list = ['iwae', 'medbgan', 'emrwgan', 'medgan']
fig, axs = plt.subplots(4, 3, figsize = (15, 20))
for run in range(0,5):
    for i in range(0,4):
        pos = np.load(f'{syn_list[i]}_pos_{run}.npy')
        pos = pos.round()
        neg = np.load(f'{syn_list[i]}_neg_{run}.npy')
        neg = neg.round()
        cob = np.load(f'{syn_list[i]}_combined_{run}.npy')
        cob = cob[:,:-1]
        cob = cob.round()
        data = [pos, neg, cob]
        title = ['positive', 'negative', 'combined']
        for j in range(0,3):
            axs[i,j].scatter(data[j].mean(axis = 0),real_data[j].mean(axis = 0), s = 8)
            axs[i,j].set_title(f'{title[j]}, {syn_list[i]} v.s. real', fontsize = 10)
            axs[i,j].set_xlim([0, 1])
            axs[i,j].set_ylim([0, 1])
    plt.savefig(f'./syn_pairwise_{run}.png', dpi = 100)