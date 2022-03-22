import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(rc={'figure.figsize':(5,2)})  #(5,3.75)
n_ex = 3
n_model = 6
x = ['IWAE', 'medGAN', 'medBGAN', 'EMR-WGAN', 'medWGAN', 'DPGAN', 'Real']
y = np.zeros((n_ex, n_model + 1))
for i in range(n_ex):
    y[i, :] = np.genfromtxt('Results_Synthetic_Reid/Reid_Risk_Ex' + str(i+1) + '.csv', delimiter=',', skip_header=0)

y = y.ravel()
x = np.tile(x, n_ex)
dataset = pd.DataFrame({'Meaningful identity disclosure risk': y, 'Model': x})
sns.barplot(x='Meaningful identity disclosure risk', y='Model', data=dataset, palette='viridis')
plt.show()