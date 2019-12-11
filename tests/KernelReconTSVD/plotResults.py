import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import numpy as np
import sys

#plt.style.use('classic')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

b = np.loadtxt("./results/b_mat.txt")
t = np.loadtxt("./results/t_mat.txt")
x_true = np.loadtxt("./results/x_true_mat.txt")
x_filt = np.loadtxt("./results/xfilt_mat.txt")


f = plt.figure(1,figsize=(12,8))
plt.plot(t, x_true, "k", linewidth=3, label="x true")
plt.plot(t, b, "ko", linewidth=3, label="b")
plt.legend()
plt.xlabel('t', fontsize=25)
plt.grid()

f = plt.figure(2,figsize=(12,8))
plt.plot(t, x_true, "b-", linewidth=3, label="x true")
plt.plot(t, x_filt, "k-", linewidth=3, label="x filt")
plt.legend()
plt.xlabel('t', fontsize=25)
plt.grid()
plt.show()
