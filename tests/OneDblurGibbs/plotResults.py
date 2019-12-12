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
x_mean = np.loadtxt("./results/x_mean_mat.txt")
credibilityBounds = np.loadtxt("./results/credibilityBounds_mat.txt")
lambda_steady = np.loadtxt("./results/lambda_steady_mat.txt")
delta_steady = np.loadtxt("./results/delta_steady_mat.txt")
acfLambda = np.loadtxt("./results/acfLambda_mat.txt")
acfDelta = np.loadtxt("./results/acfDelta_mat.txt")

f = plt.figure(1,figsize=(12,8))
plt.plot(t, x_true, "k", linewidth=3, label="x true")
plt.plot(t, b, "ko", linewidth=3, label="b")
plt.legend()
plt.xlabel('t', fontsize=25)
plt.grid()

f = plt.figure(2,figsize=(12,8))
plt.plot(t, x_true, "b", linewidth=3, label="x true")
plt.plot(t, x_mean, "k", linewidth=3, label="x mean")
plt.plot(t, credibilityBounds[:,0], "k--", linewidth=2, label="95pc credibility bound")
plt.plot(t, credibilityBounds[:,1], "k--", linewidth=2)
plt.legend()
plt.xlabel('t', fontsize=25)
plt.grid()

f = plt.figure(3,figsize=(12,8))
plt.plot(lambda_steady, "b", linewidth=3)
plt.ylabel('lambda', fontsize=25)
plt.grid()

f = plt.figure(4,figsize=(12,8))
plt.plot(delta_steady, "b", linewidth=3)
plt.ylabel('delta', fontsize=25)
plt.grid()

f = plt.figure(5,figsize=(12,8))
plt.plot(delta_steady, lambda_steady, "ob", linewidth=2)
plt.ylabel('lambda', fontsize=25)
plt.ylabel('delta', fontsize=25)
plt.grid()

f = plt.figure(6,figsize=(12,8))
plt.plot(acfLambda, "k", linewidth=3, label=r"$\lambda$")
plt.plot(acfDelta, "--k", linewidth=3, label=r"$\delta$")
plt.title('ACFs', fontsize=25)
plt.xlim(0,40)
plt.ylim(0,1)
plt.legend()
plt.grid()

plt.show()
