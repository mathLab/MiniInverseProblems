import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import numpy as np 
import sys
from scipy import linalg
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import seaborn as sns


#plt.style.use('classic')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

seedsX0 = np.loadtxt  ("./seedsX0_mat.txt") 
seedsX1 = np.loadtxt  ("./seedsX1_mat.txt") 
pdf = np.loadtxt  ("./pdf_mat.txt")         
output = np.loadtxt  ("./output_mat.txt")   
sampleMean = np.loadtxt  ("./sampleMean_mat.txt")   
autocovariance = np.loadtxt  ("./autocovariance_mat.txt")   

# Make the plot
g = plt.figure(1,figsize=(12,8))
plt.pcolormesh(seedsX0, seedsX1, pdf)
plt.colorbar()

g = plt.figure(2,figsize=(12,8))
plt.plot(output[1,:], output[0,:], 'o')
#plt.xlim(-2,2)
#plt.ylim(-2,2)

g = plt.figure(10,figsize=(12,8))
plt.plot(sampleMean[0,:], label = "x0")
plt.plot(sampleMean[1,:], label = "x1")
plt.legend()

g = plt.figure(11,figsize=(12,8))
plt.plot(autocovariance[0,:], label = "x0")
plt.plot(autocovariance[1,:], label = "x1")
plt.legend()
plt.show()

