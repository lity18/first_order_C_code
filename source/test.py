import numpy as np
from mpi4py import MPI
import time
import os
import socket
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal


data = np.loadtxt('output.txt')
plt.plot(data[:, 0], data[:, 1])
plt.savefig('test.png')