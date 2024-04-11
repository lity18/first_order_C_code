import numpy as np
from mpi4py import MPI
import time
import os
import socket
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
import yt




from yt.frontends.boxlib.data_structures import AMReXDataset

# ds = AMReXDataset("../setup/plt00000")
ds = yt.load("../setup/plt00100")
sl = yt.SlicePlot(ds, 2, ('boxlib', 'phi'))
#sl.annotate_grids(min_level=0, max_level=2)
sl.annotate_cell_edges(line_width=0.0003)
sl.save('test.png')

"""ds = yt.load("../setup/plt00040")
sl = yt.SlicePlot(ds, 2, ('boxlib', 'phi'))
#sl.annotate_grids(min_level=0, max_level=2)
sl.annotate_cell_edges(line_width=0.0003)
sl.save('test1.png')"""

"""data = np.loadtxt('output.txt')
plt.plot(data[:, 0], data[:, 1])
plt.savefig('test.png')"""