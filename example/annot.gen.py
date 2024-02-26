import numpy as np
import pandas as pd

M = 5000 ## Feature size for sim

Nset = 3

annot_interval = [[0,30],[30, 60], [60,90]]



annots = []
for interval in annot_interval:
    annot = np.zeros(M)
    start, end = interval
    annot[start:end] = 1
    annots.append(annot)

annots = np.array(annots).T # transpose it -- row denote SNP index

np.savetxt('sim.annot',annots, fmt="%i")
    