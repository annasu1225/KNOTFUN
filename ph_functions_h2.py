import ripserplusplus as rpp_py
import numpy as np

data = np.loadtxt('human_protein_data_files/5BK8/5BK8_rec_bb.txt', skiprows=1, usecols=(1, 2, 3))
print("data=", data)

dict = rpp_py.run("--format point-cloud --sparse --dim 2 --threshold 1.4", data)
print(dict)