import h5py
import os
import numpy as np

'''
This is the data remover for h5 files in order to reduce their size form an .h5 file
'''
def h5del(fname, datasets):
  with h5py.File(fname, "r+") as f:
    for d in datasets:
      if not d in f:
        raise ValueError("dataset {} does not exist in {}".format(d, fname))

    for d in datasets:
      del f[d]

if __name__ == '__main__':
  # to_delete = ['N', 'Re', 'kx', 'u', 'v', 'vort', 'x', 'xx', 'yy']
  # print(os.getcwd())
  # #path = 'main\\Kolmogorov_Re40.0_T6000_DT001_res33.h5' #This does not exist yet since the data was removed.
  # #print(list(h5py.File(path, "r+").keys()))
  # with h5py.File(path, "r+") as f:
  #   keys = list(f.keys())
  #   for key in keys:
  #     np_key = np.array(f[key])
  #     np.save(f'{key}', np_key)
  pass