import h5py
import numpy as np

from test_iceland.paths import path_file_hdf5

print('creating connection to hdf5 file ...')
f = h5py.File(path_file_hdf5, 'r')

print('loading reduced data ...')
reduced = np.asarray(f['roi_reduced'])

from spectral import ppi

ppis = ppi(reduced, 100, centered=True)
print(f'identified: {dict(zip(*np.unique(ppis, return_counts=True)))}')


import matplotlib.pyplot as plt

plt.figure()
plt.imshow(ppis, interpolation='none')
plt.savefig('temp.pdf')

idx = 4
rep = reduced[ppis == idx].mean(axis=0)

plt.figure()
plt.plot(rep)
plt.savefig('temp.pdf')
