import matplotlib.pyplot as plt
import numpy as np
import spectral
from spectral import mnf

from test_iceland.paths import path_file_raw
print('creating connection to file ...')
img = spectral.open_image(path_file_raw)
#
# x_roi = (1000, 1800)
# y_roi = (200, 7800)
#
# x_min, x_max = x_roi
# y_min, y_max = y_roi
#
# print('loading data ...')
# roi = img[y_min:y_max, x_min:x_max, :]
# print()


# use roi calibrated
import h5py
from test_iceland.paths import path_file_hdf5

print('creating connection to hdf5 file')
f = h5py.File(path_file_hdf5, 'r')
print("loading calibrated data ...")
roi = np.asarray(f['roi_calibrated'])

print('calcing stats ...')
signal = spectral.calc_stats(roi)
print('calcing noise ...')
noise = spectral.noise_from_diffs(roi)
print('calcing nmf ...')
mnfr = mnf(signal, noise)

# De-noise the data by eliminating NAPC components where SNR < 10.
# The de-noised data will be in the original coordinate space (at
# full dimensionality).
print('denoising ...')
denoised = mnfr.denoise(roi, snr=10)
print('creating plot ...')

# plot some of the eigenvectors (in columns according to documentation)
n_vecs = 10
plt.figure()
lams = img.bands.centers
for i in range(n_vecs):
    v = mnfr.napc.eigenvectors[:, i].copy()
    v -= v.min()
    v /= v.max()
    plt.plot(lams, v)
plt.savefig('temp.pdf')
plt.close()

# Reduce dimensionality, retaining top 50 NAPC components.
print('reducing dimensionality ...')
reduced = mnfr.reduce(roi, num=50)

i = 0
fig, axs = plt.subplots(ncols=2)
img = reduced[:, :, i]
axs[0].imshow(img, vmin=np.quantile(img, .05), vmax=np.quantile(img, .95))
axs[1].plot(mnfr.napc.eigenvectors[:, i], lams)
plt.savefig('temp.pdf')
plt.close()
i += 1

print('adding reduced data to hdf5 ...')
f = h5py.File(path_file_hdf5, 'a')
f.create_dataset(name='roi_reduced', data=reduced)
print('DONE!')
