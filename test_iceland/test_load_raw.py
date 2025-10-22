import numpy as np
import matplotlib.pyplot as plt

from spectral import open_image

from test_iceland.paths import path_file_raw, path_file_black, path_file_white

img = open_image(path_file_raw)

black = open_image(path_file_black).load()
background = black.sum(axis=0)
white = open_image(path_file_white).load()
normalization = white.sum(axis=0)

band_idx = 201  # red
wavelength = img.bands.centers[band_idx]

# pixel
pixel_coords = (2700, 600)
pixel_idcs = np.index_exp[pixel_coords[0], pixel_coords[1], :]

p = np.squeeze(img[pixel_idcs])
p_black = np.squeeze(black[:, pixel_idcs[1], :].mean(axis=0))
p_white = np.squeeze(white[:, pixel_idcs[1], :].mean(axis=0))
p_scaled = (p - p_black) / (p_white - p_black)

# image
i = np.squeeze(img[:, :, band_idx])
i_black = np.squeeze(black[:, :, band_idx].mean(axis=0))
i_white = np.squeeze(white[:, :, band_idx].mean(axis=0))
i_scaled = (i - i_black) / (i_white - i_black)

fig, axs = plt.subplots(nrows=4, layout='constrained')

axs[0].plot(img.bands.centers, p, label='raw')
axs[0].plot(img.bands.centers, p_black, label='raw black', c='k')
axs[0].plot(img.bands.centers, p_white, label='raw white', c='grey')
axs[0].vlines(wavelength, ymin=0, ymax=p_white.max(), colors='r')
axs[0].set_title('raw')
axs[0].legend()

axs[1].plot(img.bands.centers, p_scaled)
axs[1].vlines(wavelength, ymin=0, ymax=p_scaled.max(), colors='r')
axs[1].set_xlabel('Wavelength in nm')
axs[1].set_ylabel('Intensity')
axs[1].set_title('normalized')

axs[2].imshow(i.T)
axs[2].scatter([pixel_coords[0]], [pixel_coords[1]], c='red')
axs[2].set_title('raw')
axs[3].imshow(i_scaled.T)
axs[3].scatter([pixel_coords[0]], [pixel_coords[1]], c='red')
axs[3].set_title('normalized')

plt.savefig('temp.pdf')
plt.close()
