from matplotlib import pyplot as plt
from tqdm import tqdm
from spectral import open_image

from test_iceland.paths import path_file_raw, path_file_black, path_file_white

x_roi = (1000, 1800)
y_roi = (200, 7800)


img = open_image(path_file_raw)
lams = img.bands.centers

height, width, channels = img.shape

for i in tqdm(range(height), total=height):
    if i < y_roi[0] or i > y_roi[1]:
        continue
    for j in range(width):
        if j < x_roi[0] or j > x_roi[1]:
            continue
        spec = img.read_pixel(i, j)
        plt.plot(lams, spec)
plt.savefig('temp.pdf')
plt.close()

