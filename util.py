import numpy as np

from spectral.io.bilfile import BilFile

default_bands = (201, 127, 64)  # red, green, blue


def get_rgb(img: BilFile):
    channels = [img.read_band(b) for b in default_bands]
    channels = [c / np.max(c) for c in channels]
    rgb = np.stack(channels, axis=-1)
    return rgb


if __name__ == '__main__':
    from spectral import open_image
    from test_iceland.paths import path_file_raw
    import matplotlib.pyplot as plt

    img = open_image(path_file_raw)
    rgb = get_rgb(img)

    plt.imshow(rgb)
    plt.minorticks_on()
    plt.grid(True, which='both', alpha=.2)
    plt.savefig('temp.pdf')
    plt.close()

    x_roi = (1000, 1800)
    y_roi = (200, 7800)

    plt.imshow(rgb[y_roi[0]:y_roi[1], x_roi[0]:x_roi[1], :])
    plt.minorticks_on()
    plt.grid(True, which='both', alpha=.2)
    plt.savefig('temp.pdf')
    plt.close()

