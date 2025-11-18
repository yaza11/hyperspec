import h5py

import numpy as np
from tqdm import tqdm

from hyperspec.calib.bil_reader import BilReader
from hyperspec.file_finder import FileFinder


def reshape_line_bil(line_bil, num_columns):
    return line_bil.reshape((-1, num_columns))


def get_sensor_wise_average(path_file_header: str, path_file_binary: str) -> np.ndarray:
    reader = BilReader(path_file_header, path_file_binary)
    avg = np.zeros(reader.num_bytes_per_line // reader.num_bytes_per_item, dtype=np.uint64)

    for l in tqdm(reader.get_iterable(), desc='Calcing average reflectance of reference'):
        avg += l
    return avg.astype(float) / reader.shape_roi[0]


def calib_spec(spec: np.ndarray, j: int, white_ref, black_ref):
    """Calibrate a row of pixels"""
    white = white_ref[j, :]
    black = black_ref[j, :]

    # scale channel-wise
    scaled = (spec - black) / (white - black)
    return scaled


def calib_row_fast(line_bil, white_avg_bil, dark_avg_bil):
    line_bil_float = line_bil.astype(float)
    cal = (line_bil_float - dark_avg_bil) / (white_avg_bil - dark_avg_bil)
    return np.clip(cal, 0, 1)


def resample_bil_line(line_bil, *, wavelengths, downsampling_factor=1):
    # TODO: ahndle cases where num vals is not divisible by downscale factor

    if downsampling_factor == 1:
        print('downsampling factor is 1')
        return line_bil

    assert downsampling_factor > 1
    assert isinstance(downsampling_factor, int)

    # take the mean / sum of neighbouring bands
    num_vals = line_bil.shape[0]
    num_bands = len(wavelengths)
    num_pixels_per_line = num_vals // num_bands

    dwl = np.median(wavelengths)
    new_min = wavelengths.min() + dwl * (downsampling_factor - 1)
    new_max = wavelengths.max() - dwl * (downsampling_factor - 1)

    wavelengths_new = np.linspace(new_min, new_max, round(num_bands / downsampling_factor))
    dwl = wavelengths_new[1] - wavelengths_new[0]

    new = np.zeros_like(wavelengths_new * num_pixels_per_line, dtype=line_bil.dtype)

    # add weighted means
    for wavelength in wavelengths:
        # TODO: find next smallest and biggest new wavelength, calc dist, add weighted
        ...

    return new.astype(float) / downsampling_factor


def write_calibrated_resampled_hdf(path_file_hdf5, b_reader: BilReader, white_ref_bil, dark_ref_bil):
    hdf = h5py.File(path_file_hdf5, 'w')

    hdf.create_dataset(
        name='roi_calibrated',
        dtype=np.float16,
        shape=b_reader.shape_roi,
        chunks=(1, round(np.sqrt(b_reader.shape_roi[1])), round(np.sqrt(b_reader.shape_roi[2])))  # chunk row-wise
    )

    for i, line in tqdm(enumerate(b_reader.get_iterable()), desc='writing hdf', total=b_reader.shape_roi[0]):
        c = calib_row_fast(line, white_ref_bil, dark_ref_bil)
        hdf['roi_calibrated'][i, :, :] = reshape_line_bil(c, b_reader.shape_roi[1]).T


if __name__ == '__main__':
    from _local import path_folder_iceland, path_file_hdf5_iceland
    path_folder = path_folder_iceland
    path_file_hdf5 = path_file_hdf5_iceland

    ff = FileFinder(path_folder)

    # ref_black = get_sensor_wise_average(ff.path_dark_ref_header_file)
    # ref_white = get_sensor_wise_average(ff.path_white_ref_header_file)
    #
    # full_img = open_image(ff.)
    #
    # n_rows = full_img.shape[0]
    # rows = np.arange(n_rows)
    #
    # for row in tqdm(rows, desc='calibrating ...', total=n_rows):
    #     calib = calib_spec(full_img[row, :, :], row, ref_white, ref_black)

    white_ref_bil = get_sensor_wise_average(ff.path_white_ref_header_file, ff.path_white_ref_binary_file)
    dark_ref_bil = get_sensor_wise_average(ff.path_dark_ref_header_file, ff.path_dark_ref_binary_file)

    b_reader = BilReader(
        path_file_header=ff.path_meas_header_file,
        path_file_binary=ff.path_meas_binary_file
    )
    b_reader.set_roi(1000, 1800, 100, 7600)

    # apply roi mask to references
    if b_reader.mask_roi_per_line is not None:
        white_ref_bil = white_ref_bil[b_reader.mask_roi_per_line]
        dark_ref_bil = dark_ref_bil[b_reader.mask_roi_per_line]

    # find downsampling factor
    i, j = b_reader.shape_roi[0] // 2, b_reader.shape_roi[1] // 2
    b_reader.set_roi(j, j + 1, i, i + 1)
    line = b_reader.get_next_line()
    spec = b_reader.get_spectrum(i, j, True)

    import matplotlib.pyplot as plt

    plt.figure()
    for dsf in [1, 2, 4]:
        spec_ds = resample_bil_line(line, num_pixels_per_line=b_reader.shape_roi[1], downsampling_factor=dsf)
        plt.plot(np.arange(spec_ds.shape[0]) * dsf, spec_ds)
    plt.show()


    # img0 = hdf['roi_calibrated'][:, :, 0]
    # plt.imshow(img0)
    # plt.show()
    #
    # plt.plot(hdf['roi_calibrated'][0, 0, :])
    # plt.show()