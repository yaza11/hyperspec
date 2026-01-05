import os
import time

import h5py

import numpy as np
import zarr
from dask.distributed import Client
from distributed import as_completed
from tqdm import tqdm

from hyperspec.calib.bil_reader import BilReader
from hyperspec.file_finder import FileFinder


def reshape_line_bil(line_bil, num_columns):
    return line_bil.reshape((-1, num_columns))


def get_sensor_wise_average(path_file_header: str, path_file_binary: str, roi=None) -> np.ndarray:
    reader = BilReader(path_file_header, path_file_binary)
    if roi is not None:
        reader.set_roi(x_min=roi[0], x_max=roi[1])
    else:
        reader.set_roi()
    avg = np.zeros(reader._num_xs_roi * reader._num_bands, dtype=np.uint64)
    # if is_mask_provided := (mask_roi is not None):
    #     avg = avg[mask_roi]

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
    # TODO: handle cases where num vals is not divisible by downscale factor

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


def write_calibrated_resampled_hdf(
        path_file_hdf5: str,
        b_reader: BilReader,
        white_ref_bil: np.ndarray,
        dark_ref_bil: np.ndarray,
        dtype=np.float32
):
    def _calib_row(line_bil):
        # cal = (line_bil_float - dark_avg_bil) / (white_avg_bil - dark_avg_bil)
        cal = line_bil.astype(dtype) / span + offset
        return np.clip(cal, 0, 1)

    span = (white_ref_bil - dark_ref_bil).astype(dtype)
    offset = (-dark_ref_bil / span).astype(dtype)

    hdf = h5py.File(path_file_hdf5, 'w')

    hdf.create_dataset(
        name='roi_calibrated',
        dtype=dtype,
        shape=b_reader.shape_roi,
        chunks=(1, round(np.sqrt(b_reader.shape_roi[1])), round(np.sqrt(b_reader.shape_roi[2])))  # chunk row-wise
    )
    dset = hdf['roi_calibrated']

    num_columns = b_reader.shape_roi[1]
    for i, line in tqdm(enumerate(b_reader.get_iterable()), desc='writing hdf', total=b_reader.shape_roi[0]):
        c = _calib_row(line)
        # line_bil.reshape((-1, num_columns))
        # hdf['roi_calibrated'][i, :, :] = reshape_line_bil(c, b_reader.shape_roi[1]).T
        dset[i, :, :] = c.reshape((-1, num_columns)).T


def correct_save_to_zarr(
        imhdr_path: str,
        zarr_path: str,
        roi: tuple = None,
        white_file_path: str = None,
        dark_for_im_file_path: str = None,
        dark_for_white_file_path: str = None,
        band_indices=None,
        min_max_bands=None,
        downsample_bands=1,
        background_correction=True,
        destripe=False,
        savgol_width=100,
        savgol_order=2,
        use_dask=False,
        chunk_size=500,
        use_float=True
):
    def get_bil_reader():
        _bil_reader = BilReader(
            path_file_header=ff.path_meas_header_file,
            path_file_binary=ff.path_meas_binary_file
        )
        if roi is not None:
            _bil_reader.set_roi(*roi)
        return _bil_reader

    def correct_single_channel(idx_band_array, idx_band) -> None:
        # TODO: do we always need a new reader instance?
        _reader = get_bil_reader()
        # 2D array
        _channel = _reader.read_single_channel(idx_band=idx_band, show_progress=False)
        # calibrate
        _white_line_channel = _white_ref_roi[idx_band_array, :]
        _dark_line_channel = _dark_ref_roi[idx_band_array, :]
        _channel_calibrated = (_channel - _dark_line_channel) / (_white_line_channel - _dark_line_channel)
        # submit to zarr array
        zarr_arr[idx_band_array, :, :] = _channel_calibrated

    _folder = os.path.dirname(imhdr_path)
    _ff = FileFinder(_folder)
    bil_reader = get_bil_reader()

    _wavelengths = bil_reader.wavelengths_nm
    _num_columns = int(bil_reader._num_xs_roi)
    _num_lines = int(bil_reader._num_ys_roi)

    # each row contains values of a specific bands for all pixels in the ROI
    _white_ref_roi = get_sensor_wise_average(
        ff.path_white_ref_header_file, ff.path_white_ref_binary_file, roi=bil_reader.roi
    ).reshape((bil_reader._num_bands, -1))
    _dark_ref_roi = get_sensor_wise_average(
        ff.path_dark_ref_header_file, ff.path_dark_ref_binary_file, roi=bil_reader.roi
    ).reshape((bil_reader._num_bands, -1))

    downsample_bands = int(downsample_bands)

    if band_indices is not None:
        assert min_max_bands is None, 'band_indices and min_max_bands cannot be provided together'
        band_indices = np.array(band_indices)
        num_bands = len(band_indices)
    elif min_max_bands is not None:
        min_band = np.argmin(np.abs(_wavelengths - min_max_bands[0]))
        max_band = np.argmin(np.abs(_wavelengths - min_max_bands[1]))
        band_indices = np.arange(min_band, max_band + 1, downsample_bands)
        num_bands = len(band_indices)
    else:
        num_bands = bil_reader._num_bands
        band_indices = np.arange(0, num_bands, downsample_bands, dtype=int)
        num_bands = len(band_indices)
    band_indices.astype(int, copy=False)

    if use_float:
        dtype = 'f4'
    else:
        dtype = 'u2'

    zarr_arr = zarr.create_array(
        zarr_path,
        shape=(num_bands, _num_lines, _num_columns),
        shards=(1, chunk_size * (1 + _num_lines // chunk_size), chunk_size * (1 + _num_columns // chunk_size)),
        chunks=(1, chunk_size, chunk_size),
        dtype=dtype
    )

    if use_dask:
        with Client() as client:
            futures = client.map(
                correct_single_channel,
                range(len(band_indices)), band_indices,
                pure=False
            )

            for f in tqdm(as_completed(futures), total=len(futures), desc="correcting and saving to zarr"):
                try:
                    f.result()  # raises if the task failed
                finally:
                    f.release()  # free worker/scheduler memory
    else:
        for _idx_band_array, _idx_band in tqdm(
                enumerate(band_indices),
                desc="correcting and saving to zarr",
                total=len(band_indices)
        ):
            correct_single_channel(
                _idx_band_array, _idx_band
            )

    zarr_arr.attrs['metadata'] = {
        'wavelength': list(bil_reader.wavelengths_nm[band_indices]),
        'centers': list(bil_reader.wavelengths_nm[band_indices]),
        'roi': roi
    }


if __name__ == '__main__':
    from _local import path_folder_iceland, path_file_hdf5_iceland, path_folder_zarr_iceland

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

    roi = 1200, 1650, 290, 7600

    correct_save_to_zarr(
        imhdr_path=ff.path_meas_header_file, zarr_path=path_folder_zarr_iceland, roi=None, use_dask=False
    )

    # %% test if this is correct
    hyperzarr = zarr.open(path_folder_zarr_iceland, mode='r')

    bil_reader = BilReader(
        path_file_header=ff.path_meas_header_file,
        path_file_binary=ff.path_meas_binary_file
    )
    wavelengths = bil_reader.rgb_wavelengths
    idcs = np.array([np.argmin(np.abs(wv - bil_reader.wavelengths_nm)) for wv in wavelengths])

    rgb = hyperzarr[idcs, :, :]

    import matplotlib.pyplot as plt
    plt.imshow(np.moveaxis(rgb, 0, -1) / rgb.max())
    plt.show()

    # b_reader = BilReader(
    #     path_file_header=ff.path_meas_header_file,
    #     path_file_binary=ff.path_meas_binary_file
    # )
    # b_reader.set_roi(*roi)

    # white_ref_bil = get_sensor_wise_average(
    #     ff.path_white_ref_header_file, ff.path_white_ref_binary_file, roi=b_reader.roi)
    # dark_ref_bil = get_sensor_wise_average(
    #     ff.path_dark_ref_header_file, ff.path_dark_ref_binary_file, roi=b_reader.roi)
    #
    # write_calibrated_resampled_hdf(
    #     r"C:\Users\Yannick Zander\Downloads\calibrated.hdf5",
    #     b_reader,
    #     white_ref_bil,
    #     dark_ref_bil
    # )

    # # find downsampling factor
    # i, j = b_reader.shape_roi[0] // 2,  b_reader.shape_roi[1] // 2
    # spec = b_reader.get_spectrum(i, j, True)
    #
    # line = b_reader.get_line(i)
    # line_calib = calib_row_fast(line, white_ref_bil, dark_ref_bil)
    #
    # mask = b_reader.column_indices_bil_line_roi == (j + b_reader._roi_x_min)
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(b_reader.wavelengths_nm, spec)
    # plt.plot(b_reader.wavelengths_nm, line[mask])
    # plt.show()
    # plt.figure()
    # plt.plot(b_reader.wavelengths_nm, line_calib[mask])
    # plt.show()
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # for dsf in [1, 2, 4]:
    #     spec_ds = resample_bil_line(line, num_pixels_per_line=b_reader.shape_roi[1], downsampling_factor=dsf)
    #     plt.plot(np.arange(spec_ds.shape[0]) * dsf, spec_ds)
    # plt.show()

    # img0 = hdf['roi_calibrated'][:, :, 0]
    # plt.imshow(img0)
    # plt.show()
    #
    # plt.plot(hdf['roi_calibrated'][0, 0, :])
    # plt.show()
