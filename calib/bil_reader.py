import sys
from typing import Literal

import numpy as np
from tqdm import tqdm

from calib.header import read_header

# https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html
ENVI_DATA_TYPE_TO_NP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64
}

BYTE_ORDER = {
    0: 'little',
    1: 'big'
}


class BilReader:
    _path_file_header: str = None
    _path_file_binary: str = None

    _open_file = None

    attrs: dict = None

    shape_format = ('height', 'width', 'bands')

    # defined from header file
    _interleave: Literal['bil', 'bsq', 'bip'] = None
    _byte_order: Literal['little', 'big'] = None
    _data_type: type = None
    _num_bytes_per_item: int = None
    _initial_offset: int = None
    wavelengths_nm: np.ndarray[float] = None

    _num_bands: int = None
    _num_rows: int = None
    _num_columns: int = None

    # inferred from data format
    _num_bytes_per_line: int = None
    _num_bytes_in_file: int = None

    # inferred from ROI
    _roi_x_min: int = None
    _roi_x_max: int = None
    _roi_y_min: int = None
    _roi_y_max: int = None

    rows_to_iterate: np.ndarray[int] = None
    _mask_roi_per_line: np.ndarray[bool] = None
    _offset_roi: int = None

    def __init__(
            self,
            path_file_header: str,
            path_file_binary: str,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None
    ):
        self._path_file_header = path_file_header
        self._path_file_binary = path_file_binary

        self.attrs = read_header(self._path_file_header)

        self._set_from_attrs()
        self._set_file_parsing()
        # can be called again later by the user
        self.set_roi(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def __del__(self):
        if self._open_file is not None:
            self._open_file.close()

    @property
    def shape(self):
        return self._num_rows, self._num_columns, self._num_bands

    @property
    def _column_indices_bil_line(self):
        # defined by interleave = 'bil'
        return np.array(list(range(self._num_columns)) * self._num_bands)

    @property
    def _wavelengths_bil_line(self):
        return np.repeat(self.wavelengths_nm, self._num_columns)

    def _set_from_attrs(self):
        self._interleave = self.attrs['interleave']

        if self._interleave != 'bil':
            raise NotImplementedError

        self._byte_order = BYTE_ORDER[int(self.attrs['byte order'])]
        self._data_type = ENVI_DATA_TYPE_TO_NP[int(self.attrs['data type'])]
        self._num_bytes_per_item = np.dtype(self._data_type).itemsize
        self._initial_offset = int(self.attrs.get('header offset', 0))
        self.wavelengths_nm = np.array(self.attrs['Wavelength'][1:-1].split(',')).astype(float)
        self.rgb_wavelengths = sorted([
            self.wavelengths_nm[int(idx)]
            for idx in self.attrs['default bands'][1:-1].split(',')
        ], reverse=True)
        # determines how 3D values are stored sequentially
        # BIL = band interleaved by line
        # BIP = band interleaved by pixel
        # BSQ = band sequential

        self._num_bands = int(self.attrs['bands'])
        self._num_rows = int(self.attrs['lines'])  # number of lines taken
        self._num_columns = int(self.attrs['samples'])  # number of sensors in a row

    def _set_file_parsing(self):
        # check that byte orders are the same
        if sys.byteorder != self._byte_order:
            # adjust how bytes are interpreted by swapping from little to big endian or vice versa
            self._data_type = np.dtype(self._data_type).newbyteorder()
        self._num_bytes_per_line = self._num_bytes_per_item * self._num_bands * self._num_columns
        self._num_bytes_in_file = self._num_bytes_per_line * self._num_columns

        self._open_file = open(self._path_file_binary, 'rb')

    def reset_seek(self):
        self._open_file.seek(self._initial_offset)

    @property
    def num_bytes_per_item(self):
        return self._num_bytes_per_item

    @property
    def num_bytes_per_line(self):
        return self._num_bytes_per_line

    def set_roi(self, x_min: int = None, x_max: int = None, y_min: int = None, y_max: int = None):
        """Indices of the ROI. min values are inclusive, max values exclusive"""
        assert (x_min is None) or (x_min < self._num_columns)
        assert (x_max is None) or (x_max < self._num_columns)
        assert (y_min is None) or (y_min < self._num_rows)
        assert (y_max is None) or (y_max < self._num_rows)

        self._roi_x_min = 0 if x_min is None else x_min
        self._roi_y_min = 0 if y_min is None else y_min
        self._roi_x_max = self._num_columns if x_max is None else x_max
        self._roi_y_max = self._num_rows if y_max is None else y_max

        _xs = self._column_indices_bil_line
        # leave at None if both x_min and x_max are not defined
        if (x_min is not None) or (x_max is not None):
            self._mask_roi_per_line = (_xs >= x_min) & (_xs < x_max)
        # define row mask
        self.rows_to_iterate = np.arange(self._roi_y_min, self._roi_y_max)
        # set the offset to the right row
        self._offset_roi = self._initial_offset + self._num_bytes_per_line * self._roi_y_min
        self._open_file.seek(self._offset_roi)

    @property
    def mask_roi_per_line(self):
        return self._mask_roi_per_line

    @property
    def shape_roi(self):
        return self._roi_y_max - self._roi_y_min, self._roi_x_max - self._roi_x_min, self._num_bands

    def get_next_line(self):
        # TODO: test if filter on bytes is faster
        line_data = self._open_file.read(self._num_bytes_per_line)
        if self._mask_roi_per_line is None:
            return np.frombuffer(line_data, dtype=self._data_type)
        return np.frombuffer(line_data, dtype=self._data_type)[self._mask_roi_per_line]

    def get_spectrum(self, i, j, in_mask=False):
        """this function is not aware of ROI"""
        # set to start of line
        offset = self._offset_roi + i * self._num_bytes_per_line
        if in_mask:
            j += self._roi_x_min
            offset += self._num_bytes_per_line * self._roi_y_min
        self._open_file.seek(offset)

        # select values at right index
        mask_spec = self._column_indices_bil_line == j
        if self._mask_roi_per_line is not None:
            mask_spec = mask_spec[self._mask_roi_per_line]
        vals = self.get_next_line()[mask_spec]

        return vals


    def get_iterable(self):
        return BilIterator(self)

    def read_single_channel(self, wavelength) -> np.ndarray[int]:
        idx_wavelength = np.argmin(np.abs(wavelength - self.wavelengths_nm))

        n_pixels_x = self._roi_x_max - self._roi_x_min
        # we need to jump between lines
        num_bytes_per_chunk = self._num_bytes_per_item * self._num_columns
        num_bytes_before_roi = self._num_bytes_per_item * self._roi_x_min

        jump_nbytes_before_line = idx_wavelength * num_bytes_per_chunk + num_bytes_before_roi

        # read data of all pixels in row for one specific band
        num_bytes = self._num_bytes_per_item * n_pixels_x

        img = np.zeros((self.rows_to_iterate.shape[0], n_pixels_x), dtype=self._data_type)

        offset = self._offset_roi
        for jdx, j in tqdm(enumerate(self.rows_to_iterate),
                           total=self.rows_to_iterate.shape[0],
                           desc=f'reading data for wavelength {self.wavelengths_nm[idx_wavelength]} nm'):
            self._open_file.seek(jump_nbytes_before_line + offset)
            img[jdx, :] = np.frombuffer(
                self._open_file.read(num_bytes),
                dtype=self._data_type
            )
            offset = self._num_bytes_per_line * j + self._offset_roi
        return (img / img.max() * 255).astype(np.uint8)

    def get_rgb_img(self):
        return np.stack([self.read_single_channel(wl) for wl in self.rgb_wavelengths], axis=-1)


class BilIterator:
    def __init__(self, bil_reader: BilReader):
        self._bil_reader = bil_reader
        self._bil_reader.reset_seek()
        self._current_line = self._bil_reader.rows_to_iterate[0]
        self._last_line = self._bil_reader.rows_to_iterate[-1]

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_line > self._last_line:
            raise StopIteration
        self._current_line += 1
        return self._bil_reader.get_next_line()

    def __len__(self):
        return self._bil_reader.rows_to_iterate.shape[0]


if __name__ == '__main__':
    from calib.file_finder import FileFinder
    import matplotlib.pyplot as plt

    ff = FileFinder(
        r"\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\capture")

    b_reader = BilReader(
        path_file_header=ff.path_meas_header_file,
        path_file_binary=ff.path_meas_binary_file
    )
    b_reader.set_roi(1000, 1800, 100, 7600)
    # b_reader.set_roi(0, 1000, 100, 7600)

    spec = b_reader.get_spectrum(4000, 500, True)

    # img = b_reader.get_rgb_img()
    #
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    # b_reader.set_roi(x_min=10, x_max=11, y_min=2700, y_max=2800)
    #
    # plt.figure()
    # for l in tqdm(b_reader.get_iterable()):
        # plt.plot(l)
        # pass
    # plt.show()
