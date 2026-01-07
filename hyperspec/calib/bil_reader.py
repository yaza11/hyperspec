import sys
from functools import cached_property
from typing import Literal

import numpy as np
from tqdm import tqdm

from hyperspec.calib.header import read_header
from hyperspec.file_finder import FileFinder

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

    # connection to .raw file
    _open_file = None

    # order of dimensions in the image cube
    shape_format = ('height', 'width', 'bands')

    # defined from header file
    #  attributes read from the header file
    attrs: dict = None
    #  defines order in which values are raveled in the raw file
    _interleave: Literal['bil', 'bsq', 'bip'] = None
    #  byte order for values in .raw file
    _byte_order: Literal['little', 'big'] = None
    #  data type of values in the .raw file. Usually integer-type
    _data_type: type = None
    #  how many bytes constitute a value in the raw file
    _num_bytes_per_item: int = None
    #  byte offsets need to be stored as uint64
    _initial_offset: np.uint64 = None
    #  wavelengths of the channels
    wavelengths_nm: np.ndarray[float] = None

    num_bands: int = None
    num_rows: int = None
    num_columns: int = None

    # inferred from data format
    _num_bytes_per_line: np.uint64 = None
    _num_bytes_in_file: np.uint64 = None

    # inferred from ROI
    #  indices defining the roi, max is not included
    roi_x_min: np.uint64 = None
    roi_x_max: np.uint64 = None
    roi_y_min: np.uint64 = None
    roi_y_max: np.uint64 = None
    #  number of values in the horizontal and vertical direction
    num_xs_roi: np.uint64 = None
    num_ys_roi: np.uint64 = None
    #  indices of lines in the roi
    rows_to_iterate: np.ndarray[np.uint64] = None
    #  mask that allows discarding values from a line that are not in the roi
    _mask_roi_per_line_values: np.ndarray[bool] = None
    #  mask that allows discarding bytes from a line that are not in the roi
    _mask_roi_per_line_bytes: np.ndarray[bool] = None
    #  number of bytes to skip before reaching the first line of the roi (includes initial offset)

    _offset_roi: np.uint64 = None
    #  the current line seek points to
    _current_line: int = None
    #  current byte offset (always at the start of a line)
    _offset: np.uint64 = None
    #
    _column_indices_bil_line_roi: np.ndarray[np.uint64] = None

    def __init__(
            self,
            path_file_header: str,
            path_file_binary: str,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None
    ):
        self._path_file_header: str = path_file_header
        self._path_file_binary: str = path_file_binary

        self.attrs: dict = read_header(self._path_file_header)

        self._set_from_attrs()
        self._set_file_parsing()
        # can be called again later by the user
        self.set_roi(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def __del__(self):
        if self._open_file is not None:
            self._open_file.close()

    @classmethod
    def from_filefinder(cls, file_finder: FileFinder, **kwargs):
        return cls(
            path_file_header=file_finder.path_meas_header_file,
            path_file_binary=file_finder.path_meas_binary_file,
            **kwargs
        )

    @classmethod
    def from_path_capture_folder(cls, path_capture_folder: str, **kwargs):
        _ff = FileFinder(path_capture_folder)
        return cls.from_filefinder(_ff, **kwargs)

    @property
    def shape(self):
        return self.num_rows, self.num_columns, self.num_bands

    @cached_property
    def _column_indices_bil_line(self):
        # defined by interleave = 'bil'
        return np.array(list(range(self.num_columns)) * self.num_bands)

    @property
    def column_indices_bil_line_roi(self):
        return self._column_indices_bil_line_roi

    @property
    def _wavelengths_bil_line(self):
        return np.repeat(self.wavelengths_nm, self.num_columns)

    def _set_from_attrs(self):
        # determines how 3D values are stored sequentially
        # BIL = band interleaved by line
        # BIP = band interleaved by pixel
        # BSQ = band sequential
        self._interleave: str = self.attrs['interleave']

        if self._interleave != 'bil':
            raise NotImplementedError

        self._byte_order: str = BYTE_ORDER[int(self.attrs['byte order'])]
        self._data_type: type = ENVI_DATA_TYPE_TO_NP[int(self.attrs['data type'])]
        self._num_bytes_per_item: np.uint64 = np.uint64(np.dtype(self._data_type).itemsize)
        self._initial_offset: np.uint64 = np.uint64(self.attrs.get('header offset', 0))
        self.wavelengths_nm: np.ndarray[float] = np.array(
            self.attrs['Wavelength'][1:-1].split(',')).astype(float)
        self.rgb_wavelengths: list[float] = sorted([
            self.wavelengths_nm[int(idx)]
            for idx in self.attrs['default bands'][1:-1].split(',')
        ], reverse=True)

        self.num_bands: np.uint64 = np.uint64(self.attrs['bands'])
        self.num_rows: np.uint64 = np.uint64(self.attrs['lines'])  # number of lines taken
        self.num_columns: np.uint64 = np.uint64(self.attrs['samples'])  # number of sensors in a row

    def _set_file_parsing(self):
        # check that byte orders are the same
        if sys.byteorder != self._byte_order:
            # adjust how bytes are interpreted by swapping from little to big endian or vice versa
            self._data_type = np.dtype(self._data_type).newbyteorder()
        self._num_bytes_per_line: np.uint64 = np.uint64(
            self._num_bytes_per_item * self.num_bands * self.num_columns
        )
        self._num_bytes_in_file: np.uint64 = np.uint64(
            self._num_bytes_per_line * self.num_columns
        )
        self._open_file = open(self._path_file_binary, 'rb')

    def reset_seek(self):
        self._offset = self._offset_roi
        self._open_file.seek(self._offset_roi)

    @property
    def num_bytes_per_item(self):
        return self._num_bytes_per_item

    @property
    def num_bytes_per_line(self):
        return self._num_bytes_per_line

    def set_roi(self, x_min: int = None, x_max: int = None, y_min: int = None, y_max: int = None):
        """Indices of the ROI. min values are inclusive, max values exclusive"""
        assert (x_min is None) or (x_min < self.num_columns)
        assert (x_max is None) or (x_max <= self.num_columns)
        assert (y_min is None) or (y_min < self.num_rows)
        assert (y_max is None) or (y_max <= self.num_rows)

        self.roi_x_min: np.uint64 = np.uint64(0 if x_min is None else x_min)
        self.roi_y_min: np.uint64 = np.uint64(0 if y_min is None else y_min)
        self.roi_x_max: np.uint64 = np.uint64(self.num_columns if x_max is None else x_max)
        self.roi_y_max: np.uint64 = np.uint64(self.num_rows if y_max is None else y_max)

        self.num_xs_roi = self.roi_x_max - self.roi_x_min
        self.num_ys_roi = self.roi_y_max - self.roi_y_min

        _xs_bytes = self._column_indices_bil_line  # this is for bytes
        _xs = np.arange(0, self.num_columns, dtype=np.uint64)
        # leave at None if both x_min and x_max are not defined
        if (x_min is not None) or (x_max is not None):
            self._mask_roi_per_line_values = (_xs >= x_min) & (_xs < x_max)
            self._mask_roi_per_line_bytes = (_xs_bytes >= x_min) & (_xs_bytes < x_max)
        else:
            self._mask_roi_per_line_bytes = None
            self._mask_roi_per_line_values = None

        # define row mask
        self.rows_to_iterate: np.ndarray[np.uint64] = np.arange(
            self.roi_y_min, self.roi_y_max, dtype=np.uint64
        )
        # set the offset to the right row
        self._offset_roi: np.uint64 = np.uint64(
            self._initial_offset + self._num_bytes_per_line * self.roi_y_min
        )
        self._offset = self._offset_roi

        self._column_indices_bil_line_roi = np.array(
            list(np.arange(self.roi_x_min, self.roi_x_max, dtype=np.uint64)) * self.num_bands
        )

    @property
    def roi(self):
        return self.roi_x_min, self.roi_x_max, self.roi_y_min, self.roi_y_max

    @property
    def mask_roi_per_line_values(self):
        return self._mask_roi_per_line_values

    @property
    def mask_roi_per_line_bytes(self):
        return self._mask_roi_per_line_bytes

    @property
    def shape_roi(self):
        return self.roi_y_max - self.roi_y_min, self.roi_x_max - self.roi_x_min, self.num_bands

    def read_pixels_for_row_and_band(
            self,
            idx_row,
            idx_band,
            idx_pixel_min=None,
            idx_pixel_max=None
    ):
        """Reads values of all pixels for the specified row and band. This is in accordance with the BIL format."""
        if idx_pixel_min is None:
            idx_pixel_min = self.roi_x_min
        if idx_pixel_max is None:
            idx_pixel_max = self.roi_x_max
        idx_pixel_min = np.uint64(idx_pixel_min)
        idx_pixel_max = np.uint64(idx_pixel_max)

        # calculate offset from idcs
        offset: np.uint64 = (
                self._initial_offset +  # offset from file reserved for header
                np.uint64(idx_row) * self._num_bytes_per_line +  # shift by n rows to start of ROI
                np.uint64(idx_band) * self.num_columns * self._num_bytes_per_item +  # shift to the right band
                idx_pixel_min * self._num_bytes_per_item  # within band chunk, shift to first pixel
        )

        self._open_file.seek(offset)
        bytes = self._open_file.read((idx_pixel_max - idx_pixel_min) * self._num_bytes_per_item)
        # values = np.fromfile(
        #     self._path_file_binary,
        #     dtype=self._data_type,
        #     offset=offset,
        #     count=idx_pixel_max - idx_pixel_min
        # )

        # frombuffer is much faster than fromfile in this case
        return np.frombuffer(bytes, dtype=self._data_type)

    def get_line(self, idx_line: int):
        """Get values for all bands in a line of the ROI"""
        values = np.zeros(
            self.num_bands * self.num_xs_roi.astype(self.num_bands.dtype), dtype=self._data_type
        )
        for idx_band in range(self.num_bands):
            idx_expr = np.index_exp[(idx_band * self.num_xs_roi):((idx_band + 1) * self.num_xs_roi)]
            values[idx_expr] = self.read_pixels_for_row_and_band(
                idx_row=idx_line,
                idx_band=idx_band
            )
        return values

    def get_next_line(self):
        if self._current_line is None:
            self._current_line = self.roi_y_min
        else:
            self._current_line += 1
        line = self.get_line(self._current_line)
        return line

    def get_spectrum(self, i, j, in_mask=False):
        # set to start of line
        offset: np.uint64 = self._offset_roi + np.uint64(i) * self._num_bytes_per_line
        if in_mask:  # if indices refer to offset in mask, we need to adjust the absolute values
            assert i <= (self.roi_y_max - self.roi_y_min)
            assert j <= (self.roi_x_max - self.roi_x_min)
            j += self.roi_x_min
            offset += self._num_bytes_per_line * self.roi_y_min

        # select values at right index
        mask_spec = self.column_indices_bil_line_roi == j
        # if self._mask_roi_per_line is not None:
        #     mask_spec = mask_spec[self._mask_roi_per_line]
        vals = self.get_line(idx_line=i)[mask_spec]
        return vals

    def get_iterable(self):
        return BilIterator(self)

    def read_single_channel(
            self,
            wavelength=None,
            idx_band=None,
            convert_to_uint8=False,
            show_progress=True
    ) -> np.ndarray:
        assert (wavelength is not None) or (idx_band is not None), 'provide either wavelength or idx_band'
        if idx_band is None:
            idx_band = np.argmin(np.abs(wavelength - self.wavelengths_nm))

        img = np.zeros((self.rows_to_iterate.shape[0], self.num_xs_roi), dtype=self._data_type)

        for jdx, j in tqdm(enumerate(self.rows_to_iterate),
                           total=self.rows_to_iterate.shape[0],
                           desc=f'reading data for wavelength {self.wavelengths_nm[idx_band]} nm',
                           disable=not show_progress):
            img[jdx, :] = self.read_pixels_for_row_and_band(
                idx_row=j,
                idx_band=idx_band
            )

        if convert_to_uint8:
            return (img / img.max() * 255).astype(np.uint8)
        return img

    def get_rgb_img(self):
        return np.stack(
            [
                self.read_single_channel(wl, convert_to_uint8=True)
                for wl in self.rgb_wavelengths
            ], axis=-1
        )


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
    import matplotlib.pyplot as plt
    from hyperspec.file_finder import FileFinder

    ff = FileFinder(
        r"\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\capture")

    b_reader = BilReader(
        path_file_header=ff.path_meas_header_file,
        path_file_binary=ff.path_meas_binary_file
    )
    b_reader.set_roi(1200, 1650, 290, 7600)
    # b_reader.set_roi(0, 1000, 100, 7600)

    spec = b_reader.get_spectrum(4000, 200, True)
    plt.figure()
    plt.plot(b_reader.wavelengths_nm, spec)
    plt.show()

    # img = b_reader.get_rgb_img()
    img = b_reader.read_single_channel(400, False)
    plt.figure()
    plt.imshow(img)
    plt.show()

    # # b_reader.set_roi(x_min=10, x_max=11, y_min=2700, y_max=2800)
    # #
    # # plt.figure()
    # for l in tqdm(b_reader.get_iterable()):
    #     # plt.plot(l)
    #     pass
    # # plt.show()
