import os
import sys

import numpy as np
from scipy.io import netcdf_file





folder = r"\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\capture"
file = os.path.join(folder, r"qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46.raw")
header =  os.path.join(folder, r'qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46.hdr')

def read_line_bil():
    attrs = read_header(header)

    byte_order = BYTE_ORDER[int(attrs['byte order'])]
    data_type = ENVI_DATA_TYPE_TO_NP[int(attrs['data type'])]
    bytes_per_item = np.dtype(data_type).itemsize
    offset = int(attrs.get('header offset', 0))
    # determines how 3D values are stored sequentially
    # BIL = band interleaved by line
    # BIP = band interleaved by pixel
    # BSQ = band sequential
    interleave = attrs['interleave']

    num_bands = int(attrs['bands'])
    num_rows = int(attrs['lines'])  # number of lines taken
    num_columns = int(attrs['samples'])  # number of sensors in a row

    # check that byte orders are the same
    if sys.byteorder != byte_order:
        # adjust how bytes are interpreted by swapping from little to big endian or vice versa
        data_type = np.dtype(data_type).newbyteorder()

    if interleave != 'bil':
        raise NotImplementedError


    # read a line
    with open(file, 'rb') as f:
        f.seek(offset)
        # calculate the number of bytes per line
        num_bytes_per_line = bytes_per_item * num_bands * num_columns
        line_data = f.read(num_bytes_per_line)
        # the first bytes_per_item * num_bands entries correspond to the values of the first band for the first line
        # TODO: mask for ROI ... sadly cannot easily skip reading in superflous pixels outside the ROI in this format
        values = np.frombuffer(line_data, dtype=data_type)


    return values

import matplotlib.pyplot as plt

plt.plot(specs_in_cols[:, 5])
plt.show()