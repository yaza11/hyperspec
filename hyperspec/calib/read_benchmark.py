"""This is the time to beat"""
from tqdm import tqdm

from spectral import open_image
from hyperspec.calib.bil_reader import BilReader
from hyperspec.file_finder import FileFinder

ff = FileFinder(
    r"\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\capture")

b_reader = BilReader(
    path_file_header=ff.path_meas_header_file,
    path_file_binary=ff.path_meas_binary_file
)

for _ in tqdm(range(b_reader.num_rows)):
    l = b_reader.get_next_line()

img_ = open_image(ff.path_meas_header_file)

for j in tqdm(range(b_reader.num_rows)):
    for i in range(b_reader.num_columns):
        l = img_[i, j, :]