"""This is the time to beat"""
from tqdm import tqdm

try:
    from spectral import open_image
except ImportError:
    def open_image(*args, **kwargs):
        raise ImportError('spectral package is not installed')
from hyperspec.calib.bil_reader import BilReader
from hyperspec.file_finder import FileFinder

ff = FileFinder(
    r"\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\capture")

b_reader = BilReader(
    path_file_header=ff.path_meas_header_file,
    path_file_binary=ff.path_meas_binary_file
)

# for l in tqdm(b_reader.get_iterable()):
#     pass

img_ = open_image(ff.path_meas_header_file)
