import os

path_folder_test = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\capture'
path_folder_test_python = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\python'

path_file_raw = os.path.join(path_folder_test, 'qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46.hdr')
path_file_black = os.path.join(path_folder_test, 'DARKREF_qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46.hdr')
path_file_white = os.path.join(path_folder_test, 'WHITEREF_qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46.hdr')

path_file_hdf5 = os.path.join(path_folder_test_python, 'calibrated.hdf5')
