import os


class FileFinder:
    _white_ref_header = None
    _white_ref_binary = None

    _dark_ref_header = None
    _dark_ref_binary = None

    _meas_header = None
    _meas_binary = None

    def __init__(self, path_folder):
        self.path_folder = path_folder

        self._set_files()

    def _set_files(self):
        self._set_white_header_file()
        self._set_dark_header_file()
        self._set_meas_header_file()

        self._set_white_binary_file()
        self._set_dark_binary_file()
        self._set_meas_binary_file()

    def _find_by_start_end(self, start=None, end=None):
        files = os.listdir(self.path_folder)
        file = [
            f for f in files
            if ((start is None) or f.startswith(start)) and
               ((end is None) or f.endswith(end))
        ]
        assert (_n := len(file)) == 1, f'found {_n} files with {start=} and {end=}'
        return file[0]

    def _set_white_header_file(self):
        self._white_ref_header = self._find_by_start_end('WHITEREF', '.hdr')

    @property
    def white_ref_header(self):
        return self._white_ref_header

    @property
    def path_white_ref_header_file(self):
        return os.path.join(self.path_folder, self._white_ref_header)

    def _set_dark_header_file(self):
        self._dark_ref_header = self._find_by_start_end('DARKREF', '.hdr')

    @property
    def dark_ref_header(self):
        return self._dark_ref_header

    @property
    def path_dark_ref_header_file(self):
        return os.path.join(self.path_folder, self._dark_ref_header)


    def _set_meas_header_file(self):
        files = os.listdir(self.path_folder)
        file = [f for f in files if
                (not (f.startswith('WHITEREF') or f.startswith('DARKREF'))) and f.endswith('.hdr')]
        assert (_n := len(file)) == 1, f'found {_n} files for header file'
        self._meas_header = file[0]

    @property
    def meas_ref_header(self):
        return self._meas_header

    @property
    def path_meas_header_file(self):
        return os.path.join(self.path_folder, self._meas_header)


    def _set_white_binary_file(self):
        self._white_ref_binary = self._find_by_start_end('WHITEREF', '.raw')

    @property
    def white_ref_binary(self):
        return self._white_ref_binary

    @property
    def path_white_ref_binary_file(self):
        return os.path.join(self.path_folder, self._white_ref_binary)

    def _set_dark_binary_file(self):
        self._dark_ref_binary = self._find_by_start_end('DARKREF', '.raw')

    @property
    def dark_ref_binary(self):
        return self._dark_ref_binary

    @property
    def path_dark_ref_binary_file(self):
        return os.path.join(self.path_folder, self._dark_ref_binary)


    def _set_meas_binary_file(self):
        files = os.listdir(self.path_folder)
        file = [f for f in files if
                (not (f.startswith('WHITEREF') or f.startswith('DARKREF'))) and f.endswith('.raw')]
        assert (_n := len(file)) == 1, f'found {_n} files for binary file'
        self._meas_binary = file[0]

    @property
    def meas_binary(self):
        return self._meas_binary

    @property
    def path_meas_binary_file(self):
        return os.path.join(self.path_folder, self._meas_binary)