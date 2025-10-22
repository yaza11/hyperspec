"""
Module for hooking into napari-sediment-processed data.

Disclaimer: some methods and classes in here are directly copied from napari-sediment
TODO: import those functions properly or something similar
"""
from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml
import numpy as np
import zarr
import dask.array as da
from spectral import open_image, remove_continuum, kmeans
from tifffile import tifffile


@dataclass
class Param:
    """
    Class for keeping track of processing parameters.

    Paramters
    ---------
    project_path: str
        path where the project is saved
    file_paths: list[str]
        list of paths of files belonging to the project
    dark_for_im_path: str
        path of dark image for the files
    dark_for_white_path: str
        path of dark image for the white image
    main_roi: array
        main roi
    rois: dict of arrays
        flat list of rois
    measurement_roi: array
        roi for measurement. Note that these coordinates are relative
        to the main roi and not the image.
    scale: float
        scale of the image in scale_units/px
    scale_units: str
        units of the scale
    location: str
        location of the sample
    rgb: list
        list of rgb bands

    """
    project_path: str = None
    file_path: str = None
    white_path: str = None
    dark_for_im_path: str = None
    dark_for_white_path: str = None
    main_roi: list = field(default_factory=list)
    rois: list = field(default_factory=list)
    measurement_roi: list = field(default_factory=list)
    scale: float = 1
    scale_units: str = 'mm'
    location: str = ''
    rgb: list = field(default_factory=list)

    def __post_init__(self):
        self.rgb = [640, 545, 460]

    def save_parameters(self, alternate_path=None):
        """Save parameters as yml file.

        Parameters
        ----------
        alternate_path : str or Path, optional
            place where to save the parameters file.

        """

        if alternate_path is not None:
            save_path = Path(alternate_path).joinpath("Parameters.yml")
        else:
            save_path = Path(self.project_path).joinpath("Parameters.yml")

        with open(save_path, "w") as file:
            dict_to_save = dataclasses.asdict(self)
            for path_name in ['project_path', 'file_path', 'white_path', 'dark_for_im_path', 'dark_for_white_path']:
                if dict_to_save[path_name] is not None:
                    if not isinstance(dict_to_save[path_name], str):
                        dict_to_save[path_name] = dict_to_save[path_name].as_posix()

            yaml.dump(dict_to_save, file)

    def format_measurement_roi(self):
        """Make sure that the measurement roi is formatted correctly
        and that empy rois are replaced with defaults, i.e. original sub-rois."""

        mainroi = self.get_formatted_main_roi()
        subrois = self.get_formatted_subrois()

        if self.measurement_roi == []:
            self.measurement_roi = [[] for x in range(len(mainroi))]

        for i in range(len(mainroi)):
            if self.measurement_roi[i] == []:
                row_bounds = [
                    subrois[i][0][:, 0].min() - mainroi[i][:, 0].min(),
                    subrois[i][0][:, 0].max() - mainroi[i][:, 0].min()]
                col_bounds = [
                    subrois[i][0][:, 1].min() - mainroi[i][:, 1].min(),
                    subrois[i][0][:, 1].max() - mainroi[i][:, 1].min()]

                roi_square = np.array([
                    row_bounds[0], col_bounds[0],
                    row_bounds[1], col_bounds[0],
                    row_bounds[1], col_bounds[1],
                    row_bounds[0], col_bounds[1]
                ])
                self.measurement_roi[i] = list(roi_square)
                self.measurement_roi[i] = [x.item() for x in self.measurement_roi[i]]

    def get_formatted_main_roi(self):
        """Get mainroi as formatted array."""

        return np.array([np.array(x).reshape(4, 2) for x in self.main_roi]).astype(int)

    def get_formatted_measurement_roi(self):
        """Get mainroi as formatted array."""

        self.format_measurement_roi()
        return np.array([np.array(x).reshape(4, 2) for x in self.measurement_roi]).astype(int)

    def get_formatted_subrois(self):
        """Get rois as formatted arrays."""

        return [[np.array(x).reshape(4, 2) for x in y] for y in self.rois]

    def get_formatted_col_row_bounds(self, mainroi_index):
        """Get column and row bounds of the mainroi.

        Parameters
        ----------
        mainroi_index : int
            index of the mainroi to get bounds for.

        Returns
        -------
        row_bounds : list
            row bounds [rmin, rmax]
        col_bounds : list
            column bounds [cmin, cmax]

        """

        mainroi = self.get_formatted_main_roi()
        row_bounds = [
            mainroi[mainroi_index][:, 0].min(),
            mainroi[mainroi_index][:, 0].max()]
        col_bounds = [
            mainroi[mainroi_index][:, 1].min(),
            mainroi[mainroi_index][:, 1].max()]

        return row_bounds, col_bounds

    def get_formatted_rois(self):
        """Get rois as formatted arrays."""

        mainroi = self.get_formatted_main_roi()
        subrois = self.get_formatted_subrois()
        measurement_roi = self.get_formatted_measurement_roi()

        return mainroi, subrois, measurement_roi


def load_params_yml(params, file_name='Parameters.yml'):
    if not Path(params.project_path).joinpath(file_name).exists():
        raise FileNotFoundError(f"Project {params.project_path} does not exist")

    with open(params.project_path.joinpath(file_name)) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(params, k, documents[k])

    return params

def load_project_params(folder):
    """Load project parameters from yaml file in a given folder."""

    folder = Path(folder)
    params = Param(project_path=folder)
    params = load_params_yml(params)
    return params


@dataclass
class ImChannels:
    """
    Class for handling partial import of HDR images.

    Paramters
    ---------
    imhdr_path: str
        path where the project is saved
    channels: list of str
        list of available channels
    rois: list of arrays
        current roi of each image, None means full image
    channel_array: list of arrays
        current array of each channel. Can contain different rois
    metadata: dict
        metadata of the image
    nrows: int
        number of rows in the image
    ncols: int
        number of columns in the image
    centers: array of float
        band centers of the channels

    """
    imhdr_path: str = None
    channels: list[str] = None
    rois: list[list] = None
    channel_array: list[list] = None
    metadata: dict = field(default_factory=dict)
    nrows: int = None
    ncols: int = None
    centers: np.ndarray = None

    def __post_init__(self):

        data, metadata = read_spectral(
            path=self.imhdr_path,
            bands=[0],
            row_bounds=None,
            col_bounds=None,
        )
        self.channel_names = metadata['wavelength']
        self.rois = [None] * len(self.channel_names)
        self.channel_array = [None] * len(self.channel_names)
        self.channel_array[0] = data[:, :, 0]
        self.metadata = metadata
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]
        self.centers = np.array(metadata['centers'])

    def read_channels(self, channels=None, roi=None):
        """
        Get channels from the image.

        Parameters
        ----------
        channels: list of int
            indices of channel to get
        roi: array
            [row_start, row_end, col_start, col_end], None means full image

        """

        if channels is None:
            raise ValueError('channels must be provided')

        channels_full_image = []
        channels_partial_image = []
        for channel in channels:
            if roi is None:
                if self.rois[channel] is None:
                    if self.channel_array[channel] is None:
                        channels_full_image.append(channel)
                else:
                    channels_full_image.append(channel)
            else:
                # if a new roi is provided, reload the channel even if full frame is already loaded
                if self.rois[channel] is None:
                    # if self.channel_array[channel] is None:
                    channels_partial_image.append(channel)
                else:
                    if not np.array_equal(roi, self.rois[channel]):
                        channels_partial_image.append(channel)

        if len(channels_full_image) > 0:
            data, _ = read_spectral(
                path=self.imhdr_path,
                bands=channels_full_image,
                row_bounds=None,
                col_bounds=None,
            )
            for ind, c in enumerate(channels_full_image):
                self.channel_array[c] = data[:, :, ind]
                self.rois[c] = None

        if len(channels_partial_image) > 0:
            data, _ = read_spectral(
                path=self.imhdr_path,
                bands=channels_partial_image,
                row_bounds=[roi[0], roi[1]],
                col_bounds=[roi[2], roi[3]],
            )
            for ind, c in enumerate(channels_partial_image):
                self.channel_array[c] = data[:, :, ind]
                self.rois[c] = roi

    def get_image_cube(self, channels=None, roi=None):
        """
        Get image stack containing the selected channels indices.

        Parameters
        ----------
        channels: list of int
            indices of channel to get
        roi: array
            [row_start, row_end, col_start, col_end], None means full image

        Returns
        -------
        data: array
            array of shape (n_channels, n_rows, n_cols)

        """

        if channels is None:
            channels = np.arange(len(self.channel_names))

        # make sure data is loaded
        self.read_channels(channels, roi)

        # get data
        if roi is None:
            data = np.stack([self.channel_array[c] for c in channels], axis=0)
        else:
            full = [0, self.nrows, 0, self.ncols]
            data = []
            for ind, r in enumerate(roi):
                if r is None:
                    roi[ind] = full[ind]
            '''data = np.zeros(
                shape=(len(channels), roi[1]-roi[0], roi[3]-roi[2]),
                dtype=self.channel_array[channels[0]].dtype)
            for ind, c in enumerate(channels):
                if self.rois[c] is None:
                    data[ind,:,:] = self.channel_array[c][roi[0]:roi[1], roi[2]:roi[3]]
                else:
                    data[ind,:,:] = self.channel_array[c]'''
            for ind, c in enumerate(channels):
                if self.rois[c] is None:
                    data.append(self.channel_array[c][roi[0]:roi[1], roi[2]:roi[3]])
                else:
                    data.append(self.channel_array[c])
            data = np.stack(data, axis=0)

        return data

    def get_image_cube_bands(self, bands, roi=None):
        """
        Get image stack containing the selected bands.

        Parameters
        ----------
        bands: list of float
            list of band values for which to find the index of the closest
            bands in the dataset

        Returns
        -------
        data: array
            array of shape (n_channels, n_rows, n_cols)

        """

        bands_indices, _ = self.get_indices_of_bands(bands)
        data = self.get_image_cube(channels=bands_indices, roi=roi)

        return data

    def get_indices_of_bands(self, bands):
        """
        Given the bands centers of the dataset and a set of band values to recover
        find the indices of the closest bands in the dataset. E.g if the dataset
        has bands [450, 500, 550, 600] and bands = [460, 550], the function will
        return [0, 2]. Those bands indices can then be used e.g by get_image_cube

        Parameters
        ----------
        bands: list of float
            list of band values for which to find the index of the closest
            bands in the dataset

        Returns
        -------
        bands_indices: list of int
            list of indices of the bands in the dataset closest to the desired bands
        bands_names: list of str
            list of band names corresponding to bands_indices
        """

        bands_indices = find_index_of_band(self.centers, bands)
        # bands_indices = [np.argmin(np.abs(np.array(self.channel_names).astype(float) - x)) for x in bands]
        bands_names = [self.channel_names[x] for x in bands_indices]

        return bands_indices, bands_names


def get_im_main_roi(export_folder, bands=None, mainroi_index=None):
    """Get bands from a given main roi from an export folder, i.e.
     containing a corrected.zarr file.

    Parameters
    ----------
    export_folder : str
        Path to export folder.
    bands : list of int
        List of bands to get.
    mainroi_index : list of int
        Index of main roi to get. Default is None.
        If None, get full image.

    Returns
    -------
    new_cube : array
        Image cube containing the bands in dimensions (bands, rows, cols).

     """

    params = load_project_params(folder=export_folder)
    mainrois, subrois, measurementroi = params.get_formatted_rois()

    roi_bounds = None
    if mainroi_index is not None:
        roi_bounds = [mainrois[mainroi_index][:, 0].min(), mainrois[mainroi_index][:, 0].max(),
                      mainrois[mainroi_index][:, 1].min(), mainrois[mainroi_index][:, 1].max()]

    if not export_folder.joinpath('corrected.zarr').exists():
        raise FileNotFoundError(f"File {export_folder.joinpath('corrected.zarr')} does not exist")

    imagechannels = ImChannels(export_folder.joinpath('corrected.zarr'))

    new_cube = imagechannels.get_image_cube(channels=bands,
                                            roi=roi_bounds)

    return new_cube


def get_roi_limits(edge_coordinates: list[int]) -> tuple[int, int, int, int]:
    formatted = np.array(edge_coordinates).reshape(4, 2).astype(int)
    row_start, row_end, col_start, col_end = (
        formatted[:, 0].min(),
        formatted[:, 0].max(),
        formatted[:, 1].min(),
        formatted[:, 1].max(),
    )
    return row_start, row_end, col_start, col_end


def get_extent_roi(edge_coordinates: list[int]) -> np.index_exp:
    row_start, row_end, col_start, col_end = get_roi_limits(edge_coordinates)
    return np.index_exp[row_start:row_end, col_start:col_end, :]


def get_mask_roi(edge_coordinates: list[int], shape_main: tuple[int]) -> np.ndarray[bool]:
    idcs = get_extent_roi(edge_coordinates)

    in_roi = np.zeros(shape_main, dtype=bool)
    in_roi[idcs] = True
    return in_roi


def find_index_of_band(band_list, band_value):
    """Find index of band in list of bands

    Parameters
    ----------
    band_list : list of float
        List of real bands values
    band_value : float or list/array of float
        Band value(s) to find index of in band_list

    Returns
    -------
    band_index : int or list of int
        Index of band in band_list. If band_value is a list/array,
        returns a list of indices. Otherwise, returns a single index.

    """

    if not isinstance(band_value, (list, np.ndarray)):
        band_value = np.array([band_value])
    band_value = np.array(band_value)
    if band_value.ndim != 1:
        raise ValueError('band_value must be 1D')

    band_list = np.array(band_list)
    band_index = [np.argmin(np.abs(band_list - b)) for b in band_value]

    if len(band_index) == 1:
        band_index = band_index[0]

    return band_index


def read_spectral(path, bands=None, row_bounds=None, col_bounds=None):
    """Read spectral data from an hdr or zarr file.

    Parameters
    ----------
    path: str
        path to hdr or zarr file
    bands: list of int
        list of bands indices to read
    row_bounds: tuple of int
        (row_start, row_end)
    col_bounds: tuple of int
        (col_start, col_end)

    Returns
    -------
    data: ndarray
        spectral data
    metadata: dict
        metadata with keys 'wavelength' (list of str), 'centers' (list of float)
    """

    path = Path(path)
    if path.suffix == '.hdr':
        img = open_image(path)

        metadata = img.metadata
        metadata['centers'] = img.bands.centers

        if bands is None:
            bands = np.arange(0, len(metadata['wavelength']))

        if (row_bounds is None) and (col_bounds is None):
            data = img.read_bands(bands)
        else:
            if row_bounds is None:
                row_bounds = (0, img.nrows)
            if col_bounds is None:
                col_bounds = (0, img.ncols)

            data = img.read_subregion(row_bounds=row_bounds, col_bounds=col_bounds, bands=bands)

    elif path.suffix == '.zarr':
        zarr_image = zarr.open(path, mode='r')

        if 'metadata' in zarr_image.attrs:
            metadata = zarr_image.attrs['metadata']

            if 'index_name' in zarr_image.attrs['metadata']:
                data = np.array(zarr_image)

            elif 'wavelength' in zarr_image.attrs['metadata']:
                if bands is None:
                    bands = np.arange(zarr_image.shape[0])
                else:
                    bands = np.array(bands)

                if row_bounds is None:
                    row_bounds = (0, zarr_image.shape[1])
                if col_bounds is None:
                    col_bounds = (0, zarr_image.shape[2])

                zarr_image = da.from_zarr(path)
                data = zarr_image[bands, row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]

                data = np.moveaxis(data, 0, 2)
            else:
                data = np.array(zarr_image)
        else:
            data = np.array(zarr_image)
            metadata = {}
    else:
        raise ValueError(f'cannot import file type {path.suffix}')

    return data, metadata


def load_mask(filename):
    mask = tifffile.imread(filename)
    return mask


def get_mask_path(export_folder):

    export_folder = Path(export_folder)
    return export_folder.joinpath('mask.tif')


def get_mask_main(main_folder, main_roi_index=0):
    """Get mask from a given folder and roi index."""

    params = load_project_params(main_folder)
    mask_path = get_mask_path(main_folder.joinpath(f'roi_{main_roi_index}'))
    row_bounds, col_bounds = params.get_formatted_col_row_bounds(main_roi_index)
    if mask_path.is_file():
        mask = load_mask(mask_path)
    else:
        mask = np.zeros((row_bounds[1] - row_bounds[0], col_bounds[1] - col_bounds[0]), dtype=np.uint8)

    return mask


def get_data_red_roi_from_extent(imcube, params, row_start, row_end, col_start, col_end, debug=False):
    n_channels = imcube.shape[2]
    roi_height = row_end - row_start
    roi_width = col_end - col_start

    mask_roi = np.zeros(imcube.shape, dtype=bool)
    mask_roi[row_start:row_end, col_start:col_end, :] = True

    mask_main = ~get_mask_main(Path(params.project_path)).astype(bool)

    # convert mask main to shape that is compatible with smaller roi
    mask_valid = np.repeat(
        mask_main[mask_roi[:, :, 0]].reshape((roi_height, roi_width, 1)),
        n_channels,
        axis=2
    )

    if debug:
        print(f'obtaining {roi_height=} x {roi_width=} x {n_channels=} datapoints')
    data = (imcube[mask_roi]
            .compute_chunk_sizes()
            .reshape((roi_height, roi_width, n_channels)))
    if debug:
        print('done obtaining data')

    return data, mask_valid


def get_data_red_roi(imcube, params, debug=False):
    edge_coordinates = params.measurement_roi[0]
    row_start, row_end, col_start, col_end = get_roi_limits(edge_coordinates)

    return get_data_red_roi_from_extent(imcube, params, row_start, row_end, col_start, col_end, debug=debug)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import spectral

    path_export_folder = Path(
        r"\\hlabstorage.dmz.marum.de\scratch\Yannick\hyperspec\iceland\qd_Geld_3.1_105-130cm_04082025_5_2025-08-04_12-24-46\napari_re"
    )
    params = load_project_params(path_export_folder)

    # passing mainroi_index=None gives the entire dataset
    imcube = get_im_main_roi(path_export_folder, mainroi_index=0)
    imcube = da.moveaxis(imcube, 0, -1)
    mask = get_mask_main(path_export_folder)

    img = open_image(params.file_path)
    bands = np.array(img.bands.centers)
    metadata = img.metadata
    rgb_channels = np.array(metadata['default bands']).astype(int)

    # read reduced data
    path_denoised = path_export_folder.joinpath('roi_0', 'denoised.zarr')
    denoised = np.moveaxis(np.array(zarr.open_array(path_denoised)), 0, 2)

    path_mnf = path_export_folder.joinpath('roi_0', 'mnf.zarr')
    mnf = zarr.open_array(path_mnf)

    # perform mnf on masked subsection
    roi, mask_valid_roi = get_data_red_roi(imcube, params)
    roi = np.asarray(roi)

    print('calcing stats ...')
    signal = spectral.calc_stats(roi, mask_valid_roi[:, :, 0])
    print('calcing noise ...')
    noise = spectral.noise_from_diffs(roi)
    print('calcing nmf ...')
    mnfr = spectral.mnf(signal, noise)

    # De-noise the data by eliminating NAPC components where SNR < 10.
    # The de-noised data will be in the original coordinate space (at
    # full dimensionality).
    print('denoising ...')
    denoised_mnf = mnfr.denoise(roi, snr=10)
    print('creating plot ...')

    # plot some of the eigenvectors (in columns according to documentation)
    n_vecs = 10
    plt.figure()
    lams = img.bands.centers
    for i in range(n_vecs):
        v = mnfr.napc.eigenvectors[:, i].copy()
        v -= v.min()
        v /= v.max()
        plt.plot(lams, v)
    plt.savefig('temp.pdf')
    plt.close()

    # Reduce dimensionality, retaining top 50 NAPC components.
    print('reducing dimensionality ...')
    reduced = mnfr.reduce(roi, num=50)

    # reduction works by mixing together bands in some new way
    # TODO: is this right? I dont think so ...
    n_bands = imcube.shape[2]
    res = mnfr.reduce(np.ones((1, 1, n_bands)), num=n_bands)

    i = 0
    fig, axs = plt.subplots(ncols=2)
    img = reduced[:, :, i]
    axs[0].imshow(img, vmin=np.quantile(img, .05), vmax=np.quantile(img, .95), aspect=1/20)
    axs[1].plot(mnfr.napc.eigenvectors[:, i], lams)
    plt.savefig('temp.pdf')
    plt.close()
    i += 1

    # cluster

    out = kmeans(reduced, max_iterations=100)
    clusters, centers = out

    import matplotlib.pyplot as plt

    rgb_img = imcube[:, :, rgb_channels]
    thr = np.quantile(rgb_img, .95, axis=(0, 1))
    thr = thr.max()
    rgb_img[rgb_img > thr] = thr
    rgb_img /= thr
    fig, axs = plt.subplots(ncols=5)
    axs[0].imshow(rgb_img)
    axs[0].imshow(mask, alpha=.5)
    # draw red rectangle
    # from matplotlib import patches
    # rect = patches.Rectangle(xy=(col_start, row_start),
    #                          width=col_end-col_start,
    #                          height=row_end-row_start,
    #                          edgecolor='red')
    # axs[0].add_patch(rect)

    corrected = np.array(imcube[10, 10, :])
    cont_removed = remove_continuum(corrected, bands=bands)
    axs[1].plot(corrected, bands)
    axs[1].plot(cont_removed, bands)

    axs[2].imshow(denoised[:, :, 0], aspect=1/20)
    axs[3].imshow(clusters, aspect=1 / 20, cmap='hsv')

    for i in range(centers.shape[0]):
        # TODO: get bands of reduced from mnf
        axs[4].plot(centers[i, :])
    axs[4].set_xlabel('mnf component')

    plt.savefig('temp.pdf')
