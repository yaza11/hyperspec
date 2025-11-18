import numpy as np
from scipy.spatial import ConvexHull

def remove_continuum(bands, intensities, downsample_factor=1, plts=False):
    if downsample_factor > 1:
        bands_downsampled = np.linspace(
            np.min(bands),
            np.max(bands),
            round(intensities.shape[0] / downsample_factor)
        )
        intensities_downsampled = np.interp(bands_downsampled, bands, intensities)
    else:
        bands_downsampled = bands
        intensities_downsampled = intensities

    points = np.hstack([bands_downsampled[:, None], intensities_downsampled[:, None]])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    # only take the upper half
    neighbor_idcs = hull.vertices + 1
    # use left neighbor for last point to avoid walking out of bounds
    neighbor_idcs[-1] -= 2
    hull_point_neighbors = points[(hull.vertices + 1) % points.shape[0], :]
    is_upper_point = hull_points[:, 1] > hull_point_neighbors[:, 1]
    hull_points_upper = hull_points[is_upper_point, :]

    # interpolate between hull points
    hull_px, hull_py = hull_points_upper[:, 0], hull_points_upper[:, 1]
    o = np.argsort(hull_px)
    continuum = np.interp(bands, hull_px[o], hull_py[o])
    continuum_removed = intensities - continuum
    continuum_removed[continuum_removed < 0] = 0

    if plts:
        plt.plot(bands, intensities, 'r--')
        plt.plot(bands_downsampled, intensities_downsampled)
        plt.scatter(hull_points[:, 0], hull_points[:, 1], c='r')
        plt.scatter(hull_px, hull_py, c='g')
        plt.plot(bands, continuum)
        plt.show()

    return continuum_removed


if __name__ == '__main__':
    import h5py
    from _local import path_file_hdf5_iceland, path_file_mask_foreground_iceland, path_folder_iceland
    from hyperspec.calib.bil_reader import BilReader
    from hyperspec.file_finder import FileFinder
    import matplotlib.pyplot as plt

    ff = FileFinder(path_folder_iceland)
    bil_reader = BilReader(
        path_file_header=ff.path_meas_header_file,
        path_file_binary=ff.path_meas_binary_file
    )

    hdf = h5py.File(path_file_hdf5_iceland, 'r')

    dataset = hdf['roi_calibrated']

    intensities = dataset[0, 0, :]
    bands = bil_reader.wavelengths_nm

    remove_continuum(bands, intensities, 4, plts=True)