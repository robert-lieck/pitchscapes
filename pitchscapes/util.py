import warnings

import numpy as np
import matplotlib.pyplot as plt

from .scapes import PitchScape


pitch_classes_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
pitch_classes_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
pitch_classes = pitch_classes_sharp


def safe_int(s):
    i = int(s)
    assert i == s, f"{i} == {s}"
    return i


def random_batch_ids(n_data, n_batches):
    batch_ids = np.arange(n_data)
    np.random.shuffle(batch_ids)
    batch_ids %= n_batches
    return batch_ids


def axis_set_invisible(ax, splines=False, ticks=(), patch=False, x=False, y=False):
    if x:
        ax.get_xaxis().set_visible(False)
    if y:
        ax.get_yaxis().set_visible(False)
    if splines:
        plt.setp(ax.spines.values(), visible=False)
    if ticks:
        if 'left' in ticks or 'all' in ticks:
            ax.tick_params(left=False, labelleft=False)
        if 'right' in ticks or 'all' in ticks:
            ax.tick_params(right=False, labelright=False)
        if 'top' in ticks or 'all' in ticks:
            ax.tick_params(top=False, labeltop=False)
        if 'bottom' in ticks or 'all' in ticks:
            ax.tick_params(bottom=False, labelbottom=False)
    if patch:
        ax.patch.set_visible(False)


def start_end_to_center_width(start, end):
    """Transform (start, end) coordinates of a window to (center, width) coordinates"""
    return start + (end - start) / 2, end - start


def center_width_to_start_end(center, width):
    """Transform (center, width) coordinates of a window to (start, end) coordinates"""
    return center - width / 2, center + width / 2


def to_array(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def euclidean_distance(x, y, axis=None):
    x = to_array(x)
    y = to_array(y)
    return np.linalg.norm(x - y, axis=axis)


def multi_sample_pitch_scapes(scapes, n_samples):
    positions = None
    samples = []
    coords = None
    for s in scapes:
        positions, samp, coords = sample_pitch_scape(scape=s, n_samples=n_samples)
        samples.append(samp[None, :, :])
    if positions is None or coords is None:
        raise RuntimeWarning(f"Unable to sample pitch scapes. Did you provide a non-empty list? (provided: '{scapes}')")
    samples = np.concatenate(tuple(samples), axis=0)
    return positions, samples, coords


def sample_pitch_scape(scape, n_samples, prior_counts=None, normalize=True):
    warnings.warn("This function is deprecated as it handles prior counts inconsistently.", DeprecationWarning)
    if prior_counts is None:
        if isinstance(scape, PitchScape):
            prior_counts = 1.
        else:
            prior_counts = 0.
    positions = []
    samples = []
    coords = []
    for center, width, coo, counts in sample_discrete_scape(scape,
                                                            np.linspace(scape.min_time, scape.max_time, n_samples),
                                                            center_width=True,
                                                            coords=True):
        positions.append([center, width])
        samples.append(counts)
        coords.append(coo)
    coords = np.array(coords)
    positions = np.array(positions)
    # remove offset
    coords[:, :, 0] -= scape.min_time
    positions[:, 0] -= scape.min_time
    # rescale to unit interval [0, 1]
    coords /= (scape.max_time - scape.min_time)
    positions /= (scape.max_time - scape.min_time)
    # add prior counts
    samples = np.array(samples) + prior_counts
    # normalise
    if normalize:
        samples /= samples.sum(axis=1, keepdims=True)
    return positions, samples, coords


def sample_discrete_scape(scape,
                          times=None,
                          start_end_idx=False,
                          center_width=False,
                          start_end_time=False,
                          coords=False,
                          value=True):
    """
    Generator that returns values from scape on a discrete time grid.
    :param scape: The scape to sample from. scape[start, end] (for start < end in times) should return the respective
    value.
    :param times: set of valid points in time (optional; required if scape does not have a times attribute)
    :param start_end_idx: whether to return start and end index of window (default: False)
    :param center_width: whether to return center and width of window (default: False)
    :param start_end_time: whether to return start and end time of window (default: False)
    :param coords: whether to return the center-width coordinates for plotting (default: False). This is a 4-tuple of
    pairs with center width coordinates for windows defined by time indices [start, end-1], [start, end],
    [start+1, end], [start+1,end-1]. If the respective window contains only a single time slot, the last pair is
    (nan, nan).
    :param value: whether to return the value of the scape for the window (default: True)
    :return: yields a nested tuple ((start_idx, center, start_time), (end_idx, width, end_time), coords, value). Some
    elements may be omitted (based on start_end_idx, center_width, start_end_time, coords, and value). If the first
    tuples would be empty, they are entirely omitted instead.
    """
    # try to get times from scape object if not provided
    if times is None:
        try:
            times = scape.times
        except AttributeError:
            raise AttributeError("Provided scape object does not have 'times' attribute. "
                                 "Please provide explicitly as argument.")
    # convert times to numpy array
    if not isinstance(times, np.ndarray):
        times = np.array(times)
    # go through all possible start/end points
    for start_idx, start in enumerate(times):
        for end_idx, end in enumerate(times):
            # only process valid, non-zero size windows
            if start_idx < end_idx:
                # construct return tuple
                return_list = []
                # add window dimensions
                start_list = []
                end_list = []
                if start_end_idx:
                    start_list.append(start_idx)
                    end_list.append(end_idx)
                if center_width:
                    center, width = start_end_to_center_width(start, end)
                    start_list.append(center)
                    end_list.append(width)
                if start_end_time:
                    start_list.append(start)
                    end_list.append(end)
                # only add to return tuple if non-empty
                if start_list and end_list:
                    if len(start_list) == len(end_list) == 1:
                        return_list += [start_list[0], end_list[0]]
                    else:
                        return_list += [tuple(start_list), tuple(end_list)]
                # add coordinates
                if coords:
                    coo = [tuple(start_end_to_center_width(start, times[end_idx - 1])),
                           tuple(start_end_to_center_width(start, end)),
                           tuple(start_end_to_center_width(times[start_idx + 1], end))]
                    if end_idx - start_idx > 1:
                        coo.append(tuple(start_end_to_center_width(times[start_idx + 1], times[end_idx - 1])))
                    else:
                        coo.append((np.nan, np.nan))
                    return_list.append(tuple(coo))
                # add value
                if value:
                    return_list.append(scape[start, end])
                if len(return_list) == 1:
                    yield return_list[0]
                else:
                    yield tuple(return_list)


def coords_from_times(times,
                      start_end_idx=False,
                      start_end_time=False,
                      center_width=False,
                      coords=False,
                      remove_offset=True,
                      unit_times=True):
    """
    Generate various coordinate information from a given list of times. Given N points in time, the returned arrays will
     contain N(N-1)/2 items, corresponding to all ordered non-zero time intervals. The ordering corresponds constructing
      a list by iterating over start points in the outer loop and end points in the inner loop.
    :param times: An iterable of N points in time (has to be ordered).
    :param start_end_idx: (default False) Whether to return an array of shape Nx2 with integer (start_idx, end_idx)
    pairs.
    :param start_end_time: (default False) Whether to return an array of shape Nx2 with (start_time, end_time) pairs
    according to the values in times.
    :param center_width: (default False) Whether to return an array of shape Nx2 with (center, width) time coordinate,
    related to (start_time, end_time) via start_end_to_center_width.
    :param coords: (default False) Whether to return an array of shape Nx4x2 with four points in center-width
    coordinates, describing the rhomboid region in a scape plot that corresponds to the respective time interval. The
    top-point of the rhombus corresponds to the respective center_width coordinate, the other three points are obtained
    by moving the start and/or end point one time step forward and/or backward, respectively. The coordinates are
    ordered clock-wise starting at 9 (left, top, right, bottom), corresponding to the indices [(start, end - 1),
    (start, end), (start + 1, end), (start + 1, end - 1)]. For the bottom row of a scape plot (adjacent points in time),
    the last (bottom) coordinate will be a pair of np.nan.
    :param remove_offset: (default True) Whether the minimum time is subtracted to have time start at zero.
    :param unit_times: (default True) Whether to normalise times to be in [0, 1].
    :return: If only one of [start_end_idx, start_end_time, center_width, coords] is true, return the corresponding
    array, otherwise return a tuple of the requested arrays (in this order).
    """
    start_end_idx_list = []
    start_end_time_list = []
    center_width_list = []
    coord_list = []
    for start_idx, start in enumerate(times):
        for end_idx, end in enumerate(times):
            # only process valid, non-zero size windows
            if start_idx < end_idx:
                # get start/end index
                if start_end_idx:
                    start_end_idx_list.append([start_idx, end_idx])
                # get start/end time
                if start_end_time:
                    start_end_time_list.append([start, end])
                # get center/width
                if center_width:
                    center_width_list.append(start_end_to_center_width(start, end))
                # get coordinates
                if coords:
                    coo = [tuple(start_end_to_center_width(start, times[end_idx - 1])),
                           tuple(start_end_to_center_width(start, end)),
                           tuple(start_end_to_center_width(times[start_idx + 1], end))]
                    if end_idx - start_idx > 1:
                        coo.append(tuple(start_end_to_center_width(times[start_idx + 1], times[end_idx - 1])))
                    else:
                        coo.append((np.nan, np.nan))
                    coord_list.append(tuple(coo))
    start_end_idx_list = np.array(start_end_idx_list)
    # initialise as floats to allow for inplace division later
    start_end_time_list = np.array(start_end_time_list, dtype=float)
    center_width_list = np.array(center_width_list, dtype=float)
    coord_list = np.array(coord_list, dtype=float)
    # remove offset
    min_time = np.min(times)
    max_time = np.max(times)
    if remove_offset:
        if start_end_time:
            start_end_time_list -= min_time
        if center_width:
            center_width_list[:, 0] -= min_time
        if coords:
            coord_list[:, :, 0] -= min_time
    # rescale to unit interval [0, 1]
    if unit_times:
        if remove_offset:
            if start_end_time:
                start_end_time_list /= (max_time - min_time)
            if center_width:
                center_width_list /= (max_time - min_time)
            if coords:
                coord_list /= (max_time - min_time)
        else:
            if start_end_time:
                start_end_time_list /= max_time
            if center_width:
                center_width_list /= max_time
            if coords:
                coord_list /= max_time
    ret = []
    if start_end_idx:
        ret += [start_end_idx_list]
    if start_end_time:
        ret += [start_end_time_list]
    if center_width:
        ret += [center_width_list]
    if coords:
        ret += [coord_list]
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def key_estimates_to_str(estimates, sharp_flat='sharp', use_capitalisation=True):
    """
    Transform an array of key estimates into string representation
    :param estimates: 2D numpy array of shape Nx2 with N estimates, where the first entry in each row indicates whether it
    is major (0) or minor (1) and the second entry indicates the tonic (0-11, starting a C in chromatically ascending
    order)
    :param sharp_flat: one of ['sharp', 'flat'] (default 'sharp'); whether to use sharps or flats for chromatic notes
    (i.e. C# versus Db)
    :param use_capitalisation: True/False (default True); whether to indicate major/minor by capitalisation
    ('C' versus 'c') or spelled out ('C major' versus 'C minor')
    :return: 1D array of length N with the string descriptions
    """
    pcs = np.array(pitch_classes_sharp if sharp_flat == 'sharp' else pitch_classes_flat)
    tonic_names = pcs[estimates[:, 1]]
    if use_capitalisation:
        keys = np.array([tonic if is_major else tonic.lower()
                         for tonic, is_major in zip(tonic_names, estimates[:, 0] == 0)])
    else:
        keys = np.array([tonic + " major" if is_major else tonic + " minor"
                         for tonic, is_major in zip(tonic_names, estimates[:, 0] == 0)])
    return keys
