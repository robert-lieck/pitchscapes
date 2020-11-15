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


def coords_from_times(times, remove_offset=True, unit_times=True):
    coords = []
    for start_idx, start in enumerate(times):
        for end_idx, end in enumerate(times):
            # only process valid, non-zero size windows
            if start_idx < end_idx:
                coo = [tuple(start_end_to_center_width(start, times[end_idx - 1])),
                       tuple(start_end_to_center_width(start, end)),
                       tuple(start_end_to_center_width(times[start_idx + 1], end))]
                if end_idx - start_idx > 1:
                    coo.append(tuple(start_end_to_center_width(times[start_idx + 1], times[end_idx - 1])))
                else:
                    coo.append((np.nan, np.nan))
                coords.append(tuple(coo))
    coords = np.array(coords)
    # remove offset
    min_time = np.min(times)
    max_time = np.max(times)
    if remove_offset:
        coords[:, :, 0] -= min_time
    # rescale to unit interval [0, 1]
    if unit_times:
        if remove_offset:
            coords /= (max_time - min_time)
        else:
            coords /= max_time
    return coords
