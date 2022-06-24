from math import isclose
from warnings import warn
import numpy as np


def normalize_non_zero(a, axis=None, replace_zeros=False, skip_type_check=False):
    """For the given ND array, normalise each 1D array obtained by indexing the 'axis' dimension if the sum along the
    other dimensions (for that entry) is non-zero. Normalisation is performed in place."""
    # check that dtype is float (in place division of integer arrays silently rounds)
    if not skip_type_check:
        if not np.issubdtype(a.dtype, np.floating):
            raise TypeError(f"Cannot guarantee that normalisation works as expected on array of type '{a.dtype}'. "
                            f"Use 'skip_type_check=True' to skip this check.")
    # normalise along last axis per default
    if axis is None:
        axis = a.ndim - 1
    # make axis a tuple if it isn't
    if not isinstance(axis, tuple):
        axis = (axis,)
    # compute sum along axis, keeping dimensions
    s = a.sum(axis=axis, keepdims=True)
    # check for non-zero entries
    non_zero = (s != 0)
    if not np.any(non_zero):
        # directly return if there are no non-zero entries
        return a
    # construct an index tuple to select the appropriate entries for normalisation (the dimensions specified by axis
    # have to be replaced by full slices ':' to broadcast normalisation along these dimensions)
    non_zero_arr = tuple(slice(None) if idx in axis else n for idx, n in enumerate(non_zero.nonzero()))
    # in-place replace non-zero entries by their normalised values
    a[non_zero_arr] = a[non_zero_arr] / s[non_zero_arr]
    # add uniform distribution for zeros if requested
    if replace_zeros:
        zero_arr = tuple(slice(None) if idx in axis else n for idx, n in enumerate(np.logical_not(non_zero).nonzero()))
        a[zero_arr] += 1 / np.array(a.shape)[np.array(axis)].prod()
    # return array
    return a


class Scape:
    """
    Abstract base class for scapes objects. Which has a minimum time (min_time) a maximum time (max_time) and a
    well-defined value for any window (t1, t2) with min_time <= t1 < t2 <= max_time. The value can be retrieved via
    scape[t1, t2].
    """

    def __init__(self, min_time, max_time):
        """
        :param min_time: minimum time
        :param max_time: maximum time
        """
        self.min_time = min_time
        self.max_time = max_time

    def __getitem__(self, item):
        """
        Return value of the scape at position
        :param item: (start, end) with start < end
        :return: value of the scape for the time window (start, end)
        """
        raise NotImplementedError

    def assert_valid_time_window(self, start, end):
        if start < self.min_time and not isclose(start, self.min_time):
            raise ValueError(f"Window start is less than minimum time ({start} < {self.min_time})")
        if end > self.max_time and not isclose(end, self.max_time):
            raise ValueError(f"Window end is greater than maximum time ({end} > {self.max_time})")
        if start > end:
            raise ValueError(f"Window start is greater than or equal to window end ({start} >= {end})")


class DiscretePitchScape_(Scape):
    """
    Discrete-time scape that is only well-defined at specific points in time. It computes the weighted sum (optionally
    the mean) over the given time window i.e. it sums up the values weighted by the corresponding time span and
    (optionally) divides by the total time interval.
    """

    def __init__(self,
                 values,
                 times=None,
                 prior_counts=None,
                 strategy="left",
                 normalise=False,
                 normalise_values=False,
                 skip_type_check=False,
                 dtype=None):
        """
        Initialise the scape.
        :param values: values for each time slot (first dimension of size N runs over time intervals)
        :param times: [optional] boundaries of the time intervals (array like of length N+1).
        Defaults to [0, 1, ..., N].
        :param prior_counts: scalar prior counts added to the aggregated values. Non-zero prior counts correspond to
         a Bayesian estimate with a uniform prior; the magnitude adjusts the prior strength. A value of None (default)
         does not add any prior counts. If return values are normalised (normalise=True) there is a difference between
         None and 0: A value of 0 is interpreted as the limitting case of non-zero prior counts, which means that
         outputs will always be normalised and if no data is available (zero counts, e.g. for zero-width time intervals
         or time intervals only zero values), a uniform distribution is returned. For a value of None, you will get an
         unnormalised all-zero output instead.
        :param strategy: one of ["left", "right", "center"] (default: "left"). Determines how a time-index interval
        [K, L] is split in the recursive computations. For "left": [K,L-1]+[L-1,L]; for "right": [K,K+1]+[K+1,L]; for
        "center": [K,K+1]+[K+1,L-1]+[L-1,L]. Also see comments for parse_bottom_up concerning efficiency and run time.
        :param normalise: whether to compute the weighted sum or mean (default: False i.e. compute weighted sum)
        :param normalise_values: whether to normalise the values per time interval (default: True)
        :param skip_type_check: If the values are integers there may be problems with in-place normalisation or adding
        floating-type prior counts. The function raises a warning in these cases, suggesting to use a floating type
        for the values. This check can be disabled (which is generally a bad idea) by setting skip_type_check=True.
        Note that the warning is not raised for integer values of no normalisation is performed and the prior counts are
        None or not of a floating type.
        :param dtype: type to use for values
        """
        self.prior_counts = prior_counts
        self.strategy = strategy
        self.normalise = normalise
        # set times and initialise super class
        if times is None:
            times = list(range(len(values) + 1))
        self._len = len(times)
        self.times = np.array(times)
        if not np.all(self.times[:-1] <= self.times[1:]):
            raise ValueError(f"Times are not sorted: {self.times}")
        # get reverse time indices
        self.indices = {time: idx for idx, time in enumerate(self.times)}
        self.min_index = 0
        self.max_index = self.times.shape[0]
        super().__init__(min_time=self.times.min(), max_time=self.times.max())
        # values
        if dtype is None:
            self.values = np.array(values)
        else:
            self.values = np.array(values, dtype=dtype)
        # warn for non-float types
        if not skip_type_check and not np.issubdtype(self.values.dtype, np.floating):
            if normalise_values or normalise:
                raise TypeError(f"Using non-floating values of type '{self.values.dtype}' may lead to errors or "
                                f"unexpected results in combination with normalisation. Either use float values; or "
                                f"don't use normalisation (normalise_values=False and normalise=False); or set "
                                f"skip_type_check=True to take the risk.")
            if np.issubdtype(type(self.prior_counts), np.floating):
                raise TypeError(f"Combining non-floating values of type '{self.values.dtype}' with floating-type prior "
                                f"counts of type '{type(self.prior_counts)}' may lead to errors or unexpected results. "
                                f"Use float values; or set skip_type_check=True to take the risk.")
        # normalise_values
        if normalise_values:
            normalize_non_zero(self.values, skip_type_check=skip_type_check)
        # pre-compute the overall number of value dimensions (for prior counts and normalisation)
        self.n_value_dim = np.prod(self.values.shape[1:])
        # check dimensions
        if self.values.shape[0] != self.times.shape[0] - 1:
            raise ValueError(f"There should be n-1 values for n time steps "
                             f"(values shape: {self.values.shape}, times shape: {self.times.shape})")
        # initialise scape
        if self.normalise:
            self._accumulated_values = {(idx, idx + 1): v for idx, v in enumerate(self.values)}
        else:
            self._accumulated_values = {(idx, idx + 1): v * (self.times[idx + 1] - self.times[idx])
                                        for idx, v in enumerate(self.values)}

    def assert_valid_time_window(self, start, end):
        super().assert_valid_time_window(start, end)
        if start not in self.indices:
            raise ValueError(f"Window start ({start}) is not a valid time (not in {self.indices})")
        if end not in self.indices:
            raise ValueError(f"Window end ({end}) is not a valid time (not in {self.indices})")

    def assert_valid_index_window(self, start, end):
        if not (self.min_index <= start <= end <= self.max_index):
            raise ValueError(f"Invalid start/end index, should be "
                             f"{self.min_index} <= {start} < {end} <= {self.max_index}")

    def __getitem__(self, item):
        start, end = item
        self.assert_valid_time_window(start, end)
        start_idx, end_idx = self.indices[start], self.indices[end]
        self.assert_valid_index_window(start_idx, end_idx)
        # check for zero-size windows or return accumulated value
        if start == end:
            return self.zero_size_value()
        else:
            value = self.get_value_at_index(start_idx, end_idx)
            delta_t = self.times[end_idx] - self.times[start_idx]
            self.add_prior_counts(value, delta_t)
            return value

    def recursive_indices(self, start_idx, end_idx):
        if end_idx - start_idx < 2:
            raise ValueError(f"Recursion only defined for windows with size > 1 "
                             f"(start: {start_idx}, end: {end_idx}, size: {end_idx - start_idx})")
        if self.strategy == "center":
            index_list = [(start_idx, start_idx + 1), (end_idx - 1, end_idx)]
            if end_idx - start_idx > 2:
                index_list.append((start_idx + 1, end_idx - 1))
        elif self.strategy == "left":
            index_list = [(start_idx, end_idx - 1), (end_idx - 1, end_idx)]
        elif self.strategy == "right":
            index_list = [(start_idx, start_idx + 1), (start_idx + 1, end_idx)]
        else:
            raise ValueError(f"Unknown recursion strategy '{self.strategy}'")
        return index_list

    def zero_size_value(self):
        # return zeros, uniform distribution, or prior counts (depending on value of prior_counts)
        ret = np.zeros(self.values.shape[1:], dtype=self.values.dtype)
        if self.normalise:
            if self.prior_counts is not None:
                # for any value of prior counts (except None) return a uniform distribution
                ret += 1 / self.n_value_dim
        else:
            if self.prior_counts is not None and self.prior_counts != 0:
                ret += self.prior_counts
        return ret

    def get_value_at_index(self, start, end):
        # if value is not stored in memory: compute recursively
        if (start, end) not in self._accumulated_values:
            # accumulate indices to compute
            indices_to_compute = [(start, end)]
            unprocessed_indices = [(start, end)]
            while unprocessed_indices:
                unprocessed_start, unprocessed_end = unprocessed_indices.pop()
                for new_start, new_end in self.recursive_indices(unprocessed_start, unprocessed_end):
                    if (new_start, new_end) not in self._accumulated_values:
                        unprocessed_indices.append((new_start, new_end))
                        indices_to_compute.append((new_start, new_end))
            # compute
            for new_start, new_end in reversed(indices_to_compute):
                # double-check before computing
                # (might have reoccurred higher in the stack and already been computed)
                if (new_start, new_end) not in self._accumulated_values:
                    sum_val = None
                    for new_new_start, new_new_end in self.recursive_indices(new_start, new_end):
                        # compute value
                        val = self._accumulated_values[new_new_start, new_new_end]
                        if self.normalise:
                            # get unnormalised value by multiplying with the respective time interval
                            val = val * (self.times[new_new_end] - self.times[new_new_start])
                        # initialise or update sum
                        if sum_val is None:
                            sum_val = val
                        else:
                            sum_val = sum_val + val
                    if self.normalise:
                        # if output should be normalised, store normalised values
                        sum_val /= self.times[new_end] - self.times[new_start]
                    self._accumulated_values[new_start, new_end] = sum_val
        # return a copy of the stored value
        return self._accumulated_values[start, end].copy()

    def add_prior_counts(self, value, delta_t):
        """Add prior counts ensuring correct normalisation. NOTE: value is modified in place."""
        if self.normalise:
            # get unnormalised value from normalised to correctly add prior counts
            value *= delta_t
        if self.prior_counts is not None and self.prior_counts != 0:
            # add prior counts
            value += self.prior_counts
        if self.normalise:
            # renormalise
            if self.prior_counts == 0 and np.all(value == 0):
                # edge case of prior counts --> 0: return uniform distribution
                value += 1 / self.n_value_dim
            elif self.prior_counts is None:
                # prior counts ignored in normalisation
                value /= delta_t
            else:
                # prior counts included in normalisation
                value /= delta_t + self.n_value_dim * self.prior_counts
            if not np.all(value == 0):
                # compensate floating point errors
                n = value.sum()
                value /= n
                if not np.isclose(n, 1, atol=1e-2):
                    warn(f"Inexact normalisation corrected ({n})", RuntimeWarning)

    def parse_bottom_up(self):
        """
        Compute all values bottom up. Performs an efficient dynamic programming pass through the entire scape and avoids
        time consuming recursive look-ups for compting single values (run time ~O(n^2)). This can be used if all values
        within the scape are expected to be needed. For high-resolution scapes where only few values need to be computed
        this may be inefficient as a single lookup has only ~O(n) run time. So if fewer than n values are needed,
        separate lookup will be more efficient. Moreover, if multiple value have the same start or end time, the overall
        run time for computing these values is only ~O(n). Depending on whether star or end time are in common the
        "left" or "right" strategy is more efficient.
        """
        for width in range(2, self.max_index - self.min_index):
            for start_idx in range(self.min_index, self.max_index - width):
                self.get_value_at_index(start_idx, start_idx + width)


class DiscreteScape(Scape):
    """
    Discrete-time scape that is only well-defined at specific points in time. It computes the weighted sum over the
    given time window i.e. it sums up the values weighted by the corresponding time span.
    """

    def __init__(self,
                 values,
                 times=None,
                 weights=None,
                 strategy="left"):
        """
        Initialise the scape.
        :param values: values for each time slot (first dimension of size N runs over time intervals)
        :param times: [optional] boundaries of the time intervals (array like of length N+1).
        Defaults to [0, 1, ..., N].
        :param weights: [optional] non-negative weights for the values; if not provided, the size of respective time
        interval is used.
        :param strategy: one of ["left", "right", "center"] (default: "left"). Determines how a time-index interval
        [K, L] is split in the recursive computations. For "left": [K,L-1]+[L-1,L]; for "right": [K,K+1]+[K+1,L]; for
        "center": [K,K+1]+[K+1,L-1]+[L-1,L]. Also see comments for parse_bottom_up concerning efficiency and run time.
        """
        self.strategy = strategy
        # set times and initialise super class
        if times is None:
            times = list(range(len(values) + 1))
        self._len = len(times)
        self.times = np.array(times)
        if not np.all(self.times[:-1] <= self.times[1:]):
            raise ValueError(f"Times are not sorted: {self.times}")
        # get reverse time indices
        self.indices = {time: idx for idx, time in enumerate(self.times)}
        self.min_index = 0
        self.max_index = self.times.shape[0]
        super().__init__(min_time=self.times.min(), max_time=self.times.max())
        # get weights
        if weights is None:
            self.weights = self.times[1:] - self.times[:-1]
        else:
            self.weights = np.array(weights)
        if np.any(self.weights < 0):
            raise ValueError(f"Weights must be non-negative: {self.weights}")
        # values
        self.values = np.array(values)
        # value for windows of zero width
        self.zero_size_value = np.zeros(self.values.shape[1:], dtype=self.values.dtype)
        # check dimensions
        if self.values.shape[0] != self.times.shape[0] - 1:
            raise ValueError(f"There should be n-1 values for n time steps "
                             f"(values shape: {self.values.shape}, times shape: {self.times.shape})")
        if self.weights.shape[0] != self.values.shape[0]:
            raise ValueError(f"There should be a many weights as values "
                             f"(values shape: {self.values.shape}, weight shape: {self.weights.shape})")
        # initialise bottom level of scape with weighted values
        self._accumulated_values = {(idx, idx + 1): v * w for idx, (v, w) in enumerate(zip(self.values, self.weights))}

    def assert_valid_time_window(self, start, end):
        super().assert_valid_time_window(start, end)
        if start not in self.indices:
            raise ValueError(f"Window start ({start}) is not a valid time (not in {self.indices})")
        if end not in self.indices:
            raise ValueError(f"Window end ({end}) is not a valid time (not in {self.indices})")

    def assert_valid_index_window(self, start, end):
        if not (self.min_index <= start <= end <= self.max_index):
            raise ValueError(f"Invalid start/end index, should be "
                             f"{self.min_index} <= {start} < {end} <= {self.max_index}")

    def __getitem__(self, item):
        start, end = item
        self.assert_valid_time_window(start, end)
        start_idx, end_idx = self.indices[start], self.indices[end]
        self.assert_valid_index_window(start_idx, end_idx)
        return self.get_value_at_index(start_idx, end_idx)

    def recursive_indices(self, start_idx, end_idx):
        if end_idx - start_idx < 2:
            raise ValueError(f"Recursion only defined for windows with size > 1 "
                             f"(start: {start_idx}, end: {end_idx}, size: {end_idx - start_idx})")
        if self.strategy == "center":
            index_list = [(start_idx, start_idx + 1), (end_idx - 1, end_idx)]
            if end_idx - start_idx > 2:
                index_list.append((start_idx + 1, end_idx - 1))
        elif self.strategy == "left":
            index_list = [(start_idx, end_idx - 1), (end_idx - 1, end_idx)]
        elif self.strategy == "right":
            index_list = [(start_idx, start_idx + 1), (start_idx + 1, end_idx)]
        else:
            raise ValueError(f"Unknown recursion strategy '{self.strategy}'")
        return index_list

    def get_value_at_index(self, start, end):
        # zero-size window
        if start == end:
            return self.zero_size_value.copy()
        # if value is not stored in memory: compute recursively
        if (start, end) not in self._accumulated_values:
            # accumulate indices to compute
            indices_to_compute = [(start, end)]
            unprocessed_indices = [(start, end)]
            while unprocessed_indices:
                unprocessed_start, unprocessed_end = unprocessed_indices.pop()
                for new_start, new_end in self.recursive_indices(unprocessed_start, unprocessed_end):
                    if (new_start, new_end) not in self._accumulated_values:
                        unprocessed_indices.append((new_start, new_end))
                        indices_to_compute.append((new_start, new_end))
            # compute
            for new_start, new_end in reversed(indices_to_compute):
                # double-check before computing
                # (might have reoccurred higher in the stack and already been computed)
                if (new_start, new_end) not in self._accumulated_values:
                    sum_val = None
                    for new_new_start, new_new_end in self.recursive_indices(new_start, new_end):
                        # compute value
                        val = self._accumulated_values[new_new_start, new_new_end]
                        # initialise or update sum
                        if sum_val is None:
                            sum_val = val
                        else:
                            sum_val = sum_val + val
                    self._accumulated_values[new_start, new_end] = sum_val
        # return a copy of the stored value
        return self._accumulated_values[start, end].copy()

    def parse_bottom_up(self):
        """
        Compute all values bottom up. Performs an efficient dynamic programming pass through the entire scape and avoids
        time consuming recursive look-ups for compting single values (run time ~O(n^2)). This can be used if all values
        within the scape are expected to be needed. For high-resolution scapes where only few values need to be computed
        this may be inefficient as a single lookup has only ~O(n) run time. So if fewer than n values are needed,
        separate lookup will be more efficient. Moreover, if multiple value have the same start or end time, the overall
        run time for computing these values is only ~O(n). Depending on whether star or end time are in common the
        "left" or "right" strategy is more efficient.
        """
        for width in range(2, self.max_index - self.min_index):
            for start_idx in range(self.min_index, self.max_index - width):
                self.get_value_at_index(start_idx, start_idx + width)


class PitchScape_(Scape):
    """
    Continuous-time scape that takes a discrete-time scape and interpolates linearly for points that lie in between the
    discrete time points. This is the exact continuous generalisation when using DiscretePitchScape as discrete-time
    scape.
    """

    def __init__(self, values=None, scape=None, **kwargs):
        """
        Initialise PitchScape either from count values or from DiscretePitchScape.
        :param values: pitch-class counts (do not provide together with scape)
        :param scape: DiscretePitchScape object (do not provide together with values)
        :param kwargs: keyword arguments passed on to initialize DiscretePitchScape (use only when also providing
        values)
        """
        if (values is None) == (scape is None):
            raise ValueError("Please specify EITHER 'values' (and optional keyword arguments) OR 'scape'")
        if scape is None:
            scape = DiscretePitchScape_(values=values, **kwargs)
        elif kwargs:
            raise TypeError("Cannot take keyword arguments if scape object is given.")
        self.scape = scape
        self.normalise = scape.normalise
        super().__init__(min_time=self.scape.min_time, max_time=self.scape.max_time)

    def __getitem__(self, item):
        try:
            return self.scape[item]
        except ValueError:
            return self.interpolate(*item)

    @staticmethod
    def get_adjacent_indices(start, end, times):
        # get upper/lower adjacent indices
        upper_start_idx = np.searchsorted(times, start, side='right')
        lower_start_idx = upper_start_idx - 1
        upper_end_idx = np.searchsorted(times, end, side='left')
        lower_end_idx = upper_end_idx - 1
        n_times = len(times)
        # handle floating point round-off errors
        if lower_start_idx == -1 and isclose(start, times[0]):
            lower_start_idx = 0
            upper_start_idx = 1
            start = times[0]
        if upper_end_idx == n_times and isclose(end, times[-1]):
            lower_end_idx = n_times - 2
            upper_end_idx = n_times - 1
            end = times[-1]
        # check bounds
        if lower_start_idx < 0:
            raise ValueError("Start below valid times")
        if upper_end_idx >= n_times:
            raise ValueError("End beyond valid times")
        return start, end, lower_start_idx, upper_start_idx, lower_end_idx, upper_end_idx

    def interpolate(self, start, end):
        # check for zero-size windows
        if start == end:
            return self.scape.zero_size_value()
        # get adjacent indices
        (start,
         end,
         lower_start_idx,
         upper_start_idx,
         lower_end_idx,
         upper_end_idx) = self.get_adjacent_indices(start, end, self.scape.times)
        # get corresponding times
        upper_start_time = self.scape.times[upper_start_idx]
        lower_start_time = self.scape.times[lower_start_idx]
        upper_end_time = self.scape.times[upper_end_idx]
        lower_end_time = self.scape.times[lower_end_idx]
        if lower_start_idx == lower_end_idx and upper_start_idx == upper_end_idx:
            # window start/end lie in the same time interval
            if self.normalise:
                value = self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
            else:
                value = (end - start) / (upper_start_time - lower_start_time) * \
                      self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
        elif upper_start_time == lower_end_time:
            # window start/end lie in between two adjacent time intervals
            e = self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
            f = self.scape.get_value_at_index(lower_end_idx, upper_end_idx)
            if self.normalise:
                f1 = upper_start_time - start
                f2 = end - lower_end_time
                value = (f1 * e + f2 * f) / (f1 + f2)
            else:
                f1 = (upper_start_time - start) / (upper_start_time - lower_start_time)
                f2 = (end - lower_end_time) / (upper_end_time - lower_end_time)
                value = f1 * e + f2 * f
        else:
            # at least one complete time interval lies in between window start/end
            assert lower_start_time <= start <= upper_start_time, \
                f"NOT {lower_start_time} <= {start} <= {upper_start_time}"
            assert lower_end_time <= end <= upper_end_time, \
                f"NOT {lower_end_time} <= {end} <= {upper_end_time}"
            d = self.scape.get_value_at_index(upper_start_idx, lower_end_idx)
            e = self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
            f = self.scape.get_value_at_index(lower_end_idx, upper_end_idx)
            if self.normalise:
                f1 = upper_start_time - start
                f2 = end - lower_end_time
                f3 = lower_end_time - upper_start_time
                value = (f3 * d + f1 * e + f2 * f) / (f1 + f2 + f3)
            else:
                f1 = (upper_start_time - start) / (upper_start_time - lower_start_time)
                f2 = (end - lower_end_time) / (upper_end_time - lower_end_time)
                value = d + f1 * e + f2 * f
        # add prior counts
        self.scape.add_prior_counts(value, end - start)
        return value


class ContinuousScape(Scape):
    """
    Continuous-time scape that takes a discrete-time scape and interpolates linearly for points that lie in between the
    discrete time points. This is the exact continuous generalisation when using DiscretePitchScape as discrete-time
    scape.
    """

    def __init__(self, values=None, scape=None, **kwargs):
        """
        Initialise PitchScape either from count values or from DiscretePitchScape.
        :param values: pitch-class counts (do not provide together with scape)
        :param scape: DiscretePitchScape object (do not provide together with values)
        :param kwargs: keyword arguments passed on to initialize DiscretePitchScape (use only when also providing
        values)
        """
        if (values is None) == (scape is None):
            raise ValueError("Please specify EITHER 'values' (and optional keyword arguments) OR 'scape'")
        if scape is None:
            scape = DiscreteScape(values=values, **kwargs)
        elif kwargs:
            raise TypeError("Cannot take keyword arguments if scape object is given.")
        self.scape = scape
        super().__init__(min_time=self.scape.min_time, max_time=self.scape.max_time)

    def __getitem__(self, item):
        try:
            return self.scape[item]
        except ValueError:
            return self.interpolate(*item)

    @staticmethod
    def get_adjacent_indices(start, end, times):
        # get upper/lower adjacent indices
        upper_start_idx = np.searchsorted(times, start, side='right')
        lower_start_idx = upper_start_idx - 1
        upper_end_idx = np.searchsorted(times, end, side='left')
        lower_end_idx = upper_end_idx - 1
        n_times = len(times)
        # handle floating point round-off errors
        if lower_start_idx == -1 and isclose(start, times[0]):
            lower_start_idx = 0
            upper_start_idx = 1
            start = times[0]
        if upper_end_idx == n_times and isclose(end, times[-1]):
            lower_end_idx = n_times - 2
            upper_end_idx = n_times - 1
            end = times[-1]
        # check bounds
        if lower_start_idx < 0:
            raise ValueError("Start below valid times")
        if upper_end_idx >= n_times:
            raise ValueError("End beyond valid times")
        return start, end, lower_start_idx, upper_start_idx, lower_end_idx, upper_end_idx

    def interpolate(self, start, end):
        # check for zero-size windows
        if start == end:
            return self.scape.zero_size_value.copy()
        # get adjacent indices
        (start,
         end,
         lower_start_idx,
         upper_start_idx,
         lower_end_idx,
         upper_end_idx) = self.get_adjacent_indices(start, end, self.scape.times)
        # get corresponding times
        upper_start_time = self.scape.times[upper_start_idx]
        lower_start_time = self.scape.times[lower_start_idx]
        upper_end_time = self.scape.times[upper_end_idx]
        lower_end_time = self.scape.times[lower_end_idx]
        if lower_start_idx == lower_end_idx and upper_start_idx == upper_end_idx:
            # window start/end lie in the same time interval
            value = self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
            value *= (end - start) / (upper_start_time - lower_start_time)
        elif upper_start_time == lower_end_time:
            # window start/end lie in between two adjacent time intervals
            e = self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
            f = self.scape.get_value_at_index(lower_end_idx, upper_end_idx)
            f1 = (upper_start_time - start) / (upper_start_time - lower_start_time)
            f2 = (end - lower_end_time) / (upper_end_time - lower_end_time)
            value = f1 * e + f2 * f
        else:
            # at least one complete time interval lies in between window start/end
            assert lower_start_time <= start <= upper_start_time, \
                f"NOT {lower_start_time} <= {start} <= {upper_start_time}"
            assert lower_end_time <= end <= upper_end_time, \
                f"NOT {lower_end_time} <= {end} <= {upper_end_time}"
            d = self.scape.get_value_at_index(upper_start_idx, lower_end_idx)
            e = self.scape.get_value_at_index(lower_start_idx, upper_start_idx)
            f = self.scape.get_value_at_index(lower_end_idx, upper_end_idx)
            f1 = (upper_start_time - start) / (upper_start_time - lower_start_time)
            f2 = (end - lower_end_time) / (upper_end_time - lower_end_time)
            value = d + f1 * e + f2 * f
        return value


class PitchScape(Scape):

    def __init__(self,
                 values=None,
                 scape=None,
                 prior_counts=0,
                 normalise=False,
                 **kwargs):
        self.prior_counts = prior_counts
        self.normalise = normalise
        # use float values
        values = np.array(values, dtype=float)
        # dimensionality of the output
        self.n_dim = np.prod(values.shape[1:])
        # value of uniform distribution
        self.uniform = 1 / self.n_dim
        # custom weights not allowed with normalisation
        if self.normalise and "weights" in kwargs:
            raise ValueError("Custom weights are not supported in conjunction with normalisation.")
        # initialise underlying scape
        if (values is None) == (scape is None):
            raise ValueError("Please specify EITHER 'values' (and optional keyword arguments) OR 'scape'")
        if scape is None:
            # if required: normalise allong all axes except the first
            if self.normalise:
                axis = tuple(range(1, len(values.shape)))
                normalize_non_zero(values, axis=axis, replace_zeros=True)
            scape = ContinuousScape(values=values, **kwargs)
        elif kwargs:
            raise TypeError("Cannot take keyword arguments if scape object is given.")
        self.scape = scape
        # init super
        super().__init__(min_time=self.scape.min_time, max_time=self.scape.max_time)

    def __getitem__(self, item):
        # get value from underlying scape
        value = self.scape[item]
        # handle zeros
        if np.all(value == 0):
            if self.prior_counts is None:
                # if prior_counts is None: just return value of zeros
                pass
            elif self.normalise:
                # if prior_counts is not None and normalisation is requested: return uniform distribution
                value += self.uniform
            else:
                # if prior_counts is not None and NO normalisation is requested: add prior counts
                value += self.prior_counts
            return value
        # add prior counts
        if self.prior_counts is not None:
            value += self.prior_counts
        # normalise
        if self.normalise:
            # get time interval for normalisation
            start, end = item
            delta_t = end - start
            # if prior_counts is not None: add to normalisation
            if self.prior_counts is None:
                value /= delta_t
            else:
                value /= delta_t + self.n_dim * self.prior_counts
        return value
