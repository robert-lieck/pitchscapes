#  Copyright (c) 2020 Robert Lieck
from unittest import TestCase

import numpy as np
from numpy import nan
from numpy.testing import assert_array_equal

from pitchscapes.util import coords_from_times


class Test(TestCase):
    def test_coords_from_times(self):
        times = [1, 2, 3, 4, 5]
        for remove_offset in [False, True]:
            for unit_times in [False, True]:
                for return_all in [False, True]:
                    if return_all:
                        (start_end_idx,
                         start_end_time,
                         center_width,
                         coords) = coords_from_times(times=times,
                                                     start_end_idx=True,
                                                     start_end_time=True,
                                                     center_width=True,
                                                     coords=True,
                                                     remove_offset=remove_offset,
                                                     unit_times=unit_times)
                    else:
                        start_end_idx = coords_from_times(times=times,
                                                          start_end_idx=True,
                                                          remove_offset=remove_offset,
                                                          unit_times=unit_times)
                        start_end_time = coords_from_times(times=times,
                                                           start_end_time=True,
                                                           remove_offset=remove_offset,
                                                           unit_times=unit_times)
                        center_width = coords_from_times(times=times,
                                                         center_width=True,
                                                         remove_offset=remove_offset,
                                                         unit_times=unit_times)
                        coords = coords_from_times(times=times,
                                                   coords=True,
                                                   remove_offset=remove_offset,
                                                   unit_times=unit_times)
                    assert_array_equal([[0, 1], [0, 2], [0, 3], [0, 4],
                                        [1, 2], [1, 3], [1, 4],
                                        [2, 3], [2, 4],
                                        [3, 4]],
                                       start_end_idx)
                    if remove_offset:
                        if unit_times:
                            assert_array_equal([[0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
                                                [0.25, 0.5], [0.25, 0.75], [0.25, 1],
                                                [0.5, 0.75], [0.5, 1],
                                                [0.75, 1]],
                                               start_end_time)
                            assert_array_equal([[0.125, 0.25], [0.25, 0.5], [0.375, 0.75], [0.5, 1],
                                                [0.375, 0.25], [0.5, 0.5], [0.625, 0.75],
                                                [0.625, 0.25], [0.75, 0.5],
                                                [0.875, 0.25]],
                                               center_width)
                            assert_array_equal([[[0, 0], [0.125, 0.25], [0.25, 0], [nan, nan]],
                                                [[0.125, 0.25], [0.25, 0.5], [0.375, 0.25], [0.25, 0]],
                                                [[0.25, 0.5], [0.375, 0.75], [0.5, 0.5], [0.375, 0.25]],
                                                [[0.375, 0.75], [0.5, 1], [0.625, 0.75], [0.5, 0.5]],
                                                [[0.25, 0], [0.375, 0.25], [0.5, 0], [nan, nan]],
                                                [[0.375, 0.25], [0.5, 0.5], [0.625, 0.25], [0.5, 0]],
                                                [[0.5, 0.5], [0.625, 0.75], [0.75, 0.5], [0.625, 0.25]],
                                                [[0.5, 0], [0.625, 0.25], [0.75, 0], [nan, nan]],
                                                [[0.625, 0.25], [0.75, 0.5], [0.875, 0.25], [0.75, 0]],
                                                [[0.75, 0], [0.875, 0.25], [1, 0], [nan, nan]]],
                                               coords)
                        else:
                            assert_array_equal([[0, 1], [0, 2], [0, 3], [0, 4],
                                                [1, 2], [1, 3], [1, 4],
                                                [2, 3], [2, 4],
                                                [3, 4]],
                                               start_end_time)
                            assert_array_equal([[0.5, 1], [1, 2], [1.5, 3], [2, 4],
                                                [1.5, 1], [2, 2], [2.5, 3],
                                                [2.5, 1], [3, 2],
                                                [3.5, 1]],
                                               center_width)
                            assert_array_equal([[[0, 0], [0.5, 1], [1, 0], [nan, nan]],
                                                [[0.5, 1], [1, 2], [1.5, 1], [1, 0]],
                                                [[1, 2], [1.5, 3], [2, 2], [1.5, 1]],
                                                [[1.5, 3], [2, 4], [2.5, 3], [2, 2]],
                                                [[1, 0], [1.5, 1], [2, 0], [nan, nan]],
                                                [[1.5, 1], [2, 2], [2.5, 1], [2, 0]],
                                                [[2, 2], [2.5, 3], [3, 2], [2.5, 1]],
                                                [[2, 0], [2.5, 1], [3, 0], [nan, nan]],
                                                [[2.5, 1], [3, 2], [3.5, 1], [3, 0]],
                                                [[3, 0], [3.5, 1], [4, 0], [nan, nan]]],
                                               coords)
                    else:
                        if unit_times:
                            assert_array_equal([[1 / 5, 2 / 5], [1 / 5, 3 / 5], [1 / 5, 4 / 5], [1 / 5, 1],
                                                [2 / 5, 3 / 5], [2 / 5, 4 / 5], [2 / 5, 1],
                                                [3 / 5, 4 / 5], [3 / 5, 1],
                                                [4 / 5, 1]],
                                               start_end_time)
                            assert_array_equal([[1.5 / 5, 1 / 5], [2 / 5, 2 / 5], [2.5 / 5, 3 / 5], [3 / 5, 4 / 5],
                                                [2.5 / 5, 1 / 5], [3 / 5, 2 / 5], [3.5 / 5, 3 / 5],
                                                [3.5 / 5, 1 / 5], [4 / 5, 2 / 5],
                                                [4.5 / 5, 1 / 5]],
                                               center_width)
                            assert_array_equal([[[1 / 5, 0], [1.5 / 5, 1 / 5], [2 / 5, 0], [nan, nan]],
                                                [[1.5 / 5, 1 / 5], [2 / 5, 2 / 5], [2.5 / 5, 1 / 5], [2 / 5, 0]],
                                                [[2 / 5, 2 / 5], [2.5 / 5, 3 / 5], [3 / 5, 2 / 5], [2.5 / 5, 1 / 5]],
                                                [[2.5 / 5, 3 / 5], [3 / 5, 4 / 5], [3.5 / 5, 3 / 5], [3 / 5, 2 / 5]],
                                                [[2 / 5, 0], [2.5 / 5, 1 / 5], [3 / 5, 0], [nan, nan]],
                                                [[2.5 / 5, 1 / 5], [3 / 5, 2 / 5], [3.5 / 5, 1 / 5], [3 / 5, 0]],
                                                [[3 / 5, 2 / 5], [3.5 / 5, 3 / 5], [4 / 5, 2 / 5], [3.5 / 5, 1 / 5]],
                                                [[3 / 5, 0], [3.5 / 5, 1 / 5], [4 / 5, 0], [nan, nan]],
                                                [[3.5 / 5, 1 / 5], [4 / 5, 2 / 5], [4.5 / 5, 1 / 5], [4 / 5, 0]],
                                                [[4 / 5, 0], [4.5 / 5, 1 / 5], [1, 0], [nan, nan]]],
                                               coords)
                        else:
                            assert_array_equal([[1, 2], [1, 3], [1, 4], [1, 5],
                                                [2, 3], [2, 4], [2, 5],
                                                [3, 4], [3, 5],
                                                [4, 5]],
                                               start_end_time)
                            assert_array_equal([[1.5, 1], [2, 2], [2.5, 3], [3, 4],
                                                [2.5, 1], [3, 2], [3.5, 3],
                                                [3.5, 1], [4, 2],
                                                [4.5, 1]],
                                               center_width)
                            assert_array_equal([[[1, 0], [1.5, 1], [2, 0], [nan, nan]],
                                                [[1.5, 1], [2, 2], [2.5, 1], [2, 0]],
                                                [[2, 2], [2.5, 3], [3, 2], [2.5, 1]],
                                                [[2.5, 3], [3, 4], [3.5, 3], [3, 2]],
                                                [[2, 0], [2.5, 1], [3, 0], [nan, nan]],
                                                [[2.5, 1], [3, 2], [3.5, 1], [3, 0]],
                                                [[3, 2], [3.5, 3], [4, 2], [3.5, 1]],
                                                [[3, 0], [3.5, 1], [4, 0], [nan, nan]],
                                                [[3.5, 1], [4, 2], [4.5, 1], [4, 0]],
                                                [[4, 0], [4.5, 1], [5, 0], [nan, nan]]],
                                               coords)
