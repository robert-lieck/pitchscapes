from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

import pitchscapes.reader as rd
from pitchscapes.scapes import PitchScape


class Test(TestCase):
    def test_piano_roll(self):
        # only get piano roll
        piano_roll = rd.piano_roll('tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl')
        # also return pitch range and durations
        _, pitch_range, durations = rd.piano_roll('tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl',
                                                  return_range=True,
                                                  return_durations=True)
        # check for matching dimensions
        self.assertEqual(piano_roll.shape[0], len(durations))
        self.assertEqual(piano_roll.shape[1], len(pitch_range))
        # make sure inconsistent min/max pitch value raise
        self.assertRaises(ValueError,
                          lambda: rd.piano_roll('tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl', min_pitch=37))
        self.assertRaises(ValueError,
                          lambda: rd.piano_roll('tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl', max_pitch=80))

    def test_sample_density(self):
        data = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        for normalise in [False, True]:
            scape = PitchScape(values=data, normalise=normalise)
            if normalise:
                assert_array_almost_equal([[1, 0, 0, 0],
                                           [1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 0, 1]],
                                          rd.sample_density(n_time_intervals=8, scape=scape))
            else:
                assert_array_almost_equal([[0.5, 0, 0, 0],
                                           [0.5, 0, 0, 0],
                                           [0, 0.5, 0, 0],
                                           [0, 0.5, 0, 0],
                                           [0, 0, 0.5, 0],
                                           [0, 0, 0.5, 0],
                                           [0, 0, 0, 0.5],
                                           [0, 0, 0, 0.5]],
                                          rd.sample_density(n_time_intervals=8, scape=scape))

    def test_sample_scape(self):
        data = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        for normalise in [False, True]:
            scape = PitchScape(values=data, normalise=normalise)
            if normalise:
                assert_array_almost_equal(
                    [[1, 0, 0, 0], [1 / 2, 1 / 2, 0, 0], [1 / 3, 1 / 3, 1 / 3, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                     [0, 1, 0, 0], [0, 1 / 2, 1 / 2, 0], [0, 1 / 3, 1 / 3, 1 / 3],
                     [0, 0, 1, 0], [0, 0, 1 / 2, 1 / 2],
                     [0, 0, 0, 1]],
                    rd.sample_scape(n_time_intervals=4, scape=scape)
                )
            else:
                assert_array_almost_equal(
                    [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1],
                     [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1],
                     [0, 0, 1, 0], [0, 0, 1, 1],
                     [0, 0, 0, 1]],
                    rd.sample_scape(n_time_intervals=4, scape=scape)
                )
