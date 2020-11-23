#  Copyright (c) 2020 Robert Lieck
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from pitchscapes.keyfinding import KeyEstimator
import pitchscapes.plotting as pt


class Test(TestCase):
    def test_key_scores_to_color(self):
        # key estimator
        k = KeyEstimator()
        # prototypical major and minor profile
        major = np.array([[2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1]])  # C major
        minor = np.array([[2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 1, 0]])  # C minor
        # key estimator
        k = KeyEstimator()
        # colour palette
        palette = np.zeros((12, 2, 3))
        palette[:, :, 0] = np.linspace(0, 1, 12)[:, None]
        palette[:, 0, 1] = 0
        palette[:, 1, 1] = 1
        # roll through all transpositions
        for trans in range(12):
            major_scores = k.get_score(np.roll(major, shift=trans, axis=1))
            minor_scores = k.get_score(np.roll(minor, shift=trans, axis=1))
            # circle of fifths: False (via parameter)
            assert_array_equal(palette[trans, 0], pt.key_scores_to_color(major_scores,
                                                                         soft_max_temperature=1e-10,
                                                                         palette=palette,
                                                                         circle_of_fifths=False)[0][:3])
            assert_array_equal(palette[trans, 1], pt.key_scores_to_color(minor_scores,
                                                                         soft_max_temperature=1e-10,
                                                                         palette=palette,
                                                                         circle_of_fifths=False)[0][:3])
            # circle of fifths: True (via parameter)
            assert_array_equal(palette[trans * 7 % 12, 0], pt.key_scores_to_color(major_scores,
                                                                                  soft_max_temperature=1e-10,
                                                                                  palette=palette,
                                                                                  circle_of_fifths=True)[0][:3])
            assert_array_equal(palette[(trans * 7 - 3) % 12, 1], pt.key_scores_to_color(minor_scores,
                                                                                        soft_max_temperature=1e-10,
                                                                                        palette=palette,
                                                                                        circle_of_fifths=True)[0][:3])
            # circle of fifths: False (via default)
            pt.set_circle_of_fifths(False)
            assert_array_equal(palette[trans, 0], pt.key_scores_to_color(major_scores,
                                                                         soft_max_temperature=1e-10,
                                                                         palette=palette)[0][:3])
            assert_array_equal(palette[trans, 1], pt.key_scores_to_color(minor_scores,
                                                                         soft_max_temperature=1e-10,
                                                                         palette=palette)[0][:3])
            # circle of fifths: True (via default)
            pt.set_circle_of_fifths(True)
            assert_array_equal(palette[trans * 7 % 12, 0], pt.key_scores_to_color(major_scores,
                                                                                  soft_max_temperature=1e-10,
                                                                                  palette=palette)[0][:3])
            assert_array_equal(palette[(trans * 7 - 3) % 12, 1], pt.key_scores_to_color(minor_scores,
                                                                                        soft_max_temperature=1e-10,
                                                                                        palette=palette)[0][:3])
