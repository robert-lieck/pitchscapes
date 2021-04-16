#  Copyright (c) 2020 Robert Lieck
from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
from pitchscapes.keyfinding import KeyEstimator
from pitchscapes.util import pitch_classes_sharp, key_estimates_to_str


class TestKeyEstimator(TestCase):

    def test_score(self):
        profiles = np.array([[1, 2, 3, 4, 5],
                             [5, 4, 3, 2, 1]])
        counts = np.array([[1, 2, 3, 4, 5],
                           [5, 1, 2, 3, 4],
                           [4, 5, 1, 2, 3],
                           [5, 4, 3, 2, 1],
                           [1, 5, 4, 3, 2],
                           [2, 1, 5, 4, 3]])
        k = KeyEstimator(profiles=profiles, normalise_counts=False, normalise_profiles=False)
        scores = KeyEstimator.score(counts=counts, profiles=profiles, normalise_counts=False, normalise_profiles=False)
        np.testing.assert_array_equal(scores, k.get_score(counts))
        self.assertEqual((counts.shape[0], profiles.shape[0], 5), scores.shape)
        estimates = k.get_estimate(counts=counts)
        np.testing.assert_array_equal([[0, 0], [0, 1], [0, 2],
                                       [1, 0], [1, 1], [1, 2]], estimates)

    def test_classification(self):
        # prototypical major and minor profile
        major = np.array([[2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1]])  # C major
        minor = np.array([[2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 1, 0]])  # C minor
        # key estimator
        k = KeyEstimator()
        # roll through all transpositions
        for trans in range(12):
            # major
            assert_array_equal([[0, trans]], k.get_estimate(np.roll(major, shift=trans, axis=1)))
            assert_array_equal([pitch_classes_sharp[trans]],
                               key_estimates_to_str(k.get_estimate(np.roll(major, shift=trans, axis=1)),
                                                    use_capitalisation=True))
            assert_array_equal([pitch_classes_sharp[trans] + " major"],
                               key_estimates_to_str(k.get_estimate(np.roll(major, shift=trans, axis=1)),
                                                    use_capitalisation=False))
            # minor
            assert_array_equal([[1, trans]], k.get_estimate(np.roll(minor, shift=trans, axis=1)))
            assert_array_equal([pitch_classes_sharp[trans].lower()],
                               key_estimates_to_str(k.get_estimate(np.roll(minor, shift=trans, axis=1)),
                                                    use_capitalisation=True))
            assert_array_equal([pitch_classes_sharp[trans] + " minor"],
                               key_estimates_to_str(k.get_estimate(np.roll(minor, shift=trans, axis=1)),
                                                    use_capitalisation=False))
