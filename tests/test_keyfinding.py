#  Copyright (c) 2020 Robert Lieck
from unittest import TestCase
import numpy as np
from pitchscapes.keyfinding import KeyEstimator


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
