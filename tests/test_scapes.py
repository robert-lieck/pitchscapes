#  Copyright (c) 2020 Robert Lieck
from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_almost_equal
from pitchscapes.scapes import DiscreteScape, ContinuousScape, PitchScape


class TestPitchScape(TestCase):

    def test_scape(self):
        # data (as int)
        data = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
        for strategy in ['left', 'right', 'center']:
            for weighting in ["uniform", "times", "weights"]:
                if weighting == "uniform":
                    # |x|x|x|x|x|
                    times = None
                    weights = None
                elif weighting == "times":
                    # |xx|x|xx|x|xx|
                    times = [0, 2, 3, 5, 6, 8]
                    weights = None
                elif weighting == "weights":
                    times = None
                    # |xx|x|xx|x|xx|
                    weights = [2, 1, 2, 1, 2]
                else:
                    self.fail("Unhandled weighting scheme")
                discrete_scape = DiscreteScape(values=data, times=times, weights=weights, strategy=strategy)
                continuous_scape = ContinuousScape(scape=discrete_scape)
                # check both discrete and continuous scape
                for discrete in [True, False]:
                    if discrete:
                        pitch_scape = discrete_scape
                    else:
                        pitch_scape = continuous_scape
                    if weighting == "uniform":
                        # check width-5 time slot
                        assert_array_almost_equal([2, 1, 1, 1], pitch_scape[0, 5])
                        # check width-0 time slots
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[0, 0])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[1, 1])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[2, 2])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[3, 3])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[4, 4])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[5, 5])
                        # check width-1 time slots
                        assert_array_almost_equal([1, 0, 0, 0], pitch_scape[0, 1])
                        assert_array_almost_equal([0, 1, 0, 0], pitch_scape[1, 2])
                        assert_array_almost_equal([0, 0, 1, 0], pitch_scape[2, 3])
                        assert_array_almost_equal([0, 0, 0, 1], pitch_scape[3, 4])
                        assert_array_almost_equal([1, 0, 0, 0], pitch_scape[4, 5])
                        if not discrete:
                            # check width-1 time slots shifted by 0.5
                            assert_array_almost_equal([0.5, 0.5, 0, 0],
                                                      pitch_scape[0.5, 1.5])
                            assert_array_almost_equal([0, 0.5, 0.5, 0],
                                                      pitch_scape[1.5, 2.5])
                            assert_array_almost_equal([0, 0, 0.5, 0.5],
                                                      pitch_scape[2.5, 3.5])
                            assert_array_almost_equal([0.5, 0, 0, 0.5],
                                                      pitch_scape[3.5, 4.5])
                        # check width-2 time slots
                        assert_array_almost_equal([1, 1, 0, 0], pitch_scape[0, 2])
                        assert_array_almost_equal([0, 1, 1, 0], pitch_scape[1, 3])
                        assert_array_almost_equal([0, 0, 1, 1], pitch_scape[2, 4])
                        assert_array_almost_equal([1, 0, 0, 1], pitch_scape[3, 5])
                        # check width-3 time slots
                        assert_array_almost_equal([1, 1, 1, 0], pitch_scape[0, 3])
                        assert_array_almost_equal([0, 1, 1, 1], pitch_scape[1, 4])
                        assert_array_almost_equal([1, 0, 1, 1], pitch_scape[2, 5])
                        # check width-4 time slots
                        assert_array_almost_equal([1, 1, 1, 1], pitch_scape[0, 4])
                        assert_array_almost_equal([1, 1, 1, 1], pitch_scape[1, 5])
                    elif weighting == "times":
                        # check width-5 time slot
                        assert_array_almost_equal([4, 1, 2, 1], pitch_scape[0, 8])
                        # check width-0 time slots
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[0, 0])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[2, 2])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[3, 3])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[5, 5])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[6, 6])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[8, 8])
                        # check width-1 time slots
                        assert_array_almost_equal([2, 0, 0, 0], pitch_scape[0, 2])
                        assert_array_almost_equal([0, 1, 0, 0], pitch_scape[2, 3])
                        assert_array_almost_equal([0, 0, 2, 0], pitch_scape[3, 5])
                        assert_array_almost_equal([0, 0, 0, 1], pitch_scape[5, 6])
                        assert_array_almost_equal([2, 0, 0, 0], pitch_scape[6, 8])
                        if not discrete:
                            # check width-1 time slots shifted by 0.5
                            assert_array_almost_equal([1, 0.5, 0, 0],
                                                      pitch_scape[1, 2.5])
                            assert_array_almost_equal([0, 0.5, 1, 0],
                                                      pitch_scape[2.5, 4])
                            assert_array_almost_equal([0, 0, 1, 0.5],
                                                      pitch_scape[4, 5.5])
                            assert_array_almost_equal([1, 0, 0, 0.5],
                                                      pitch_scape[5.5, 7])
                        # check width-2 time slots
                        assert_array_almost_equal([2, 1, 0, 0], pitch_scape[0, 3])
                        assert_array_almost_equal([0, 1, 2, 0], pitch_scape[2, 5])
                        assert_array_almost_equal([0, 0, 2, 1], pitch_scape[3, 6])
                        assert_array_almost_equal([2, 0, 0, 1], pitch_scape[5, 8])
                        # check width-3 time slots
                        assert_array_almost_equal([2, 1, 2, 0], pitch_scape[0, 5])
                        assert_array_almost_equal([0, 1, 2, 1], pitch_scape[2, 6])
                        assert_array_almost_equal([2, 0, 2, 1], pitch_scape[3, 8])
                        # check width-4 time slots
                        assert_array_almost_equal([2, 1, 2, 1], pitch_scape[0, 6])
                        assert_array_almost_equal([2, 1, 2, 1], pitch_scape[2, 8])
                    elif weighting == "weights":
                        # check width-5 time slot
                        assert_array_almost_equal([4, 1, 2, 1], pitch_scape[0, 5])
                        # check width-0 time slots
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[0, 0])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[1, 1])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[2, 2])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[3, 3])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[4, 4])
                        assert_array_almost_equal([0, 0, 0, 0], pitch_scape[5, 5])
                        # check width-1 time slots
                        assert_array_almost_equal([2, 0, 0, 0], pitch_scape[0, 1])
                        assert_array_almost_equal([0, 1, 0, 0], pitch_scape[1, 2])
                        assert_array_almost_equal([0, 0, 2, 0], pitch_scape[2, 3])
                        assert_array_almost_equal([0, 0, 0, 1], pitch_scape[3, 4])
                        assert_array_almost_equal([2, 0, 0, 0], pitch_scape[4, 5])
                        if not discrete:
                            # check width-1 time slots shifted by 0.5
                            assert_array_almost_equal([1, 0.5, 0, 0],
                                                      pitch_scape[0.5, 1.5])
                            assert_array_almost_equal([0, 0.5, 1, 0],
                                                      pitch_scape[1.5, 2.5])
                            assert_array_almost_equal([0, 0, 1, 0.5],
                                                      pitch_scape[2.5, 3.5])
                            assert_array_almost_equal([1, 0, 0, 0.5],
                                                      pitch_scape[3.5, 4.5])
                        # check width-2 time slots
                        assert_array_almost_equal([2, 1, 0, 0], pitch_scape[0, 2])
                        assert_array_almost_equal([0, 1, 2, 0], pitch_scape[1, 3])
                        assert_array_almost_equal([0, 0, 2, 1], pitch_scape[2, 4])
                        assert_array_almost_equal([2, 0, 0, 1], pitch_scape[3, 5])
                        # check width-3 time slots
                        assert_array_almost_equal([2, 1, 2, 0], pitch_scape[0, 3])
                        assert_array_almost_equal([0, 1, 2, 1], pitch_scape[1, 4])
                        assert_array_almost_equal([2, 0, 2, 1], pitch_scape[2, 5])
                        # check width-4 time slots
                        assert_array_almost_equal([2, 1, 2, 1], pitch_scape[0, 4])
                        assert_array_almost_equal([2, 1, 2, 1], pitch_scape[1, 5])
                    else:
                        self.fail("Unhandled weighting scheme")

    def test_multidimensional(self):
        data = np.array([[[1, 0, 0, 0], [1, 0, 0, 0]],
                         [[0, 1, 0, 0], [0, 1, 0, 0]],
                         [[0, 0, 1, 0], [0, 0, 1, 0]],
                         [[0, 0, 0, 1], [0, 0, 0, 1]]], dtype=float)
        zeros = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        for normalise in [False, True]:
            for prior_counts in [None, 0, 1]:
                pitch_scape = PitchScape(values=data, normalise=normalise, prior_counts=prior_counts)
                # check width-4 time slot
                if normalise:
                    pitch_scape = PitchScape(values=data, normalise=normalise, prior_counts=prior_counts)
                    assert_array_almost_equal(zeros + 0.125, pitch_scape[0, 4])
                else:
                    value = zeros + 1
                    if prior_counts is not None:
                        value += prior_counts
                    assert_array_almost_equal(value, pitch_scape[0, 4])
                # check width-0 time slots
                if prior_counts is None or (prior_counts == 0 and not normalise):
                    value = zeros
                else:
                    if normalise:
                        value = zeros + 0.125
                    else:
                        value = zeros + prior_counts
                assert_array_almost_equal(value, pitch_scape[0, 0])
                assert_array_almost_equal(value, pitch_scape[1, 1])
                assert_array_almost_equal(value, pitch_scape[2, 2])
                assert_array_almost_equal(value, pitch_scape[3, 3])
                assert_array_almost_equal(value, pitch_scape[4, 4])
                # check width-1 time slots
                val_0_1 = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
                val_1_2 = np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=float)
                val_2_3 = np.array([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=float)
                val_3_4 = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=float)
                for v in [val_0_1, val_1_2, val_2_3, val_3_4]:
                    if normalise:
                        v /= v.sum()
                    if prior_counts is not None:
                        v += prior_counts
                    if normalise:
                        v /= v.sum()
                assert_array_almost_equal(val_0_1, pitch_scape[0, 1])
                assert_array_almost_equal(val_1_2, pitch_scape[1, 2])
                assert_array_almost_equal(val_2_3, pitch_scape[2, 3])
                assert_array_almost_equal(val_3_4, pitch_scape[3, 4])
                # check width-1 time slots shifted by 0.5
                val_1 = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]])
                val_2 = np.array([[0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0]])
                val_3 = np.array([[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]])
                for v in [val_1, val_2, val_3]:
                    if normalise:
                        v /= 2
                    if prior_counts is not None:
                        v += prior_counts
                    if normalise:
                        v /= v.sum()
                assert_array_almost_equal(val_1, pitch_scape[0.5, 1.5])
                assert_array_almost_equal(val_2, pitch_scape[1.5, 2.5])
                assert_array_almost_equal(val_3, pitch_scape[2.5, 3.5])
                # check width-2 time slots
                val_1 = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=float)
                val_2 = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=float)
                val_3 = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=float)
                for v in [val_1, val_2, val_3]:
                    if normalise:
                        v /= 2
                    if prior_counts is not None:
                        v += prior_counts
                    if normalise:
                        v /= v.sum()
                assert_array_almost_equal(val_1, pitch_scape[0, 2])
                assert_array_almost_equal(val_2, pitch_scape[1, 3])
                assert_array_almost_equal(val_3, pitch_scape[2, 4])
                # check width-3 time slots
                val_1 = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=float)
                val_2 = np.array([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=float)
                for v in [val_1, val_2]:
                    if normalise:
                        v /= 2
                    if prior_counts is not None:
                        v += prior_counts
                    if normalise:
                        v /= v.sum()
                assert_array_almost_equal(val_1, pitch_scape[0, 3])
                assert_array_almost_equal(val_2, pitch_scape[1, 4])

    def test_pitch_scape(self):
        # data (as int)
        data = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
        for strategy in ['left', 'right', 'center']:
            for normalise in [False, True]:
                for pcs in [None, 0, 0.5, 1]:
                    if pcs is None:
                        # have to use zero in most places for None
                        pcs_ = 0
                    else:
                        pcs_ = pcs
                    pitch_scape = PitchScape(values=data,
                                             normalise=normalise,
                                             strategy=strategy,
                                             prior_counts=pcs)
                    try:
                        if normalise:
                            # check width-5 time slot
                            assert_array_almost_equal([(2 + pcs_) / (5 + 4 * pcs_),
                                                       (1 + pcs_) / (5 + 4 * pcs_),
                                                       (1 + pcs_) / (5 + 4 * pcs_),
                                                       (1 + pcs_) / (5 + 4 * pcs_)], pitch_scape[0, 5])
                            # check width-0 time slots
                            if pcs is None:
                                # should return zeros even for normalisation
                                assert_array_almost_equal([0, 0, 0, 0], pitch_scape[0, 0])
                                assert_array_almost_equal([0, 0, 0, 0], pitch_scape[1, 1])
                                assert_array_almost_equal([0, 0, 0, 0], pitch_scape[2, 2])
                                assert_array_almost_equal([0, 0, 0, 0], pitch_scape[3, 3])
                                assert_array_almost_equal([0, 0, 0, 0], pitch_scape[4, 4])
                                assert_array_almost_equal([0, 0, 0, 0], pitch_scape[5, 5])
                            else:
                                # should return uniform distribution (even for zeros prior counts)
                                assert_array_almost_equal([1/4, 1/4, 1/4, 1/4], pitch_scape[0, 0])
                                assert_array_almost_equal([1/4, 1/4, 1/4, 1/4], pitch_scape[1, 1])
                                assert_array_almost_equal([1/4, 1/4, 1/4, 1/4], pitch_scape[2, 2])
                                assert_array_almost_equal([1/4, 1/4, 1/4, 1/4], pitch_scape[3, 3])
                                assert_array_almost_equal([1/4, 1/4, 1/4, 1/4], pitch_scape[4, 4])
                                assert_array_almost_equal([1/4, 1/4, 1/4, 1/4], pitch_scape[5, 5])
                            # check width-1 time slots
                            assert_array_almost_equal([(1 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_)], pitch_scape[0, 1])
                            assert_array_almost_equal([pcs_ / (1 + 4 * pcs_),
                                                       (1 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_)], pitch_scape[1, 2])
                            assert_array_almost_equal([pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       (1 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_)], pitch_scape[2, 3])
                            assert_array_almost_equal([pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       (1 + pcs_) / (1 + 4 * pcs_)], pitch_scape[3, 4])
                            assert_array_almost_equal([(1 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_)], pitch_scape[4, 5])
                            # check width-1 time slots shifted by 0.5
                            assert_array_almost_equal([(0.5 + pcs_) / (1 + 4 * pcs_),
                                                       (0.5 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_)], pitch_scape[0.5, 1.5])
                            assert_array_almost_equal([pcs_ / (1 + 4 * pcs_),
                                                       (0.5 + pcs_) / (1 + 4 * pcs_),
                                                       (0.5 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_)], pitch_scape[1.5, 2.5])
                            assert_array_almost_equal([pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       (0.5 + pcs_) / (1 + 4 * pcs_),
                                                       (0.5 + pcs_) / (1 + 4 * pcs_)], pitch_scape[2.5, 3.5])
                            assert_array_almost_equal([(0.5 + pcs_) / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       pcs_ / (1 + 4 * pcs_),
                                                       (0.5 + pcs_) / (1 + 4 * pcs_)], pitch_scape[3.5, 4.5])
                            # check width-2 time slots
                            assert_array_almost_equal([(1 + pcs_) / (2 + 4 * pcs_),
                                                       (1 + pcs_) / (2 + 4 * pcs_),
                                                       pcs_ / (2 + 4 * pcs_),
                                                       pcs_ / (2 + 4 * pcs_)], pitch_scape[0, 2])
                            assert_array_almost_equal([pcs_ / (2 + 4 * pcs_),
                                                       (1 + pcs_) / (2 + 4 * pcs_),
                                                       (1 + pcs_) / (2 + 4 * pcs_),
                                                       pcs_ / (2 + 4 * pcs_)], pitch_scape[1, 3])
                            assert_array_almost_equal([pcs_ / (2 + 4 * pcs_),
                                                       pcs_ / (2 + 4 * pcs_),
                                                       (1 + pcs_) / (2 + 4 * pcs_),
                                                       (1 + pcs_) / (2 + 4 * pcs_)], pitch_scape[2, 4])
                            assert_array_almost_equal([(1 + pcs_) / (2 + 4 * pcs_),
                                                       pcs_ / (2 + 4 * pcs_),
                                                       pcs_ / (2 + 4 * pcs_),
                                                       (1 + pcs_) / (2 + 4 * pcs_)], pitch_scape[3, 5])
                            # check width-3 time slots
                            assert_array_almost_equal([(1 + pcs_) / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_),
                                                       pcs_ / (3 + 4 * pcs_)], pitch_scape[0, 3])
                            assert_array_almost_equal([pcs_ / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_)], pitch_scape[1, 4])
                            assert_array_almost_equal([(1 + pcs_) / (3 + 4 * pcs_),
                                                       pcs_ / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_),
                                                       (1 + pcs_) / (3 + 4 * pcs_)], pitch_scape[2, 5])
                            # check width-4 time slots
                            assert_array_almost_equal([(1 + pcs_) / (4 + 4 * pcs_),
                                                       (1 + pcs_) / (4 + 4 * pcs_),
                                                       (1 + pcs_) / (4 + 4 * pcs_),
                                                       (1 + pcs_) / (4 + 4 * pcs_)], pitch_scape[0, 4])
                            assert_array_almost_equal([(1 + pcs_) / (4 + 4 * pcs_),
                                                       (1 + pcs_) / (4 + 4 * pcs_),
                                                       (1 + pcs_) / (4 + 4 * pcs_),
                                                       (1 + pcs_) / (4 + 4 * pcs_)], pitch_scape[1, 5])
                        else:
                            # check width-5 time slot
                            assert_array_almost_equal([2 + pcs_, 1 + pcs_, 1 + pcs_, 1 + pcs_], pitch_scape[0, 5])
                            # check width-0 time slots
                            assert_array_almost_equal([pcs_, pcs_, pcs_, pcs_], pitch_scape[0, 0])
                            assert_array_almost_equal([pcs_, pcs_, pcs_, pcs_], pitch_scape[1, 1])
                            assert_array_almost_equal([pcs_, pcs_, pcs_, pcs_], pitch_scape[2, 2])
                            assert_array_almost_equal([pcs_, pcs_, pcs_, pcs_], pitch_scape[3, 3])
                            assert_array_almost_equal([pcs_, pcs_, pcs_, pcs_], pitch_scape[4, 4])
                            assert_array_almost_equal([pcs_, pcs_, pcs_, pcs_], pitch_scape[5, 5])
                            # check width-1 time slots
                            assert_array_almost_equal([1 + pcs_, pcs_, pcs_, pcs_], pitch_scape[0, 1])
                            assert_array_almost_equal([pcs_, 1 + pcs_, pcs_, pcs_], pitch_scape[1, 2])
                            assert_array_almost_equal([pcs_, pcs_, 1 + pcs_, pcs_], pitch_scape[2, 3])
                            assert_array_almost_equal([pcs_, pcs_, pcs_, 1 + pcs_], pitch_scape[3, 4])
                            assert_array_almost_equal([1 + pcs_, pcs_, pcs_, pcs_], pitch_scape[4, 5])
                            # check width-1 time slots shifted by 0.5
                            assert_array_almost_equal([0.5 + pcs_, 0.5 + pcs_, pcs_, pcs_],
                                                      pitch_scape[0.5, 1.5])
                            assert_array_almost_equal([pcs_, 0.5 + pcs_, 0.5 + pcs_, pcs_],
                                                      pitch_scape[1.5, 2.5])
                            assert_array_almost_equal([pcs_, pcs_, 0.5 + pcs_, 0.5 + pcs_],
                                                      pitch_scape[2.5, 3.5])
                            assert_array_almost_equal([0.5 + pcs_, pcs_, pcs_, 0.5 + pcs_],
                                                      pitch_scape[3.5, 4.5])
                            # check width-2 time slots
                            assert_array_almost_equal([1 + pcs_, 1 + pcs_, pcs_, pcs_], pitch_scape[0, 2])
                            assert_array_almost_equal([pcs_, 1 + pcs_, 1 + pcs_, pcs_], pitch_scape[1, 3])
                            assert_array_almost_equal([pcs_, pcs_, 1 + pcs_, 1 + pcs_], pitch_scape[2, 4])
                            assert_array_almost_equal([1 + pcs_, pcs_, pcs_, 1 + pcs_], pitch_scape[3, 5])
                            # check width-3 time slots
                            assert_array_almost_equal([1 + pcs_, 1 + pcs_, 1 + pcs_, pcs_], pitch_scape[0, 3])
                            assert_array_almost_equal([pcs_, 1 + pcs_, 1 + pcs_, 1 + pcs_], pitch_scape[1, 4])
                            assert_array_almost_equal([1 + pcs_, pcs_, 1 + pcs_, 1 + pcs_], pitch_scape[2, 5])
                            # check width-4 time slots
                            assert_array_almost_equal([1 + pcs_, 1 + pcs_, 1 + pcs_, 1 + pcs_], pitch_scape[0, 4])
                            assert_array_almost_equal([1 + pcs_, 1 + pcs_, 1 + pcs_, 1 + pcs_], pitch_scape[1, 5])
                    except Exception:
                        print(f"strategy: {strategy}\nnormalise: {normalise}\npcs: {pcs}")
                        raise

    def test_normalisation(self):
        # np.random.seed(0)
        length = 100
        for prior_counts in [None, 0] + list(np.random.uniform(0, 100, 10)):
            counts = np.random.randint(0, 3, (length, 5)).astype(float)
            times = np.linspace(0, length, length + 1)
            times[1:-1] += np.random.uniform(-0.1, 0.1, length - 1)
            scape = PitchScape(values=counts,
                               times=times,
                               normalise=True,
                               prior_counts=prior_counts)
            scape.scape.scape.parse_bottom_up()
            # check non-zero intervals
            for a, b in np.random.uniform(0, length, (100, 2)):
                start, end = sorted([a, b])
                self.assertAlmostEqual(1, scape[start, end].sum())
            # check zero intervals
            for a, b in [(x, x) for x in np.random.uniform(0, length, 100)]:
                start, end = sorted([a, b])
                if prior_counts is None:
                    self.assertAlmostEqual(0, scape[start, end].sum())
                else:
                    self.assertAlmostEqual(1, scape[start, end].sum())
