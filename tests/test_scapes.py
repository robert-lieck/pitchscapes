#  Copyright (c) 2020 Robert Lieck
from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_almost_equal
from pitchscapes.scapes import PitchScape, DiscretePitchScape


class TestPitchScape(TestCase):

    def test_scape(self):
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
                    # initialisation from int with normalisation should raise TypeError
                    self.assertRaises(TypeError, lambda: DiscretePitchScape(values=data,
                                                                            normalise_values=True,
                                                                            strategy=strategy,
                                                                            normalise=normalise,
                                                                            prior_counts=pcs))
                    if normalise or pcs == 0.5:
                        # without value normalisation but with output normalisation, or with float prior counts it
                        # should still raise
                        self.assertRaises(TypeError, lambda: DiscretePitchScape(values=data,
                                                                                normalise_values=False,
                                                                                strategy=strategy,
                                                                                normalise=normalise,
                                                                                prior_counts=pcs))
                        # use float values in that case
                        discrete_scape = DiscretePitchScape(values=data.astype(np.float),
                                                            normalise_values=False,
                                                            strategy=strategy,
                                                            normalise=normalise,
                                                            prior_counts=pcs)
                    else:
                        # otherwise it should be fine with ints
                        discrete_scape = DiscretePitchScape(values=data,
                                                            normalise_values=False,
                                                            strategy=strategy,
                                                            normalise=normalise,
                                                            prior_counts=pcs)
                    continuous_scape = PitchScape(scape=discrete_scape)
                    # check both discrete and continuous scape
                    for discrete in [True, False]:
                        if discrete:
                            pitch_scape = discrete_scape
                        else:
                            pitch_scape = continuous_scape
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
                                if not discrete:
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
                                if not discrete:
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
