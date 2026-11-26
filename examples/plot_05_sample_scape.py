"""
Sampling a Pitch Scape: Prior Counts and Normalisation
======================================================

:func:`~pitchscapes.reader.sample_scape` divides a piece into ``n_time_intervals`` equally sized time intervals and
returns the pitch-class counts for every possible combination of start and end time (ordered first by start time,
then by end time), just like we did manually in :doc:`plot_03_pitch_class_distributions`.

The point of this example is to understand the role of the ``prior_counts`` and ``normalise`` parameters (which are
passed on to :class:`~pitchscapes.scapes.PitchScape`). To make that easy, we only print the first three samples, so
you can compare the numbers directly.
"""

import numpy as np
import pitchscapes.reader as rd

np.set_printoptions(precision=3, linewidth=1000)  # for more compact printing
file_path = '../tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl'

# %%
# Just counts
# -----------
#
# By default, you get the raw counts: the sum of all notes of a pitch class in a window, weighted by their duration
# (in quarter notes). Larger windows have larger counts, and pitch classes that do not occur have zero counts. The
# first three samples all start at the beginning of the piece and end after one, two, and three time intervals,
# respectively, so the counts grow from one row to the next.

samples = rd.sample_scape(n_time_intervals=10, file_path=file_path)
print(samples.shape)
print(samples[:3])

# %%
# With prior counts
# -----------------
#
# ``prior_counts`` adds a constant to all counts. This corresponds to a Bayesian estimate with a uniform prior, where
# the value adjusts the strength of the prior. It ensures that no pitch class has zero probability, which matters if
# you compute things like logarithms or divergences (e.g. the Kullback-Leibler or Jensen-Shannon divergence) on the
# distributions. Compare with the output above: all values are larger by exactly one.

print(rd.sample_scape(n_time_intervals=10, file_path=file_path, prior_counts=1)[:3])

# %%
# With prior counts and normalisation
# -----------------------------------
#
# ``normalise=True`` turns the counts into probability distributions that sum to one. Note that the prior counts are
# added *before* normalising, so their effect is stronger for small windows (few counts) and weaker for large windows
# (many counts), which is exactly what you want from a prior.

normalised = rd.sample_scape(n_time_intervals=10, file_path=file_path, prior_counts=1, normalise=True)
print(normalised[:3])
print(normalised[:3].sum(axis=1))

# %%
# **Note:** If you use ``normalise=True`` there is a difference between ``prior_counts=None`` (the default) and
# ``prior_counts=0``. For windows without any notes (e.g. zero-width windows or rests), the former returns all zeros,
# while the latter treats this as the limit of very small prior counts and returns a uniform distribution.
