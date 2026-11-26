"""
Comparing Key Profiles
======================

Keys are estimated by comparing pitch-class distributions to *key profiles*: typical pitch-class distributions for
major and minor keys (see :doc:`plot_07_key_estimates`). The :class:`~pitchscapes.keyfinding.KeyEstimator` comes with
several sets of profiles (``"albrecht"``, the default, ``"krumhansl"``, ``"temperley"``, ``"bayes"`` and
``"metadata"``), each with a major and a minor profile for the tonic C. In this example, we look at what these
profiles look like using :func:`~pitchscapes.plotting.compare_key_profiles`, which plots two distributions side by
side as a bar chart.
"""

import matplotlib.pyplot as plt
import numpy as np
import pitchscapes.plotting as pt
from pitchscapes.keyfinding import KeyEstimator

print(list(KeyEstimator.profiles))

# %%
# C major and A minor
# -------------------
#
# First, we compare the empirical (Albrecht) profiles of C major and A minor (the minor profile is defined for C minor,
# so we roll it by -3 semitones to transpose it to A minor). These two keys are *relative* keys: they use the same
# notes, essentially the white keys on the piano. Accordingly, both profiles have a high weight on the white keys and
# a low weight on the black keys. But there are clear differences in *how much* weight each note gets: C major puts
# most weight on C and G (its tonic and fifth), while A minor puts most weight on A and E (its tonic and fifth), and A
# minor also has some weight on G# (the leading tone, which is raised in minor).

major_albrecht = KeyEstimator.profiles["albrecht"]["major"]
minor_albrecht = KeyEstimator.profiles["albrecht"]["minor"]
pt.compare_key_profiles(empirical=major_albrecht,
                        empirical_label="C major profile",
                        profile=np.roll(minor_albrecht, -3),
                        profile_label="A minor profile",
                        transpose_profile=False)
plt.show()

# %%
# Coarse hand-made profiles vs empirical profiles
# -----------------------------------------------
#
# A very coarse approximation of a key profile is to give a weight of 1 to all notes of the scale, a weight of 2 to
# the tonic and fifth, and 0 to everything else. Let's define such a profile for B major (B, C#, D#, E, F#, G#, A#
# with B and F# doubled) and compare it to the empirical profile.
#
# If no ``profile`` is passed to :func:`~pitchscapes.plotting.compare_key_profiles`, it uses a
# :class:`~pitchscapes.keyfinding.KeyEstimator` to find the best-matching key and shows the corresponding profile
# (named on the x-axis). With ``transpose_profile=False``, the hand-made distribution is transposed to the estimated
# key, rather than the profile to B major, so the tonic is always the first bar. The two are quite similar: the
# coarse profile captures the overall shape of the empirical one, though the empirical one is more nuanced (e.g. the
# third gets more weight than other scale degrees).

major = [0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1, 2]  # B major
pt.compare_key_profiles(empirical=major, transpose_profile=False)
plt.show()

# %%
# The same for A minor (natural minor scale: A, B, C, D, E, F, G with A and E doubled). Note how the empirical minor
# profile differs from the coarse one on the 6th and 7th scale degrees: in minor, these occur both in their natural and
# in their raised form (e.g. G and G# in A minor), so the weight is spread over both.

minor = [1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 0, 1]  # A minor
pt.compare_key_profiles(empirical=minor, transpose_profile=False)
plt.show()
