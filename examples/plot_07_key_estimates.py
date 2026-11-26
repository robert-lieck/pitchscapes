"""
Estimating Keys
===============

This example walks step by step through estimating keys from pitch-class distributions with the
:class:`~pitchscapes.keyfinding.KeyEstimator`, what the different outputs mean, and how to extract the relevant
information from them. This is also what happens behind the scenes in key-scape plots.

The key estimator compares a pitch-class distribution to the major and minor key profiles (see
:doc:`plot_06_compare_profiles`) in all twelve transpositions. Each of these 24 comparisons gives a *score*, which is
a distance (by default the Euclidean distance between the normalised distribution and the normalised profile), so
*lower is better*. The best estimate is the key with the lowest score.
"""

import matplotlib.pyplot as plt
import numpy as np
import pitchscapes.plotting as pt
import pitchscapes.util as ut
from pitchscapes.keyfinding import KeyEstimator

np.random.seed(0)  # make reproducible
np.set_printoptions(precision=3, linewidth=1000)  # for more compact printing

# %%
# The data
# --------
#
# We generate some random pitch-class counts as data (with real data, you would get these from a pitch scape, see
# :doc:`plot_03_pitch_class_distributions`). Raising them to the 5th power makes them more "peaked" so that some
# pitch classes dominate. The data may be normalised or not, this does not matter as the key estimator normalises
# internally by default. We also set up a key estimator, which uses the ``"albrecht"`` profiles by default.

N = 5  # number of data points
pitch_class_counts = np.random.uniform(0, 10, (N, 12)) ** 5
k = KeyEstimator()

# %%
# Scores
# ------
#
# The scores have shape Nx2x12: for each of the N data points, there is one score for each mode (index 0: major,
# index 1: minor) and transposition (index 0: C, 1: C#, ..., 11: B). Lower scores mean a better match.

scores = k.get_score(pitch_class_counts)
print("scores\n", scores)

# %%
# Best estimates
# --------------
#
# The best matching keys have shape Nx2, with the mode (0: major, 1: minor) in the first column and the transposition,
# that is, the tonic as a pitch class (0: C, 1: C#, ..., 11: B), in the second column.

best_match = k.get_estimate(pitch_class_counts)
major_minor = best_match[:, 0]
transposition = best_match[:, 1]
print("best match\n", best_match)
print("major/minor\n", major_minor)
print("transposition\n", transposition)

# %%
# Estimates can be converted to strings, either using capitalisation (upper case for major, lower case for minor) or
# by spelling out major/minor.

print("as string (capitalisation)\n", ut.key_estimates_to_str(best_match, use_capitalisation=True))
print("as string (major/minor)\n", ut.key_estimates_to_str(best_match, use_capitalisation=False))

# %%
# Colours
# -------
#
# For visualisation, scores can be converted to colours (Nx4 in RGBA). This is how key-scape plots are coloured. The
# scores are first turned into probabilities over the 24 keys with a soft-max function, and the colour is the
# probability-weighted average of the key colours. The *uncertainty* of this distribution (its entropy) is coded as
# transparency: if one key clearly matches best, the colour is opaque; if several keys match similarly well, it gets
# more transparent. The soft-max temperature (default: 0.005) controls this: lower temperatures make the
# probabilities more peaked (more opaque), higher temperatures make them flatter (more transparent). Here we use a very
# low temperature, so the colours are essentially those of the best estimate. Colours can be chromatic (neighbouring
# keys on the chromatic scale have similar colours) or fifths-based (neighbouring keys on the circle of fifths have
# similar colours).

colours_chrom = pt.key_scores_to_color(scores, soft_max_temperature=0.000001, circle_of_fifths=False)
colours_fifth = pt.key_scores_to_color(scores, soft_max_temperature=0.000001, circle_of_fifths=True)
print("colours (chromatic, RGBA)\n", colours_chrom)
print("colours (fifths, RGBA)\n", colours_fifth)

# %%
# Finally, we plot the colours of the estimates next to the corresponding key legends (chromatic on top,
# fifths-based below). In the legends, upper-case labels are major keys, lower-case labels minor keys.

fig, axes = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw=dict(width_ratios=[1, 3]))
for row, colours, cof in zip(axes, [colours_chrom, colours_fifth], [False, True]):
    row[0].bar(np.arange(N), np.ones(N), color=colours)
    row[0].set_xticks(np.arange(N))
    row[0].set_xticklabels(ut.key_estimates_to_str(best_match))
    row[0].set_yticks([])
    pt.key_legend(ax=row[1], circle_of_fifths=cof)
fig.tight_layout()
plt.show()
