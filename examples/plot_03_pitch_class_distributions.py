"""
Extracting Pitch-Class Distributions
====================================

A pitch scape gives you the pitch-class counts for *any* time window of a piece. This makes it easy to extract a
rich data set from a single piece: by going over a grid of start and end times, we get pitch-class distributions for
windows at different points in time *and* of different sizes, from short local sections to the whole piece. This
gives a much more diverse picture of a piece than, say, a single global distribution or a sequence of local
distributions, and is a good starting point for further analysis (for instance, clustering, embedding or other
machine learning methods).
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pitchscapes.reader as rd

np.set_printoptions(precision=3, linewidth=1000)  # for more compact printing

# %%
# Read the score into a pitch scape.

pitch_scape = rd.get_pitch_scape('../tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl')

# %%
# Define a grid of ``n_times`` equally spaced points in time from the beginning to the end of the piece. Every pair
# of points with ``start < end`` defines a window, so we get ``n_times * (n_times - 1) / 2`` windows. Indexing the
# pitch scape with ``pitch_scape[start, end]`` returns the pitch-class counts in that window (time is measured in
# quarter notes, and counts are weighted by duration). Normalising them gives pitch-class distributions.

n_times = 10
times = np.linspace(pitch_scape.min_time, pitch_scape.max_time, n_times)
windows = []
pitch_class_distributions = []
for start, end in product(times, times):
    if end <= start:
        continue
    windows.append((start, end))
    pitch_class_distributions.append(pitch_scape[start, end])
pitch_class_distributions = np.array(pitch_class_distributions)
pitch_class_distributions /= pitch_class_distributions.sum(axis=1, keepdims=True)
print(pitch_class_distributions.shape)
print(pitch_class_distributions)

# %%
# Each row is the distribution over the twelve pitch classes (C, C#, D, ..., B) of one window. Let's look at a few
# of them: the whole piece, a window from the beginning to somewhere in the middle, one from there to the end, and a
# short window in the middle. While the overall
# distribution is clearly C major, the windows differ in their details, depending on the local harmonic content.

labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
fig, ax = plt.subplots(figsize=(10, 4))
for idx in [windows.index((times[0], times[-1])),
            windows.index((times[0], times[4])),
            windows.index((times[4], times[-1])),
            windows.index((times[4], times[5]))]:
    start, end = windows[idx]
    ax.plot(labels, pitch_class_distributions[idx], 'o-', label=f"[{start:.0f}, {end:.0f}]")
ax.set_ylabel("probability")
ax.legend(title="window (quarter notes)")
plt.show()

# %%
# **Note:** :func:`~pitchscapes.reader.sample_scape` does the same in one go (see :doc:`plot_05_sample_scape`). To
# do this for a whole folder of pieces, see :doc:`plot_08_reading_a_corpus`.
