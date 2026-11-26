"""
Getting a Piano Roll
====================

Pitch scapes work with pitch *classes*, that is, they ignore the octave. If you need the actual pitches, you can
read a score into a piano-roll representation instead: a boolean matrix with one row per time slice and one column
per pitch, indicating which pitches sound in which time slice. A new time slice starts whenever any note starts or
ends, so the slices have different durations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pitchscapes.reader as rd

# %%
# Read the piano roll, also returning the range of pitches (as MIDI numbers, 60 is middle C) that correspond to the
# columns and the duration of each time slice (in quarter notes). By default, the pitch range goes from the lowest to
# the highest pitch in the piece; use ``min_pitch`` and ``max_pitch`` to set it explicitly (e.g. to get piano rolls of
# the same size for different pieces).

piano_roll, pitch_range, durations = rd.piano_roll('../tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl',
                                                   return_range=True,
                                                   return_durations=True)
print("piano roll:", piano_roll.shape)
print(piano_roll)
print("pitch range:", pitch_range.shape)
print(pitch_range)
print("durations:", durations.shape)
print(durations)

# %%
# Plot the first bars of the piano roll, with each time slice stretched according to its duration.

n_slices = 64
onsets = np.concatenate([[0], np.cumsum(durations[:n_slices])])
fig, ax = plt.subplots(figsize=(10, 4))
ax.pcolormesh(onsets, np.concatenate([pitch_range, [pitch_range[-1] + 1]]) - 0.5, piano_roll[:n_slices].T,
              cmap='Greys', shading='flat')
ax.set_xlabel("time (quarter notes)")
ax.set_ylabel("MIDI pitch")
plt.show()
