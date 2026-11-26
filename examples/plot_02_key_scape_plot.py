"""
Reading a Score and Plotting a Key Scape
========================================

This is the minimal example to get started: read a score, turn it into a pitch scape, and look at it as a key-scape
plot.

A *pitch scape* represents the pitch content of a piece on all time scales at once: for any time window
from a start time to an end time it gives you the pitch-class distribution of that section of the piece. A
*key-scape plot* visualises this as a triangle: the horizontal axis is the centre of the window, the vertical axis its
width. The bottom row shows small, local windows, while further up the windows get larger, until the single point at
the top covers the whole piece. Each point is coloured by the key that best matches the pitch-class distribution in
that window (see :doc:`plot_07_key_estimates`). For details, see the paper
`Modelling Hierarchical Key Structure With Pitch Scapes <https://doi.org/10.5281/zenodo.4245558>`_
(Lieck & Rohrmeier, ISMIR 2020).
"""

import matplotlib.pyplot as plt
import pitchscapes.reader as rd
import pitchscapes.plotting as pt

# %%
# Read the score into a pitch scape. Besides MIDI files, any format that `music21 <https://web.mit.edu/music21/>`_
# can read (such as MusicXML) works.

pitch_scape = rd.get_pitch_scape(file_path='../tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl')

# %%
# Show the key-scape plot. The top of the triangle shows the key of the whole piece (C major), while the lower parts
# show the local keys of shorter sections. Here we use fifths-based colouring and no legend.
# ``n_samples`` controls the resolution (the number of windows along the bottom). Higher values give smoother
# plots, but the computation time grows quadratically.

pt.key_scape_plot(pitch_scape, n_samples=200, size=10, legend=False, circle_of_fifths=True)
plt.show()

# %%
# Alternatively, show the separate pitch-scape plots, one for each pitch class. Here, the colour intensity shows how
# much weight a pitch class has in each window. In C major, the white keys (C, D, E, F, G, A, B) dominate.

pt.pitch_scape_plots(pitch_scape, n_samples=50, size=2)
plt.show()
