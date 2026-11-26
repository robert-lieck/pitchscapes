"""
Reading a Corpus
================

For most analyses you will want to extract pitch-class distributions not just from a single piece but from a whole
corpus. This example defines a few convenience functions for reading all pieces in a folder into one data set, with
the pitch-class distributions of all windows of all pieces (as in :doc:`plot_03_pitch_class_distributions`), along
with the piece each distribution belongs to.

.. note::
   These functions are not (yet) part of the library. Just copy them into your own code if you find them useful.
"""

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy

import pitchscapes.reader as rd

# %%
# Reading pieces and folders
# --------------------------
#
# ``extract_pitchscape_for_piece`` reads a single piece using :func:`~pitchscapes.reader.sample_scape` with prior
# counts and normalisation (see :doc:`plot_05_sample_scape`). ``read_pitchscapes_in_folder`` does this for all files
# in a folder (optionally only including or excluding certain file extensions) and stacks the results into one data
# set. It returns the pitch-class distributions as a torch tensor of shape Nx12, plus the file name and an index of
# the piece for each of the N distributions.


def extract_pitchscape_for_piece(path, n_times):
    """Extract pitch-class distributions from a score file given by path."""
    return rd.sample_scape(n_time_intervals=n_times, file_path=path, prior_counts=1, normalise=True)


def read_pitchscapes_in_folder(path, n_times, include_extensions=None, exclude_extensions=None):
    """
    Extract pitch-class distributions from all the pieces in a folder given by path and return a data set with the
    pitch-class distributions of all these pieces.

    :param path: path to folder containing music files
    :param n_times: number of time intervals for extracting pitch-class distributions
    :param include_extensions: list/tuple of file extensions to include (e.g. ['.mxl', '.xml']); if None, all files
     are considered
    :param exclude_extensions: list/tuple of file extensions to exclude (e.g. ['.txt', '.md']); if None, no files are
     excluded
    :return: tensor of pitch-class distributions (Nx12), list of piece names and list of piece indices (both of
     length N)
    """
    piece_names = []
    X = np.array([])
    curr_piece_index = 0
    pieces_indexes = []

    # normalise extensions to ensure they start with a dot
    if include_extensions is not None:
        include_extensions = tuple(ext if ext.startswith('.') else f'.{ext}' for ext in include_extensions)
    if exclude_extensions is not None:
        exclude_extensions = tuple(ext if ext.startswith('.') else f'.{ext}' for ext in exclude_extensions)

    for file_name in sorted(os.listdir(path)):
        _, ext = os.path.splitext(file_name)
        # skip if include_extensions is specified and file doesn't match
        if include_extensions is not None and ext not in include_extensions:
            continue
        # skip if exclude_extensions is specified and file matches
        if exclude_extensions is not None and ext in exclude_extensions:
            continue
        print(f"reading {file_name}")
        pitch_class_dist = extract_pitchscape_for_piece(os.path.join(path, file_name), n_times)
        pieces_indexes.extend([curr_piece_index] * len(pitch_class_dist))
        X = np.vstack([X, pitch_class_dist]) if X.size else pitch_class_dist
        piece_names.extend([file_name] * pitch_class_dist.shape[0])
        curr_piece_index = curr_piece_index + 1
    return torch.from_numpy(X), piece_names, pieces_indexes


# %%
# Here, we read the MIDI files from the ``tests/data`` folder of the repository (only including ``.mid`` files, as
# the folder also contains a MusicXML version of the Bach prelude). With ``n_times=10`` we get
# ``10 * 11 / 2 = 55`` distributions per piece.

X, piece_names, piece_indices = read_pitchscapes_in_folder('../tests/data/', n_times=10, include_extensions=['.mid'])
print(X.shape)
print(sorted(set(piece_names)))

# %%
# Keeping track of the window length
# ----------------------------------
#
# The distributions above do not tell you the size of the window they come from. If you need that (for instance, to
# analyse or plot the data separately for different time scales), this variant additionally returns the window length
# (in quarter notes) of each distribution.


def extract_pitchscape_for_piece_with_window_length(path, n_times):
    """Extract pitch-class distributions and the corresponding window lengths from a score file given by path."""
    scape = rd.get_pitch_scape(path, prior_counts=1, normalise=True)
    times = np.linspace(scape.min_time, scape.max_time, n_times + 1)
    samples = []
    window_len = []
    for start, end in product(times, times):
        if start >= end:
            continue
        samples.append(scape[start, end])
        window_len.append(end - start)
    return np.array(samples), window_len


# %%
# As an example, we look at the entropy of the distributions in the two pieces as a function of window length. Low
# entropy means the distribution is concentrated on a few pitch classes, high entropy means it is spread out (the
# maximum for twelve pitch classes is log(12) ≈ 2.48).

fig, ax = plt.subplots(figsize=(8, 4))
for file_name in ['Prelude_No_1_BWV_846_in_C_Major.mid', 'Sonata_No._14_Op._27_No._2_-_Ludwig_van_Beethoven.mid']:
    samples, window_len = extract_pitchscape_for_piece_with_window_length('../tests/data/' + file_name, n_times=20)
    ax.scatter(window_len, entropy(samples, axis=1), s=5, label=file_name)
ax.set_xlabel("window length (quarter notes)")
ax.set_ylabel("entropy")
ax.legend(fontsize=8)
plt.show()
