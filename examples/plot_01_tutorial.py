"""
Pitch Scape Tutorial
====================

This tutorial walks you through computing and plotting pitch scapes and training pitch-scape mixture models (clusters).
The methods are described in the paper
`Modelling Hierarchical Key Structure With Pitch Scapes <https://doi.org/10.5281/zenodo.4245558>`_
(Lieck & Rohrmeier, ISMIR 2020); please cite it when using this library for publications.

Throughout this tutorial, I will use two pieces as examples: Johann Sebastian Bach's *Prelude in C major (BWV 846)*
and Ludwig van Beethoven's "Moonlight Sonata" *Sonata No. 14 in C# minor (Op. 27, No. 2)*. Here is the first bar from
both:

.. image:: /Prelude_No_1_BWV_846_in_C_Major.png
   :alt: Johann Sebastian Bach, Prelude in C major, BWV 846
   :width: 45%

.. image:: /Sonata_No._14_Op._27_No._2_-_Ludwig_van_Beethoven.png
   :alt: Ludwig van Beethoven, Sonata No. 14 in C# minor, Op. 27, No. 2
   :width: 45%

.. note::
   The MIDI files used here are in the ``tests/data`` folder of the
   `repository <https://github.com/robert-lieck/pitchscapes>`_. If you download this example as a Python file or
   Jupyter notebook, adapt ``data_dir`` below to point to where you saved them.
"""

# %%
# Quick Start
# -----------
#
# The quickest way to some nice plots is to load a piece and do a key-scape plot.

import numpy as np
import pitchscapes.reader as rd
import pitchscapes.plotting as pt

np.random.seed(0)  # make reproducible
data_dir = '../tests/data/'

scape_JSB = rd.get_pitch_scape(data_dir + 'Prelude_No_1_BWV_846_in_C_Major.mid')
scape_LvB = rd.get_pitch_scape(data_dir + 'Sonata_No._14_Op._27_No._2_-_Ludwig_van_Beethoven.mid')
pt.key_scape_plot(scape=scape_JSB, n_samples=100, size=5)
pt.key_scape_plot(scape=scape_LvB, n_samples=100, size=5)

# %%
# And here are the separate pitch-scape plots, you can see quite clearly that one is in C-major and the other in
# c#-minor.

pt.pitch_scape_plots(scape=scape_JSB, n_samples=30)
pt.pitch_scape_plots(scape=scape_LvB, n_samples=30)

# %%
# Now, for a quick start, let's train a mixture model with two clusters on these pieces. This can take a minute – in
# the background there are 2904 variables being optimised.

from pitchscapes import PitchScapeMixtureModel

model = PitchScapeMixtureModel(n_clusters=2)
model.set_data(scapes=[scape_JSB, scape_LvB], n_samples=20)
model.optimize(progress=False)  # use progress=True (the default) to see the training progress
for c in model.clusters():
    pt.key_scape_plot(scape=c, n_samples=20, size=5)

# %%
# Do you recognise which cluster represents which piece?
#
# **Hint:** *Clusters are transposition invariant but we have to choose a transposition to visualise them. The default
# initialisation gives them a bias towards the diatonic without accidentals (C-major / a-minor). So pieces in minor
# mode will appear to be transposed to a-minor...*
#
# Step-by-Step
# ------------
#
# Now let's understand a bit more of what is going on behind the scenes. First some more imports and settings for
# later.

import matplotlib.pyplot as plt
from pitchscapes import PitchScape

np.set_printoptions(precision=3, linewidth=1000)  # for more compact printing

# %%
# Reading Files
# ^^^^^^^^^^^^^
#
# The ``get_pitch_scape`` function is a shortcut for doing two things:
#
# 1. using ``pitch_class_counts`` to read pitch-class counts from a file and
# 2. constructing a ``PitchScape`` object from these data.

counts, times = rd.pitch_class_counts(data_dir + 'Prelude_No_1_BWV_846_in_C_Major.mid')
print("data of the fist quarter note of the first bar:")
print(counts[:6])
print(times[:7])
print(len(counts), len(times))

# %%
# The ``pitch_class_counts`` function returns a 2D-array of pitch-class counts for all time intervals where no
# note-on or note-off event happens (i.e. where the counts are constant) and a 1D-array of times that correspond to
# the interval boundaries. So there is one more element in times than in counts. The extremely short time intervals
# come from the MIDI exporter adding short silence between two consecutive notes, this can safely be ignored. From
# these data, we can construct a ``PitchScape`` object and plot it as above.

scape = PitchScape(values=counts, times=times)
pt.key_scape_plot(scape=scape, n_samples=100, size=5)

# %%
# Pitch scapes give you access to the pitch-class counts of any time interval between ``min_time`` and ``max_time``
# (time is measured in beats or quarter notes).

start, end = sorted(np.random.uniform(scape.min_time, scape.max_time, 2))
print(f"pitch counts in [{start}, {end}]:")
print(scape[start, end])

# %%
# As per default, these values correspond to the sum of all pitch-class counts within the specified time interval
# (the integral over the pitch-class density). If you want to have normalised values, just say so:

normalised_scape = PitchScape(values=counts, times=times, normalise=True)
print(f"normalised pitch counts in [{start}, {end}]:")
print(normalised_scape[start, end])

# %%
# In a key-scape plot you wouldn't see the difference because normalisation does not affect the key estimates (values
# are actually internally normalised before computing the estimates). But in the separate pitch-scape plots you can
# see the difference if you disable the internal normalisation of the plotting function:

# normalisation done by pitch scape, not by the plotting function
pt.pitch_scape_plots(scape=normalised_scape, n_samples=30, normalise=False)
# no normalisation: you see counts summing up towards the top
pt.pitch_scape_plots(scape=scape, n_samples=30, normalise=False)
# normalisation done by the plotting function, not by the pitch scape (default)
pt.pitch_scape_plots(scape=scape, n_samples=30)

# %%
# **Note:** *Unnormalised* pitch-scape plots of a *normalised* pitch scape (first figure) look exactly like
# *normalised* pitch-scape plots (default) of an *unnormalised* (default) pitch scape (third figure).
#
# Plotting
# ^^^^^^^^
#
# You can control several aspects of the plots, such as changing the resolution

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[0], legend=False)
pt.key_scape_plot(scape=scape, n_samples=30, ax=axes[1], legend=False)
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[2], legend=False)
pt.key_scape_plot(scape=scape, n_samples=150, ax=axes[3], legend=False)
fig.tight_layout()

# %%
# changing properties of the legend

fig, axes = plt.subplots(1, 10, figsize=(25, 2.5))
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[0], legend=False)
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[1], location='top')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[2], location='bottom')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[3], location='left')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[4], location='right')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[5], location='top left')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[6], location='top right')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[7], location='left small')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[8], location='right small')
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[9], x_scale=0.5, y_scale=4, x_offset=0.7, y_offset=0.3,
                  horizontal=True, fontsize=5, label_size=0.5, aspect=0.5)
fig.tight_layout()

# %%
# choosing chromatic or fifths-based colouring (note that the legend also changes) – fifths-based colouring is more
# "harmonic" and less contrasting, both have their pros and cons

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[0], circle_of_fifths=False)
pt.key_scape_plot(scape=scape, n_samples=50, ax=axes[1], circle_of_fifths=True)
fig.tight_layout()

# %%
# **Note:** You can set this property globally to ``True`` or ``False`` with ``pt.set_circle_of_fifths()``.
#
# Or choosing an entirely different colour palette.

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
pt.key_scape_plot(scape=scape, n_samples=100, ax=axes[0], palette=np.random.uniform(0, 1, (12, 2, 3)))
pt.key_scape_plot(scape=scape, n_samples=100, ax=axes[1], palette=np.random.uniform(0, 1, (12, 2, 3)))
fig.tight_layout()

# %%
# Clustering
# ^^^^^^^^^^
#
# Now let's do clustering with more than one piece per cluster. For this we will cut random parts off the beginning
# and end of Bach's and Beethoven's piece and randomly transpose them to artificially generate two clusters of similar
# but not identical pieces.

# read the original pieces
counts_JSB, times_JSB = rd.pitch_class_counts(data_dir + 'Prelude_No_1_BWV_846_in_C_Major.mid')
counts_LvB, times_LvB = rd.pitch_class_counts(data_dir + 'Sonata_No._14_Op._27_No._2_-_Ludwig_van_Beethoven.mid')
# generate artificial pieces by cutting start/end and transposing
n_pieces = 5
pieces = [[PitchScape(values=np.roll(c[s:e], shift=shift, axis=1), times=t[s:e + 1])
           for s, e, shift in zip(np.random.randint(0, 200, n_pieces),                # random start
                                  np.random.randint(len(c) - 200, len(c), n_pieces),  # random end
                                  np.random.randint(0, 12, n_pieces))]                # random transposition
          for c, t in [(counts_JSB, times_JSB), (counts_LvB, times_LvB)]]
# plot the pieces
fig, axes = plt.subplots(2, n_pieces + 1, figsize=(3 * (n_pieces + 1), 3 * 2))
for ps, axs in zip(pieces, axes):
    for p, ax in zip(ps, axs):
        pt.key_scape_plot(scape=p, n_samples=30, ax=ax, legend=False)
pt.key_legend(horizontal=False, ax=plt.subplot2grid((2, n_pieces + 1), (0, n_pieces), rowspan=2, fig=fig))
fig.tight_layout()

# %%
# Now we set up a model with two clusters, provide the pieces as data and train it.

# set up model and data
data = pieces[0] + pieces[1]
model = PitchScapeMixtureModel(n_clusters=2)
model.set_data(scapes=data, n_samples=20)
# train
model.optimize(progress=False)
# plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, c in zip(axes, model.clusters()):
    pt.key_scape_plot(scape=c, n_samples=20, size=5, ax=ax, legend=False)
pt.key_legend(horizontal=False, ax=axes[2])
fig.tight_layout()

# %%
# You can look at the learning progress – the downwards kink, this is where the two clusters (which were initialised
# almost identically) separated to capture the two different groups of pieces. There might also be some upward
# wiggles or spikes towards the end. This happens if the gradient step is too large and can be counteracted with a
# smaller learning rate.
#
# **Note:** The best parameter set (with smallest loss) is remembered and restored after the optimisation.

pt.plot_loss(model.loss)

# %%
# We can also look at the assignments, first separated by cluster:

pt.plot_cluster_assignments(model.assignments())

# %%
# The marginal on the right indicates the probability of a piece being assigned to the specific cluster, the
# "interior" shows how this probability is distributed over different transpositions. Very clearly, the two groups of
# pieces are correctly assigned to the two clusters. Also the transposition was correctly inferred – you can check
# with the key-scape plots above. We can also look at the assignment separated by pieces:

pt.plot_piece_assignments(model.assignments())

# %%
# Progressive and Hierarchical Clustering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# OK, this was good to get an intuition of how training the model works. But in practice, we will usually have
# larger data sets and more than two clusters (maybe we don't even know in advance how many clusters we want to have)
# and these models take a while to train. So we may want to progressively refine the model and split clusters
# hierarchically, starting with a very simple model.

model = PitchScapeMixtureModel(n_clusters=1,  # one cluster
                               n_center=1,    # low horizontal resolution
                               n_width=1)     # low vertical resolution
model.set_data(scapes=data, n_samples=10)  # low number of samples, which is enough for the low resolution
model.optimize(progress=False)
pt.plot_loss(model.loss)
pt.key_scape_plot(model.cluster(0), 10, size=5)

# %%
# Now – and this is the important new technique – we can derive new and more complex models from the pretrained one.
# Let's first simply increase the resolution.

# only increase the resolution – everything else is copied from the old model, including the already learned parameters
model = model.new(n_center=3, n_width=3)
model.set_data(scapes=data, n_samples=30)  # more samples for higher resolution
model.optimize(progress=False)
pt.plot_loss(model.loss)
pt.key_scape_plot(model.cluster(0), 30, size=5)

# %%
# Now it's time to have more clusters. This can be done by cloning the existing ones.

model = model.new(clone=[2])  # the one and only cluster is cloned twice
model.set_data(scapes=data, n_samples=30)
model.optimize(progress=False)
pt.plot_loss(model.loss)
for c in model.clusters():
    pt.key_scape_plot(c, 30, size=5)

# %%
# Here we are, this looks like the clusters matching the two original pieces. We can confirm this by looking at the
# cluster assignments.

pt.plot_cluster_assignments(model.assignments())

# %%
# With a real data set we might now go on splitting and refining the clusters. With our small artificial data set,
# just for fun, we will end by splitting each of the two clusters into five and see whether they can separate the
# single pieces – which might not be possible, depending on how similar they turned out to be.

model = model.new(clone=[5, 5])  # both clusters split into five
model.set_data(scapes=data, n_samples=30)
model.optimize(progress=False)
pt.plot_loss(model.loss)
for c in model.clusters():
    pt.key_scape_plot(c, 30, size=5)
pt.plot_cluster_assignments(model.assignments())

# %%
# Some technical details...
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# There are some technical details about the training procedure worth knowing.
#
# Empty Clusters
# """"""""""""""
#
# The optimisation space is very high-dimensional, which means that any random initialisation is likely to be totally
# off any of the data points. This can cause a problem: If multiple clusters are very similar but all very much off
# the data, they will enter a "race" towards the data, which probably only one of them will win. The cluster that is
# nearest to the data will be most attracted and therefore in the next step be even more ahead of the others. The
# result is "empty" clusters that do not have any data points assigned to them. This can be avoided by starting with
# only a single cluster and initialising new clusters by splitting/cloning existing ones, as demonstrated above. The
# safest is to only do binary splits; when splitting into more components (as in the last example above) this may
# also lead to clusters being "left behind" and ending up empty.
#
# Splitting Clusters and Learning Rate
# """"""""""""""""""""""""""""""""""""
#
# A challenge with progressive and hierarchical clustering is that when splitting a cluster, the initial parameters
# are very close to each other *and* very close to an optimum (after all they correspond to the optimum before
# splitting). This implies two things:
#
# 1. We need to separate them a bit so they can take different paths in the subsequent optimisation. This is done by
#    adding some noise to the parameters to shift them a bit into some random direction. Now, as they were close to an
#    optimum, a random direction is highly unlikely to improve the model or even leave the loss unchanged. So you will
#    see the model to drop a bit in quality (increase in loss) after splitting. This is not a problem and will quickly
#    vanish at the beginning of the subsequent optimisation. The point where the split clusters really start
#    separating shows up as a downward kink in the loss: Before that point they jointly sit on a saddle point of the
#    objective function; after that point one runs down one side of the saddle and the other one the other side.
# 2. As the clusters already are close to an optimum, we want to make sure not to mess up their parameters by taking
#    huge gradient steps in the wrong direction. Especially, because the noise we have to add for separating them will
#    also introduce noise in the gradients. We therefore employ a custom adaptation of the Adam optimiser (it's called
#    *WarmAdam* because it's for warm starts) that has an initial learning rate and then slowly transitions to a final
#    learning rate. The initial learning rate is set to zero by default, which allows Adam to get an idea of how the
#    gradients look like before starting to run after it. It will then slowly transition to the final learning rate to
#    actually make progress. In the loss plots you see the small initial learning rate at the beginning where there is
#    little progress before the model dives into the optima as the learning rate increases and the loss starts
#    decreasing more rapidly. The final flattening is not induced by a change in learning rate but is the usual and
#    desired convergence to an optimum.

# sphinx_gallery_thumbnail_number = 1
