import numpy as np
from numpy.testing import assert_allclose
from scipy.special import xlogy, softmax
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as color_map

from .keyfinding import KeyEstimator
from .util import sample_pitch_scape, axis_set_invisible, pitch_classes_sharp, pitch_classes_flat, coords_from_times


_default_circle_of_fifths = False


def set_circle_of_fifths(value):
    global _default_circle_of_fifths
    _default_circle_of_fifths = bool(value)


def get_circle_of_fifths():
    return _default_circle_of_fifths


def minor_shift(circle_of_fifths=None):
    if circle_of_fifths is None:
        circle_of_fifths = _default_circle_of_fifths
    return -3 if circle_of_fifths else 0


def set_axis_off(ax):
    axis_set_invisible(ax=ax, splines=True, ticks='all', patch=True)


def get_key_colour(pitch, maj_min, circle_of_fifths=None, palette=None):
    if palette is None:
        # hand-tuned hue circle of bright and dark colours
        palette = np.array([(255, 55, 121), (213, 0, 70),
                            (255, 114, 37), (162, 58, 0),
                            (255, 190, 40), (200, 130, 0),
                            (255, 234, 59), (171, 153, 0),
                            (212, 255, 62), (138, 178, 0),
                            (21, 255, 85), (0, 159, 43),
                            (44, 255, 237), (0, 187, 171),
                            (5, 199, 255), (0, 126, 162),
                            (33, 122, 255), (0, 74, 186),
                            (130, 100, 255), (100, 41, 255),
                            (200, 0, 234), (125, 0, 157),
                            (255, 82, 213), (184, 0, 140)],
                           dtype=float).reshape((12, 2, 3)) / 255
    else:
        assert palette.shape == (12, 2, 3), palette.shape
    assert maj_min == 0 or maj_min == 1, maj_min
    if circle_of_fifths is None:
        circle_of_fifths = get_circle_of_fifths()
    if circle_of_fifths:
        pitch = (pitch * 7) % 12
    if maj_min == 1:
        pitch = (pitch + minor_shift(circle_of_fifths)) % 12
    return palette[pitch, maj_min]


def key_scores_to_color(scores,
                        circle_of_fifths=None,
                        soft_max_temperature=0.005,
                        alpha_remapping=lambda x: x ** 2,
                        palette=None):
    """
    Transform score values from a key finder into colors
    :param scores:
    :param soft_max_temperature:
    :param alpha_remapping:
    :return:
    """
    if circle_of_fifths is None:
        circle_of_fifths = get_circle_of_fifths()
    # get scores as numpy array
    scores = np.array(scores)
    # apply soft-max (also normalises)
    scores = softmax(-scores / soft_max_temperature, axis=(1, 2))
    # mask for non-nan values
    is_nan = np.any(np.isnan(scores), axis=(1, 2))
    is_not_nan = np.logical_not(is_nan)
    # get colours
    key_colours = np.zeros((12, 2, 3))
    key_colours[:, 0, :] = get_key_colour(pitch=np.arange(12),
                                          maj_min=0,
                                          circle_of_fifths=circle_of_fifths,
                                          palette=palette)
    key_colours[:, 1, :] = get_key_colour(pitch=np.arange(12),
                                          maj_min=1,
                                          circle_of_fifths=circle_of_fifths,
                                          palette=palette)
    # compute weighted colours for each estimate
    colours = scores[is_not_nan, :, :, None] * key_colours[None, :, :, :]
    # take average colour
    colours = colours.sum(axis=(1, 2))
    # compute normalised entropy
    ent = -xlogy(scores, scores).sum(axis=(1, 2)) / np.log(scores.shape[1] * scores.shape[2])
    # compose colours by combining rgb value, alpha value from entropy, and default nan colour
    full_colours = np.zeros((scores.shape[0], 4))
    full_colours[is_not_nan, 0:3] = colours
    full_colours[:, 3] = alpha_remapping(1 - ent)
    full_colours[is_nan] = (0, 0, 0, 0)
    # for very low temperatures in the softmax, some value may evaluate greater than one (1.0000000e+00)
    # check they are actually close to one (and one below zero) and then clip
    assert_allclose(full_colours[full_colours > 1], 1, atol=1e-10)
    assert_allclose(full_colours[full_colours < 0], 0, atol=1e-10)
    full_colours = np.clip(full_colours, 0, 1)
    return full_colours


def key_legend(ax=None,
               sharp_flat='sharp',
               location='top',
               circle_of_fifths=None,
               horizontal=None,
               x_scale=None,
               y_scale=None,
               aspect=1.,
               x_offset=None,
               y_offset=None,
               palette=None,
               label_size=None,
               fontsize=None,
               axis_off=True,
               equal_axes_aspect=True):
    if circle_of_fifths is None:
        circle_of_fifths = get_circle_of_fifths()
    # set defaults
    label_size_ = 0.8
    x_scale_ = 1.
    y_scale_ = 1.
    x_offset_ = 0.5
    y_offset_ = 0.5
    fontsize_ = 10
    # set location specific defaults
    if location == 'top':
        horizontal_ = True
        y_offset_ = 1 - y_scale_ / 12
    elif location == 'bottom':
        horizontal_ = True
        y_offset_ = y_scale_ / 12
    elif location == 'left':
        horizontal_ = False
        x_offset_ = x_scale_ / 12
    elif location == 'right':
        horizontal_ = False
        x_offset_ = 1 - x_scale_ / 12
    elif location == 'top left':
        label_size_ = 0.35
        fontsize = 7
        x_scale_ = 0.43
        y_scale_ = 0.43
        horizontal_ = True
        x_offset_ = 0.25
        y_offset_ = 1 - y_scale_ / 12
    elif location == 'top right':
        label_size_ = 0.35
        fontsize = 7
        x_scale_ = 0.43
        y_scale_ = 0.43
        horizontal_ = True
        x_offset_ = 0.75
        y_offset_ = 1 - y_scale_ / 12
    elif location == 'left small':
        label_size_ = 0.65
        x_scale_ = 0.75
        y_scale_ = 0.75
        horizontal_ = False
        x_offset_ = x_scale_ / 12
        y_offset_ = 1 - y_scale_ / 2
    elif location == 'right small':
        label_size_ = 0.65
        x_scale_ = 0.75
        y_scale_ = 0.75
        horizontal_ = False
        x_offset_ = 1 - x_scale_ / 12
        y_offset_ = 1 - y_scale_ / 2
    else:
        raise ValueError(f"'location' should be one of ['top', 'bottom', 'left', 'right', "
                         f"'top left', 'top right', 'left small', 'right small'] but is '{location}'")
    # set explicitly specified values
    if horizontal is not None:
        horizontal_ = horizontal
    if x_offset is not None:
        x_offset_ = x_offset
    if y_offset is not None:
        y_offset_ = y_offset
    if x_scale is not None:
        x_scale_ = x_scale
    if y_scale is not None:
        y_scale_ = y_scale
    if label_size is not None:
        label_size_ = label_size
    if fontsize is not None:
        fontsize_ = fontsize
    # get axis
    if ax is None:
        fig, ax_ = plt.subplots(1, 1)
    else:
        ax_ = ax
    if equal_axes_aspect:
        ax_.set_aspect('equal')
    # generate legend
    for pitch, key in enumerate(zip(pitch_classes_sharp, pitch_classes_flat)):
        for maj_min in [0, 1]:
            # x coordinate {0, 1}
            x = maj_min
            # y coordinate {0,...,11}
            if circle_of_fifths:
                y = (pitch * 7) % 12
            else:
                y = pitch
            # adapt capitalisation of label and shift y coordinate for minor
            if maj_min == 0:
                key = [key[0].capitalize(), key[1].capitalize()]
            elif maj_min == 1:
                key = [key[0].lower(), key[1].lower()]
                y = (y + minor_shift(circle_of_fifths)) % 12
            else:
                raise ValueError(f"'maj_min' should be 0 or 1 but is {maj_min}. This is a bug.")
            # use sharps, flats or both for label
            if sharp_flat == 'sharp':
                txt = key[0]
            elif sharp_flat == 'flat':
                txt = key[1]
            elif sharp_flat == 'both':
                txt = f"{key[0]}/{key[1]}" if key[0] != key[1] else key[0]
            else:
                raise ValueError(f"'sharp_flat should be one of ['sharp', 'flat', 'both'] but is '{sharp_flat}'")
            # scale y-coord to [0, 1]
            y = (y + 0.5) / 12
            # transpose for horizontal orientation
            if horizontal_:
                x, y = y, x
            # invert y direction
            y = 1 - y
            # move to origin
            x -= 0.5
            y -= 0.5
            # scale short coord (in maj/min direction)
            if horizontal_:
                y /= 12
            else:
                x /= 12
            # apply scale
            x *= x_scale_
            y *= y_scale_
            # add offset
            x += x_offset_
            y += y_offset_
            # put text
            ax_.text(x, y, txt, ha='center', va='center', fontsize=fontsize_)
            # put circles
            width = label_size_ / 12 * np.sqrt(aspect)
            height = label_size_ / 12 / np.sqrt(aspect)
            ax_.add_patch(Ellipse((x, y), width, height,
                                  color=get_key_colour(pitch=pitch,
                                                       maj_min=maj_min,
                                                       circle_of_fifths=circle_of_fifths,
                                                       palette=palette)))
            ax_.plot([x - width / 2, x + width / 2], [y - height / 2, y + height / 2], alpha=0)
    if axis_off:
        set_axis_off(ax_)
    if ax is None:
        return fig, ax_


def samples_to_key(samples, circle_of_fifths=None, palette=None):
    if circle_of_fifths is None:
        circle_of_fifths = get_circle_of_fifths()
    key_estimator = KeyEstimator()
    scores = key_estimator.get_score(samples)
    colours = key_scores_to_color(scores=scores, circle_of_fifths=circle_of_fifths, palette=palette)
    return colours


def pitch_scape_plots(scape=None,
                      n_samples=None,
                      samples=None,
                      coords=None,
                      axes=None,
                      prior_counts=0.,
                      normalise=True,
                      size=1,
                      axes_off=True,
                      pitch_labels="sharp",
                      **scape_plot_kwargs):
    if (samples is None) != (coords is None) or \
            (scape is None) != (n_samples is None) or \
            (scape is None) == (samples is None):
        raise ValueError("Please specify EITHER 'scape' and 'n_samples' OR 'samples' and 'coords'")
    # sample scape if samples are not provided
    if scape is not None:
        positions, samples, coords = sample_pitch_scape(scape=scape,
                                                        n_samples=n_samples,
                                                        prior_counts=prior_counts,
                                                        normalize=normalise)
    # create axes if not provided
    if axes is None:
        fig, axes_ = plt.subplots(1, 12, figsize=(12 * size, size))
    else:
        axes_ = axes
    # get global scale for all plots
    scale = np.amax(np.abs(samples[np.logical_not(np.isnan(samples))]))
    if isinstance(pitch_labels, str):
        if pitch_labels == "sharp":
            labels = pitch_classes_sharp
        elif pitch_labels == "flat":
            labels = pitch_classes_flat
        else:
            raise ValueError(f"'pitch_labels' should be 'sharp', 'flat' or an iterable with explicit labels "
                             f"(is: {pitch_labels})")
    else:
        labels = pitch_labels
    # scape plot kwargs
    scape_plot_kwargs = {**dict(vmin=-scale, vmax=scale, axis_off=axes_off), **scape_plot_kwargs}
    # plot different dimensions into separate axes
    for idx, (ax, l) in enumerate(zip(axes_, labels)):
        scape_plot(samples=samples[:, idx],
                   coords=coords,
                   ax=ax,
                   **scape_plot_kwargs)
        if pitch_labels:
            ax.set_xlabel(l)
    if axes is None:
        for ax in axes_:
            ax.set_aspect('equal')
        fig.tight_layout()
        return fig, axes_


def key_scape_plot(scape=None,
                   n_samples=None,
                   samples=None,
                   coords=None,
                   ax=None,
                   size=1,
                   axis_off=True,
                   legend=True,
                   circle_of_fifths=None,
                   palette=None,
                   xlim=(0, 1),
                   ylim=(0, 1),
                   **legend_kwargs):
    if circle_of_fifths is None:
        circle_of_fifths = get_circle_of_fifths()
    if (samples is None) != (coords is None) or \
            (scape is None) != (n_samples is None) or \
            (scape is None) == (samples is None):
        raise ValueError("Please specify EITHER 'scape' and 'n_samples' OR 'samples' and 'coords'")
    # sample scape if samples are not provided
    if scape is not None:
        positions, samples, coords = sample_pitch_scape(scape=scape, n_samples=n_samples)
        # create axes if not provided
    if ax is None:
        fig, ax_ = plt.subplots(1, 1, figsize=(size, size))
    else:
        ax_ = ax
    scape_plot(samples=samples_to_key(samples, circle_of_fifths=circle_of_fifths, palette=palette),
               coords=coords, ax=ax_, axis_off=axis_off, xlim=xlim, ylim=ylim)
    if legend:
        legend_kwargs = {
            **dict(circle_of_fifths=circle_of_fifths,
                   palette=palette,
                   axis_off=False,
                   aspect=(xlim[1] - xlim[0]) / (ylim[1] - ylim[0]),
                   equal_axes_aspect=False),
            **legend_kwargs}
        key_legend(ax_, **legend_kwargs)
    if ax is None:
        return fig, ax_


def scape_plot_from_array(arr: np.ndarray, ax=None, times=None, check=True, coord_kwargs=None, plot_kwargs=None):
    """
    Generate a scape plot from a numpy array of values or colours.
    :param arr: array of values or colors; must be of shape (L,), (L, 3), or (L, 4), where L = n*(n+1)/2 for some
    integer n; for the two dimensional versions the second dimension is interpreted as RGB or RGBA values, respectively.
    :param ax: [optional] axis to plot to
    :param times: optional array of length n + 1 with boundaries between the time slots; if not provided n uniformly
    spaced slots of width 1/n in [0, 1] are assumed.
    :param check: [default=True] check dimensions of arr and times.
    :param coord_kwargs: optional kwargs passed to `coords_from_times` function.
    :param plot_kwargs: optional kwargs passed to `scape_plot` function.
    """
    # get number of time slots
    n = int(np.sqrt(1 / 4 + 2 * arr.shape[0]) - 1 / 2)
    # get times if not provided
    if times is None:
        times = np.linspace(0, 1, n + 1)
    # get coords
    if coord_kwargs is None:
        coord_kwargs = {}
    coords = coords_from_times(times, **coord_kwargs)
    # do checks
    if check:
        if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] in [3, 4])):
            raise ValueError(f"arr must be a one-dimensional array or a two-dimensional array with second dimension of "
                             f"length 3 or 4 (has {arr.ndim} dimensions with shape {arr.shape})")
        if n * (n + 1) / 2 != arr.shape[0]:
            raise ValueError(f"array length does not correspond to triangular matrix; "
                             f"lower match: n = {n} --> {int(n * (n + 1) / 2)}; "
                             f"upper match: n = {n + 1} --> {int((n + 1) * (n + 2) / 2)}; "
                             f"array length: {arr.shape[0]}")
        if coords.shape[0] != arr.shape[0]:
            raise ValueError("coord length and times length do not match")
    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if plot_kwargs is None:
        plot_kwargs = {}
    scape_plot(samples=arr, coords=coords, ax=ax, **plot_kwargs)


def scape_plot(samples,
               coords,
               ax,
               xlim=(0, 1),
               ylim=(0, 1),
               cmap=color_map.RdBu_r,
               vmin=None,
               vmax=None,
               axis_off=True):
    # get scale
    scale = np.amax(np.abs(samples[np.logical_not(np.isnan(samples))]))
    if vmin is None:
        vmin = -scale
    if vmax is None:
        vmax = scale
    # if last dimension of samples is only a single value or samples is one-dimensional apply cmap to get colour
    if samples.shape[-1] == 1 or samples.ndim == 1:
        samples = cmap((samples - vmin) / (vmax - vmin))
    # do plotting
    polygons = PatchCollection([Polygon(np.array(coo), facecolor=color, edgecolor=color, linewidth=0)
                                for color, coo in zip(samples, coords)],
                               match_original=True)
    ax.add_collection(polygons)
    # set axis limits and turn axis of
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if axis_off:
        set_axis_off(ax)


def plot_loss(loss, ax=None):
    """
    Plot the loss for batch optimisation
    :param loss: nested list with loss for each batch per epoch
    :param ax: axis to plot to or None (default) to create new figure
    :return: figure and axis
    """
    if ax is None:
        fig, ax_ = plt.subplots(1, 1)
    else:
        ax_ = ax
    # plot progress
    ax_.set_xlabel("Iteration / Epoch")
    ax_.set_ylabel("Loss")
    color = next(ax_._get_lines.prop_cycler)['color']
    # all iterations
    flat_loss = [l for ll in loss for l in ll]
    ax_.plot(range(1, len(flat_loss) + 1), flat_loss, '-o', markersize=2, color=color)
    # epochs
    epoch_index = [len(loss[0])]
    epoch_value = [loss[0][-1]]
    for l in loss[1:]:
        epoch_index.append(epoch_index[-1] + len(l))
        epoch_value.append(l[-1])
    ax_.plot(epoch_index, epoch_value, 'o', markersize=5, color=color)
    if ax is None:
        return fig, ax_


def plot_piece_assignments(assignments, with_marginals=True, axes=None, size=5):
    n_pieces, n_clusters, n_transpositions = assignments.shape
    # create axes
    if axes is None:
        fig, axes_ = plt.subplots(n_pieces, 1, figsize=(size * (n_transpositions + 1) / n_transpositions,
                                                        size * (n_pieces * (n_clusters + 1)) / n_transpositions))
    else:
        axes_ = axes
    # plot pieces
    for idx in range(n_pieces):
        # plot cluster assignments
        plot_assignment(assignments[idx, :, :],
                        axes_[idx],
                        add_marginal_col=with_marginals)
        # set labels
        axes_[idx].set_xlabel("transposition")
        axes_[idx].set_ylabel("cluster")
    if axes is None:
        return fig, axes_


def plot_cluster_assignments(assignments, with_marginals=True, axes=None, size=5):
    n_pieces, n_clusters, n_transpositions = assignments.shape
    # create axes
    if axes is None:
        fig, axes_ = plt.subplots(n_clusters, 1, figsize=(size * (n_transpositions + 1) / n_transpositions,
                                                        size * (n_clusters * (n_pieces + 1)) / n_transpositions))
    else:
        axes_ = axes
    # plot clusters
    for idx in range(n_clusters):
        # plot assignment
        plot_assignment(assignments[:, idx, :],
                        axes_[idx],
                        add_marginal_col=with_marginals)
        # set labels
        axes_[idx].set_xlabel("transposition")
        axes_[idx].set_ylabel("piece")
    if axes is None:
        return fig, axes_


def plot_assignment(assignments, ax, add_marginal_row=False, add_marginal_col=False):
    # get dimensions
    n_rows, n_cols = assignments.shape
    shape_height, shape_width = n_rows, n_cols
    # add row/column for marginals
    if add_marginal_row:
        shape_height += 1
    if add_marginal_col:
        shape_width += 1
    # assign data
    data = np.zeros((shape_height, shape_width))
    data[:n_rows, :n_cols] = assignments
    # compute marginals
    if add_marginal_row:
        data[n_rows, :n_cols] = data[:n_rows, :n_cols].sum(axis=0)
    if add_marginal_col:
        data[:n_rows, n_cols] = data[:n_rows, :n_cols].sum(axis=1)
    # show image
    ax.imshow(data,
              cmap=color_map.RdBu_r,
              vmin=-1,
              vmax=1,
              extent=(-0.5, shape_width - 0.5, -0.5, shape_height - 0.5),
              origin="lower")
    # remove frame
    axis_set_invisible(ax, splines=True)
    # add separation lines for marginals
    if add_marginal_row:
        ax.plot([-0.5, shape_width - 0.5], [n_rows - 0.5, n_rows - 0.5], color='k')
    if add_marginal_col:
        ax.plot([n_cols - 0.5, n_cols - 0.5], [-0.5, shape_height - 0.5], color='k')
    # add tick labels
    ax.set_xticks(list(range(n_cols)))
    ax.set_yticks(list(range(n_rows)))
