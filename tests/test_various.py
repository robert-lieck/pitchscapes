from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

import pitchscapes.reader as rd
import pitchscapes.plotting as pt
from pitchscapes import PitchScape, PitchScapeMixtureModel


class TestVarious(TestCase):

    _print_count = None

    def test_reader(self):
        p1 = rd.get_pitch_scape('./tests/data/Prelude_No_1_BWV_846_in_C_Major.mid')
        p2 = rd.get_pitch_scape('./tests/data/Prelude_No_1_BWV_846_in_C_Major.mxl')
        pt.key_scape_plot(scape=p1, n_samples=10, size=5)
        pt.key_scape_plot(scape=p2, n_samples=10, size=5)

    def test_various_1(self):
        self.print_count = 0
        def pp():
            print(self.print_count)
            self.print_count += 1

        pp()

        # load some data
        scape_JSB = rd.get_pitch_scape('./tests/data/Prelude_No_1_BWV_846_in_C_Major.mid')
        scape_LvB = rd.get_pitch_scape('./tests/data/Sonata_No._14_Op._27_No._2_-_Ludwig_van_Beethoven.mid')

        pp()

        # plot key-scape
        pt.key_scape_plot(scape=scape_JSB, n_samples=100, size=5)
        pt.key_scape_plot(scape=scape_LvB, n_samples=100, size=5)

        pp()

        # plot pitch scapes
        pt.pitch_scape_plots(scape=scape_JSB, n_samples=30)
        pt.pitch_scape_plots(scape=scape_LvB, n_samples=30)

        pp()

        # optimise a model
        model = PitchScapeMixtureModel(n_clusters=2)
        model.set_data(scapes=[scape_JSB, scape_LvB], n_samples=20)
        model.optimize()
        # plot results
        # for c in model.clusters():
        #     pt.key_scape_plot(scape=c, n_samples=20, size=5)

        pp()

        # get counts an times
        counts, times = rd.pitch_class_counts('./tests/data/Prelude_No_1_BWV_846_in_C_Major.mid')
        print("data of the fist quarter note of the first bar:")
        print(counts[:6])
        print(times[:7])
        print(len(counts), len(times))

        pp()

        # pitch scape from counts and times
        scape = PitchScape(values=counts, times=times)
        # plot
        # pt.key_scape_plot(scape=scape, n_samples=100, size=5)

        pp()

        # get counts
        start, end = sorted(np.random.uniform(scape.min_time, scape.max_time, 2))
        #print(f"pitch counts in [{start}, {end}]:")
        print(scape[start, end])

        pp()

        # normalised version
        normalised_scape = PitchScape(values=counts, times=times, normalise=True)
        print(f"normalised pitch counts in [{start}, {end}]:")
        print(normalised_scape[start, end])

        pp()

        # normalisation done by pitch scape, not by the plotting function
        pt.pitch_scape_plots(scape=normalised_scape, n_samples=30, normalise=False)
        # no normalisation: you see counts summing up towards the top
        pt.pitch_scape_plots(scape=scape, n_samples=30, normalise=False)
        # normalisation done by the plotting function, not by the pitch scape (default)
        pt.pitch_scape_plots(scape=scape, n_samples=30)

        pp()

        # plotting parameters

        # resolution
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        pt.key_scape_plot(scape=scape, n_samples=5, ax=axes[0], legend=False)
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[1], legend=False)
        pp()
        # legend
        fig, axes = plt.subplots(1, 10, figsize=(25, 2.5))
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[0], legend=False)
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[1], location='top')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[2], location='bottom')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[3], location='left')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[4], location='right')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[5], location='top left')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[6], location='top right')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[7], location='left small')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[8], location='right small')
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[9], x_scale=0.5, y_scale=4, x_offset=0.7, y_offset=0.3,
                          horizontal=True, fontsize=5, label_size=0.5, aspect=0.5)
        pp()
        # chromatic vs circle of fifths
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[0], circle_of_fifths=False)
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[1], circle_of_fifths=True)
        fig.tight_layout()
        pp()
        # colour palette
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[0], palette=np.random.uniform(0, 1, (12, 2, 3)))
        pt.key_scape_plot(scape=scape, n_samples=10, ax=axes[1], palette=np.random.uniform(0, 1, (12, 2, 3)))
        pp()

    def test_various_2(self):
        self.print_count = 0
        def pp():
            print(self.print_count)
            self.print_count += 1

        # read the original pieces
        counts_JSB, times_JSB = rd.pitch_class_counts('./tests/data/Prelude_No_1_BWV_846_in_C_Major.mid')
        counts_LvB, times_LvB = rd.pitch_class_counts('./tests/data/Sonata_No._14_Op._27_No._2_-_Ludwig_van_Beethoven.mid')
        # generate artificial pieces by cutting start/end and transposing
        n_pieces = 2
        pieces = [[PitchScape(values=np.roll(c[s:e], shift=shift, axis=1), times=t[s:e + 1])
                   for s, e, shift in zip(np.random.randint(0, 200, n_pieces),  # random start
                                          np.random.randint(len(c) - 200, len(c), n_pieces),  # random end
                                          np.random.randint(0, 12, n_pieces))]  # random transposition
                  for c, t in [(counts_JSB, times_JSB), (counts_LvB, times_LvB)]]
        # # plot the pieces
        # fig, axes = plt.subplots(2, n_pieces + 1, figsize=(3 * (n_pieces + 1), 3 * 2))
        # for ps, axs in zip(pieces, axes):
        #     for p, ax in zip(ps, axs):
        #         pt.key_scape_plot(scape=p, n_samples=30, ax=ax, legend=False)
        # pt.key_legend(horizontal=False, ax=plt.subplot2grid((2, n_pieces + 1), (0, n_pieces), rowspan=2, fig=fig))

        pp()

        # set up model and data
        data = pieces[0] + pieces[1]
        model = PitchScapeMixtureModel(n_clusters=2)
        model.set_data(scapes=data, n_samples=10)
        # train
        model.optimize()
        # plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, c in zip(axes, model.clusters()):
            pt.key_scape_plot(scape=c, n_samples=20, size=5, ax=ax, legend=False)
        pt.key_legend(horizontal=False, ax=axes[2])

        pp()

        # some plotting
        pt.plot_loss(model.loss)
        pt.plot_cluster_assignments(model.assignments())
        pt.plot_piece_assignments(model.assignments())

        pp()

        # single-cluster low-resolution model
        model = PitchScapeMixtureModel(n_clusters=1,  # one cluster
                                       n_center=1,  # low horizontal resolution
                                       n_width=1)  # low vertical resolution
        model.set_data(scapes=data, n_samples=10)  # low number of samples, which is enough for the low resolution
        model.optimize()
        # pt.plot_loss(model.loss)
        # pt.key_scape_plot(model.cluster(0), 10, size=5)

        pp()

        # only increase the resolution
        # everything else is copied from the old model, including the already learned parameters
        model = model.new(n_center=3,
                          n_width=3)
        model.set_data(scapes=data, n_samples=10)  # more samples for higher resolution
        model.optimize()
        # pt.plot_loss(model.loss)
        # pt.key_scape_plot(model.cluster(0), 30, size=5)

        pp()

        # the one (and only) cluster is cloned twice
        model = model.new(clone=[2])
        model.set_data(scapes=data, n_samples=10)
        model.optimize()
        # pt.plot_loss(model.loss)
        # for c in model.clusters():
        #     pt.key_scape_plot(c, 30, size=5)

        pp()

        # both clusters split into two
        model = model.new(clone=[2, 2])
        model.set_data(scapes=data, n_samples=10)
        model.optimize()
        # pt.plot_loss(model.loss)
        # for c in model.clusters():
        #     pt.key_scape_plot(c, 30, size=5)
        # pt.plot_cluster_assignments(model.assignments())

    def test_key_scape_plot(self):
        scape = rd.get_pitch_scape('./tests/data/Prelude_No_1_BWV_846_in_C_Major.mid')
        pt.key_scape_plot(scape=scape, n_samples=10)
        # plt.show()

    def test_scape_plot_from_array(self):
        # check for wrong array shape (3,5)
        self.assertRaises(ValueError, lambda: pt.scape_plot_from_array(arr=np.array([[1, 2, 3, 4, 5],
                                                                                     [6, 7, 8, 9, 10],
                                                                                     [11, 12, 13, 14, 15]])))
        # check for wrong array length (1D case)
        self.assertRaises(ValueError, lambda: pt.scape_plot_from_array(arr=np.array([1, 2, 3, 4])))
        # plot with scalar values
        pt.scape_plot_from_array(arr=np.array([1, 2, 3]))
        # plot with RGB values
        pt.scape_plot_from_array(arr=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        # plot with RGBA values
        pt.scape_plot_from_array(arr=np.array([[1, 0, 0, 0.2], [0, 1, 0, 0.4], [0, 0, 1, 0.6]]))
        # plot with random RGBA values
        n = 5
        values = np.random.uniform(0, 1, (int(n * (n + 1) / 2), 4))
        pt.scape_plot_from_array(arr=values)
        # plt.show()
        # check for mismatch of times and array length
        self.assertRaises(ValueError, lambda: pt.scape_plot_from_array(arr=values,
                                                                       times=np.linspace(0, 1, n),  # should be n + 1
                                                                       ))
