import numpy as np

from .util import euclidean_distance


class KeyEstimator:
    """
    Compute key estimates. Custom parameters can be provided on initialisation.
    """

    # different major and minor key profiles
    profiles = {
        "metadata": {
            "major": [0.16728176182048993, 0.02588894338557735, 0.1171404498466961, 0.0356495908305276,
                      0.11123089694868556, 0.09004464377374272, 0.0382659859401006, 0.16751964441157266,
                      0.03408440554363338, 0.09272715883619839, 0.03950089152204068, 0.08066562714073418],
            "minor": [0.1556462295111984, 0.03462755036633632, 0.09903157107324087, 0.1044334346698825,
                      0.048477125346135526, 0.10397079256257924, 0.034020023696857145, 0.15515432119981398,
                      0.07456760938348618, 0.05334836234206381, 0.08307367815831439, 0.05364930169009233]},
        "bayes": {
            "major": [0.19965448155346477, 0.008990613278155785, 0.12994202213313824, 0.01242816337359103,
                      0.12638864214266632, 0.08545131087389726, 0.0263734257934788, 0.20655914875192785,
                      0.013393451149384972, 0.08825555654425607, 0.01458718179430356, 0.08797600261173529],
            "minor": [0.17930236689047313, 0.017073963004507538, 0.10822708838359425, 0.11953464160572477,
                      0.025539463722006557, 0.11184032733554274, 0.020187990226227148, 0.181798715257532,
                      0.07218923098556179, 0.0341394852891066, 0.08257135177438128, 0.047595375525342154]},
        "krumhansl": {
            "major": [0.15195022732711172, 0.0533620483369227, 0.08327351040918879, 0.05575496530270399,
                      0.10480976310122037, 0.09787030390045463, 0.06030150753768843, 0.1241923905240488,
                      0.05719071548217276, 0.08758076094759511, 0.05479779851639147, 0.06891600861450106],
            "minor": [0.14221523253201526, 0.06021118849696697, 0.07908335205571781, 0.12087171422152324,
                      0.05841383958660975, 0.07930802066951245, 0.05706582790384183, 0.1067175915524601,
                      0.08941810829027184, 0.06043585711076162, 0.07503931700741405, 0.07121995057290496]},
        "temperley": {
            "major": [0.12987012987012989, 0.05194805194805195, 0.09090909090909091, 0.05194805194805195,
                      0.1168831168831169, 0.1038961038961039, 0.05194805194805195, 0.1168831168831169,
                      0.05194805194805195, 0.09090909090909091, 0.03896103896103896, 0.1038961038961039],
            "minor": [0.12987012987012989, 0.05194805194805195, 0.09090909090909091, 0.1168831168831169,
                      0.05194805194805195, 0.1038961038961039, 0.05194805194805195, 0.1168831168831169,
                      0.09090909090909091, 0.05194805194805195, 0.03896103896103896, 0.1038961038961039]},
        "albrecht": {
            "major": [0.23800000000000004, 0.006000000000000002, 0.11100000000000003, 0.006000000000000002,
                      0.13700000000000004, 0.09400000000000003, 0.016000000000000004, 0.21400000000000005,
                      0.009000000000000001, 0.08000000000000002, 0.008000000000000002, 0.08100000000000002],
            "minor": [0.22044088176352702, 0.0060120240480961915, 0.10420841683366731, 0.12324649298597193,
                      0.019038076152304607, 0.10320641282565128, 0.012024048096192383, 0.21442885771543083,
                      0.06212424849699398, 0.022044088176352703, 0.06112224448897795, 0.05210420841683366]}
    }

    @staticmethod
    def score(counts, profiles, scoring_func=euclidean_distance, normalise_counts=True, normalise_profiles=True):
        """
        Compute scores for the given count statistics and profiles.
        :param counts: array-like of shape MxK with count statistics; M is the number of independent statistics to
        compute scores for and K the number of categories, for instance, 12 for pitch-classes.
        :param profiles: array-like of shape NxK with profiles to match the count statistics against; N is the
        number of different profiles and K has to be of the same as for the counts.
        :param scoring_func: the scoring function taking two arrays....
        :param normalise_counts: whether to normalise counts before computing scores
        :param normalise_profiles: whether to normalise profiles before computing scores
        :return: array of shape MxNxK with scores for all statistics, profiles, and transpositions
        """
        # get counts and profiles as numpy arrays and normalise
        counts = np.array(counts, dtype=float)
        profiles = np.array(profiles, dtype=float)
        if normalise_counts:
            counts /= counts.sum(axis=1, keepdims=True)
        if normalise_profiles:
            profiles /= profiles.sum(axis=1, keepdims=True)
        # assert matching dimensions
        if counts.shape[1:] != profiles.shape[1:]:
            raise ValueError(f"Count and profile arrays have different size ({counts.shape[1]} vs {profiles.shape[1]})")
        # initialise scores
        scores = np.full((counts.shape[0], profiles.shape[0],) + counts.shape[1:], np.nan)
        # compute scores by iterating over transpositions
        for roll_idx in range(profiles.shape[1]):
            # roll/transpose profiles
            profs = np.roll(profiles, roll_idx, axis=1)
            scores[:, :, roll_idx] = scoring_func(counts[:, None, ...], profs[None, :, ...], axis=2)
        return scores

    @staticmethod
    def argsort_scores(scores):
        return scores.argsort(axis=None).reshape(scores.shape)

    @staticmethod
    def argmin_scores(scores):
        """
        Compute the best match (minimal score).
        :param scores: scores as computed by score function; shape MxNxK (data points x profiles x transpositions)
        :return: array with indices of best match; shape Nx2 (first index is the profile index; second the
        transposition)
        """
        # get list of min indices from flattened scores
        idx = np.array([s.argmin() for s in scores])
        # unravel them into multidimensional indices (grouped by dimension)
        idx = np.unravel_index(idx, scores.shape[1:])
        # concatenate along first dimension and transpose (i-th entry in return correspond to indices of minimum entry
        # in the i-th score array)
        idx = np.concatenate([i[None, :] for i in idx]).T
        # reconstruct nan values
        contains_nans = np.any(np.isnan(scores), axis=tuple(range(1, len(scores.shape))))
        # if there are nans: convert to float and reconstruct
        if np.any(contains_nans):
            idx = idx.astype(float)
            idx[contains_nans, ...] = np.nan
        return idx

    def __init__(self,
                 # profiles="krumhansl",
                 # profiles="bayes",
                 # profiles="metadata",
                 # profiles="temperley",
                 profiles="albrecht",
                 scoring_func=euclidean_distance,
                 normalise_counts=True,
                 normalise_profiles=True):
        """
        Defaults to use
        :param profiles: a string in KeyEstimator.profiles or profiles of shape Nx12 (for N profiles)
        :param scoring_func: scoring function (default Euclidean distance)
        :param normalise_counts: whether to normalise counts before computing scores (default: True)
        :param normalise_profiles: whether to normalise profiles before computing scores (default: True)
        """
        # get profiles
        if isinstance(profiles, str):
            try:
                profiles = KeyEstimator.profiles[profiles]
            except KeyError:
                raise KeyError(f"Could not find source '{profiles}', "
                               f"available sources are {list(KeyEstimator.profiles.keys())}")
            self.profiles = np.array([profiles["major"], profiles["minor"]], dtype=float)
        else:
            self.profiles = np.array(profiles, dtype=float)
        self.scoring_func = scoring_func
        self.normalise_counts = normalise_counts
        self.normalise_profiles = normalise_profiles

    def get_estimate(self, counts):
        """
        Get best key estimate for given counts. Shortcut for using KeyEstimator.argmin_scores with given defaults.
        :param counts: counts to compute scores from
        :return: estimates as return by KeyEstimator.argmin_scores
        """
        return KeyEstimator.argmin_scores(self.get_score(counts))

    def get_score(self, counts):
        """
        Get scores for given counts. Shotcut to using KeyEstimator.score with given defaults.
        :param counts: counts to compute
        :return: scores as returned by KeyEstimator.score
        """
        return KeyEstimator.score(counts,
                                  profiles=self.profiles,
                                  scoring_func=self.scoring_func,
                                  normalise_counts=self.normalise_counts,
                                  normalise_profiles=self.normalise_profiles)
