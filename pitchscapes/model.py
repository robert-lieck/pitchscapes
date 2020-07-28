import numpy as np
from itertools import product
from datetime import datetime

import torch
from torch.nn import Module, Parameter
from torch.distributions.dirichlet import Dirichlet
from IPython.display import clear_output

from .util import sample_discrete_scape, multi_sample_pitch_scapes, random_batch_ids, start_end_to_center_width, safe_int
from .optimization import WarmAdam
from .scapes import Scape


class ScapeWrapper(Scape):
    """
    Wrapper class for PitchScapeMixtureModel, so that a specific cluster can be accessed as Scape object.
    """
    def __init__(self, model, cluster):
        super().__init__(0, 1)
        self.model = model
        self.cluster = cluster

    def __getitem__(self, item):
        with torch.no_grad():
            center, width = start_end_to_center_width(*item)
            val = self.model.f(positions=np.array([[center, width]]), positive=True, normalise=True, log_rep=False)
            return val[0, :, self.cluster].numpy()


class PitchScapeMixtureModel(Module):

    def __init__(self,
                 n_center=5,
                 n_width=5,
                 n_clusters=1,
                 n_pitch=12,
                 offset=0.,
                 init_noise=1e-8,
                 periodic=True,
                 c_major=1.):
        super().__init__()
        if not n_pitch % 2 == 0:
            raise ValueError("Not implemented for uneven number of pitches")
        # basic parameters
        self.n_pitch = n_pitch
        self.n_clusters = n_clusters
        self.n_center = n_center
        self.n_width = n_width
        self.periodic = periodic

        # Fourier representations
        self.coefficients = np.random.uniform(-1, 1, (2 * n_center + 1,
                                                      2 * n_width + 1,
                                                      n_pitch,
                                                      n_clusters)) * init_noise
        self.coefficients[0, 0, 0, :] = offset
        # add bias towards C-major (Fourier coefficients from diatonic scale)
        cos_coeff = np.array([3.5, 0.1339, 0.5, 1, -0.5, 1.8660, -0.5]) / 3.5
        sin_coeff = np.array([-0.2320, 0.8660, 0., 0.8660, 3.2320]) / 3.5
        self.coefficients[0, 0, 0, :] += c_major
        self.coefficients[0, 0, :, :] = c_major * np.concatenate((cos_coeff, sin_coeff))[:, None]
        self.coefficients = Parameter(torch.from_numpy(self.coefficients))
        self.wave_vectors = []
        self.phase_shifts = []
        # center and width dimensions
        for n in [n_center, n_width]:
            # first n+1 cos; second n sin (i.e. cos with phase shift)
            if self.periodic:
                wv = np.arange(0, n + 1)
            else:
                wv = np.arange(0, n + 1) * (1 - 1 / (2 * n))
            self.wave_vectors.append(np.concatenate((wv, wv[1:])))
            self.phase_shifts.append(np.concatenate((np.zeros(n + 1),
                                                     np.ones(n) / 4)))
        # pitch dimension (first n+1 cos; second n-1 sin; n = n_pitch / 2 for even n_pitch)
        self.wave_vectors.append(np.concatenate((np.arange(0, safe_int(n_pitch / 2) + 1) / n_pitch,
                                                 np.arange(1, safe_int(n_pitch / 2)) / n_pitch)))
        self.phase_shifts.append(np.concatenate((np.zeros(safe_int(n_pitch / 2) + 1),
                                                 np.ones(safe_int(n_pitch / 2) - 1) / 4)))
        # to torch
        self.wave_vectors = [torch.from_numpy(wv) for wv in self.wave_vectors]
        self.phase_shifts = [torch.from_numpy(ps) for ps in self.phase_shifts]

        # mixture model
        # log joint has shape (n_pieces, n_clusters, n_transpositions) and equals p(piece | cluster, transposition),
        # which is proportional to the joint p(piece, cluster, transposition) when assuming uniform priors over clusters
        # and transpositions; the probability for a piece is a DENSITY because it comes from an average Dirichlet, i.e.
        # it is not normalised and can be larger than one (positive log)
        self.log_joint = None
        self.positions = None
        self.samples = None
        self.coords = None
        self.piece_weights = None
        self.loss = None

    def set_data(self, scapes=None, n_samples=None, positions=None, samples=None, coords=None, piece_weights=None):
        if not ((scapes is None) == (n_samples is None)) or \
                not ((positions is None) == (samples is None) == (coords is None)) or \
                ((scapes is None) == (positions is None)):
            raise ValueError("Please specify EITHER 'scapes' and 'n_samples' OR 'positions', 'samples' and 'coords'")
        if scapes is not None:
            positions, samples, coords = multi_sample_pitch_scapes(scapes=scapes, n_samples=n_samples)
        # get dimensions
        (n_positions, n_dims) = positions.shape
        (n_pieces, n_positions_, n_pitches) = samples.shape
        n_transpositions = n_pitches
        # store
        self.positions = torch.from_numpy(positions.copy())
        # check consistency
        assert n_dims == 2, f"{n_dims}"
        assert n_pitches == self.n_pitch, f"{n_pitches} | {self.n_pitch}"
        assert n_positions == n_positions_, f"{n_positions} | {n_positions_}"
        # check and assign weights
        if piece_weights is not None:
            assert piece_weights.shape == (n_pieces,), f"{piece_weights.shape} == {n_pieces}"
            self.piece_weights = torch.from_numpy(piece_weights.copy())
        else:
            self.piece_weights = None
        # store samples with precomputed transpositions
        self.samples = torch.from_numpy(np.full((n_pieces, n_positions, n_pitches, n_transpositions), np.nan))
        samples = torch.from_numpy(samples)
        for transp_idx in range(n_transpositions):
            self.samples[:, :, :, transp_idx] = samples.roll(shifts=transp_idx, dims=2)

    def set_coefficients(self, cluster_idx, coefficients):
        # helper function to extract indices
        def coef_indices(n):
            """
            Return the start and end index of the model coefficients for cosine and sine components of the continuous
            dimensions (center and width). Additionally, the difference that needs to be added to start in slicing to get
            the full slice, i.e. [start:start + diff] such that start + diff = end + 1.
            :param n: order of Fourier series (number of complex-valued parameters)
            :return: (start_cos, end_cos, diff_cos), (start_sin, end_sin, diff_sin)
            """
            return (0, n, n + 1), (n + 1, 2 * n, n)
        # indices for this cluster
        this_n_center = self.n_center
        this_n_width = self.n_width
        ((this_center_start_cos, this_center_end_cos, this_center_diff_cos),
         (this_center_start_sin, this_center_end_sin, this_center_diff_sin)) = coef_indices(this_n_center)
        ((this_width_start_cos, this_width_end_cos, this_width_diff_cos),
         (this_width_start_sin, this_width_end_sin, this_width_diff_sin)) = coef_indices(this_n_width)
        # indices for other cluster
        other_n_center = safe_int((coefficients.shape[1] - 1) / 2)
        other_n_width = safe_int((coefficients.shape[0] - 1) / 2)
        ((other_center_start_cos, other_center_end_cos, other_center_diff_cos),
         (other_center_start_sin, other_center_end_sin, other_center_diff_sin)) = coef_indices(other_n_center)
        ((other_width_start_cos, other_width_end_cos, other_width_diff_cos),
         (other_width_start_sin, other_width_end_sin, other_width_diff_sin)) = coef_indices(other_n_width)
        # use the shorter of both for copying
        center_diff_cos = min(this_center_diff_cos, other_center_diff_cos)
        center_diff_sin = min(this_center_diff_sin, other_center_diff_sin)
        width_diff_cos = min(this_width_diff_cos, other_width_diff_cos)
        width_diff_sin = min(this_width_diff_sin, other_width_diff_sin)
        # do assignment on numpy level
        self_coefficients = self.coefficients.data.numpy()
        # assign all combinations of sine and cosine blocks
        for this_center_start, other_center_start, center_diff in [
            (this_center_start_cos, other_center_start_cos, center_diff_cos),
            (this_center_start_sin, other_center_start_sin, center_diff_sin)
        ]:
            for this_width_start, other_width_start, width_diff in [
                (this_width_start_cos, other_width_start_cos, width_diff_cos),
                (this_width_start_sin, other_width_start_sin, width_diff_sin)
            ]:
                self_coefficients[
                this_center_start:this_center_start + center_diff,
                this_width_start:this_width_start + width_diff,
                :,
                cluster_idx] = coefficients[
                               other_center_start:other_center_start + center_diff,
                               other_width_start:other_width_start + width_diff, :]
        self.coefficients.data = torch.from_numpy(self_coefficients)

    def new(self,
            clone=None,
            n_center=None,
            n_width=None,
            n_clusters=None,
            n_pitch=None,
            clone_noise=1e-8,
            **kwargs):
        if n_pitch is not None and n_pitch != self.n_pitch:
            raise ValueError(f"Changing the number of pitches is not supported (n_pitch={n_pitch} does not match value "
                             f"of model: {self.n_pitch})")
        # initialise new model
        if n_center is None:
            n_center = self.n_center
        if n_width is None:
            n_width = self.n_width
        if n_pitch is None:
            n_pitch = self.n_pitch
        if clone is None:
            if n_clusters is None:
                n_clusters = self.n_clusters
            clone = np.ones(n_clusters, dtype=int)
        else:
            if n_clusters is None:
                n_clusters = safe_int(np.sum(clone))
            elif np.sum(clone) != n_clusters:
                raise ValueError(f"The total number of clusters to create by cloning (sum of 'clone') must equal the "
                                 f"number of initialised clusters. But we have n_clusters={n_clusters} and "
                                 f"clone={clone}")
        new_model = PitchScapeMixtureModel(n_center=n_center,
                                           n_width=n_width,
                                           n_clusters=n_clusters,
                                           n_pitch=n_pitch,
                                           **kwargs)
        # set coefficients
        clone_indices = [clone_index
                         for clone_index, clone_multiplicity in enumerate(clone)
                         for _ in range(clone_multiplicity)]
        assert len(clone_indices) == n_clusters, f"{n_clusters}, {clone_indices}"
        for to_idx, from_idx in enumerate(clone_indices):
            coef = self.coefficients[:, :, :, from_idx].data.numpy()
            new_model.set_coefficients(cluster_idx=to_idx,
                                       coefficients=coef + np.random.uniform(-clone_noise, clone_noise, coef.shape))
        return new_model


    def log_joint_pdf(self, samples, log_f):
        """
        Returns (the log of) p(piece | cluster, transposition); assuming uniform prior over clusters and transpositions,
        this is proportional to the joint p(piece, cluster, transposition)
        :param samples: array of shape (n_pieces, n_samples, n_pitches, n_clusters, n_transpositions)
        :param log_f: array of shape (n_pieces, n_samples, n_pitches, n_clusters, n_transpositions)
        :return: array of shape (...)
        """
        # construct Dirichlet distributions (move pitch dimension last
        dir = Dirichlet(torch.einsum('abcde->abdec', log_f.exp()))
        # get point-wise probabilities and multiply up (log-sum) samples for each piece
        probs = dir.log_prob(torch.einsum('abcde->abdec', samples))
        return probs.sum(dim=1)

    def log_assignments(self):
        # marginalise out cluster and transposition to get piece likelihood
        # (ignore constant priors fo cluster and transposition as they cancel out in the division below)
        piece_log_like = self.log_joint.logsumexp(dim=(1, 2), keepdim=True)
        # divide (log-minus) joint by piece likelihood to get assigment (cluster and transposition) probability
        # for each piece
        return (self.log_joint - piece_log_like).data.cpu().numpy()

    def assignments(self):
        return np.exp(self.log_assignments())

    def piece_log_like(self):
        # marginalise out cluster and transposition to get piece likelihood
        piece_log_like = self.log_joint.logsumexp(dim=(1, 2)).data.cpu().numpy() - np.log(self.n_clusters * self.n_pitch)
        return piece_log_like

    def cluster_entropy(self):
        # compute the expected piece log-likelihood per cluster, or cluster entropy:
        # sum p(piece | cluster) log p(piece | cluster)

        # only include pieces that have a non-zero weight
        if self.piece_weights is not None:
            include = np.logical_not(np.isclose(self.piece_weights.data.cpu().numpy(), 0, atol=1e-50))
            piece_weights = self.piece_weights[include]
            log_joint = self.log_joint[include]
        else:
            piece_weights = torch.ones(self.log_joint.shape[0])
            log_joint = self.log_joint
        # marginalise out transposition to get p(piece, cluster)
        piece_log_like = log_joint.logsumexp(dim=(2,)) - np.log(self.n_pitch)
        # multiply by piece weight
        piece_log_like = piece_log_like * piece_weights[:, None]
        # marginalise out piece to get p(cluster) and normalise to get p(piece | cluster)
        piece_log_like -= piece_log_like.logsumexp(dim=(0,)) - np.log(piece_weights.sum())
        # compute entropy
        entropy = -(piece_log_like.exp() * piece_log_like).sum(dim=(0,))
        return entropy.data.cpu().numpy()

    def f(self, positions, positive=True, normalise=True, log_rep=False):
        """
        returns the function value described by Fourier coefficients of the model at the specified positions
        :param positions: array of shape (n_data, 2) or (n_data, n_pitch, 3)
        :param positive: whether to apply exp to make output positive (default: True)
        :param normalise: whether to normalise along pitch dimensions (default: True)
        :param log_rep: whether to return result in log representation (default: False)
        :return: array of shape (n_data, n_pitch, n_clusters) with function values
        """
        if len(positions.shape) == 2 and positions.shape[1] == 2:
            n_data = positions.shape[0]
            # add pitch dimension
            full_positions = np.empty((n_data, self.n_pitch, 3))
            full_positions[:, :, 0] = positions[:, None, 0]
            full_positions[:, :, 1] = positions[:, None, 1]
            full_positions[:, :, 2] = np.arange(self.n_pitch)[None, None, :]
            full_positions = torch.from_numpy(full_positions)
        elif len(positions.shape) == 3 and positions.shape[1] == self.n_pitch and positions.shape[2] == 3:
            # full positions
            n_data = positions.shape[0]
            full_positions = positions
        else:
            raise ValueError(
                f"Positions should have shape (n_data, 2) or (n_data, n_pitch, 3) with n_pitch={self.n_pitch}, but "
                f"found a shape of {positions.shape}.")
        # reshape from point_position x pitch_position x dimension to point_and_pitch_position x dimension
        full_positions = full_positions.view(n_data * self.n_pitch, 3)
        # compute phases
        phases = [full_positions[:, None, dim] * self.wave_vectors[dim][None, :] + self.phase_shifts[dim][None, :]
                  for dim in range(3)]
        # compute amplitudes
        amplitudes = [np.cos(2 * np.pi * ph) for ph in phases]
        amplitudes = amplitudes[0][:, :, None, None] * amplitudes[1][:, None, :, None] * amplitudes[2][:, None,
                                                                                         None, :]
        # multiply coefficients and sum up
        amplitudes = (self.coefficients[None, ...] * amplitudes[..., None]).sum(dim=(1, 2, 3))
        # reshape
        amplitudes = amplitudes.view(n_data, self.n_pitch, self.n_clusters)
        # make positive and apply log (if requested)
        if positive and not log_rep:
            amplitudes = amplitudes.exp()
        if not positive and log_rep:
            amplitudes = amplitudes.log()
        # normalise (if requested)
        if normalise:
            if log_rep:
                amplitudes = amplitudes - amplitudes.logsumexp(dim=1, keepdim=True)
            else:
                amplitudes = amplitudes / amplitudes.sum(dim=1, keepdim=True)
        return amplitudes

    def log_likelihood(self, batch=None):
        """
        parameters that control computation:
            full_positions, precompute_trans, joint_clusters, shared_positions

        dimensions that may be iterated over:
            n_pieces, n_transpositions, n_clusters

        tensors and shapes:
            positions: (n_pieces, n_positions, 2)
            samples: (n_pieces, n_positions, n_pitches, n_transpositions)
            log_joint: (n_pieces, self.n_clusters, n_transpositions)
            log_f: (n_positions, n_pitches)

        Positions are are flattened for computing model predictions and the results is reshaped to the correct
        dimensions afterwards.
        """
        # whether values are computed successively
        # get dimensions
        (n_pieces, n_positions, n_pitches, n_transpositions) = self.samples.shape
        # use all data if batch was not provided
        if batch is None:
            batch = np.ones(n_pieces, dtype=bool)
        # exclude pieces with zero weight from batch
        if self.piece_weights is not None:
            batch = np.logical_and(batch, np.logical_not(np.isclose(self.piece_weights.data.cpu().numpy(),
                                                                    0,
                                                                    atol=1e-50)))
        n_pieces_in_batch = batch.sum()
        if n_pieces_in_batch == 0:
            raise ValueError("Cannot compute log-likelihood: empty batch and/or only zero weights")
        # iterators, slicing, effective dimentions, dropping of singleton dimensions
        # pieces
        piece_it = [None]
        piece_slice = batch
        positions_piece_slice = slice(None)
        log_joint_piece_slice = slice(None)
        eff_n_pieces = n_pieces_in_batch
        piece_drop = slice(None)
        # clusters
        cluster_it = [None]
        cluster_slice = slice(None)
        eff_n_clusters = self.n_clusters
        cluster_drop = slice(None)
        # transposition
        trans_it = [None]
        trans_slice = slice(None)
        eff_n_trans = n_transpositions
        trans_drop = slice(None)
        # allocate model log-joint if shape does not match
        if self.log_joint is None or self.log_joint.shape != (n_pieces, self.n_clusters, n_transpositions):
            self.log_joint = torch.from_numpy(np.full((n_pieces, self.n_clusters, n_transpositions), np.nan))
            self.log_joint.requires_grad_(False)
        # pre-allocate total log-joint (of this evaluation) if values are computed iteratively
        total_log_joint = None
        # do iteration
        log_f_dict = {}
        log_f = None
        rolled_log_f = None
        for piece_idx, cluster_idx, trans_idx in product(piece_it, cluster_it, trans_it):
            # whether to compute log_f
            if (piece_idx, cluster_idx) not in log_f_dict:
                # shape: (n_pieces, n_positions, 2)
                positions = self.positions[positions_piece_slice].view(-1, 2)
                # get values
                log_f = self.f(positions,
                               positive=True,
                               normalise=False,
                               log_rep=True)
                # reshape: make compatible with (n_pieces, n_positions, n_pitches, n_clusters, n_transpositions)
                log_f = log_f.view((1, n_positions, n_pitches, eff_n_clusters, 1))
                # store
                log_f_dict[(piece_idx, cluster_idx)] = None
            # compute transposition
            if (piece_idx, cluster_idx, trans_idx) not in log_f_dict:
                rolled_log_f = log_f
                log_f_dict[(piece_idx, cluster_idx, trans_idx)] = None
            # pick piece and add dimension for clusters
            samples = self.samples[piece_slice].view(eff_n_pieces, n_positions, n_pitches, 1, eff_n_trans)
            # compute log joint
            partial_log_joint = self.log_joint_pdf(samples=samples, log_f=rolled_log_f)
            # drop singleton dimensions
            partial_log_joint = partial_log_joint[piece_drop, cluster_drop, trans_drop]
            # assign to total log joint
            total_log_joint = partial_log_joint
        # normalise by number of samples per piece
        total_log_joint /= n_positions
        # assign total log-joint (of this evaluation) to corresponding entries in model's log-joint
        self.log_joint[batch] = total_log_joint
        # marginalise out latent variables (cluster and tranposition) to get log likelihood per piece
        # take product (i.e. sum in log representation) for log likelihood;
        # normalise to get cross-entropy
        piece_log_like = total_log_joint.logsumexp(dim=(1, 2)) - np.log(self.n_clusters * n_transpositions)
        if self.piece_weights is None:
            eff_piece_weight_sum = eff_n_pieces
        else:
            # compute effective piece weight
            eff_piece_weight_sum = self.piece_weights.sum()
            # weight piece log-like (weighted cross-entropy)
            piece_log_like = piece_log_like * self.piece_weights[batch]
        # multiply piece likelihoods (log-sum) and normalise (log-divide; root in dirict representation)
        # by number of data points (effective number of pieces and positions per piece)
        data_log_like = piece_log_like.sum() / eff_piece_weight_sum
        if np.isnan(data_log_like.data.cpu().numpy()):
            raise RuntimeWarning("Encounterd nan in data log-likelihood")
        return data_log_like

    def closure(self, batch=None):
        self.zero_grad()
        loss = -self.log_likelihood(batch=batch)
        loss.backward()
        return loss

    def optimize(self, init_lr=0, final_lr=1e-2, lr_beta=0.99,
                 n_batches=1, max_epochs=np.inf, latency=100, delta=1.,
                 progress=True, same_line=True, restore_best=True):
        # initialise optimizer
        optimizer = WarmAdam(self.parameters(), init_lr=init_lr, lr=final_lr, lr_beta=lr_beta)
        # training
        self.loss = []
        batch_ids = random_batch_ids(n_data=self.samples.shape[0], n_batches=n_batches)
        epoch_idx = 0
        start_time = datetime.now()
        if same_line:
            def short(n, prec=5):
                return np.format_float_positional(n, precision=prec)
        else:
            def short(n):
                return n
        best_state_dict = None
        best_loss = np.inf
        while True:  # epoch loop
            epoch_idx += 1
            if max_epochs is not None and epoch_idx > max_epochs:
                break
            self.loss.append([])
            if progress and not same_line:
                print(f"epoch {epoch_idx}")
            for batch_idx in range(n_batches):  # batch loop
                it = len(self.loss[-1]) + 1
                if progress and not same_line:
                    print(f"    batch {it}/{n_batches}")
                self.loss[-1].append(float(optimizer.step(closure=lambda: self.closure(batch_ids == batch_idx))))
                if progress and not same_line:
                    print(f"        batch loss: {short(self.loss[-1][-1])}")
                    print(f"        time elapsed {datetime.now()-start_time}")
                # same line output
                if progress and same_line:
                    clear_output(wait=True)
                    print(f"\repoch {epoch_idx} | "
                          f"batch {it}/{n_batches} | "
                          f"time elapsed {datetime.now()-start_time} | "
                          f"batch loss: {short(self.loss[-1][-1])}", end="")
            if progress:
                if same_line:
                    print(f" | epoch loss: {short(np.mean(self.loss[-1]))}", end="")
                else:
                    print(f"    epoch loss: {short(np.mean(self.loss[-1]))}")
            if restore_best:
                epoch_loss = np.mean(self.loss[-1])
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_state_dict = self.state_dict()
            if epoch_idx > latency:
                d = np.mean(self.loss[-1]) - np.mean(self.loss[-latency - 1])
                if progress:
                    if same_line:
                        print(f" | delta[{latency}]: {short(d)}", end="")
                    else:
                        print(f"    delta[{latency}]: {short(d)}")
                if -d <= delta:
                    break
        if restore_best:
            if best_state_dict is None:
                raise RuntimeWarning("Cannot restore best parameters, not state dict stored.")
            self.load_state_dict(best_state_dict)
        if progress and same_line:
            print("")

    def get_samples(self, n_samples, positive=True, normalise=True, log_rep=False):
        with torch.no_grad():
            # get positions (and possibly coords)
            positions = []
            coords = []
            for center, width, c in sample_discrete_scape(None, np.linspace(0, 1, n_samples),
                                                          center_width=True, coords=True, value=False):
                positions.append([center, width])
                coords.append(c)
            coords = np.array(coords)
            positions = np.array(positions)
            with torch.no_grad():
                samples = self.f(positions=positions,
                                 positive=positive,
                                 normalise=normalise,
                                 log_rep=log_rep).data.cpu().numpy()
            return samples, coords

    def cluster(self, idx):
        return ScapeWrapper(self, idx)

    def clusters(self):
        for idx in range(self.n_clusters):
            yield self.cluster(idx)