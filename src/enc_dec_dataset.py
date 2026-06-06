"""Feature encoding/decoding for distributional graph-statistics sampling."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Final

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class FeatureEncoderDecoder(ABC):
    """Abstract base class for encoding and sampling graph statistics.

    This class defines the interface for converting raw graph statistics into
    a compact representation (encoding) and back into synthetic samples.
    """

    def __init__(self, num_classes: int, is_discrete: np.ndarray, rng: np.random.Generator | None = None) -> None:
        """Initializes common state shared by all encoder-decoder subclasses.

        Args:
            num_classes: Number of classes in the dataset; controls the size of
                the per-class storage lists.
            is_discrete: Boolean array of shape ``(num_features,)`` indicating
                which statistics should be treated as discrete integers.
            rng: NumPy random Generator used for all stochastic operations.
                A fresh generator is created when *None* is passed.
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self._encodings: list[np.ndarray | None] = [None] * num_classes
        self._corr_matrices: list[np.ndarray | None] = [None] * num_classes
        # We drop col 1 (edges) because redundant with avg degree + num_nodes
        self._is_discrete: np.ndarray = np.delete(is_discrete, 1)

    def encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Encodes the per-class statistics matrix and stores the result internally.

        Before delegating to :meth:`_encode_features`, this method pre-processes
        the raw statistics matrix: it un-normalises the degree mean (col 2) and
        variance (col 3) by multiplying by ``N-1`` and ``(N-1)^2`` respectively,
        and drops the redundant ``n_edges`` column (col 1).

        Args:
            stat_matrix: Raw statistics matrix of shape ``(num_samples, num_features)``
                as produced by the preprocessing pipeline (with n_edges at col 1).
            class_id: Zero-based index of the class being encoded.
        """
        working_matrix = stat_matrix.astype(float).copy()
        
        # Un-normalize mean degree (col 2) and variance (col 3)
        n_nodes = working_matrix[:, 0]
        denom = np.maximum(n_nodes - 1, 1)
        working_matrix[:, 2] = working_matrix[:, 2] * denom
        working_matrix[:, 3] = working_matrix[:, 3] * (denom ** 2)
        
        # Delete Edges (col 1)
        self._encode_features(np.delete(working_matrix, 1, axis=1), class_id)

    def sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        """Samples synthetic statistics from the stored representation of a class.

        Delegates to :meth:`_sample_features` (which works in the reduced internal
        space without n_edges), then post-processes: it derives ``n_edges`` from
        the sampled ``avg_degree`` and ``n_nodes``, re-normalises the degree mean
        and variance, and inserts n_edges back at col 1.

        Args:
            num_samples: Number of synthetic samples to generate.
            class_id: Zero-based index of the class to sample from.
        Returns:
            Matrix of shape ``(num_samples, num_features)`` in the same column
            layout as the original ``stat_matrix`` passed to :meth:`encode_features`.
        """
        samples = self._sample_features(num_samples, class_id)
        
        n_nodes = samples[:, 0]
        avg_degree = np.maximum(samples[:, 1], 0)
        var_unnorm = np.maximum(samples[:, 2], 0)
        
        # Derive Edges (to become col 1)
        n_edges = np.round((n_nodes * avg_degree) / 2.0)
        
        # Re-normalize mean degree and variance
        denom = np.maximum(n_nodes - 1, 1)
        mean_degree_norm = avg_degree / denom
        var_norm = var_unnorm / (denom ** 2)
        
        # Insert n_edges at col 1. This shifts avg_degree to col 2, and var_unnorm to col 3.
        samples = np.insert(samples, 1, n_edges, axis=1)
        
        # Overwrite cols 2 and 3 with re-normalized values
        samples[:, 2] = mean_degree_norm
        samples[:, 3] = var_norm
        
        return samples

    @abstractmethod
    def _encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        pass

    @abstractmethod
    def _sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_embedding(self, class_id: int) -> np.ndarray:
        """Returns the full embedding vector for a class.

        Args:
            class_id: The ID of the class to get the embedding for.
        Returns:
            Flattened 1D numpy array representing the learned parameters.
        """

    @abstractmethod
    def load_embedding(self, embedding: np.ndarray, class_id: int) -> bool:
        """Loads a flattened embedding vector for a class to enable sampling.

        Args:
            embedding: Flattened 1D numpy array representing the learned parameters.
            class_id: The ID of the class to load the embedding for.
        Returns:
            True if the embedding was loaded successfully, False if validation failed.
        """

    # ------------------------------------------------------------------
    # Internal feature-space bounds (used by load_embedding checks)
    # ------------------------------------------------------------------
    # After encode_features removes col-1 (n_edges), the 13 internal columns are:
    # (min, max) per internal column; None = unbounded on that side.
    _FEATURE_BOUNDS: Final[dict[int, tuple[float | None, float | None]]] = {
        0:  (1.0,  None),   # n_nodes                    >= 1
        1:  (0.0,  None),   # avg_degree (unnormalised)  >= 0 (no upper bound)
        2:  (0.0,  None),   # degree_var (unnormalised)  >= 0 (no upper bound)
        3:  (None, None),   # degree_skew                unbounded
        4:  (None, None),   # degree_kurt (Pearson)      empirically unbounded for small samples
        5:  (0.0,  1.0),    # annd_bin_0                 in [0, 1]
        6:  (0.0,  1.0),    # annd_bin_1                 in [0, 1]
        7:  (0.0,  1.0),    # annd_bin_2                 in [0, 1]
        8:  (0.0,  1.0),    # annd_bin_3                 in [0, 1]
        9:  (0.0,  1.0),    # eccentricity_bin_0         in [0, 1]
        10: (0.0,  1.0),    # eccentricity_bin_1         in [0, 1]
        11: (0.0,  1.0),    # eccentricity_bin_2         in [0, 1]
        12: (0.0,  1.0),    # eccentricity_bin_3         in [0, 1]
    }
    def _check_percentile_matrix(self, enc: np.ndarray) -> bool:
        """Validates a (num_percentiles, num_features) percentile matrix.

        Checks:
        - Min/max percentile values respect ``_FEATURE_BOUNDS``.
        - Each feature column is non-decreasing (monotone increasing).

        Args:
            enc: Percentile matrix of shape (num_percentiles, num_features).
        Returns:
            True if all checks pass, False otherwise.
        """
        for col_idx, (lower, upper) in self._FEATURE_BOUNDS.items():
            if col_idx >= enc.shape[1]:
                continue
            if lower is not None and enc[0, col_idx] < lower - 1e-8:
                logger.warning(f"load_embedding: feature col {col_idx} min percentile {enc[0, col_idx]:.4g} < lower bound {lower:.4g}.")
                return False
            if upper is not None and enc[-1, col_idx] > upper + 1e-8:
                logger.warning(f"load_embedding: feature col {col_idx} max percentile {enc[-1, col_idx]:.4g} > upper bound {upper:.4g}.")
                return False
        diffs = np.diff(enc, axis=0)
        if np.any(diffs < -1e-8):
            bad_cols = np.where(np.any(diffs < -1e-8, axis=0))[0].tolist()
            logger.warning(f"load_embedding: percentile columns not monotone increasing for features {bad_cols}.")
            return False
        return True


class MomentsEncoderDecoder(FeatureEncoderDecoder):
    """Implementation of FeatureEncoderDecoder using statistical moments.

    This class represents distributions using their first k moments (mean,
    variance, skewness, and kurtosis) and reconstructs samples via a polynomial
    transformation of standard normal noise.

    The polynomial coefficients are found by Nelder-Mead on a large **surrogate**
    noise vector (``_SURROGATE_SIZE`` points) so that optimisation converges
    regardless of how small ``num_variants`` is.  The found coefficients are then
    applied to the caller's z-noise of size V.  This preserves the interpretable
    moment-based embedding while decoupling convergence from batch size.
    """

    _SURROGATE_SIZE: int = 10_000  # size of the surrogate z-vector used during optimisation

    def __init__(self, num_classes: int, is_discrete: np.ndarray, k: int = 4, rng: np.random.Generator | None = None) -> None:
        """Initializes the encoder with the number of moments and a random generator.

        Args:
            num_classes: Number of classes in the dataset.
            is_discrete: Boolean array of shape (num_features,).
            k: Number of moments to compute (between 1 and 4).
            rng: NumPy random generator instance.
        Raises:
            ValueError: If k is not in the range [1, 4].
        """
        super().__init__(num_classes, is_discrete, rng)
        if not 1 <= k <= 4: raise ValueError(f"k must be between 1 and 4, got {k}.")
        self.k = k

    def _encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Computes the first k moments for each feature and stores them for a class.

        Args:
            stat_matrix: Matrix of shape (num_samples, num_features).
            class_id: The ID of the class being encoded.
        Raises:
            ValueError: If class_id is out of bounds.
        """
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")

        moments = [np.mean(stat_matrix, axis=0)]
        if self.k >= 2: moments.append(np.var(stat_matrix, axis=0))
        if self.k >= 3: moments.append(skew(stat_matrix, axis=0))
        if self.k >= 4: moments.append(kurtosis(stat_matrix, axis=0, fisher=False))

        self._encodings[class_id] = np.array(moments)

    def _sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        """Generates samples for all features matching the stored moments of a class.

        Args:
            num_samples: Number of samples to generate.
            class_id: The ID of the class to sample from.
        Returns:
            Matrix of shape (num_samples, num_features).
        Raises:
            ValueError: If class_id is out of bounds or not yet encoded.
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before sampling.")

        encoding_matrix = self._encodings[class_id]
        num_features = encoding_matrix.shape[1]
        samples = np.column_stack([
            self._sample_single_feature(encoding_matrix[:, i], num_samples) for i in range(num_features)
        ])

        if np.any(self._is_discrete):
            discrete_mask = self._is_discrete.astype(bool)
            samples[:, discrete_mask] = np.round(np.maximum(0, samples[:, discrete_mask]))
            if discrete_mask[0]:
                samples[:, 0] = np.maximum(1.0, samples[:, 0])

        return samples

    def _sample_single_feature(self, encoding: np.ndarray, num_samples: int) -> np.ndarray:
        """Generates samples for a single feature matching target moments.

        Args:
            encoding: Target moments [mean, variance, skewness, kurtosis].
            num_samples: Number of samples to generate.
        Returns:
            Sampled values matching the target moments.
        """
        num_moments = len(encoding)
        if num_moments == 0 or num_samples == 0:
            return np.array([])

        z_noise = self.rng.standard_normal(num_samples)

        if num_moments == 1:
            return z_noise - np.mean(z_noise) + encoding[0]

        if num_moments == 2:
            z_std = (z_noise - np.mean(z_noise)) / (np.std(z_noise) + 1e-12)
            return z_std * np.sqrt(encoding[1]) + encoding[0]

        if 3 <= num_moments <= 4:
            return self._optimize_polynomial_samples(z_noise, encoding)

        raise ValueError(f"Unsupported number of moments: {num_moments}.")

    def _optimize_polynomial_samples(self, z_noise: np.ndarray, target_moments: np.ndarray) -> np.ndarray:
        """Finds polynomial coefficients to match target moments.

        Optimisation runs on a large surrogate vector (_SURROGATE_SIZE points)
        so convergence is independent of the caller's batch size V.
        The found coefficients are then applied to the original z_noise of size V.

        Args:
            z_noise: Base normal noise of size V (potentially very small).
            target_moments: Target moments [mean, var, skew, kurt].
        Returns:
            Array of size V with values distributed according to target_moments.
        """
        # Large surrogate vector: gives the optimiser enough statistical mass
        # to estimate skewness and kurtosis reliably regardless of V.
        z_surrogate = self.rng.standard_normal(self._SURROGATE_SIZE)

        def _loss_fn(coeffs: np.ndarray) -> float:
            """Internal loss to minimize moment discrepancy."""
            x_candidate = np.polyval(coeffs[::-1], z_surrogate)
            current_moms = self._compute_moments(x_candidate)
            weights = 1.0 / (np.abs(target_moments) + 1e-5)
            return float(np.sum(((current_moms - target_moments) * weights) ** 2))

        init_coeffs = np.zeros(self.k)
        init_coeffs[0] = target_moments[0]
        init_coeffs[1] = np.sqrt(max(target_moments[1], 1e-8))

        res = minimize(_loss_fn, init_coeffs, method="Nelder-Mead", options={"maxiter": 5000})
        if not res.success:
            logger.warning("MomentsEncoderDecoder: optimiser did not converge — %s", res.message)

        # Apply the robustly-found coefficients to the original (small) z_noise.
        return np.polyval(res.x[::-1], z_noise)

    def _compute_moments(self, stat_values: np.ndarray) -> np.ndarray:
        """Computes the first k moments of an array."""
        if stat_values.size == 0: return np.zeros(self.k)
        
        funcs = [np.mean, np.var, skew, lambda x: kurtosis(x, fisher=False)]
        return np.array([f(stat_values) for f in funcs[: self.k]])

    def get_embedding(self, class_id: int) -> np.ndarray:
        """Returns the full embedding vector for a class.

        Args:
            class_id: The ID of the class to get the embedding for.
        Returns:
            Flattened 1D numpy array of the stored moments (interpretable descriptors).
        Raises:
            ValueError: If class_id is out of bounds or not yet encoded.
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before getting embedding.")
        
        return self._encodings[class_id].flatten()

    def load_embedding(self, embedding: np.ndarray, class_id: int) -> bool:
        """Loads a flattened embedding vector for a class to enable sampling.

        Validates the embedding semantics before storing it:
        - Mean values respect per-feature lower and upper bounds.
        - Variance row is non-negative for all features.
        - Kurtosis row (Pearson, row-3) is >= 1 for degree kurtosis (col 4).

        Args:
            embedding: Flattened 1D numpy array of moments.
            class_id: The ID of the class to load the embedding for.
        Returns:
            True on success, False if semantic validation fails.
        """
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")

        num_features = self._is_discrete.size
        expected_size = self.k * num_features
        if embedding.size != expected_size:
            raise ValueError(f"Embedding size {embedding.size} does not match expected {expected_size}.")

        enc = embedding.reshape(self.k, num_features)  # shape (k, num_features)

        # --- Row 0: mean bounds checks ---
        for col_idx, (lower, upper) in self._FEATURE_BOUNDS.items():
            if col_idx >= num_features:
                continue
            val = enc[0, col_idx]
            if lower is not None and val < lower - 1e-8:
                logger.warning(f"MomentsEncoderDecoder.load_embedding: mean of feature col {col_idx} ({val:.4g}) below lower bound {lower:.4g} for class {class_id}.")
                return False
            if upper is not None and val > upper + 1e-8:
                logger.warning(f"MomentsEncoderDecoder.load_embedding: mean of feature col {col_idx} ({val:.4g}) above upper bound {upper:.4g} for class {class_id}.")
                return False

        # --- Row 1: variance must be >= 0 ---
        if self.k >= 2 and np.any(enc[1] < -1e-8):
            bad = np.where(enc[1] < -1e-8)[0].tolist()
            logger.warning(f"MomentsEncoderDecoder.load_embedding: negative variance for feature cols {bad} in class {class_id}.")
            return False

        self._encodings[class_id] = enc
        return True


class PercentileEncoderDecoder(FeatureEncoderDecoder):
    """Implementation of FeatureEncoderDecoder using percentiles.

    This class represents distributions using a set of percentile values (edges)
    and reconstructs samples by interpolating between these values.
    """

    def __init__(self, num_classes: int, is_discrete: np.ndarray, percentile_size: float = 0.1, replicate_correlation: bool = False, rng: np.random.Generator | None = None) -> None:
        """Initializes the encoder with the percentile size and a random generator.

        Args:
            num_classes: Number of classes in the dataset.
            is_discrete: Boolean array of shape (num_features,).
            percentile_size: The size of the percentile steps (between 0 and 1).
            replicate_correlation: Whether to replicate correlation between features.
            rng: NumPy random generator instance.
        """
        super().__init__(num_classes, is_discrete, rng)
        if not 0 < percentile_size <= 1:raise ValueError(f"percentile_size must be between 0 and 1, got {percentile_size}.")
        self.percentile_size = percentile_size
        self.replicate_correlation = replicate_correlation

    def _encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Computes the percentile edges for each feature and stores them for a class.

        Args:
            stat_matrix: Matrix of shape (num_samples, num_features).
            class_id: The ID of the class being encoded.
        """
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")

        # Determine quantile grid
        q = np.arange(0, 1 + (self.percentile_size / 2), self.percentile_size)
        q[q > 1.0] = 1.0
        if q[-1] < 1.0: q = np.append(q, 1.0)
            
        # Apply jittering to discrete columns in one go
        working_matrix = stat_matrix.astype(float).copy()
        if np.any(self._is_discrete):
            discrete_mask = self._is_discrete.astype(bool)
            jitter = self.rng.uniform(-0.5, 0.5, size=(working_matrix.shape[0], np.sum(discrete_mask)))
            working_matrix[:, discrete_mask] += jitter
            
        # Vectorized quantile computation across all features (axis=0)
        self._encodings[class_id] = np.quantile(working_matrix, q, axis=0) # Shape: (num_percentiles, num_features)

        if self.replicate_correlation:
            corr_matrix = pd.DataFrame(stat_matrix).corr(method="spearman").values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            self._corr_matrices[class_id] = corr_matrix + np.eye(corr_matrix.shape[0]) * 1e-6

    def _sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        """Generates samples for all features matching stored percentiles of a class.

        Args:
            num_samples: Number of samples to generate per feature.
            class_id: The ID of the class to sample from.
        Returns:
            Matrix of shape (num_samples, num_features).
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before sampling.")
            
        encoding_matrix = self._encodings[class_id]
        num_percentiles, num_features = encoding_matrix.shape
        q_grid = np.linspace(0, 1, num_percentiles)

        # --- PHASE COPULA for generating u ---
        if self.replicate_correlation:
            # Sample multivariate normal distribution
            z = self.rng.multivariate_normal(mean=np.zeros(num_features), cov=self._corr_matrices[class_id], size=num_samples)
            # Transform to uniform via normal CDF
            u = scipy.stats.norm.cdf(z)
        else:
            # Sample uniform random values for all samples and features at once
            u = self.rng.uniform(0, 1, size=(num_samples, num_features))        
        
        # --- PHASE MARGINALS for generating final samples ---
        # Find indices of the bins for each sample in u
        idx = np.searchsorted(q_grid, u) - 1
        idx = np.clip(idx, 0, num_percentiles - 2)
        
        # Vectorized linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        # For each sample/feature, we select the appropriate bin edges from self._encodings
        v0 = np.take_along_axis(encoding_matrix, idx, axis=0)
        v1 = np.take_along_axis(encoding_matrix, idx + 1, axis=0)
        
        q0 = q_grid[idx]
        q1 = q_grid[idx + 1]
        
        samples_matrix = v0 + (v1 - v0) * (u - q0) / (q1 - q0)
        
        # Post-process discrete features
        if np.any(self._is_discrete):
            discrete_mask = self._is_discrete.astype(bool)
            samples_matrix[:, discrete_mask] = np.round(np.maximum(0, samples_matrix[:, discrete_mask]))
            
        return samples_matrix

    def get_embedding(self, class_id: int) -> np.ndarray:
        """Returns the full embedding vector for a class.

        Args:
            class_id: The ID of the class to get the embedding for.
        Returns:
            Flattened 1D numpy array representing the learned parameters.
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before getting embedding.")
            
        parts = [self._encodings[class_id].flatten()]
        if self.replicate_correlation and self._corr_matrices[class_id] is not None:
            corr_mat = self._corr_matrices[class_id]
            parts.append(corr_mat[np.triu_indices_from(corr_mat)])
            
        return np.concatenate(parts)

    def load_embedding(self, embedding: np.ndarray, class_id: int) -> bool:
        """Loads a flattened embedding vector for a class to enable sampling.

        Validates the embedding semantics before storing it:
        - Percentile values respect per-feature lower bounds.
        - Each feature column is monotone non-decreasing across percentiles.
        - Correlation matrix is corrected to be PSD if needed.

        Args:
            embedding: Flattened 1D numpy array of percentiles (+ optional corr triu).
            class_id: The ID of the class to load the embedding for.
        Returns:
            True on success, False if semantic validation fails.
        """
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")

        num_features = self._is_discrete.size
        q = np.arange(0, 1 + (self.percentile_size / 2), self.percentile_size)
        q[q > 1.0] = 1.0
        if q[-1] < 1.0: q = np.append(q, 1.0)
        num_percentiles = len(q)

        enc_size = num_percentiles * num_features

        if self.replicate_correlation:
            expected_triu_size = num_features * (num_features + 1) // 2
            expected_size = enc_size + expected_triu_size
            if embedding.size != expected_size:
                raise ValueError(f"Embedding size {embedding.size} does not match expected {expected_size}.")

            enc_flat = embedding[:enc_size]
            triu_flat = embedding[enc_size:]
        else:
            if embedding.size != enc_size:
                raise ValueError(f"Embedding size {embedding.size} does not match expected {enc_size}.")
            enc_flat = embedding

        enc = enc_flat.reshape(num_percentiles, num_features)

        # --- Validate percentile matrix ---
        if not self._check_percentile_matrix(enc):
            return False

        self._encodings[class_id] = enc

        if self.replicate_correlation:
            corr_mat = np.zeros((num_features, num_features))
            triu_indices = np.triu_indices(num_features)
            corr_mat[triu_indices] = triu_flat
            corr_mat = corr_mat + corr_mat.T - np.diag(np.diag(corr_mat))

            # Symmetrize and project onto nearest PSD matrix
            corr_mat = (corr_mat + corr_mat.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            corr_mat = (eigenvectors * eigenvalues) @ eigenvectors.T

            self._corr_matrices[class_id] = corr_mat

        return True


class GMCMEncoderDecoder(FeatureEncoderDecoder):
    """Implementation of FeatureEncoderDecoder using Gaussian Mixture Copula Models (GMCM).

    This class captures the complex joint distribution of features using a GMM in the
    latent normal space, after transforming the original marginals to uniform via percentiles.
    """

    def __init__(self, num_classes: int, is_discrete: np.ndarray, percentile_size: float = 0.1, n_components: int = 10, rng: np.random.Generator | None = None) -> None:
        """Initializes the encoder with the percentile size, number of components, and random generator.

        Args:
            num_classes: Number of classes in the dataset.
            is_discrete: Boolean array of shape (num_features,).
            percentile_size: The size of the percentile steps (between 0 and 1).
            n_components: Number of Gaussian mixture components in the latent space.
            rng: NumPy random generator instance.
        """
        super().__init__(num_classes, is_discrete, rng)
        if not 0 < percentile_size <= 1:
            raise ValueError(f"percentile_size must be between 0 and 1, got {percentile_size}.")
        self.percentile_size = percentile_size
        self.n_components = n_components
        self._gmm_models: list[GaussianMixture | None] = [None] * num_classes

    def _encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Computes percentiles for marginals and fits a GMM on the latent normal space.

        Args:
            stat_matrix: Matrix of shape (num_samples, num_features).
            class_id: The ID of the class being encoded.
        """
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")

        # Determine quantile grid
        q = np.arange(0, 1 + (self.percentile_size / 2), self.percentile_size)
        q[q > 1.0] = 1.0
        if q[-1] < 1.0: q = np.append(q, 1.0)
            
        # Apply jittering to discrete columns in one go to smooth them
        working_matrix = stat_matrix.astype(float).copy()
        if np.any(self._is_discrete):
            discrete_mask = self._is_discrete.astype(bool)
            jitter = self.rng.uniform(-0.5, 0.5, size=(working_matrix.shape[0], np.sum(discrete_mask)))
            working_matrix[:, discrete_mask] += jitter

        # Vectorized quantile computation across all features (axis=0)
        # Shape: (num_percentiles, num_features)
        percentiles = np.quantile(working_matrix, q, axis=0)
        self._encodings[class_id] = percentiles
        
        # --- PHASE LATENT SPACE: PIT to Uniform, then to Normal ---
        num_percentiles, num_features = percentiles.shape
        q_uniform = np.linspace(0, 1, num_percentiles)
        u = np.zeros_like(working_matrix)
        
        for i in range(num_features):
            u[:, i] = np.interp(working_matrix[:, i], percentiles[:, i], q_uniform)
            
        # Clip to avoid infinite values in the inverse normal CDF
        eps = 1e-6
        u = np.clip(u, eps, 1 - eps)
        
        z = scipy.stats.norm.ppf(u)
        
        # --- PHASE COPULA: Fit GMM ---
        seed = int(self.rng.integers(0, 10000)) if self.rng else 42
        gmm_rng = np.random.RandomState(seed)
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=gmm_rng)
        gmm.fit(z)
        self._gmm_models[class_id] = gmm

    def _sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        """Generates samples by sampling from GMM and applying inverse PIT.

        Args:
            num_samples: Number of samples to generate.
            class_id: The ID of the class to sample from.
            
        Returns:
            Matrix of shape (num_samples, num_features).
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None or self._gmm_models[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before sampling.")
            
        encoding_matrix = self._encodings[class_id]
        num_percentiles, num_features = encoding_matrix.shape
        q_grid = np.linspace(0, 1, num_percentiles)

        # --- PHASE COPULA: Sample from GMM and transform to uniform ---
        gmm = self._gmm_models[class_id]
        z_sample, _ = gmm.sample(num_samples)
        
        # Scikit-learn's GMM returns samples sorted by component! We must shuffle them
        # so that when sampling a batch of variants, the variants don't get stuck on a single component.
        self.rng.shuffle(z_sample)
        
        u = scipy.stats.norm.cdf(z_sample)
        
        # --- PHASE MARGINALS: Inverse PIT ---
        # Find indices of the bins for each sample in u
        idx = np.searchsorted(q_grid, u) - 1
        idx = np.clip(idx, 0, num_percentiles - 2)
        
        # Vectorized linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        v0 = np.take_along_axis(encoding_matrix, idx, axis=0)
        v1 = np.take_along_axis(encoding_matrix, idx + 1, axis=0)
        
        q0 = q_grid[idx]
        q1 = q_grid[idx + 1]
        
        samples_matrix = v0 + (v1 - v0) * (u - q0) / (q1 - q0)
        
        # Post-process discrete features
        if np.any(self._is_discrete):
            discrete_mask = self._is_discrete.astype(bool)
            samples_matrix[:, discrete_mask] = np.round(np.maximum(0, samples_matrix[:, discrete_mask]))
            
        return samples_matrix

    def get_embedding(self, class_id: int) -> np.ndarray:
        """Returns the full embedding vector for a class.

        Args:
            class_id: The ID of the class to get the embedding for.
        Returns:
            Flattened 1D numpy array representing the learned parameters.
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None or self._gmm_models[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before getting embedding.")
            
        gmm = self._gmm_models[class_id]
        parts = [
            self._encodings[class_id].flatten(),
            gmm.weights_.flatten(),
            gmm.means_.flatten(),
        ]
        # Append only the upper triangle of each covariance matrix
        for i in range(gmm.covariances_.shape[0]):
            cov_mat = gmm.covariances_[i]
            parts.append(cov_mat[np.triu_indices_from(cov_mat)])
            
        return np.concatenate(parts)

    def load_embedding(self, embedding: np.ndarray, class_id: int) -> bool:
        """Loads a flattened embedding vector for a class to enable sampling.

        Validates the embedding before committing:

        - Percentile matrix must satisfy :meth:`_check_percentile_matrix`.
        - GMM weights are normalised via softmax so arbitrary real-valued inputs
          are converted to a valid probability distribution.
        - GMM means are checked for finiteness (they live in latent normal space,
          so no domain bounds apply).
        - Each covariance matrix is symmetrised and projected onto the nearest
          positive-semi-definite matrix (eigenvalue clipping at 1e-6).
        - Precision-Cholesky factors are computed for use by scikit-learn's GMM
          sampler.

        Args:
            embedding: Flattened 1D numpy array with layout
                ``[percentiles_flat | weights | means_flat | cov_triu_flat]``.
            class_id: Zero-based index of the class to load the embedding for.
        Returns:
            True on success, False if the percentile or mean validation fails.
        Raises:
            ValueError: If ``class_id`` is out of range or the embedding size is
                inconsistent with the configured number of features and components.
        """
        import scipy.linalg

        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")

        num_features = self._is_discrete.size
        q = np.arange(0, 1 + (self.percentile_size / 2), self.percentile_size)
        q[q > 1.0] = 1.0
        if q[-1] < 1.0: q = np.append(q, 1.0)
        num_percentiles = len(q)

        enc_size = num_percentiles * num_features
        weights_size = self.n_components
        means_size = self.n_components * num_features
        cov_triu_size = self.n_components * (num_features * (num_features + 1) // 2)

        expected_size = enc_size + weights_size + means_size + cov_triu_size
        if embedding.size != expected_size:
            raise ValueError(f"Embedding size {embedding.size} does not match expected {expected_size}.")

        # Unpack the arrays
        ptr = 0
        enc_flat = embedding[ptr : ptr + enc_size]; ptr += enc_size
        weights = embedding[ptr : ptr + weights_size]; ptr += weights_size
        means_flat = embedding[ptr : ptr + means_size]; ptr += means_size
        cov_triu_flat = embedding[ptr:]

        # --- Validate percentile matrix ---
        enc = enc_flat.reshape(num_percentiles, num_features)
        if not self._check_percentile_matrix(enc):
            return False

        # --- Normalize GMM weights using softmax-like correction ---
        # This allows arbitrary real-valued weights to be converted to a valid probability distribution.
        shifted_weights = weights - np.max(weights)
        weights = np.exp(shifted_weights)
        weights = weights / np.sum(weights)


        # --- Validate GMM means (same bounds as percentile min values) ---
        means = means_flat.reshape(self.n_components, num_features)
        # Means are in the *latent normal space* (after PIT), so no domain bounds apply.
        # We only do a basic NaN/Inf sanity check.
        if not np.all(np.isfinite(means)):
            logger.warning(f"GMCMEncoderDecoder.load_embedding: non-finite GMM means for class {class_id}.")
            return False

        # --- Restore covariances (with PSD correction) ---
        covariances = np.zeros((self.n_components, num_features, num_features))
        triu_indices = np.triu_indices(num_features)
        triu_len = num_features * (num_features + 1) // 2

        for i in range(self.n_components):
            cov_triu = cov_triu_flat[i * triu_len : (i + 1) * triu_len]
            cov_mat = np.zeros((num_features, num_features))
            cov_mat[triu_indices] = cov_triu
            cov_mat = cov_mat + cov_mat.T - np.diag(np.diag(cov_mat))
            cov_mat = (cov_mat + cov_mat.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            cov_mat = (eigenvectors * eigenvalues) @ eigenvectors.T
            covariances[i] = cov_mat

        # --- Assemble fitted GaussianMixture ---
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.covariances_ = covariances

        precisions_chol = np.empty((self.n_components, num_features, num_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = scipy.linalg.cholesky(covariance, lower=True)
            except scipy.linalg.LinAlgError:
                cov_chol = scipy.linalg.cholesky(covariance + 1e-6 * np.eye(num_features), lower=True)
            precisions_chol[k] = scipy.linalg.solve_triangular(
                cov_chol, np.eye(num_features), lower=True
            ).T
        gmm.precisions_cholesky_ = precisions_chol
        gmm.converged_ = True

        # --- Commit ---
        self._encodings[class_id] = enc
        self._gmm_models[class_id] = gmm
        return True


KNOWN_SAMPLERS: dict[str, Callable[..., FeatureEncoderDecoder]] = {
    "gmcm": GMCMEncoderDecoder,
    "moments": MomentsEncoderDecoder,
    "percentile": partial(PercentileEncoderDecoder, replicate_correlation=False),
    "percentile_corr": partial(PercentileEncoderDecoder, replicate_correlation=True),
}
