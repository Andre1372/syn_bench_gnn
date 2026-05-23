"""Module for encoding and decoding graph features using statistical approaches.

This module provides an abstraction for feature encoding/decoding and a specific
implementation based on statistical moments.
"""

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
        """Initializes the encoder with a random generator."""
        self.rng = rng if rng is not None else np.random.default_rng()
        self._encodings: list[np.ndarray | None] = [None] * num_classes
        self._corr_matrices: list[np.ndarray | None] = [None] * num_classes
        # We drop col 1 (edges) because redundant with avg degree + num_nodes
        self._is_discrete: np.ndarray = np.delete(is_discrete, 1)

    def encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Encodes multiple features into compact representations and stores them for a class.

        Args:
            stat_matrix: Matrix of shape (num_samples, num_features).
            class_id: The ID of the class being encoded.
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
        """Samples synthetic feature values from the stored representations of a class.

        Args:
            num_samples: Number of synthetic samples to generate.
            class_id: The ID of the class to sample from.
        Returns:
            Matrix of shape (num_samples, num_features).
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
    def load_embedding(self, embedding: np.ndarray, class_id: int) -> None:
        """Loads a flattened embedding vector for a class to enable sampling.

        Args:
            embedding: Flattened 1D numpy array representing the learned parameters.
            class_id: The ID of the class to load the embedding for.
        """


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

    def load_embedding(self, embedding: np.ndarray, class_id: int) -> None:
        """Loads a flattened embedding vector for a class to enable sampling."""
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        
        num_features = self._is_discrete.size
        expected_size = self.k * num_features
        if embedding.size != expected_size:
            raise ValueError(f"Embedding size {embedding.size} does not match expected {expected_size}.")
        
        self._encodings[class_id] = embedding.reshape(self.k, num_features)


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

    def load_embedding(self, embedding: np.ndarray, class_id: int) -> None:
        """Loads a flattened embedding vector for a class to enable sampling."""
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
            
            self._encodings[class_id] = enc_flat.reshape(num_percentiles, num_features)
            
            corr_mat = np.zeros((num_features, num_features))
            triu_indices = np.triu_indices(num_features)
            corr_mat[triu_indices] = triu_flat
            corr_mat = corr_mat + corr_mat.T - np.diag(np.diag(corr_mat))
            
            # Symmetrize
            corr_mat = (corr_mat + corr_mat.T) / 2.0
            
            # Project onto the nearest Positive Semi-Definite (PSD) matrix using eigenvalue clipping
            eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            corr_mat = (eigenvectors * eigenvalues) @ eigenvectors.T
            
            self._corr_matrices[class_id] = corr_mat
        else:
            if embedding.size != enc_size:
                raise ValueError(f"Embedding size {embedding.size} does not match expected {enc_size}.")
            self._encodings[class_id] = embedding.reshape(num_percentiles, num_features)


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

    def load_embedding(self, embedding: np.ndarray, class_id: int) -> None:
        """Loads a flattened embedding vector for a class to enable sampling."""
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
        idx = 0
        enc_flat = embedding[idx : idx + enc_size]
        idx += enc_size
        
        weights = embedding[idx : idx + weights_size]
        idx += weights_size
        
        means_flat = embedding[idx : idx + means_size]
        idx += means_size
        
        cov_triu_flat = embedding[idx:]
        
        # 1. Restore encodings
        self._encodings[class_id] = enc_flat.reshape(num_percentiles, num_features)
        
        # 2. Restore GMM weights, means, covariances
        means = means_flat.reshape(self.n_components, num_features)
        
        covariances = np.zeros((self.n_components, num_features, num_features))
        triu_indices = np.triu_indices(num_features)
        triu_len = num_features * (num_features + 1) // 2
        
        for i in range(self.n_components):
            cov_triu = cov_triu_flat[i * triu_len : (i + 1) * triu_len]
            cov_mat = np.zeros((num_features, num_features))
            cov_mat[triu_indices] = cov_triu
            cov_mat = cov_mat + cov_mat.T - np.diag(np.diag(cov_mat))
            
            # Symmetrize
            cov_mat = (cov_mat + cov_mat.T) / 2.0
            
            # Project onto the nearest Positive Semi-Definite (PSD) matrix using eigenvalue clipping
            eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            cov_mat = (eigenvectors * eigenvalues) @ eigenvectors.T
            
            covariances[i] = cov_mat
            
        # 3. Create a fitted GaussianMixture instance
        import scipy.linalg
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.covariances_ = covariances
        
        # Compute precisions_cholesky_
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
        
        self._gmm_models[class_id] = gmm


KNOWN_SAMPLERS: dict[str, Callable[..., FeatureEncoderDecoder]] = {
    "gmcm": GMCMEncoderDecoder,
    "moments": MomentsEncoderDecoder,
    "percentile": partial(PercentileEncoderDecoder, replicate_correlation=False),
    "percentile_corr": partial(PercentileEncoderDecoder, replicate_correlation=True),
}
