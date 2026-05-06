"""Module for encoding and decoding graph features using statistical approaches.

This module provides an abstraction for feature encoding/decoding and a specific
implementation based on statistical moments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Final

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew


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
        self._is_discrete: np.ndarray = is_discrete

    @abstractmethod
    def encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Encodes multiple features into compact representations and stores them for a class.

        Args:
            stat_matrix: Matrix of shape (num_samples, num_features).
            class_id: The ID of the class being encoded.
        """

    @abstractmethod
    def sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        """Samples synthetic feature values from the stored representations of a class.

        Args:
            num_samples: Number of synthetic samples to generate.
            class_id: The ID of the class to sample from.
        Returns:
            Matrix of shape (num_samples, num_features).
        """


class MomentsEncoderDecoder(FeatureEncoderDecoder):
    """Implementation of FeatureEncoderDecoder using statistical moments.

    This class represents distributions using their first k moments (mean,
    variance, skewness, and kurtosis) and reconstructs samples via a polynomial
    transformation of standard normal noise.
    """

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

    def encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
        """Computes the first k moments for each feature and stores them for a class.

        Args:
            stat_matrix: Matrix of shape (num_samples, num_features).
            class_id: The ID of the class being encoded.
        """
        if not 0 <= class_id < len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        
        # Vectorized moments calculation
        moments = [np.mean(stat_matrix, axis=0)]
        if self.k >= 2: moments.append(np.var(stat_matrix, axis=0))
        if self.k >= 3: moments.append(skew(stat_matrix, axis=0))
        if self.k >= 4: moments.append(kurtosis(stat_matrix, axis=0, fisher=False))
        
        self._encodings[class_id] = np.array(moments)

    def sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
        """Generates samples for all features matching the stored moments of a class.

        Args:
            num_samples: Number of samples to generate.
            class_id: The ID of the class to sample from.
        Returns:
            Matrix of shape (num_samples, num_features).
        """
        if class_id >= len(self._encodings):
            raise ValueError(f"class_id {class_id} out of bounds (0-{len(self._encodings)-1}).")
        if self._encodings[class_id] is None:
            raise ValueError(f"Class {class_id} must be encoded before sampling.")
        
        encoding_matrix = self._encodings[class_id]
        num_features = encoding_matrix.shape[1]
        # Optimization is performed per-feature
        samples = [self._sample_single_feature(encoding_matrix[:, i], num_samples) for i in range(num_features)]
        samples_matrix = np.column_stack(samples)
        
        # Post-process discrete features
        if np.any(self._is_discrete):
            discrete_mask = self._is_discrete.astype(bool)
            samples_matrix[:, discrete_mask] = np.round(np.maximum(0, samples_matrix[:, discrete_mask]))
            
        return samples_matrix

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

        # Base noise from a standard normal distribution
        z_noise = self.rng.standard_normal(num_samples)

        # If k=1 (only mean), simply shift the sample
        if num_moments == 1:
            return z_noise - np.mean(z_noise) + encoding[0]

        # If k=2 (mean and variance), standardize and scale (Z-score re-scaling)
        if num_moments == 2:
            z_std = (z_noise - np.mean(z_noise)) / (np.std(z_noise) + 1e-12)
            return z_std * np.sqrt(encoding[1]) + encoding[0]

        # General case using optimization for k=3 and k=4
        if 3 <= num_moments <= 4:
            return self._optimize_polynomial_samples(z_noise, encoding)

        raise ValueError(f"Unsupported number of moments: {num_moments}.")

    def _compute_moments(self, stat_values: np.ndarray) -> np.ndarray:
        """Computes the first k moments of an array."""
        if stat_values.size == 0: return np.zeros(self.k)

        # Mapping of moment functions (Fisher=False for Pearson kurtosis)
        moment_funcs = [np.mean, np.var, skew, lambda x: kurtosis(x, fisher=False)][:self.k]

        return np.array([func(stat_values) for func in moment_funcs])

    def _optimize_polynomial_samples(self, z_noise: np.ndarray, target_moments: np.ndarray) -> np.ndarray:
        """Finds polynomial coefficients to match target moments.

        Args:
            z_noise: Base normal noise.
            target_moments: Target moments to match.
        Returns:
            Optimized sample array matching the moments.
        """
        def _loss_fn(coeffs: np.ndarray) -> float:
            """Internal loss to minimize moment discrepancy."""
            # Use np.polyval with reversed coeffs to treat coeffs[i] as c_i for z^i
            x_candidate = np.polyval(coeffs[::-1], z_noise)
            current_moms = self._compute_moments(x_candidate)
            # Weighted squared error to handle different scales of moments
            weights = 1.0 / (np.abs(target_moments) + 1e-5)
            return float(np.sum(((current_moms - target_moments) * weights) ** 2))

        # Heuristic initialization: match mean and scale by standard deviation
        init_coeffs = np.zeros(self.k)
        init_coeffs[0] = target_moments[0]
        init_coeffs[1] = np.sqrt(max(target_moments[1], 1e-8))

        # Check for mathematical feasibility of skewness/kurtosis combination
        if self.k == 4:
            target_skew, target_kurt = target_moments[2], target_moments[3]
            if target_kurt < (target_skew**2 + 1):
                # The optimizer will struggle as this violates mathematical limits
                pass

        res = minimize(_loss_fn, init_coeffs, method="Nelder-Mead", options={"maxiter": 5000})

        if not res.success:
            print(f"Warning: Optimizer failed to converge. Status: {res.message}")

        return np.polyval(res.x[::-1], z_noise)


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

    def encode_features(self, stat_matrix: np.ndarray, class_id: int) -> None:
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

    def sample_features(self, num_samples: int, class_id: int) -> np.ndarray:
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
