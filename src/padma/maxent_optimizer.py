"""
Maximum Entropy Optimization for Discrete Distributions
========================================================

Provides discrete maximum entropy optimization using simulated annealing
to generate distributions matching target moments (mean, variance, skewness, kurtosis).

Used for:
- Degree distribution generation (Step 5)
- Feature moment fitting (Step 6)
"""

import numpy as np


def maxent_optimize_discrete(target_sum, target_var, n_samples, max_value,
                             target_skew=None, target_kurt=None, debug=False, rng=None,
                             var_penalty_weight=1.0,
                             skew_penalty_weight=0.5,
                             kurt_penalty_weight=0.5,
                             min_value=0):
    """
    Generate discrete distribution using simulated annealing with entropy maximization.
    
    Approach:
    1. Objective: maximize entropy subject to moment constraints (mean, var, skew, kurt)
    2. Moves: large random transfers (k values moved, log-distributed k)
    3. Acceptance: probabilistic for all moves, favoring balanced multi-metric improvements
    
    This combines entropy framework with large-scale exploration moves.
    
    Parameters:
        target_sum: exact total sum of all samples (e.g., 2×n_edges for degrees, total_ones for features)
        target_var: target variance
        n_samples: number of samples to generate (e.g., n_nodes for degrees, n_features for feature means)
        max_value: maximum possible value (e.g., max_degree, or n_nodes for feature counts)
        min_value: minimum possible value for each sample
        target_skew: optional target skewness
        target_kurt: optional target kurtosis
        var_penalty_weight: penalty weight for variance loss
        skew_penalty_weight: penalty weight for skewness loss
        kurt_penalty_weight: penalty weight for kurtosis loss
        
    Returns:
        x_values: array of possible values [min_value, ..., max_value]
        counts: array of length n_samples with the actual sampled values (each in [min_value, max_value])
    """
    if rng is None: rng = np.random.default_rng()

    # Caller must ensure target_sum is valid (e.g., even for degrees)
    min_value = int(min_value)
    max_value = int(max_value)
    if min_value < 0:
        raise ValueError(f"min_value must be >= 0, got {min_value}")
    if max_value < min_value:
        raise ValueError(f"max_value must be >= min_value, got max={max_value}, min={min_value}")

    original_target_sum = int(target_sum)
    shifted_max_value = max_value - min_value
    shifted_target_sum = original_target_sum - n_samples * min_value
    if shifted_target_sum < 0 or shifted_target_sum > n_samples * shifted_max_value:
        raise ValueError(
            "Infeasible target_sum for the given min/max support: "
            f"target_sum={original_target_sum}, n_samples={n_samples}, "
            f"min_value={min_value}, max_value={max_value}"
        )

    target_sum = int(shifted_target_sum)
    target_mean = shifted_target_sum / n_samples
    
    x_values = np.arange(0, shifted_max_value + 1)
    
    def compute_objective(values):
        """
        Compute objective = -entropy + penalties for moment violations.
        Lower is better (minimize -entropy = maximize entropy).
        
        Returns:
            (neg_entropy, penalty): objective components before temperature-aware blending
        """
        # Value counts and probabilities
        value_counts = np.bincount(values, minlength=shifted_max_value + 1)
        total_count = np.sum(value_counts)
        if total_count == 0:
            return 1e10, 1e10
        
        probs = value_counts / total_count
        
        # Entropy: -Σ p log p (we minimize -entropy, so this is positive contribution)
        probs_nz = probs[probs > 1e-10]
        if len(probs_nz) == 0:
            entropy = 0
        else:
            entropy = -np.sum(probs_nz * np.log(probs_nz))
        neg_entropy = -entropy
        
        # Compute statistics
        actual_mean = np.mean(values)
        actual_var = np.var(values, ddof=0)
        var_eps = 1e-12
        
        # Per-moment losses aggregated with an Lp norm.
        # p>2 emphasizes the largest residual smoothly (cleaner than hard max).
        moment_losses = []

        def _compute_metric_loss(actual_value, target_value):
            a = np.arcsinh(actual_value)
            b = np.arcsinh(target_value)
            loss = np.abs(a - b)**1.5

            # scale = min(abs(actual_value), abs(target_value)) + 1e-6
            # error = abs(actual_value - target_value) / scale
            # loss = np.log1p(error) ** 2.5
            return loss
        
        var_loss = var_penalty_weight * _compute_metric_loss(actual_var, target_var)
        moment_losses.append(var_loss)
        
        # Skewness (secondary constraint): log of percentage error
        # Skew is undefined for (near-)zero variance, so only evaluate when numerically stable.
        if target_skew is not None and actual_var > var_eps:
            actual_std = np.sqrt(actual_var)
            denom_skew = actual_std**3
            if denom_skew > var_eps:
                m3 = np.sum((values - actual_mean)**3) / n_samples
                actual_skew = m3 / denom_skew

                if np.isfinite(actual_skew):
                    skew_loss = skew_penalty_weight * _compute_metric_loss(actual_skew, target_skew)
                    moment_losses.append(skew_loss)
        
        # Kurtosis (secondary constraint): log of percentage error
        # Point-mass (near-zero variance) → actual_kurt = 0.0; still penalise vs target.
        if target_kurt is not None:
            if actual_var > var_eps:
                m4 = np.sum((values - actual_mean)**4) / n_samples
                actual_kurt = m4 / (actual_var**2)
                if not np.isfinite(actual_kurt):
                    actual_kurt = 0.0
            else:
                actual_kurt = 0.0
            kurt_loss = kurt_penalty_weight * _compute_metric_loss(actual_kurt, target_kurt)
            moment_losses.append(kurt_loss)

        p_norm = 2.0
        losses = np.asarray(moment_losses, dtype=float)
        penalty = float(np.mean(losses ** p_norm) ** (1.0 / p_norm))
        
        return neg_entropy, penalty

    def combine_objective(neg_entropy, penalty_raw, temp_norm, penalty_ref):
        """
        Blend normalized penalty and entropy cost.

        - penalty_norm in [0,1], relative to initial penalty scale
        - entropy_cost in [0,1], where 0 means max entropy
        - fixed blend: 90% penalty, 10% entropy cost
        """
        _ = temp_norm  # kept for API compatibility; fixed blend does not use temperature.
        penalty_norm = penalty_raw / (penalty_raw + penalty_ref + 1e-12)

        support_size = max(shifted_max_value + 1, 2)
        max_entropy = np.log(support_size)
        entropy = -neg_entropy
        entropy_norm = np.clip(entropy / (max_entropy + 1e-12), 0.0, 1.0)
        entropy_cost = 1.0 - entropy_norm

        entropy_weight = 1e-5

        objective = (1.0 - entropy_weight) * penalty_norm + entropy_weight * entropy_cost
        return float(objective), float(penalty_norm), float(entropy_cost), float(entropy_weight)

    def _sample_without_replacement(pool_indices, weights, k, rng):
        """Sample indices without replacement using non-negative weights."""
        pool_indices = np.asarray(pool_indices, dtype=int)
        if pool_indices.size == 0 or k <= 0:
            return np.array([], dtype=int)

        k = min(int(k), int(pool_indices.size))
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 0.0, None)
        if weights.size != pool_indices.size or np.sum(weights) <= 1e-12:
            return rng.choice(pool_indices, size=k, replace=False)

        positive_mask = weights > 1e-12
        positive_count = int(np.sum(positive_mask))
        if positive_count <= 0:
            return rng.choice(pool_indices, size=k, replace=False)

        if positive_count < k:
            pool_indices = pool_indices[positive_mask]
            weights = weights[positive_mask]
            k = positive_count

        probs = weights / np.sum(weights)
        return rng.choice(pool_indices, size=k, replace=False, p=probs)

    def _random_allocate(amount, capacities, rng):
        """Randomly allocate integer mass across slots without exceeding capacities."""
        capacities = np.asarray(capacities, dtype=int)
        allocation = np.zeros_like(capacities, dtype=int)
        remaining = int(amount)

        while remaining > 0:
            room = capacities - allocation
            active = np.where(room > 0)[0]
            if active.size == 0:
                break

            weights = np.log(room[active] + 2.0).astype(float)
            if np.sum(weights) <= 1e-12:
                weights = np.ones(active.size, dtype=float)
            probs = weights / np.sum(weights)

            proposal = rng.multinomial(remaining, probs)
            proposal = np.minimum(proposal, room[active])
            assigned = int(np.sum(proposal))

            if assigned <= 0:
                chosen = int(rng.choice(active, p=probs))
                proposal = np.zeros(active.size, dtype=int)
                proposal[np.where(active == chosen)[0][0]] = 1
                assigned = 1

            allocation[active] += proposal
            remaining -= assigned

        return allocation, remaining
    
    # Initialize value sequence with uniform distribution
    # All samples start at mean value, then SA optimization adjusts for target moments
    mean_val_int = int(round(target_mean))
    values = np.full(n_samples, mean_val_int, dtype=int)
    
    # Adjust to match exact sum with weighted selection
    current_sum = int(np.sum(values))
    diff = target_sum - current_sum
    
    if diff > 0:
        for _ in range(diff):
            candidates = np.where(values < shifted_max_value)[0]
            if len(candidates) == 0:
                break
            # Weight by log(value+2) for gentler spreading
            weights = np.log(values[candidates] + 2).astype(float)
            weights = weights / np.sum(weights)
            idx = rng.choice(candidates, p=weights)
            values[idx] += 1
    elif diff < 0:
        for _ in range(-diff):
            candidates = np.where(values > 0)[0]
            if len(candidates) == 0:
                break
            # Weight by log(value+2) for gentler concentration
            weights = np.log(values[candidates] + 2).astype(float)
            weights = weights / np.sum(weights)
            idx = rng.choice(candidates, p=weights)
            values[idx] -= 1
    
    # Initial objective
    current_neg_entropy, current_penalty = compute_objective(values)
    initial_penalty = max(current_penalty, 1e-12)
    current_obj, _, _, _ = combine_objective(
        current_neg_entropy,
        current_penalty,
        temp_norm=1.0,
        penalty_ref=initial_penalty,
    )
    best_values = values.copy()
    best_obj = current_obj
    best_penalty = current_penalty
    
    # Simulated annealing parameters
    # Scale iterations with problem size: larger n_samples needs more moves
    # to spread all bins far enough from the initial uniform state.
    base_iterations = 1000
    max_iterations = max(base_iterations, int(n_samples * np.log(n_samples)) * 2)
    max_transfer = min(int(target_sum * 0.5), shifted_max_value)
    initial_temp = 1.0
    # Final temp set so that at cooldown end, max_step = 1
    # max_step = final_temp * target_sum, so final_temp = 1 / target_sum
    final_temp = 0
    no_improve_limit = 500
    no_improve_count = 0
    penalty_threshold = 1e-4  # Early stop if penalty is negligible
    good_enough_penalty = 1e-4
    worse_accept_prob = 0.10
    worse_accept_scale = 0.02
    concentration_limit_frac = 0.70
    
    if debug:
        print(
            f"      [MaxEnt] Starting SA optimization: obj={current_obj:.4f}, "
            f"penalty={current_penalty:.6f}, support=[{min_value}, {max_value}]"
        )
    
    # Early stopping if already perfect
    if best_penalty < penalty_threshold:
        if debug:
            print(f"      [MaxEnt] Already optimal: penalty={best_penalty:.2e} < {penalty_threshold:.2e}")
        final_mean = np.mean(best_values)
        final_var = np.var(best_values, ddof=0)
        if debug:
            print(
                f"      [MaxEnt] Final: mean={final_mean + min_value:.4f} "
                f"(target={original_target_sum / n_samples:.4f}), "
                f"var={final_var:.4f} (target={target_var:.4f})"
            )
        return np.arange(min_value, max_value + 1), (best_values + min_value).astype(int)
    
    # Per-200-iter diagnostic counters
    _block_accepted = 0
    _block_n_many_sum = 0
    _block_total = 0
    _block_improve_total = 0
    _block_improve_accept = 0
    _block_worse_total = 0
    _block_worse_accept = 0
    # Piecewise-linear cooling state. Each accepted improvement resets the
    # line segment from current temperature to 0 at final iteration.
    segment_start_iter = 0
    segment_start_temp = float(initial_temp)

    for iteration in range(max_iterations):
        # Temperature schedule: linear from current segment start to final_temp
        # at the last iteration. Slope is reset whenever improvement is accepted.
        remaining_span = max((max_iterations - 1) - segment_start_iter, 1)
        seg_elapsed = max(iteration - segment_start_iter, 0)
        frac = min(seg_elapsed / remaining_span, 1.0)
        t = segment_start_temp + (final_temp - segment_start_temp) * frac
        t = float(np.clip(t, final_temp, initial_temp))
        
        # Unified move: one ↔ many
        # Normalize temperature so both fan-out and transfer cap collapse to 1 at final_temp.
        if n_samples < 2:
            continue
        temp_norm_den = max(initial_temp - final_temp, 1e-12)
        temp_norm = (t - final_temp) / temp_norm_den
        temp_norm = float(np.clip(temp_norm, 0.0, 1.0))

        # Recompute current scalar objective under this iteration's temperature blend.
        current_obj, _, _, _ = combine_objective(
            current_neg_entropy,
            current_penalty,
            temp_norm=temp_norm,
            penalty_ref=initial_penalty,
        )

        many_max_theoretical = n_samples - 1
        # Linear cooling of the cap; log-distributed sampling inside [1, max_many].
        temp_linear = temp_norm
        max_many = 1 + int(round((many_max_theoretical - 1) * temp_linear))
        max_many = max(1, min(max_many, many_max_theoretical))
        if max_many > 1:
            n_many = int(np.exp(rng.uniform(0.0, np.log(max_many + 1))))
            n_many = max(1, min(n_many, max_many))
        else:
            n_many = 1

        # Pick "one" node and "many" nodes (disjoint), both with log(value+2) weighting
        one_candidates = np.arange(n_samples)
        one_weights = np.log(values + 2).astype(float)
        one_weights = one_weights / np.sum(one_weights)
        one_idx = int(rng.choice(one_candidates, p=one_weights))

        # Linear cooling of transfer cap; moved amount is sampled log-wise later.
        temp_cap = 1 + int(round((max_transfer - 1) * temp_linear))
        temp_cap = max(1, min(temp_cap, max_transfer))

        pool = np.concatenate([np.arange(0, one_idx), np.arange(one_idx + 1, n_samples)])
        n_many = min(n_many, len(pool))
        if n_many <= 0:
            continue

        new_values = values.copy()
        values_moved = 0

        if rng.random() < 0.50:
            capacity_weights = (shifted_max_value - values[pool]).astype(float)
            many_indices = _sample_without_replacement(pool, capacity_weights, n_many, rng)
            if many_indices.size == 0:
                continue

            # Spread: distribute amount from one → many
            capacity_vec = (shifted_max_value - values[many_indices]).astype(int)
            capacity_many_in = int(np.sum(capacity_vec))
            M_neg = min(values[one_idx], temp_cap, capacity_many_in)
            if M_neg <= 0:
                continue
            if M_neg > 1:
                amount = int(np.exp(rng.uniform(0.0, np.log(M_neg + 1))))
                amount = max(1, min(amount, M_neg))
            else:
                amount = 1
            new_values[one_idx] -= amount
            allocation, remaining = _random_allocate(amount, capacity_vec, rng)
            new_values[many_indices] += allocation
            if remaining > 0:
                new_values[one_idx] += remaining  # return what couldn't be placed
            values_moved = amount - remaining
        else:
            donor_weights = values[pool].astype(float)
            many_indices = _sample_without_replacement(pool, donor_weights, n_many, rng)
            if many_indices.size == 0:
                continue

            # Concentrate: gather amount from many → one
            donor_capacity = values[many_indices].astype(int)
            capacity_many_out = int(np.sum(donor_capacity))
            receiver_room = int(shifted_max_value - values[one_idx])
            available_concentration = min(capacity_many_out, receiver_room)
            capped_concentration = 0
            if available_concentration > 0:
                capped_concentration = max(
                    1,
                    int(np.ceil(concentration_limit_frac * available_concentration)),
                )
            M_pos = min(capped_concentration, temp_cap)
            if M_pos <= 0:
                continue
            if M_pos > 1:
                amount = int(np.exp(rng.uniform(0.0, np.log(M_pos + 1))))
                amount = max(1, min(amount, M_pos))
            else:
                amount = 1
            allocation, remaining = _random_allocate(amount, donor_capacity, rng)
            new_values[many_indices] -= allocation
            values_moved = amount - remaining
            new_values[one_idx] += values_moved
        
        if values_moved == 0:
            continue
        
        # Compute new objective
        new_neg_entropy, new_penalty = compute_objective(new_values)
        new_obj, new_penalty_norm, new_entropy_cost, new_entropy_weight = combine_objective(
            new_neg_entropy,
            new_penalty,
            temp_norm=temp_norm,
            penalty_ref=initial_penalty,
        )
        delta_obj = new_obj - current_obj
        
        # Accept move?
        # Improving moves are weighted by temperature:
        # - high temp: be selective, prefer substantial improvements
        # - low temp: accept almost any improvement
        accept = False
        if delta_obj < 0:
            improve = -delta_obj
            improve_rel = improve / (abs(current_obj) + 1e-12)
            required_rel = 2e-6 + 2e-3 * temp_norm
            ratio = np.clip(improve_rel / max(required_rel, 1e-12), 0.0, 1.0)
            # Low-temp floor near 0.98, high-temp floor near 0.02.
            floor = 0.02 + 0.96 * (1.0 - temp_norm)
            prob_improve = floor + (1.0 - floor) * ratio
            accept = (rng.random() < prob_improve)
            _block_improve_total += 1
            if accept:
                _block_improve_accept += 1
        else:
            # Temperature-independent worsening acceptance with a fixed ceiling.
            # Small degradations are accepted more often; larger degradations
            # decay exponentially from the same 5% cap.
            damage_rel = delta_obj / (abs(current_obj) + 1e-12)
            prob_worse = worse_accept_prob * np.exp(-damage_rel / max(worse_accept_scale, 1e-12))
            prob_worse = float(np.clip(prob_worse, 0.0, worse_accept_prob))
            accept = (rng.random() < prob_worse)
            _block_worse_total += 1
            if accept:
                _block_worse_accept += 1
        
        if accept:
            values = new_values
            current_obj = new_obj
            current_neg_entropy = new_neg_entropy
            current_penalty = new_penalty
            _block_accepted += 1
            _block_n_many_sum += n_many
            
            if new_obj < best_obj:
                best_values = new_values.copy()
                best_obj = new_obj
                best_penalty = new_penalty
                no_improve_count = 0
                
                # Early stopping if penalty is negligible (moments matched perfectly)
                if best_penalty < penalty_threshold:
                    if debug:
                        print(f"      [MaxEnt] Early stopping at iteration {iteration+1}: penalty={best_penalty:.2e} < {penalty_threshold:.2e} (perfect match)")
                    break
            else:
                # Local progress should count as progress for patience,
                # even if it does not beat the global best yet.
                if delta_obj < 0:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

        # Reset linear cooling slope when an accepted improving move occurs.
        if accept and delta_obj < 0:
            segment_start_iter = iteration
            segment_start_temp = t

        _block_total += 1

        if current_penalty < good_enough_penalty:
            if debug:
                print(f"      [MaxEnt] Early stopping at iteration {iteration+1}: current_penalty={current_penalty:.2e} < {good_enough_penalty:.2e} (good enough)")
            break

        if no_improve_count >= no_improve_limit:
            if debug:
                print(f"      [MaxEnt] Early stopping at iteration {iteration+1}: no improvement for {no_improve_limit} steps")
            break
        
        if (iteration + 1) % 200 == 0 and debug:
            var_check = np.var(best_values, ddof=0)
            var_scale_dbg = min(var_check, target_var) + 1e-6
            var_err_dbg = abs(var_check - target_var) / var_scale_dbg
            skew_check = float(np.mean((best_values - np.mean(best_values))**3) / (np.std(best_values, ddof=0)**3 + 1e-12))
            kurt_check = float(np.mean((best_values - np.mean(best_values))**4) / (np.var(best_values, ddof=0)**2 + 1e-12))
            skew_err_dbg = np.nan
            if target_skew is not None:
                skew_scale_dbg = min(abs(skew_check), abs(target_skew)) + 1e-6
                skew_err_dbg = abs(skew_check - target_skew) / skew_scale_dbg
            kurt_err_dbg = np.nan
            if target_kurt is not None:
                kurt_scale_dbg = min(abs(kurt_check), abs(target_kurt)) + 1e-6
                kurt_err_dbg = abs(kurt_check - target_kurt) / kurt_scale_dbg
            skew_target_str = f"{target_skew:.3f}" if target_skew is not None else "None"
            kurt_target_str = f"{target_kurt:.3f}" if target_kurt is not None else "None"
            acc_rate = _block_accepted / max(_block_total, 1)
            avg_n_many = _block_n_many_sum / max(_block_accepted, 1)
            acc_improve = _block_improve_accept / max(_block_improve_total, 1)
            acc_worse = _block_worse_accept / max(_block_worse_total, 1)
            print(f"      [MaxEnt] Iter {iteration+1:4d}: bobj={best_obj:.4f} cobj={current_obj:.4f} pen={best_penalty:.2e} "
                                f"t={t:.4f} tn={temp_norm:.3f} span={remaining_span} | "
                  f"var={var_check:.4f}/{target_var:.4f} verr={var_err_dbg:.2e} "
                  f"skew={skew_check:.3f}/{skew_target_str} serr={skew_err_dbg:.2e} "
                  f"kurt={kurt_check:.3f}/{kurt_target_str} kerr={kurt_err_dbg:.2e} | "
                f"acc={acc_rate:.2f} ai={acc_improve:.2f} aw={acc_worse:.3f} "
                f"avg_n_many={avg_n_many:.2f} max_many={max_many} | "
                  f"pen_norm={new_penalty_norm:.3f} ent_cost={new_entropy_cost:.3f} w_ent={new_entropy_weight:.3f}")
            _block_accepted = 0
            _block_n_many_sum = 0
            _block_total = 0
            _block_improve_total = 0
            _block_improve_accept = 0
            _block_worse_total = 0
            _block_worse_accept = 0
    
    # Final statistics
    final_mean = np.mean(best_values)
    final_var = np.var(best_values, ddof=0)
    
    if debug:
        print(
            f"      [MaxEnt] Final: mean={final_mean + min_value:.4f} "
            f"(target={original_target_sum / n_samples:.4f}), "
            f"var={final_var:.4f} (target={target_var:.4f})"
        )
    
    # Return the actual sampled values (shape: n_samples)
    # NOT a histogram - each element is the count for that sample
    return np.arange(min_value, max_value + 1), (best_values + min_value).astype(int)
