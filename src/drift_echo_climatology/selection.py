"""Select and classify the study's longest neighboring energy-period paths."""

import numpy as np
import pandas as pd

from .detection import COLUMNS, _typed


MIN_NEIGHBOR_POINTS = 4
ENERGY_THRESHOLD_KEV = 1050
CATEGORIES = ("valid", "edge", "invalid", "single")
SELECTED_COLUMNS = COLUMNS + ["category", "event_id", "sequence_length"]


def longest_sequences(frame, min_len=MIN_NEIGHBOR_POINTS):
    """Return all longest paths across adjacent *available* effective energies.

    Period must decrease strictly between neighboring energy buckets. Paths
    rounded to the same six-decimal energy/period pairs are deduplicated before
    ranking, retaining the original sorting and stack traversal order.
    """
    if frame.empty:
        return []
    local = frame.reset_index(drop=False)
    local = local.sort_values(["energy_channel"]).reset_index(drop=True)
    energies = local["energy_channel"].unique()
    buckets = [local[local["energy_channel"] == energy].index.to_list()
               for energy in energies]
    periods = local["max_echo_cand_per"].astype(float)
    lengths, predecessors = {}, {}
    for index in buckets[0]:
        lengths[index], predecessors[index] = 1, []
    for bucket in range(1, len(buckets)):
        for index in buckets[bucket]:
            length, previous = 1, []
            for predecessor in buckets[bucket - 1]:
                if periods.iloc[predecessor] > periods.iloc[index]:
                    candidate = lengths[predecessor] + 1
                    if candidate > length:
                        length, previous = candidate, [predecessor]
                    elif candidate == length:
                        previous.append(predecessor)
            lengths[index], predecessors[index] = length, previous
    longest = max(lengths.values()) if lengths else 0
    if longest < min_len:
        return []
    ends = [index for index, length in lengths.items() if length == longest]
    paths = []
    for end in ends:
        stack = [(end,)]
        while stack:
            path = stack.pop()
            first = path[0]
            if lengths[first] == 1:
                paths.append(path)
                continue
            for predecessor in predecessors[first]:
                stack.append((predecessor,) + path)
    seen, sequences = set(), []
    energy_values, period_values = local["energy_channel"].values, periods.values
    for path in paths:
        indices = list(path)
        key = tuple(zip(np.round(energy_values[indices], 6),
                        np.round(period_values[indices], 6)))
        if key in seen or len(path) != longest:
            continue
        seen.add(key)
        sequence = local.iloc[indices].copy()
        if "index" in sequence.columns:
            sequence = sequence.drop(columns=["index"])
        sequence = sequence.reindex(columns=frame.columns, fill_value=np.nan)
        sequences.append(sequence.reset_index(drop=True))
    return sequences, longest


def envelope_metrics(y_obs, y_lower, y_upper, y_exp=None, tol_frac=0.15, margin=0.30):
    """Compute the original finite-envelope metrics and optional expected fit."""
    observed = np.asarray(y_obs, dtype=float)
    lower = np.asarray(y_lower, dtype=float)
    upper = np.asarray(y_upper, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(lower) & np.isfinite(upper)
    observed, lower, upper = observed[finite], lower[finite], upper[finite]
    count = int(len(observed))
    band = np.maximum(upper - lower, 1e-12)
    positions = (observed - lower) / band
    mean_position = float(np.mean(positions))
    orange_used, orange_count = False, 0
    mae = rmse = bias = within_expected = None
    if y_exp is not None:
        expected = np.asarray(y_exp, dtype=float)[finite]
        available = np.isfinite(expected)
        orange_count = int(np.sum(available))
        if orange_count > 0:
            orange_used = True
            error = (observed[available] - expected[available]) / band[available]
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            bias = float(np.mean(error))
            tolerance = tol_frac * band[available]
            within_expected = float(np.mean(np.abs(observed[available] - expected[available]) <= tolerance))
    below_strict, above_strict = observed < lower, observed > upper
    count_below_strict, count_above_strict = int(np.sum(below_strict)), int(np.sum(above_strict))
    outside_strict = count_below_strict + count_above_strict
    soft_lower, soft_upper = lower - margin * band, upper + margin * band
    below_soft, above_soft = observed < soft_lower, observed > soft_upper
    count_below_soft, count_above_soft = int(np.sum(below_soft)), int(np.sum(above_soft))
    outside_soft = count_below_soft + count_above_soft
    distance_below = np.maximum(0.0, (soft_lower - observed) / band)
    distance_above = np.maximum(0.0, (observed - soft_upper) / band)
    severity = distance_below + distance_above
    return {"n_points": count, "ned_mean": mean_position, "ned_values": positions,
            "orange_used": orange_used, "orange_n": orange_count,
            "exp_norm_mae": mae, "exp_norm_rmse": rmse, "exp_norm_bias": bias,
            "frac_within_expected_band": within_expected,
            "frac_outside_envelope_strict": float(outside_strict / count) if count else 0.0,
            "frac_below_lower_strict": float(count_below_strict / count) if count else 0.0,
            "frac_above_upper_strict": float(count_above_strict / count) if count else 0.0,
            "num_outside_strict": outside_strict,
            "frac_outside_envelope_soft": float(outside_soft / count) if count else 0.0,
            "frac_below_lower_soft": float(count_below_soft / count) if count else 0.0,
            "frac_above_upper_soft": float(count_above_soft / count) if count else 0.0,
            "num_outside_soft": outside_soft,
            "mean_severity_outside_soft": float(np.mean(severity)) if count else 0.0,
            "max_severity_outside_soft": float(np.max(severity)) if count else 0.0,
            "params": {"tol_frac": tol_frac, "margin": margin}}


def score_sequence(frame):
    """Return the lexicographic score (lower wins) and envelope metrics."""
    observed = frame["max_echo_cand_per"].values
    expected = frame["expected_drift_per"].values if "expected_drift_per" in frame else None
    metrics = envelope_metrics(observed, frame["est_drift_per_a90_adj"].values,
                               frame["est_drift_per_a0_adj"].values, expected)
    expected_mae = (metrics["exp_norm_mae"]
                    if metrics["orange_used"] and metrics["orange_n"] > 0 else np.inf)
    # Preserve the study's index-based log-period curvature; energy spacing
    # is not used in its second finite differences.
    log_period = np.log(np.maximum(np.asarray(observed, float), 1e-9))
    curvature = (float(np.sum(np.abs(log_period[2:] - 2 * log_period[1:-1] + log_period[:-2])))
                 if len(log_period) >= 3 else 0.0)
    negative_snr = (-float(np.sum(10.0 * np.log10(np.maximum(np.asarray(frame["amp"], float), 1e-30))))
                    if "amp" in frame else 0.0)
    score = (metrics["frac_outside_envelope_soft"], metrics["mean_severity_outside_soft"],
             metrics["frac_outside_envelope_strict"], expected_mae, curvature, negative_snr)
    return score, metrics


def classify_metrics(metrics, strict_outside_hard=0.10, soft_outside_hard=0.15,
                     severity_soft_hard=0.08, strict_outside_edge=0.01,
                     soft_outside_edge=0.03, severity_soft_edge=0.05,
                     ned_min_edge=0.08, min_points_for_multi=4,
                     use_orange=False, min_orange_points=3):
    """Classify the selected path using the study's exact envelope gates."""
    count = metrics.get("n_points", len(metrics.get("ned_values", [])))
    if count < min_points_for_multi:
        return "single", {"n": count}
    outside = metrics["frac_outside_envelope_strict"]
    outside_soft = metrics["frac_outside_envelope_soft"]
    severity = metrics["mean_severity_outside_soft"]
    position = metrics["ned_mean"]
    details = {"fos": outside, "fos_soft": outside_soft, "sev_soft": severity, "ned_mean": position}
    if outside > strict_outside_hard or outside_soft > soft_outside_hard or severity > severity_soft_hard:
        return "invalid", dict(details, reason="hard envelope gate")
    if (0 < outside <= strict_outside_edge or 0 < outside_soft <= soft_outside_edge
            or 0 < severity <= severity_soft_edge or position < ned_min_edge):
        return "edge", dict(details, reason="minor envelope issues or hugging blue")
    orange = (use_orange and metrics.get("orange_used", False)
              and metrics.get("orange_n", 0) >= min_orange_points)
    if orange:
        return "valid_gray", {"reason": "envelope ok; orange consulted",
                              "exp_norm_bias": metrics["exp_norm_bias"],
                              "frac_within_expected_band": metrics["frac_within_expected_band"]}
    return "valid_gray", {"reason": "envelope ok; orange not used"}


def select_candidates(frame):
    """Choose one longest path per satellite/species/date/telescope/UTC start.

    All 24 candidate fields are preserved. The added event ID records the group
    identity; sequence length counts selected rows, including nonfinite envelope
    rows retained by the original finite-subset classification rule.
    """
    candidates = _typed(frame.copy())
    if not candidates["species"].isin(("fpdu", "fedu")).all():
        raise ValueError("Candidates must contain fedu and/or fpdu species")
    if not candidates["satellite"].eq("goes16").all():
        raise ValueError("Selection currently supports goes16 candidates only")
    retained = candidates.loc[candidates["energy_channel"] < ENERGY_THRESHOLD_KEV]
    stats = {"candidates_input": len(candidates), "candidates_below_energy_threshold": len(retained),
             "groups_total": 0, "groups_without_sequence": 0, "sequences_considered": 0,
             "events_selected": 0, "rows_selected": 0, "categories": {name: 0 for name in CATEGORIES}}
    selected = []
    # Preserve the original species order, encountered dates/UTC starts, and
    # sorted telescope numbers rather than changing tie-sensitive row order.
    for species in ("fpdu", "fedu"):
        particle = retained.loc[retained["species"] == species]
        for date in particle["date"].unique():
            by_date = particle.loc[particle["date"] == date].reset_index(drop=True)
            for telescope in sorted(by_date["telescope_no"].unique()):
                by_telescope = by_date.loc[by_date["telescope_no"] == telescope].reset_index(drop=True)
                for start in by_telescope["date1_utc_echo"].unique():
                    group = by_telescope.loc[by_telescope["date1_utc_echo"] == start].reset_index(drop=True)
                    stats["groups_total"] += 1
                    result = longest_sequences(group)
                    if not result:
                        stats["groups_without_sequence"] += 1
                        continue
                    sequences, length = result
                    stats["sequences_considered"] += len(sequences)
                    scored = [(score_sequence(sequence), sequence) for sequence in sequences]
                    (score, metrics), best = min(scored, key=lambda item: item[0][0])
                    category, _ = classify_metrics(metrics)
                    category = "valid" if category == "valid_gray" else category
                    best = best.reset_index(drop=True)
                    best["category"] = category
                    best["event_id"] = (f"goes16:{species}:{pd.Timestamp(date):%Y%m%d}:t{int(telescope)}:"
                                        f"{pd.Timestamp(start).isoformat()}")
                    best["sequence_length"] = length
                    selected.append(best)
                    stats["events_selected"] += 1
                    stats["categories"][category] += 1
    if selected:
        result = pd.concat(selected).reset_index(drop=True)
    else:
        result = candidates.iloc[:0].copy()
        result["category"] = pd.Series(dtype=str)
        result["event_id"] = pd.Series(dtype=str)
        result["sequence_length"] = pd.Series(dtype="int64")
    stats["rows_selected"] = len(result)
    return result.loc[:, SELECTED_COLUMNS], stats
