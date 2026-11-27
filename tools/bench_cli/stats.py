"""Descriptive statistics for measured samples and matched case medians."""

import math
import random
from collections import defaultdict
from statistics import fmean, median


def percentile(values, p):
    """Linearly interpolate the p-th percentile (0..100)."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lo = math.floor(index)
    return ordered[lo] + (ordered[math.ceil(index)] - ordered[lo]) * (index - lo)


def geometric_mean(values):
    return math.exp(fmean(math.log(value) for value in values))


def paired_bootstrap_ci(ratios, seed=42, resamples=2000):
    if not ratios:
        return dict(geomean=None, ci95_low=None, ci95_high=None, n=0)
    rng = random.Random(seed)
    logs = [math.log(value) for value in ratios]
    samples = [math.exp(fmean(rng.choices(logs, k=len(logs)))) for _ in range(resamples)]
    return dict(
        geomean=geometric_mean(ratios),
        ci95_low=percentile(samples, 2.5),
        ci95_high=percentile(samples, 97.5),
        n=len(ratios),
    )


def summarize(rows):
    columns = defaultdict(list)
    histograms = []
    for row in rows:
        if "spec_accept_hist" in row:
            histograms.append(row["spec_accept_hist"])
        for key, value in row.items():
            if type(value) in (int, float):
                columns[key].append(value)
    summary = {
        key: {"median": median(values), "mean": fmean(values), "n": len(values)}
        for key, values in columns.items()
    }
    if histograms:
        summary["spec_accept_hist"] = [sum(counts) for counts in zip(*histograms, strict=True)]
    return summary
