"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(metrics, aligned_segments, align_report):
    base = clip_evaluation_report(metrics, aligned_segments)

    def clamp(value):
        if value > 1:
            return 1
        elif value < 0:
            return 0
        else:
            return value

    mades = base.get("mean_abs_duration_error_s", 0.0)
    pss = base.get("pct_severe_stretch", 0.0)
    tcds = base.get("total_cumulative_drift_s", 0.0)
    ntr = base.get("n_translation_retries", 0)
    segments = align_report.get("segments", [])

    timing_error_score = clamp(1.0 - (mades / 2))
    severe_stretch_score = clamp(1.0 - (pss / 100))
    drift_score = clamp(1.0 - (abs(tcds) / 10))
    timing_accuracy = (timing_error_score + severe_stretch_score + drift_score) / 3

    speed_factors = []
    for segment in segments:
        if "speed_factor" in segment:
            speed_factors.append(segment["speed_factor"])

    if len(speed_factors) > 0:
        speed_error = 0
        for speed_factor in speed_factors:
            speed_error += abs(speed_factor - 1.0)
        speed_error = speed_error / len(speed_factors)
        naturalness = clamp(1.0 - (speed_error / 0.75))
    else:
        stretch_error = 0
        for segment in aligned_segments:
            stretch_error += abs(segment.stretch_factor - 1.0)
        if len(aligned_segments) > 0:
            stretch_error = stretch_error / len(aligned_segments)
        naturalness = clamp(1.0 - (stretch_error / 0.75))

    fail_count = 0
    severe_count = 0
    for segment in aligned_segments:
        if segment.action == AlignAction.FAIL:
            fail_count += 1
        if segment.stretch_factor > 1.4:
            severe_count += 1

    total_aligned = max(len(aligned_segments), 1)
    fail_ratio = fail_count / total_aligned
    severe_ratio = severe_count / total_aligned

    fail_score = clamp(1.0 - fail_ratio)
    severe_score = clamp(1.0 - severe_ratio)
    intelligibility = (fail_score + severe_score) / 2

    retry_ratio = ntr / max(len(metrics), 1)
    retry_score = clamp(1.0 - retry_ratio)
    fail_semantic_score = clamp(1.0 - fail_ratio)
    semantic_fidelity = (retry_score + fail_semantic_score) / 2

    overall = (
        timing_accuracy
        + naturalness
        + intelligibility
        + semantic_fidelity
    ) / 4

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "naturalness": round(naturalness, 3),
        "intelligibility": round(intelligibility, 3),
        "semantic_fidelity": round(semantic_fidelity, 3),
        "overall": round(overall, 3),
        "details": base,
    }
