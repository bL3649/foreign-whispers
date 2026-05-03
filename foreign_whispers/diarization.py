"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []

def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    # ---- YOUR CODE HERE ----

    segments_copy = []

    for segment in segments:
        segment_copy = {}
        for key, value in segment.items():
            segment_copy.update({key:value})
        segments_copy.append(segment_copy)

    if len(diarization) == 0:
        for segment in segments_copy:
            segment.update({"speaker":"SPEAKER_00"})
    else:
        for segment in segments_copy:
            speaker_candidates = []
            for speaker in diarization:
                overlap = 0
                if speaker["start_s"] >= segment["start"] and speaker["end_s"] <= segment["end"]:
                    overlap = speaker["end_s"] - speaker["start_s"]
                elif speaker["start_s"] >= segment["start"] and speaker["end_s"] > segment["end"]:
                    overlap = segment["end"] - speaker["start_s"]
                elif speaker["start_s"] < segment["start"] and speaker["end_s"] <= segment["end"]:
                    overlap = speaker["end_s"] - segment["start"]
                elif speaker["start_s"] < segment["start"] and speaker["end_s"] > segment["end"]:
                    overlap = segment["end"] - segment["start"]
                speaker_candidates.append({"speaker":speaker["speaker"], "overlap":overlap})
            if sorted(speaker_candidates, key=lambda x: x["overlap"])[len(speaker_candidates) - 1]["overlap"] != 0:
                segment.update({"speaker":sorted(speaker_candidates, key=lambda x: x["overlap"])[len(speaker_candidates) - 1]["speaker"]})
            else:
                segment.update({"speaker":"SPEAKER_00"})

    return segments_copy
    # ---- END YOUR CODE ----
