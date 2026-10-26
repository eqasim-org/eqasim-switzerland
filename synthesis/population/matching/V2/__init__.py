"""V2 whole-diary matching.

Public API: DiaryMatcher, MatchingOptions and MatchingPrediction. Importing
feature/reporting helpers does not load Torch; estimator imports are lazy.
"""

__all__ = ["DiaryMatcher", "MatchingOptions", "MatchingPrediction"]


def __getattr__(name):
    if name == "MatchingOptions":
        from .configuration import MatchingOptions
        return MatchingOptions
    if name in ("DiaryMatcher", "MatchingPrediction"):
        from .wrappers import DiaryMatcher, MatchingPrediction
        return {"DiaryMatcher": DiaryMatcher, "MatchingPrediction": MatchingPrediction}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
