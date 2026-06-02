"""High-level prediction interfaces consumed by scripts and the API."""

from src.prediction.daily_update import DailyUpdater
from src.prediction.predictor import MatchPredictor
from src.prediction.quiniela_strategy import QuinielaStrategy
from src.prediction.score_predictor import ScorePredictor

__all__ = [
    "MatchPredictor",
    "ScorePredictor",
    "QuinielaStrategy",
    "DailyUpdater",
]
