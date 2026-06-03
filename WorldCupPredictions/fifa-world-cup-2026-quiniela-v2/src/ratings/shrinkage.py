"""Bayesian shrinkage of rating systems toward confederation priors.

Teams with few international matches against strong opposition have unreliable
ratings: Elo starts at 1500 and barely moves; Poisson attack/defense looks
strong because they faced weak regional rivals. Without correction, the Monte
Carlo simulator inflates their championship probabilities far above reality.

Shrinkage pulls every team's rating toward a **confederation-based prior**
with weight proportional to the inverse of match count:

    effective = (n / (n + K)) * actual + (K / (n + K)) * prior

K controls "how much evidence is needed to outweigh the prior":
- K=10  -> a team trusts its rating after ~10 matches
- K=30  -> default; at n=30, weight is 50/50
- K=100 -> very conservative; needs lots of evidence

See ``CONFEDERATION_ELO_PRIOR`` and ``CONFEDERATION_POISSON_*_PRIOR`` for the
empirical priors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.models.poisson_model import PoissonScoreModel
    from src.ratings.elo import EloRating
    from src.ratings.pi_rating import PIRating

logger = get_logger(__name__)

CONFEDERATION_ELO_PRIOR: dict[str, float] = {
    "UEFA": 1620.0,
    "CONMEBOL": 1620.0,
    "CONCACAF": 1430.0,
    "AFC": 1400.0,
    "CAF": 1420.0,
    "OFC": 1200.0,
    "UNKNOWN": 1450.0,
}

CONFEDERATION_POISSON_ATTACK_PRIOR: dict[str, float] = {
    "UEFA": 1.05,
    "CONMEBOL": 1.05,
    "CONCACAF": 0.85,
    "AFC": 0.80,
    "CAF": 0.85,
    "OFC": 0.65,
    "UNKNOWN": 0.85,
}

CONFEDERATION_POISSON_DEFENSE_PRIOR: dict[str, float] = {
    "UEFA": 0.95,
    "CONMEBOL": 0.95,
    "CONCACAF": 1.20,
    "AFC": 1.25,
    "CAF": 1.20,
    "OFC": 1.40,
    "UNKNOWN": 1.20,
}

TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA", "Croatia": "UEFA",
    "Cyprus": "UEFA", "Czech Republic": "UEFA", "Czechia": "UEFA",
    "Denmark": "UEFA", "England": "UEFA", "Estonia": "UEFA",
    "Faroe Islands": "UEFA", "Finland": "UEFA", "France": "UEFA",
    "Georgia": "UEFA", "Germany": "UEFA", "Gibraltar": "UEFA",
    "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Ireland": "UEFA", "Israel": "UEFA", "Italy": "UEFA",
    "Kazakhstan": "UEFA", "Kosovo": "UEFA", "Latvia": "UEFA",
    "Liechtenstein": "UEFA", "Lithuania": "UEFA", "Luxembourg": "UEFA",
    "Malta": "UEFA", "Moldova": "UEFA", "Montenegro": "UEFA",
    "Netherlands": "UEFA", "North Macedonia": "UEFA",
    "Northern Ireland": "UEFA", "Norway": "UEFA", "Poland": "UEFA",
    "Portugal": "UEFA", "Republic of Ireland": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "San Marino": "UEFA", "Scotland": "UEFA",
    "Serbia": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA",
    "Turkey": "UEFA", "Türkiye": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "Anguilla": "CONCACAF", "Antigua and Barbuda": "CONCACAF",
    "Aruba": "CONCACAF", "Bahamas": "CONCACAF", "Barbados": "CONCACAF",
    "Belize": "CONCACAF", "Bermuda": "CONCACAF", "Bonaire": "CONCACAF",
    "British Virgin Islands": "CONCACAF", "Canada": "CONCACAF",
    "Cayman Islands": "CONCACAF", "Costa Rica": "CONCACAF", "Cuba": "CONCACAF",
    "Curacao": "CONCACAF", "Curaçao": "CONCACAF", "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF", "El Salvador": "CONCACAF",
    "French Guiana": "CONCACAF", "Grenada": "CONCACAF",
    "Guadeloupe": "CONCACAF", "Guatemala": "CONCACAF", "Guyana": "CONCACAF",
    "Haiti": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Martinique": "CONCACAF", "Mexico": "CONCACAF", "Montserrat": "CONCACAF",
    "Nicaragua": "CONCACAF", "Panama": "CONCACAF", "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF", "Sint Maarten": "CONCACAF",
    "Suriname": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF", "U.S. Virgin Islands": "CONCACAF",
    "United States": "CONCACAF", "USA": "CONCACAF",
    # AFC
    "Afghanistan": "AFC", "Australia": "AFC", "Bahrain": "AFC",
    "Bangladesh": "AFC", "Bhutan": "AFC", "Brunei": "AFC", "Cambodia": "AFC",
    "China": "AFC", "China PR": "AFC", "Chinese Taipei": "AFC", "Guam": "AFC",
    "Hong Kong": "AFC", "India": "AFC", "Indonesia": "AFC", "Iran": "AFC",
    "Iraq": "AFC", "Japan": "AFC", "Jordan": "AFC", "Korea DPR": "AFC",
    "Korea Republic": "AFC", "Kuwait": "AFC", "Kyrgyz Republic": "AFC",
    "Kyrgyzstan": "AFC", "Laos": "AFC", "Lebanon": "AFC", "Macau": "AFC",
    "Malaysia": "AFC", "Maldives": "AFC", "Mongolia": "AFC", "Myanmar": "AFC",
    "Nepal": "AFC", "North Korea": "AFC", "Oman": "AFC", "Pakistan": "AFC",
    "Palestine": "AFC", "Philippines": "AFC", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "Singapore": "AFC", "South Korea": "AFC",
    "Sri Lanka": "AFC", "Syria": "AFC", "Tajikistan": "AFC",
    "Thailand": "AFC", "Timor-Leste": "AFC", "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC", "Vietnam": "AFC",
    "Yemen": "AFC",
    # CAF
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Botswana": "CAF",
    "Burkina Faso": "CAF", "Burundi": "CAF", "Cameroon": "CAF",
    "Cape Verde": "CAF", "Central African Republic": "CAF", "Chad": "CAF",
    "Comoros": "CAF", "Congo": "CAF", "Congo DR": "CAF",
    "Côte d'Ivoire": "CAF", "DR Congo": "CAF",
    "Democratic Republic of the Congo": "CAF", "Djibouti": "CAF",
    "Egypt": "CAF", "Equatorial Guinea": "CAF", "Eritrea": "CAF",
    "Eswatini": "CAF", "Ethiopia": "CAF", "Gabon": "CAF", "Gambia": "CAF",
    "Ghana": "CAF", "Guinea": "CAF", "Guinea-Bissau": "CAF",
    "Ivory Coast": "CAF", "Kenya": "CAF", "Lesotho": "CAF",
    "Liberia": "CAF", "Libya": "CAF", "Madagascar": "CAF", "Malawi": "CAF",
    "Mali": "CAF", "Mauritania": "CAF", "Mauritius": "CAF", "Morocco": "CAF",
    "Mozambique": "CAF", "Namibia": "CAF", "Niger": "CAF", "Nigeria": "CAF",
    "Rwanda": "CAF", "Sao Tome and Principe": "CAF",
    "São Tomé and Príncipe": "CAF", "Senegal": "CAF", "Seychelles": "CAF",
    "Sierra Leone": "CAF", "Somalia": "CAF", "South Africa": "CAF",
    "South Sudan": "CAF", "Sudan": "CAF", "Swaziland": "CAF",
    "Tanzania": "CAF", "Togo": "CAF", "Tunisia": "CAF", "Uganda": "CAF",
    "Zambia": "CAF", "Zimbabwe": "CAF",
    # OFC
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "New Caledonia": "OFC", "New Zealand": "OFC", "Papua New Guinea": "OFC",
    "Samoa": "OFC", "Solomon Islands": "OFC", "Tahiti": "OFC",
    "Tonga": "OFC", "Vanuatu": "OFC",
}


def count_matches_per_team(matches: pd.DataFrame) -> dict[str, int]:
    """Count how many matches each team has played in *matches*.

    Args:
        matches: Canonical match table.

    Returns:
        Mapping ``team -> count``.
    """
    counts: dict[str, int] = defaultdict(int)
    if matches.empty:
        return dict(counts)
    for col in ("team_a", "team_b"):
        for team, n in matches[col].dropna().astype(str).value_counts().items():
            counts[team] += int(n)
    return dict(counts)


@dataclass
class RatingShrinker:
    """Apply Bayesian shrinkage to rating systems toward confederation priors.

    Defaults are tuned empirically: K=30 means a team needs ~30 matches to
    weight its observed rating equally with the prior.
    """

    k_elo: float = 30.0
    k_pi: float = 30.0
    k_poisson: float = 20.0
    confederation_elo: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_ELO_PRIOR)
    )
    confederation_poisson_attack: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_ATTACK_PRIOR)
    )
    confederation_poisson_defense: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_DEFENSE_PRIOR)
    )
    team_confederation: dict[str, str] = field(
        default_factory=lambda: dict(TEAM_CONFEDERATION)
    )

    def get_confederation(self, team: str) -> str:
        """Resolve a team's confederation, falling back to ``"UNKNOWN"``."""
        return self.team_confederation.get(str(team), "UNKNOWN")

    def get_elo_prior(self, team: str) -> float:
        """Confederation-derived Elo prior for *team*."""
        conf = self.get_confederation(team)
        return self.confederation_elo.get(
            conf, self.confederation_elo.get("UNKNOWN", 1450.0)
        )

    def get_attack_prior(self, team: str) -> float:
        """Confederation-derived Poisson attack prior for *team*."""
        conf = self.get_confederation(team)
        return self.confederation_poisson_attack.get(
            conf, self.confederation_poisson_attack.get("UNKNOWN", 0.85)
        )

    def get_defense_prior(self, team: str) -> float:
        """Confederation-derived Poisson defense prior for *team*."""
        conf = self.get_confederation(team)
        return self.confederation_poisson_defense.get(
            conf, self.confederation_poisson_defense.get("UNKNOWN", 1.20)
        )

    def _alpha(self, n_matches: int, k: float) -> float:
        return float(n_matches) / (float(n_matches) + float(k))

    def shrink_elo(self, elo: "EloRating", match_counts: dict[str, int]) -> "EloRating":
        """Apply shrinkage to an :class:`EloRating` in place.

        Args:
            elo: Fitted Elo rating system.
            match_counts: Mapping ``team -> n_matches``.

        Returns:
            The (mutated) ``elo``.
        """
        n_total = 0
        n_shrunk = 0
        for team in list(elo.ratings.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_elo)
            prior = self.get_elo_prior(team)
            old = elo.ratings[team]
            new = alpha * old + (1.0 - alpha) * prior
            elo.ratings[team] = new
            n_total += 1
            if abs(new - old) > 1.0:
                n_shrunk += 1
        logger.info(
            "Elo shrinkage applied: %d teams, %d meaningfully shrunk (K=%s)",
            n_total,
            n_shrunk,
            self.k_elo,
        )
        return elo

    def shrink_pi(self, pi: "PIRating", match_counts: dict[str, int]) -> "PIRating":
        """Apply shrinkage to a :class:`PIRating` in place. Prior is 0.0."""
        for team in list(pi.home_rating.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_pi)
            pi.home_rating[team] = alpha * pi.home_rating[team]
        for team in list(pi.away_rating.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_pi)
            pi.away_rating[team] = alpha * pi.away_rating[team]
        logger.info("PI shrinkage applied (K=%s)", self.k_pi)
        return pi

    def shrink_poisson(
        self,
        poisson: "PoissonScoreModel",
        match_counts: dict[str, int],
    ) -> "PoissonScoreModel":
        """Apply shrinkage to a fitted :class:`PoissonScoreModel` in place.

        Pulls attack/defense toward confederation-specific priors. Teams with
        few matches converge to their confederation average.

        Args:
            poisson: Fitted Poisson scoreline model.
            match_counts: Mapping ``team -> n_matches``.

        Returns:
            The (mutated) ``poisson``.
        """
        fit = poisson.fit_result
        if fit is None:
            logger.warning("Poisson model is not fitted; skipping shrinkage")
            return poisson
        for team in list(fit.attack.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_poisson)
            prior = self.get_attack_prior(team)
            fit.attack[team] = alpha * fit.attack[team] + (1.0 - alpha) * prior
        for team in list(fit.defense.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_poisson)
            prior = self.get_defense_prior(team)
            fit.defense[team] = alpha * fit.defense[team] + (1.0 - alpha) * prior
        logger.info("Poisson shrinkage applied (K=%s)", self.k_poisson)
        return poisson
