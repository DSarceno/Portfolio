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

# --- Mixture priors (A.4): split each confederation into an "elite" tier and a
# "regular" tier so shrinkage stops collapsing genuine powers toward the same
# confederation mean. The elite prior sits clearly above the single-tier value,
# the regular prior at or below it, widening the spread between strong and weak
# teams within the same confederation (Baio & Blangiardo 2010).
CONFEDERATION_ELO_PRIOR_ELITE: dict[str, float] = {
    "UEFA": 1740.0,
    "CONMEBOL": 1740.0,
    "CONCACAF": 1520.0,
    "AFC": 1500.0,
    "CAF": 1520.0,
    "OFC": 1300.0,
    "UNKNOWN": 1520.0,
}

CONFEDERATION_ELO_PRIOR_REGULAR: dict[str, float] = {
    "UEFA": 1560.0,
    "CONMEBOL": 1540.0,
    "CONCACAF": 1400.0,
    "AFC": 1370.0,
    "CAF": 1390.0,
    "OFC": 1170.0,
    "UNKNOWN": 1430.0,
}

CONFEDERATION_POISSON_ATTACK_PRIOR_ELITE: dict[str, float] = {
    "UEFA": 1.20,
    "CONMEBOL": 1.20,
    "CONCACAF": 0.95,
    "AFC": 0.92,
    "CAF": 0.95,
    "OFC": 0.75,
    "UNKNOWN": 0.95,
}

CONFEDERATION_POISSON_ATTACK_PRIOR_REGULAR: dict[str, float] = {
    "UEFA": 0.98,
    "CONMEBOL": 0.98,
    "CONCACAF": 0.82,
    "AFC": 0.78,
    "CAF": 0.82,
    "OFC": 0.62,
    "UNKNOWN": 0.82,
}

CONFEDERATION_POISSON_DEFENSE_PRIOR_ELITE: dict[str, float] = {
    "UEFA": 0.82,
    "CONMEBOL": 0.82,
    "CONCACAF": 1.05,
    "AFC": 1.05,
    "CAF": 1.05,
    "OFC": 1.20,
    "UNKNOWN": 1.05,
}

CONFEDERATION_POISSON_DEFENSE_PRIOR_REGULAR: dict[str, float] = {
    "UEFA": 1.00,
    "CONMEBOL": 1.00,
    "CONCACAF": 1.25,
    "AFC": 1.30,
    "CAF": 1.25,
    "OFC": 1.45,
    "UNKNOWN": 1.25,
}

TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA
    "Albania": "UEFA",
    "Andorra": "UEFA",
    "Armenia": "UEFA",
    "Austria": "UEFA",
    "Azerbaijan": "UEFA",
    "Belarus": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Bulgaria": "UEFA",
    "Croatia": "UEFA",
    "Cyprus": "UEFA",
    "Czech Republic": "UEFA",
    "Czechia": "UEFA",
    "Denmark": "UEFA",
    "England": "UEFA",
    "Estonia": "UEFA",
    "Faroe Islands": "UEFA",
    "Finland": "UEFA",
    "France": "UEFA",
    "Georgia": "UEFA",
    "Germany": "UEFA",
    "Gibraltar": "UEFA",
    "Greece": "UEFA",
    "Hungary": "UEFA",
    "Iceland": "UEFA",
    "Ireland": "UEFA",
    "Israel": "UEFA",
    "Italy": "UEFA",
    "Kazakhstan": "UEFA",
    "Kosovo": "UEFA",
    "Latvia": "UEFA",
    "Liechtenstein": "UEFA",
    "Lithuania": "UEFA",
    "Luxembourg": "UEFA",
    "Malta": "UEFA",
    "Moldova": "UEFA",
    "Montenegro": "UEFA",
    "Netherlands": "UEFA",
    "North Macedonia": "UEFA",
    "Northern Ireland": "UEFA",
    "Norway": "UEFA",
    "Poland": "UEFA",
    "Portugal": "UEFA",
    "Republic of Ireland": "UEFA",
    "Romania": "UEFA",
    "Russia": "UEFA",
    "San Marino": "UEFA",
    "Scotland": "UEFA",
    "Serbia": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkey": "UEFA",
    "Türkiye": "UEFA",
    "Ukraine": "UEFA",
    "Wales": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "Anguilla": "CONCACAF",
    "Antigua and Barbuda": "CONCACAF",
    "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF",
    "Barbados": "CONCACAF",
    "Belize": "CONCACAF",
    "Bermuda": "CONCACAF",
    "Bonaire": "CONCACAF",
    "British Virgin Islands": "CONCACAF",
    "Canada": "CONCACAF",
    "Cayman Islands": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF",
    "Curacao": "CONCACAF",
    "Curaçao": "CONCACAF",
    "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF",
    "French Guiana": "CONCACAF",
    "Grenada": "CONCACAF",
    "Guadeloupe": "CONCACAF",
    "Guatemala": "CONCACAF",
    "Guyana": "CONCACAF",
    "Haiti": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Martinique": "CONCACAF",
    "Mexico": "CONCACAF",
    "Montserrat": "CONCACAF",
    "Nicaragua": "CONCACAF",
    "Panama": "CONCACAF",
    "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF",
    "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Sint Maarten": "CONCACAF",
    "Suriname": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "U.S. Virgin Islands": "CONCACAF",
    "United States": "CONCACAF",
    "USA": "CONCACAF",
    # AFC
    "Afghanistan": "AFC",
    "Australia": "AFC",
    "Bahrain": "AFC",
    "Bangladesh": "AFC",
    "Bhutan": "AFC",
    "Brunei": "AFC",
    "Cambodia": "AFC",
    "China": "AFC",
    "China PR": "AFC",
    "Chinese Taipei": "AFC",
    "Guam": "AFC",
    "Hong Kong": "AFC",
    "India": "AFC",
    "Indonesia": "AFC",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Korea DPR": "AFC",
    "Korea Republic": "AFC",
    "Kuwait": "AFC",
    "Kyrgyz Republic": "AFC",
    "Kyrgyzstan": "AFC",
    "Laos": "AFC",
    "Lebanon": "AFC",
    "Macau": "AFC",
    "Malaysia": "AFC",
    "Maldives": "AFC",
    "Mongolia": "AFC",
    "Myanmar": "AFC",
    "Nepal": "AFC",
    "North Korea": "AFC",
    "Oman": "AFC",
    "Pakistan": "AFC",
    "Palestine": "AFC",
    "Philippines": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Singapore": "AFC",
    "South Korea": "AFC",
    "Sri Lanka": "AFC",
    "Syria": "AFC",
    "Tajikistan": "AFC",
    "Thailand": "AFC",
    "Timor-Leste": "AFC",
    "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC",
    "Uzbekistan": "AFC",
    "Vietnam": "AFC",
    "Yemen": "AFC",
    # CAF
    "Algeria": "CAF",
    "Angola": "CAF",
    "Benin": "CAF",
    "Botswana": "CAF",
    "Burkina Faso": "CAF",
    "Burundi": "CAF",
    "Cameroon": "CAF",
    "Cape Verde": "CAF",
    "Central African Republic": "CAF",
    "Chad": "CAF",
    "Comoros": "CAF",
    "Congo": "CAF",
    "Congo DR": "CAF",
    "Côte d'Ivoire": "CAF",
    "DR Congo": "CAF",
    "Democratic Republic of the Congo": "CAF",
    "Djibouti": "CAF",
    "Egypt": "CAF",
    "Equatorial Guinea": "CAF",
    "Eritrea": "CAF",
    "Eswatini": "CAF",
    "Ethiopia": "CAF",
    "Gabon": "CAF",
    "Gambia": "CAF",
    "Ghana": "CAF",
    "Guinea": "CAF",
    "Guinea-Bissau": "CAF",
    "Ivory Coast": "CAF",
    "Kenya": "CAF",
    "Lesotho": "CAF",
    "Liberia": "CAF",
    "Libya": "CAF",
    "Madagascar": "CAF",
    "Malawi": "CAF",
    "Mali": "CAF",
    "Mauritania": "CAF",
    "Mauritius": "CAF",
    "Morocco": "CAF",
    "Mozambique": "CAF",
    "Namibia": "CAF",
    "Niger": "CAF",
    "Nigeria": "CAF",
    "Rwanda": "CAF",
    "Sao Tome and Principe": "CAF",
    "São Tomé and Príncipe": "CAF",
    "Senegal": "CAF",
    "Seychelles": "CAF",
    "Sierra Leone": "CAF",
    "Somalia": "CAF",
    "South Africa": "CAF",
    "South Sudan": "CAF",
    "Sudan": "CAF",
    "Swaziland": "CAF",
    "Tanzania": "CAF",
    "Togo": "CAF",
    "Tunisia": "CAF",
    "Uganda": "CAF",
    "Zambia": "CAF",
    "Zimbabwe": "CAF",
    # OFC
    "American Samoa": "OFC",
    "Cook Islands": "OFC",
    "Fiji": "OFC",
    "New Caledonia": "OFC",
    "New Zealand": "OFC",
    "Papua New Guinea": "OFC",
    "Samoa": "OFC",
    "Solomon Islands": "OFC",
    "Tahiti": "OFC",
    "Tonga": "OFC",
    "Vanuatu": "OFC",
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
    mixture_prior: bool = True
    elite_fraction: float = 0.30
    min_matches_for_elite: int = 20
    confederation_elo: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_ELO_PRIOR)
    )
    confederation_poisson_attack: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_ATTACK_PRIOR)
    )
    confederation_poisson_defense: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_DEFENSE_PRIOR)
    )
    confederation_elo_elite: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_ELO_PRIOR_ELITE)
    )
    confederation_elo_regular: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_ELO_PRIOR_REGULAR)
    )
    confederation_attack_elite: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_ATTACK_PRIOR_ELITE)
    )
    confederation_attack_regular: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_ATTACK_PRIOR_REGULAR)
    )
    confederation_defense_elite: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_DEFENSE_PRIOR_ELITE)
    )
    confederation_defense_regular: dict[str, float] = field(
        default_factory=lambda: dict(CONFEDERATION_POISSON_DEFENSE_PRIOR_REGULAR)
    )
    team_confederation: dict[str, str] = field(default_factory=lambda: dict(TEAM_CONFEDERATION))

    def get_confederation(self, team: str) -> str:
        """Resolve a team's confederation, falling back to ``"UNKNOWN"``."""
        return self.team_confederation.get(str(team), "UNKNOWN")

    @staticmethod
    def _lookup(table: dict[str, float], conf: str, fallback: float) -> float:
        return table.get(conf, table.get("UNKNOWN", fallback))

    def get_elo_prior(self, team: str, is_elite: bool = False) -> float:
        """Confederation-derived Elo prior for *team*.

        When ``mixture_prior`` is enabled, routes to the elite or regular tier
        depending on *is_elite*; otherwise returns the single-tier prior.
        """
        conf = self.get_confederation(team)
        if not self.mixture_prior:
            return self._lookup(self.confederation_elo, conf, 1450.0)
        table = self.confederation_elo_elite if is_elite else self.confederation_elo_regular
        return self._lookup(table, conf, 1450.0)

    def get_attack_prior(self, team: str, is_elite: bool = False) -> float:
        """Confederation-derived Poisson attack prior for *team*."""
        conf = self.get_confederation(team)
        if not self.mixture_prior:
            return self._lookup(self.confederation_poisson_attack, conf, 0.85)
        table = self.confederation_attack_elite if is_elite else self.confederation_attack_regular
        return self._lookup(table, conf, 0.85)

    def get_defense_prior(self, team: str, is_elite: bool = False) -> float:
        """Confederation-derived Poisson defense prior for *team*."""
        conf = self.get_confederation(team)
        if not self.mixture_prior:
            return self._lookup(self.confederation_poisson_defense, conf, 1.20)
        table = self.confederation_defense_elite if is_elite else self.confederation_defense_regular
        return self._lookup(table, conf, 1.20)

    def classify_elite(
        self,
        strength: dict[str, float],
        match_counts: dict[str, int],
    ) -> dict[str, bool]:
        """Flag teams as elite within their confederation by a strength signal.

        A team is elite only if it (a) has at least ``min_matches_for_elite``
        recorded matches and (b) ranks in the top ``elite_fraction`` of its
        confederation among other established teams. The match-count gate is
        the safety valve: a minnow with a few inflated results (e.g. high Poisson
        attack from beating weak regional rivals) can never reach the elite tier.

        Args:
            strength: Mapping ``team -> strength`` (Elo for Elo shrinkage, net
                attack-defense for Poisson shrinkage). Higher means stronger.
            match_counts: Mapping ``team -> n_matches``.

        Returns:
            Mapping ``team -> is_elite``. Teams absent or below the gate are
            implicitly regular (not present / ``False``).
        """
        eligible_by_conf: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for team, value in strength.items():
            if match_counts.get(team, 0) >= self.min_matches_for_elite:
                eligible_by_conf[self.get_confederation(team)].append((team, float(value)))

        elite: dict[str, bool] = {}
        for items in eligible_by_conf.values():
            # Need a few established teams to define a meaningful percentile;
            # otherwise treat the whole (tiny) group as regular.
            if len(items) < 4:
                for team, _ in items:
                    elite[team] = False
                continue
            values = sorted(v for _, v in items)
            idx = int(round((1.0 - self.elite_fraction) * (len(values) - 1)))
            threshold = values[idx]
            for team, value in items:
                elite[team] = value >= threshold
        return elite

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
        elite = self.classify_elite(dict(elo.ratings), match_counts) if self.mixture_prior else {}
        n_total = 0
        n_shrunk = 0
        for team in list(elo.ratings.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_elo)
            prior = self.get_elo_prior(team, is_elite=elite.get(team, False))
            old = elo.ratings[team]
            new = alpha * old + (1.0 - alpha) * prior
            elo.ratings[team] = new
            n_total += 1
            if abs(new - old) > 1.0:
                n_shrunk += 1
        logger.info(
            "Elo shrinkage applied: %d teams, %d meaningfully shrunk (K=%s, "
            "mixture=%s, elite=%d)",
            n_total,
            n_shrunk,
            self.k_elo,
            self.mixture_prior,
            sum(1 for v in elite.values() if v),
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
        # Elite membership uses net strength (attack - defense), which is robust
        # to the weak-opposition inflation that raw attack alone suffers from,
        # combined with the match-count gate inside classify_elite.
        if self.mixture_prior:
            net_strength = {
                team: float(fit.attack.get(team, 0.0)) - float(fit.defense.get(team, 0.0))
                for team in fit.attack
            }
            elite = self.classify_elite(net_strength, match_counts)
        else:
            elite = {}
        for team in list(fit.attack.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_poisson)
            prior = self.get_attack_prior(team, is_elite=elite.get(team, False))
            fit.attack[team] = alpha * fit.attack[team] + (1.0 - alpha) * prior
        for team in list(fit.defense.keys()):
            n = match_counts.get(team, 0)
            alpha = self._alpha(n, self.k_poisson)
            prior = self.get_defense_prior(team, is_elite=elite.get(team, False))
            fit.defense[team] = alpha * fit.defense[team] + (1.0 - alpha) * prior
        logger.info(
            "Poisson shrinkage applied (K=%s, mixture=%s, elite=%d)",
            self.k_poisson,
            self.mixture_prior,
            sum(1 for v in elite.values() if v),
        )
        return poisson
