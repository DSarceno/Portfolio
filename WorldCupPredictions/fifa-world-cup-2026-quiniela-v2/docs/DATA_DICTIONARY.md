# Data Dictionary

## Canonical match table (`data/interim/matches_unified.csv`)

| Column          | Type     | Source     | Description                          | Nullable |
| --------------- | -------- | ---------- | ------------------------------------ | -------- |
| `match_id`      | int/str  | source     | Source-specific identifier           | yes      |
| `date`          | date     | source     | UTC kick-off date                    | no       |
| `competition`   | str      | source     | Competition code (WC, EURO, ...)     | no       |
| `season`        | str      | source     | Season label                         | yes      |
| `stage`         | str      | source     | Stage description                    | yes      |
| `team_a`        | str      | source     | First / home team                    | no       |
| `team_b`        | str      | source     | Second / away team                   | no       |
| `score_a`       | int      | source     | Final goals for team_a               | yes      |
| `score_b`       | int      | source     | Final goals for team_b               | yes      |
| `neutral_venue` | bool     | source     | Neutral venue indicator              | yes      |
| `host_country`  | str      | source     | Host country / venue                 | yes      |
| `source`        | str      | pipeline   | Source identifier                    | no       |

## FIFA ranking snapshot (`data/raw/fifa_rankings/*.csv`)

| Column           | Type   | Description                |
| ---------------- | ------ | -------------------------- |
| `snapshot_date`  | date   | Snapshot publication date  |
| `team`           | str    | Team name                  |
| `rank`           | int    | Official position          |
| `points`         | float  | Ranking points             |
| `confederation`  | str    | UEFA, CONMEBOL, ...        |

## Engineered feature matrix (`data/processed/feature_matrix.csv`)

### Team-strength features

| Column                          | Type  | Description                                |
| ------------------------------- | ----- | ------------------------------------------ |
| `fifa_rank_points_a/b`          | float | FIFA points per team                       |
| `fifa_rank_position_a/b`        | float | FIFA rank position per team                |
| `elo_pre_match_a/b`             | float | Elo rating before match                    |
| `pi_rating_pre_match_a/b`       | float | PI rating before match                     |
| `rolling_form_points_last_5_a/b`| float | Rolling form score                         |
| `attack_strength_a/b`           | float | Avg. goals scored                          |
| `defense_strength_a/b`          | float | Defence inverse goal rate                  |
| `clean_sheet_rate_a/b`          | float | Clean-sheet rate                           |
| `draw_rate_a/b`                 | float | Draw rate                                  |
| `tournament_experience_a/b`     | float | Tournament-match count                     |
| `world_cup_experience_a/b`      | float | World Cup match count                      |
| `host_flag_a/b`                 | int   | 1 when team hosts the tournament           |
| `confederation_strength_index_a/b` | float | Confederation prior                     |

### Match-context features

| Column                  | Type  | Description                                  |
| ----------------------- | ----- | -------------------------------------------- |
| `stage_group`           | int   | 1 for group stage                            |
| `stage_knockout`        | int   | 1 for knockout stage                         |
| `round_of_tournament`   | int   | Round index                                  |
| `rest_days_team_a/b`    | float | Days since previous match                    |
| `rest_diff`             | float | Rest-day differential                        |
| `travel_burden_proxy_team_a/b` | float | Travel proxy `[0, 1]`                  |
| `fatigue_index_a/b`     | float | Fatigue proxy `[0, 1.5]`                     |
| `host_continent_advantage` | float | Continent advantage delta                 |
| `same_confederation_flag`  | int  | 1 when both teams share confederation     |

### Aggressive-strategy features

| Column                       | Type  | Description                              |
| ---------------------------- | ----- | ---------------------------------------- |
| `upset_window_score`         | float | Upset opportunity `[0, 1]`               |
| `volatility_index`           | float | Match volatility `[0, 1]`                |
| `draw_trap_score`            | float | Draw-trap propensity `[0, 1]`            |
| `public_bias_proxy`          | float | Public bias signed proxy                 |
| `favorite_fragility_score`   | float | Favorite fragility `[0, 1]`              |
| `underdog_live_value_score`  | float | Underdog live value `[0, 1]`             |
| `match_entropy`              | float | Outcome entropy `[0, 1]`                 |
| `confidence_gap`             | float | Confidence gap `[0, 1]`                  |
| `penalty_risk_score`         | float | Penalty risk indicator                   |

### Tournament-state features

| Column                              | Type  | Description                       |
| ----------------------------------- | ----- | --------------------------------- |
| `world_cup_points_so_far_state_a/b` | float | Tournament points accumulated     |
| `world_cup_goal_diff_so_far_state_a/b` | float | Tournament goal difference     |
| `world_cup_goals_for_so_far_state_a/b` | float | Tournament goals scored        |
| `world_cup_goals_against_so_far_state_a/b` | float | Tournament goals conceded |
| `world_cup_clean_sheets_state_a/b`  | float | Tournament clean sheets           |

### Diff features

| Column            | Type  | Description                          |
| ----------------- | ----- | ------------------------------------ |
| `elo_diff`        | float | `elo_a - elo_b`                      |
| `pi_diff`         | float | PI rating differential               |
| `form_diff`       | float | Rolling form differential            |
| `fifa_points_diff`| float | FIFA points differential             |
| `attack_diff`     | float | Attack vs. opponent defense          |
| `defense_diff`    | float | Defense vs. opponent attack          |
| `wc_points_diff`  | float | Tournament points differential       |
| `wc_goal_diff_diff`| float | Tournament goal-diff differential   |

### Labels

| Column    | Values         | Description           |
| --------- | -------------- | --------------------- |
| `outcome` | `H`, `D`, `A`  | Match result label    |
