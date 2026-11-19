# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sandivisi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sandivisi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for how many GTA VI players earn each trophy in week one, built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty. It reads the house's recording of Exophase's GTA VI PS5 trophy page up to a checkpoint and forecasts every trophy's earn rate at the target as five quantiles (5%, 25%, 50%, 75%, 95%).

## How it works

Exophase's page is a staircase: it sits frozen for hours, then jumps at a refresh. The model is built around that.

- **States, not reads.** Consecutive identical reads are collapsed into one state, so a long flat stretch counts once instead of many times.
- **Trend when there is history.** With three or more distinct states, each trophy's rate is tracked on the logit scale against its own ceiling — `logit(rate / ceiling)` against `ln(1 + hours since release)` — with a recency-weighted least-squares line (24-hour half-life). The slope is shrunk toward its tier's mean slope, and the forecast is capped just below the ceiling, so growth saturates instead of running away. Ceilings come from the trophy's tier and description: story chapters saturate high (40–55%), the final mission around 30%, grind and collectible trophies low (6–20%), the platinum and 100%-completion trophy around 2%.
- **Refresh prior when the page is frozen.** With fewer than three states, the forecast expects the next refresh: Exophase refreshes roughly every 270 minutes and each refresh multiplies tracked players by about 6. Newcomers have barely started playing, so they earn each trophy at only a small fraction (β ≈ 0.0665) of the incumbents' rate — a figure calibrated from the one refresh the spec records (638 players / 943 trophies earned frozen, then 3,868 players / 4,276 earned). At most two refreshes are compounded, the second with β halved, so long horizons cannot collapse every rate to zero. A trophy already at 100% stays at 100%.
- **Honest spreads.** Quantile widths grow with the horizon (random walk), with fit uncertainty, and for very small rates; in the frozen case the upper side is wider, because the refresh may arrive late or not at all. A trophy with no usable history falls back to its tier's median rate. The program never crashes on a malformed case.

## Run it

```sh
MAKEFUN_INPUT_DIR=<case-dir> MAKEFUN_OUTPUT_DIR=<out-dir> python3 main.py
```

It reads every `<id>.in` case in the input directory and writes `<id>.out` (JSON: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory. Standard library only, deterministic, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=gnkp9jxa&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
