"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads every <id>.in case in the input directory (the house's reads of Exophase's
Grand Theft Auto VI PS5 trophy page up to a checkpoint) and writes <id>.out as
JSON: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, one row per
trophy, five quantiles of its earn rate (percent) at the case's target.

Model (standard library only, deterministic, no network):

1. The page is a staircase: Exophase holds its counts frozen, then jumps at a
   refresh. Consecutive identical reads are collapsed into one state, so a long
   flat stretch contributes a single observation instead of many.
2. With 3+ distinct states, each trophy's rate is tracked on the logit scale
   against its own ceiling: y = logit(rate / C), x = ln(1 + hours since
   release). A recency-weighted least-squares line (24 h half-life, fitted
   on the last 24 h of states, since growth decelerates in log-time) is
   fitted per trophy, its slope shrunk toward its tier's mean slope, and
   read at the target. The logit-over-ceiling form saturates by itself, and
   a forecast is additionally capped just below its ceiling.
3. With fewer than 3 states the page is (nearly) frozen, so the forecast is a
   refresh prior. Exophase refreshes roughly every 270 minutes and each refresh
   multiplies the tracked players by ~6. A fresh cohort has had little time to
   play, so it earns each trophy at only a small fraction beta of the rate the
   incumbents show; the forecast rate is the player-weighted mix of the two
   cohorts. beta = 0.0665 comes from the one refresh the spec records (launch
   night: 638 players / 943 trophies earned frozen, then 3,868 players /
   4,276 earned; leaving out the trophy already at 100%, the newcomers earned
   103 trophies over 3,230 new players, i.e. 6.65% of the incumbents' rate).
   At most two refreshes are compounded, the second with beta halved, so a
   long horizon cannot collapse every rate toward zero. A trophy at 100 stays
   at 100.
4. Spreads widen with the horizon (random walk), with fit uncertainty, and
   with how small the rate is; in the frozen case the upper side is wider,
   because the refresh might arrive late or not at all before the target.
   A trophy with no usable history falls back to its tier's median rate.

The program never crashes on a malformed case: any failure in one case falls
back to tier-median forecasts for that case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; nothing beside
# main.py is imported, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# --- tunables (env overrides exist so a local sweep harness can tune them) ---
try:
    EXTRA_FROZEN = float(os.environ.get("EXTRA_FROZEN", "0.30"))
except ValueError:
    EXTRA_FROZEN = 0.30
try:
    SLOPE_PRIOR_SD = float(os.environ.get("SLOPE_PRIOR_SD", "0.20"))
except ValueError:
    SLOPE_PRIOR_SD = 0.20
try:
    TREND_RW_SIGMA = float(os.environ.get("TREND_RW_SIGMA", "0.30"))
except ValueError:
    TREND_RW_SIGMA = 0.30
try:
    FIT_WINDOW_H = float(os.environ.get("FIT_WINDOW_H", "24"))
except ValueError:
    FIT_WINDOW_H = 24.0

REFRESH_CADENCE_MIN = 270.0   # Exophase refreshes a game's counts hours apart
PLAYER_MULT = 6.0             # 638 -> 3,868 tracked players at the first refresh
BETA = 0.0665                 # fresh cohort earns each trophy at beta x incumbents' rate
BETA_DECAY = 0.5              # second compounded refresh uses beta * 0.5
MAX_REFRESHES = 2
HALF_LIFE_H = 24.0            # recency half-life of the trend fit's weights
RW_SIGMA = 0.5                # random-walk spread, logit units per sqrt(day)
MODEL_SIGMA = 0.10            # irreducible model allowance, logit units


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def parse_time(value):
    """Parse an ISO-8601 timestamp (or a plain date) to an aware datetime."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if len(text) == 10:  # a bare date: midnight UTC
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(y):
    if y >= 0:
        p = 1.0 / (1.0 + math.exp(-y)) if y < 700 else 1.0
    else:
        e = math.exp(y) if y > -700 else 0.0
        p = e / (1.0 + e)
    return p * 100.0


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    e = math.exp(x) if x > -700 else 0.0
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


# --------------------------------------------------------------------------
# ceilings: the eventual earn rate each trophy saturates toward
# --------------------------------------------------------------------------

def ceiling(trophy):
    """Ceiling (fraction 0..1) from tier + description keywords."""
    tier = str(trophy.get("tier") or "bronze").lower()
    desc = str(trophy.get("description") or "").lower()
    name = str(trophy.get("name") or "").lower()

    if tier == "platinum":
        return 0.02
    if "100% completion" in desc:
        return 0.02
    if "complete chapter" in desc:
        # Chapter 1 is reached by about half the players in week one, later
        # chapters by fewer: 55% / 50% / 45% / 40% ...
        num = 1
        for tok in desc.replace("chapter", " chapter ").split():
            if tok.isdigit():
                num = int(tok)
                break
        return max(0.35, 0.60 - 0.05 * num)
    if "complete the final mission" in desc:
        return 0.30
    if "final mission for" in desc:
        return 0.20
    # Grind / completionist trophies first: several of their descriptions
    # also start with "Complete", and must not fall through to the story
    # mission rate below.
    if "gold medal in all" in desc or "100 locations" in desc:
        return 0.06
    if "250 stunts" in desc or "collectible card sets" in desc \
            or "all garage properties" in desc:
        return 0.08
    if "every spot" in desc or "every location" in desc \
            or "earn $250,000 laundering" in desc:
        return 0.10
    if "50 challenges" in desc or "5 bank vaults" in desc \
            or "8 mission strands" in desc:
        return 0.12
    if "20 different races" in desc:
        return 0.15
    if "explore all of leonida" in desc:
        return 0.18
    if "discover 100 vehicles" in desc or "register 10 vehicles" in desc:
        return 0.20
    if "hunt a unique animal" in desc:
        return 0.20
    if "beat cal in a fishing challenge" in desc:
        return 0.45
    if "complete an arcade game" in desc:
        return 0.30
    if "try each type of activity once" in desc:
        return 0.30
    if "spend $100,000" in desc:
        return 0.25
    if desc.startswith("complete ") and tier in ("bronze", "silver"):
        return 0.35  # named story missions
    if tier == "gold":
        return 0.12
    if tier == "silver":
        return 0.18
    return 0.25


def tier_default_rate(tier):
    return {"platinum": 0.5, "gold": 0.5, "silver": 0.7}.get(
        str(tier or "bronze").lower(), 1.0)


# --------------------------------------------------------------------------
# history -> states
# --------------------------------------------------------------------------

def collapse_states(history):
    """Collapse consecutive identical reads into states.

    A state is (first_time, players, rates). Two reads belong to the same state
    when both the players-tracked count and every rate match.
    """
    states = []
    for read in history or []:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        players = read.get("players_tracked")
        t = parse_time(read.get("at")) or parse_time(read.get("slot"))
        key = (players, tuple(rates))
        if states and states[-1]["key"] == key:
            states[-1]["last_time"] = t or states[-1]["last_time"]
            continue
        states.append({"key": key, "first_time": t, "last_time": t,
                       "players": players, "rates": list(rates)})
    return states


# --------------------------------------------------------------------------
# trend fit (3+ states): weighted least squares of logit(rate/C) on ln(1+h)
# --------------------------------------------------------------------------

def wls_fit(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, resid_sd, xbar)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0, mx)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw
    s2 = s2 * len(xs) / dof if len(xs) > 2 else s2
    se_b = math.sqrt(max(s2, 0.0) / sxx) if sxx > 0 else 0.0
    resid_sd = math.sqrt(max(s2, 0.0))
    return (a, b, se_b, resid_sd, mx)


def trend_forecast(case, states, trophies, target, release):
    """Forecast medians + sigmas from the saturating log-time trend."""
    n = len(trophies)
    horizon_days = max(case.get("horizon_minutes") or 0, 0) / 1440.0
    x_target = math.log1p(max((target - release).total_seconds(), 0.0) / 3600.0) \
        if target is not None else 0.0
    last_time = states[-1]["first_time"]

    ceilings = []
    for i, tr in enumerate(trophies):
        c = ceiling(tr)
        # A ceiling must sit above every rate ever seen for the trophy.
        seen = [s["rates"][i] for s in states
                if i < len(s["rates"]) and isinstance(s["rates"][i], (int, float))]
        if seen:
            c = max(c, min(0.999, 1.02 * (max(seen) / 100.0) + 0.01))
        ceilings.append(min(0.999, max(0.005, c)))

    # First pass: raw per-trophy slopes, for the tier-mean shrinkage target.
    # The fit uses only recent states (growth decelerates in log-time, so old
    # states would steepen the line): states within FIT_WINDOW_H of the last
    # state, or all states if that leaves fewer than three points.
    raw = []
    for i in range(n):
        pts = []
        for s in states:
            if i >= len(s["rates"]):
                continue
            r = s["rates"][i]
            if not isinstance(r, (int, float)) or r is None:
                continue
            if r >= 99.995 or r <= 0:
                continue
            t = s["first_time"] or last_time
            if t is None or last_time is None:
                age_h = 0.0
                hours = 0.0
            else:
                age_h = max((last_time - t).total_seconds(), 0.0) / 3600.0
                hours = max((t - release).total_seconds(), 0.0) / 3600.0
            pts.append((math.log1p(hours),
                        math.log((r / 100.0 / ceilings[i]) /
                                 (1.0 - r / 100.0 / ceilings[i])),
                        0.5 ** (age_h / HALF_LIFE_H), age_h))
        if FIT_WINDOW_H > 0:
            recent = [p for p in pts if p[3] <= FIT_WINDOW_H]
            if len(recent) >= 3:
                pts = recent
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ws = [p[2] for p in pts]
        fit = wls_fit(xs, ys, ws) if len(xs) >= 2 else None
        raw.append((xs, ys, ws, fit))

    # Tier-mean (blended with the global mean) slope per tier.
    slopes_by_tier = {}
    all_slopes = []
    for i, (_, _, _, fit) in enumerate(raw):
        if fit is None:
            continue
        tier = str(trophies[i].get("tier") or "bronze").lower()
        slopes_by_tier.setdefault(tier, []).append(fit[1])
        all_slopes.append(fit[1])
    global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    tier_mean = {}
    for tier, vals in slopes_by_tier.items():
        m = sum(vals) / len(vals)
        tier_mean[tier] = 0.5 * m + 0.5 * global_mean if len(vals) < 3 else m

    medians, sigmas = [], []
    for i in range(n):
        xs, ys, ws, fit = raw[i]
        tier = str(trophies[i].get("tier") or "bronze").lower()
        if fit is None:
            medians.append(None)
            sigmas.append(None)
            continue
        _, b_raw, se_b, resid_sd, _ = fit
        prior = tier_mean.get(tier, global_mean)
        # Empirical-Bayes shrinkage of the slope toward its tier's mean. The
        # standard error is floored: a perfectly flat history (zero residual)
        # says nothing about the slope, so it must not count as certainty.
        se2 = max(se_b, 0.20) ** 2
        pr2 = max(SLOPE_PRIOR_SD, 0.01) ** 2
        b = (b_raw / se2 + prior / pr2) / (1.0 / se2 + 1.0 / pr2)
        # Refit the intercept around the shrunk slope.
        sw = sum(ws)
        a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sw if sw > 0 else 0.0
        y_t = a + b * x_target
        rate = ceilings[i] * sigmoid(y_t) * 100.0
        # The logit form already saturates at the ceiling; only shave a hair
        # off, and never below the last rate seen (the ceiling was raised
        # above every observed rate, so this cap cannot bind below it).
        rate = min(rate, 0.999 * ceilings[i] * 100.0)
        rate = max(rate, 0.01)
        medians.append(rate)
        # Spread: random walk + slope uncertainty over the extrapolation gap
        # + residual scatter + model allowance. The trend already carries the
        # systematic move, so its random-walk term is smaller than the
        # persistence baseline's.
        xbar = sum(w * x for w, x in zip(ws, xs)) / sw if sw > 0 else x_target
        gap = max(x_target - xbar, 0.0)
        var = (TREND_RW_SIGMA * math.sqrt(horizon_days)) ** 2 \
            + (se_b * gap) ** 2 + (0.5 * resid_sd) ** 2 + MODEL_SIGMA ** 2
        if rate < 0.5:
            var += 0.25 ** 2
        elif rate < 2.0:
            var += 0.10 ** 2
        sigmas.append(min(max(math.sqrt(var), 0.12), 1.20))
    return medians, sigmas


# --------------------------------------------------------------------------
# refresh prior (<3 states): frozen page, refreshes due before the target
# --------------------------------------------------------------------------

def refresh_forecast(case, states, trophies, target):
    """Forecast medians + (down, up) sigmas from the refresh prior."""
    n = len(trophies)
    horizon_days = max(case.get("horizon_minutes") or 0, 0) / 1440.0
    last = states[-1] if states else None

    if last is not None and last["first_time"] is not None and target is not None:
        frozen_min = (target - last["first_time"]).total_seconds() / 60.0
    else:
        frozen_min = float(case.get("horizon_minutes") or 0)
    n_refresh = int(min(MAX_REFRESHES, max(0, math.floor(frozen_min / REFRESH_CADENCE_MIN))))

    # Per-refresh multiplier of a trophy's rate: the player-weighted mix of
    # incumbents (rate r) and a fresh cohort (rate beta * r).
    factor = 1.0
    for k in range(n_refresh):
        beta_k = BETA * (BETA_DECAY ** k)
        factor *= (1.0 + beta_k * (PLAYER_MULT - 1.0)) / PLAYER_MULT

    medians, sigmas = [], []
    for i in range(n):
        r = None
        if last is not None and i < len(last["rates"]):
            r = last["rates"][i]
        if not isinstance(r, (int, float)) or r is None:
            medians.append(None)
            sigmas.append(None)
            continue
        if r >= 99.995:
            medians.append(100.0)
            sigmas.append((0.0, 0.0))
            continue
        med = max(r * factor, 0.01)
        var = (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + EXTRA_FROZEN ** 2 \
            + MODEL_SIGMA ** 2
        if med < 0.5:
            var += 0.25 ** 2
        elif med < 2.0:
            var += 0.10 ** 2
        sd = min(max(math.sqrt(var), 0.15), 1.20)
        # Upper side wider: the refresh may arrive late or not at all.
        sigmas.append((sd, min(sd * 1.5, 1.80)))
        medians.append(med)
    return medians, sigmas


# --------------------------------------------------------------------------
# fallbacks
# --------------------------------------------------------------------------

def fallback_rows(trophies, last_rates):
    """Tier-median fallback for trophies with no usable history."""
    by_tier = {}
    for i, tr in enumerate(trophies):
        r = last_rates[i] if i < len(last_rates) else None
        if isinstance(r, (int, float)) and 0 < r < 99.995:
            by_tier.setdefault(str(tr.get("tier") or "bronze").lower(), []).append(r)
    medians = []
    for i, tr in enumerate(trophies):
        tier = str(tr.get("tier") or "bronze").lower()
        vals = sorted(by_tier.get(tier, []))
        if vals:
            med = vals[len(vals) // 2]
        else:
            allv = sorted(v for vs in by_tier.values() for v in vs)
            med = allv[len(allv) // 2] if allv else tier_default_rate(tier)
        medians.append(max(med, 0.01))
    return medians


# --------------------------------------------------------------------------
# per-case driver
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    n = len(trophies)
    target = parse_time(case.get("target"))
    release = parse_time((case.get("game") or {}).get("release_date")) or RELEASE
    states = collapse_states(case.get("history") or [])

    if len(states) >= 3:
        medians, sigmas = trend_forecast(case, states, trophies, target, release)
        spread_kind = "trend"
    elif states:
        medians, sigmas = refresh_forecast(case, states, trophies, target)
        spread_kind = "refresh"
    else:
        medians, sigmas = [None] * n, [None] * n
        spread_kind = "none"

    last_rates = states[-1]["rates"] if states else [None] * n
    fb = fallback_rows(trophies, last_rates)
    horizon_days = max(case.get("horizon_minutes") or 0, 0) / 1440.0

    rows = []
    for i, tr in enumerate(trophies):
        med = medians[i]
        sig = sigmas[i]
        if med is None:
            med = fb[i]
            sig = 1.0 if spread_kind != "refresh" else (1.0, 1.0)
        if med >= 99.995:
            rows.append({"id": tr.get("id"), "q": [99.9, 99.99, 100, 100, 100]})
            continue
        lg = logit_pct(med)
        if isinstance(sig, tuple):
            sd_down, sd_up = sig
        else:
            sd_down = sd_up = sig if sig else 0.5
        qs = []
        for z in Z:
            sd = sd_down if z < 0 else (sd_up if z > 0 else 0.0)
            qs.append(round4(min(100.0, max(0.0, inv_logit_pct(lg + z * sd)))))
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr.get("id"), "q": qs})
    return {"trophies": rows}


def safe_forecast_case(case):
    """Never crash on a case: fall back to tier defaults on any error."""
    try:
        return forecast_case(case)
    except Exception:
        trophies = []
        try:
            trophies = case.get("trophies") or []
        except Exception:
            trophies = []
        rows = []
        for tr in trophies:
            try:
                tid = tr.get("id")
            except Exception:
                tid = None
            med = tier_default_rate(tr.get("tier") if isinstance(tr, dict) else None)
            lg = logit_pct(med)
            qs = [round4(inv_logit_pct(lg + z * 1.0)) for z in Z]
            for k in range(1, 5):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
            rows.append({"id": tid, "q": qs})
        return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not an object")
        except Exception:
            case = {"trophies": []}
        forecast = safe_forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
