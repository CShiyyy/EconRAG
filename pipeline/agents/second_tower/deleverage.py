"""Tiered portfolio deleveraging based on drawdown depth.

Replaces the binary drawdown stop with 4 graduated tiers of exposure
reduction plus a final hard stop.  Designed to prevent whipsaw via
hysteresis bands and cooldown periods.

T+1 execution model: drawdown observed at close of bar T sets the
exposure multiplier applied at the *start* of bar T+1.  This ensures
zero look-ahead bias — the controller never uses future information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Tier schedules per risk level ────────────────────────────────────────────
# Each tuple: (entry_threshold, recovery_threshold, exposure)
#   entry_threshold   : drawdown must be WORSE (more negative) to enter this tier
#   recovery_threshold: drawdown must be BETTER (less negative) for cooldown_days
#                       consecutive days to step back UP from this tier
#   exposure          : weight multiplier while in this tier

_TIER_SCHEDULES: dict[str, list[tuple[float, float, float]]] = {
    "low": [
        (0.0,    0.0,    1.00),   # Tier 0: Normal
        (-0.03, -0.015,  0.75),   # Tier 1: Caution
        (-0.05, -0.035,  0.50),   # Tier 2: Warning
        (-0.07, -0.055,  0.25),   # Tier 3: Critical
        (-0.10, -0.10,   0.00),   # Tier 4: Hard Stop (permanent)
    ],
    "medium": [
        (0.0,    0.0,    1.00),
        (-0.05, -0.03,   0.75),
        (-0.10, -0.08,   0.50),
        (-0.15, -0.13,   0.25),
        (-0.20, -0.20,   0.00),
    ],
    "high": [
        (0.0,    0.0,    1.00),
        (-0.10, -0.07,   0.75),
        (-0.20, -0.17,   0.50),
        (-0.30, -0.27,   0.25),
        (-0.40, -0.40,   0.00),
    ],
}

# Cooldown: consecutive days above recovery threshold before stepping up
_COOLDOWN_DAYS: dict[str, int] = {
    "low": 5,
    "medium": 5,
    "high": 3,
}

# Regime multipliers — scale entry/recovery thresholds (tighter = triggers sooner)
_REGIME_THRESHOLD_SCALE: dict[str, float] = {
    "Bull": 1.0,
    "Bear": 0.8,
    "WalkingOnIce": 0.7,
    "Crisis": 0.6,
}

# Maximum number of tier steps (up or down) per bar
_MAX_STEPS_PER_BAR = 1


@dataclass
class _TierLevel:
    """A single tier in the schedule (may be regime-adjusted)."""
    entry_threshold: float
    recovery_threshold: float
    exposure: float


class DeleverageController:
    """Tiered portfolio deleveraging driven by peak-to-trough drawdown.

    Usage in the daily backtest loop::

        controller = DeleverageController("medium", regime_aware=True)
        exposure_mult = 1.0          # T+1 model: starts at full exposure

        for t in range(T):
            effective_w = target_w * exposure_mult   # apply previous bar's decision
            ...                                       # SL / TP / signal-decay
            ...                                       # compute daily return, update peak
            dd = (capital - peak) / peak
            exposure_mult = controller.step(dd)       # sets mult for NEXT bar
            if controller.is_hard_stopped:
                portfolio_stopped = True
    """

    def __init__(self, risk_level: str, regime_aware: bool = False) -> None:
        self._risk_level = risk_level
        self._regime_aware = regime_aware
        self._cooldown_days = _COOLDOWN_DAYS.get(risk_level, 5)

        # Build base tier schedule (unscaled by regime)
        base = _TIER_SCHEDULES.get(risk_level, _TIER_SCHEDULES["medium"])
        self._base_tiers = [
            _TierLevel(entry, recovery, exposure)
            for entry, recovery, exposure in base
        ]
        # Active tiers (may be regime-adjusted)
        self._tiers: list[_TierLevel] = list(self._base_tiers)
        self._current_regime_scale = 1.0

        # State
        self._current_tier: int = 0
        self._exposure_mult: float = 1.0
        self._cooldown_counter: int = 0
        self._hard_stopped: bool = False
        self._bar_counter: int = 0

        # Diagnostics
        self._tier_transitions: list[dict] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def update_regime(self, regime_name: str) -> None:
        """Update tier thresholds based on detected market regime.

        Called at rebalancing events when a new regime label is available.
        Only has effect if ``regime_aware=True`` was set at construction.
        """
        if not self._regime_aware:
            return
        scale = _REGIME_THRESHOLD_SCALE.get(regime_name, 1.0)
        if scale == self._current_regime_scale:
            return
        self._current_regime_scale = scale
        self._tiers = self._apply_regime_scale(scale)

    def step(self, dd: float) -> float:
        """Process one bar's drawdown and return the exposure multiplier for the NEXT bar.

        Parameters
        ----------
        dd : float
            Current drawdown as a negative fraction (e.g. -0.08 = -8%).

        Returns
        -------
        float
            Exposure multiplier in [0.0, 1.0] to apply on the next bar.
        """
        if self._hard_stopped:
            return 0.0

        self._bar_counter += 1
        steps_taken = 0

        # ── Check for tier DOWNGRADE (deeper drawdown) ───────────────────────
        while (
            steps_taken < _MAX_STEPS_PER_BAR
            and self._current_tier < len(self._tiers) - 1
        ):
            next_tier = self._current_tier + 1
            if dd <= self._tiers[next_tier].entry_threshold:
                old_tier = self._current_tier
                self._current_tier = next_tier
                self._cooldown_counter = 0  # reset recovery cooldown
                steps_taken += 1
                self._tier_transitions.append({
                    "bar": self._bar_counter,
                    "from_tier": old_tier,
                    "to_tier": self._current_tier,
                    "dd": round(dd, 6),
                    "exposure": self._tiers[self._current_tier].exposure,
                    "direction": "down",
                })
            else:
                break

        # ── Check for tier UPGRADE (recovery) ────────────────────────────────
        if steps_taken == 0 and self._current_tier > 0:
            recovery_thresh = self._tiers[self._current_tier].recovery_threshold
            if dd > recovery_thresh:
                self._cooldown_counter += 1
                if self._cooldown_counter >= self._cooldown_days:
                    old_tier = self._current_tier
                    self._current_tier -= 1
                    self._cooldown_counter = 0
                    self._tier_transitions.append({
                        "bar": self._bar_counter,
                        "from_tier": old_tier,
                        "to_tier": self._current_tier,
                        "dd": round(dd, 6),
                        "exposure": self._tiers[self._current_tier].exposure,
                        "direction": "up",
                    })
            else:
                # Drawdown worsened or stayed in tier band — reset cooldown
                self._cooldown_counter = 0

        # ── Update exposure ──────────────────────────────────────────────────
        self._exposure_mult = self._tiers[self._current_tier].exposure

        # Tier 4 is permanent hard stop
        if self._current_tier == len(self._tiers) - 1:
            self._hard_stopped = True

        return self._exposure_mult

    @property
    def is_hard_stopped(self) -> bool:
        """True if the portfolio has reached the final tier (permanent stop)."""
        return self._hard_stopped

    @property
    def current_tier(self) -> int:
        """Current tier index (0 = normal, 4 = hard stop)."""
        return self._current_tier

    @property
    def exposure(self) -> float:
        """Current exposure multiplier."""
        return self._exposure_mult

    @property
    def transitions(self) -> list[dict]:
        """Log of tier transitions for diagnostics."""
        return list(self._tier_transitions)

    # ── Internals ────────────────────────────────────────────────────────────

    def _apply_regime_scale(self, scale: float) -> list[_TierLevel]:
        """Return a new tier list with entry/recovery thresholds scaled.

        A scale < 1.0 tightens the thresholds (triggers at shallower drawdowns).
        Tier 0 (normal) is never adjusted.  The hard stop tier's entry threshold
        is also scaled so that it triggers proportionally sooner.
        """
        adjusted = [self._base_tiers[0]]  # Tier 0 unchanged
        for tier in self._base_tiers[1:]:
            adjusted.append(_TierLevel(
                entry_threshold=tier.entry_threshold * scale,
                recovery_threshold=tier.recovery_threshold * scale,
                exposure=tier.exposure,
            ))
        return adjusted
