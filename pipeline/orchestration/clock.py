"""Market calendar helpers for slot detection and phase annotation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import pytz

ET = pytz.timezone("America/New_York")

_MARKET_OPEN_H = 9
_MARKET_OPEN_M = 30
_MARKET_CLOSE_H = 16
_MARKET_CLOSE_M = 0


class NonTradingDayError(Exception):
    def __init__(self, session_date: str, reason: str) -> None:
        self.session_date = session_date
        self.reason = reason
        super().__init__(f"Not a trading day ({session_date}): {reason}")


def _get_calendar():
    import exchange_calendars as ec
    return ec.get_calendar("XNYS")


def _now_et(now_utc: datetime | None) -> datetime:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return now_utc.astimezone(ET)


def detect_slot(now_utc: datetime | None = None) -> tuple[str, str]:
    """Return (session_date YYYY-MM-DD, run_type) based on current ET time.

    Raises NonTradingDayError if today is not an NYSE trading session.
    Slot rule: ET < 09:30 -> pre_open; ET >= 09:30 -> post_close.
    """
    now_et = _now_et(now_utc)
    session_date = now_et.strftime("%Y-%m-%d")

    cal = _get_calendar()
    import pandas as pd
    if not cal.is_session(pd.Timestamp(session_date)):
        raise NonTradingDayError(session_date, "weekend or US market holiday")

    if now_et.hour < _MARKET_OPEN_H or (now_et.hour == _MARKET_OPEN_H and now_et.minute < _MARKET_OPEN_M):
        return session_date, "pre_open"
    return session_date, "post_close"


def previous_close(now_utc: datetime | None = None) -> datetime:
    """Return the UTC datetime of the most recent NYSE market close before now."""
    import pandas as pd
    now_et = _now_et(now_utc)
    cal = _get_calendar()
    check_date = now_et.date()
    for _ in range(10):
        ts = pd.Timestamp(check_date)
        if cal.is_session(ts):
            close_et = ET.localize(
                datetime(check_date.year, check_date.month, check_date.day,
                         _MARKET_CLOSE_H, _MARKET_CLOSE_M)
            )
            if close_et <= now_et:
                return close_et.astimezone(timezone.utc)
        check_date = check_date - timedelta(days=1)
    raise RuntimeError("Could not find a previous NYSE close within 10 calendar days")


def next_close(now_utc: datetime | None = None) -> datetime:
    """Return the UTC datetime of the next NYSE market close at or after now."""
    import pandas as pd
    now_et = _now_et(now_utc)
    check_date = now_et.date()
    cal = _get_calendar()
    for _ in range(10):
        ts = pd.Timestamp(check_date)
        if cal.is_session(ts):
            close_et = ET.localize(
                datetime(check_date.year, check_date.month, check_date.day,
                         _MARKET_CLOSE_H, _MARKET_CLOSE_M)
            )
            if close_et > now_et:
                return close_et.astimezone(timezone.utc)
        check_date = check_date + timedelta(days=1)
    raise RuntimeError("Could not find a next NYSE close within 10 calendar days")


def market_phase(now_utc: datetime | None = None) -> Literal["pre_open", "open", "post_close", "closed_day"]:
    """Return the current market phase for UI display purposes.

    - pre_open:   trading day, before 09:30 ET
    - open:       trading day, 09:30-16:00 ET (market currently trading)
    - post_close: trading day, after 16:00 ET
    - closed_day: weekend or US holiday
    """
    now_et = _now_et(now_utc)
    session_date = now_et.strftime("%Y-%m-%d")

    cal = _get_calendar()
    import pandas as pd
    if not cal.is_session(pd.Timestamp(session_date)):
        return "closed_day"

    h, m = now_et.hour, now_et.minute
    if h < _MARKET_OPEN_H or (h == _MARKET_OPEN_H and m < _MARKET_OPEN_M):
        return "pre_open"
    if h < _MARKET_CLOSE_H or (h == _MARKET_CLOSE_H and m < _MARKET_CLOSE_M):
        return "open"
    return "post_close"
