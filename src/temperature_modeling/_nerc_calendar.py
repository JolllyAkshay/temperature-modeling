"""
NERC on-peak/off-peak hour classification for PJM Western Hub.

Matches ICE's PJM Western Hub Peak futures contract specification (verified
against ICE's published contract specs, not assumed): on-peak hours are all
hours ending 0800-2300 Eastern Prevailing Time, Monday-Friday, excluding
NERC holidays. Every hour on a Saturday, Sunday, or NERC holiday is off-peak,
as are hours ending 0100-0700 and 0000/2400 (i.e. hour-ending 2400/midnight)
on any other day.

PJM DataMiner2 timestamps (datetime_beginning_ept) are hour-*beginning*, not
hour-ending, and already in Eastern Prevailing Time (DST-aware) — no timezone
conversion needed here, just an hour-ending vs hour-beginning offset: hour
ending 0800 is the hour beginning at 07:00, hour ending 2300 is the hour
beginning at 22:00. So in hour-beginning terms the on-peak window is
07:00-22:00 inclusive (16 distinct starting hours).

Public API
----------
is_nerc_holiday(d: date) -> bool
is_peak_hour(dt: datetime) -> bool
    dt must be a naive datetime already in Eastern Prevailing Time, with
    dt.hour representing the *start* of the hour (PJM's datetime_beginning_ept
    convention).
"""

from datetime import date, datetime, timedelta


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """n-th occurrence (1-indexed) of `weekday` (Mon=0..Sun=6) in the given month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d += timedelta(days=offset + 7 * (n - 1))
    return d


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Last occurrence of `weekday` (Mon=0..Sun=6) in the given month."""
    if month == 12:
        next_month_start = date(year + 1, 1, 1)
    else:
        next_month_start = date(year, month + 1, 1)
    d = next_month_start - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def nerc_holidays(year: int) -> set:
    """
    The 6 standard NERC holidays observed for on-peak/off-peak classification:
    New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving,
    Christmas. Fixed-date holidays are used as-is (no weekend observed-date
    shifting) — PJM/ICE's peak calendar treats the calendar date itself as
    off-peak regardless of which day of the week it falls on; a holiday that
    lands on a Sunday doesn't need special handling since Sundays are already
    off-peak in full.
    """
    return {
        date(year, 1, 1),                              # New Year's Day
        _last_weekday_of_month(year, 5, 0),             # Memorial Day — last Monday in May
        date(year, 7, 4),                                # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),            # Labor Day — first Monday in September
        _nth_weekday_of_month(year, 11, 3, 4),           # Thanksgiving — 4th Thursday in November
        date(year, 12, 25),                              # Christmas Day
    }


def is_nerc_holiday(d: date) -> bool:
    return d in nerc_holidays(d.year)


def is_peak_hour(dt: datetime) -> bool:
    """
    True if `dt` (naive, Eastern Prevailing Time, hour-beginning convention —
    see module docstring) falls within PJM/ICE's on-peak window: Mon-Fri,
    hour-beginning 07:00 through 22:00 inclusive, excluding NERC holidays.
    """
    if dt.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    if is_nerc_holiday(dt.date()):
        return False
    return 7 <= dt.hour <= 22
