import datetime
import holidays


def get_us_market_holidays(years=None):
    """Return a set of US market holidays for given years."""
    if years is None:
        years = [datetime.datetime.now().year]
    us_holidays = holidays.US(years=years)
    return set(us_holidays.keys())


def is_trading_day(date):
    """Return True if date is trading day (Mon-Fri and not holiday)."""
    if date.weekday() >= 5:
        return False
    current_year = date.year
    holidays_set = get_us_market_holidays([current_year])
    return date not in holidays_set


def next_trading_day(date):
    """Return next trading day after the given date."""
    next_day = date + datetime.timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += datetime.timedelta(days=1)
    return next_day


def last_trading_day_of_week(date):
    """Return True if date is the last trading day of its week."""
    if not is_trading_day(date):
        return False
    next_day = date + datetime.timedelta(days=1)
    while next_day.weekday() < 5:
        if is_trading_day(next_day):
            return False  # next trading day still in week
        next_day += datetime.timedelta(days=1)
    return True


def last_trading_day_of_month(date):
    """Return True if date is the last trading day of its month."""
    if not is_trading_day(date):
        return False
    next_day = date + datetime.timedelta(days=1)
    while next_day.month == date.month:
        next_day += datetime.timedelta(days=1)
    last_day = next_day - datetime.timedelta(days=1)
    while not is_trading_day(last_day):
        last_day -= datetime.timedelta(days=1)
    return date == last_day


def find_expiry_within_trading_days(expirations, start_date, max_trading_days):
    """
    Find earliest expiry in expirations occurring within max_trading_days trading days from start_date.
    expirations: iterable of datetime.date
    start_date: datetime.date
    max_trading_days: int
    Returns datetime.date or None
    """
    candidate_dates = []
    current = start_date
    count = 0
    while count < max_trading_days:
        if is_trading_day(current):
            candidate_dates.append(current)
            count += 1
        current += datetime.timedelta(days=1)
    for d in candidate_dates:
        if d in expirations:
            return d
    return None
