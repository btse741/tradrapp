from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
import yfinance as yf


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:  
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:  
            return float(spline(dte))

    return term_spline


def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    if not todays_data.empty:
        return todays_data['Close'].iloc[0]
    else:
        return None


def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."
        
        try:
            stock = yf.Ticker(ticker)
            if len(stock.options) < 2:
                raise KeyError()
        except KeyError:
            return f"Error: Not enough options found for stock symbol '{ticker}'. Need at least two expiry dates."
        
        exp_dates = sorted(stock.options, key=lambda d: datetime.strptime(d, "%Y-%m-%d").date())
        
        try:
            exp_dates = filter_dates(exp_dates)
            if len(exp_dates) < 2:
                return "Error: Not enough option data after filtering for calendar spreads."
        except:
            return "Error: Not enough option data."
        
        options_chains = {exp_date: stock.option_chain(exp_date) for exp_date in exp_dates}
        
        underlying_price = get_current_price(stock)
        if underlying_price is None:
            return "Error: Unable to retrieve underlying stock price."
        
        atm_iv = {}
        atm_strike = None
        straddle = None
        
        for i, exp_date in enumerate(exp_dates):
            chain = options_chains[exp_date]
            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty:
                continue

            if i == 0:
                call_diffs = (calls['strike'] - underlying_price).abs()
                call_idx = call_diffs.idxmin()
                atm_strike = calls.loc[call_idx, 'strike']
                
            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = {
                'iv': atm_iv_value,
                'strike': calls.loc[call_idx, 'strike'],
                'call_bid': calls.loc[call_idx, 'bid'],
                'call_ask': calls.loc[call_idx, 'ask'],
                'put_bid': puts.loc[put_idx, 'bid'],
                'put_ask': puts.loc[put_idx, 'ask']
            }

            if i == 0:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']
                
                call_mid = (call_bid + call_ask) / 2.0 if (call_bid is not None and call_ask is not None) else None
                put_mid = (put_bid + put_ask) / 2.0 if (put_bid is not None and put_ask is not None) else None
                if call_mid is not None and put_mid is not None:
                    straddle = call_mid + put_mid

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, data in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(data['iv'])
        
        term_spline = build_term_structure(dtes, ivs)
        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])

        price_history = stock.history(period='3mo')
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
        expected_move_straddle = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None

        # Calendar Spread Calculation
        near_expiry = exp_dates[0]
        far_expiry = exp_dates[1]

        near_chain = options_chains[near_expiry]
        far_chain = options_chains[far_expiry]

        cal_strike = atm_strike

        near_calls = near_chain.calls
        far_calls = far_chain.calls
        near_puts = near_chain.puts
        far_puts = far_chain.puts

        near_call_idx = near_calls[near_calls['strike'] == cal_strike].index
        far_call_idx = far_calls[far_calls['strike'] == cal_strike].index
        near_put_idx = near_puts[near_puts['strike'] == cal_strike].index
        far_put_idx = far_puts[far_puts['strike'] == cal_strike].index

        def safe_get_iv(df, idx):
            if len(idx) == 0:
                return None
            iv = df.loc[idx[0], 'impliedVolatility']
            return iv if not np.isnan(iv) else None

        near_call_iv = safe_get_iv(near_calls, near_call_idx)
        far_call_iv = safe_get_iv(far_calls, far_call_idx)
        near_put_iv = safe_get_iv(near_puts, near_put_idx)
        far_put_iv = safe_get_iv(far_puts, far_put_idx)

        if None not in (near_call_iv, far_call_iv, near_put_iv, far_put_iv):
            cal_call_iv_avg = (near_call_iv + far_call_iv) / 2
            cal_put_iv_avg = (near_put_iv + far_put_iv) / 2
            # rough calendar edge metric (just an example)
            cal_iv_diff = cal_put_iv_avg - cal_call_iv_avg

            near_straddle = None
            far_straddle = None

            try:
                near_call_bid = near_calls.loc[near_call_idx[0], 'bid']
                near_call_ask = near_calls.loc[near_call_idx[0], 'ask']
                near_put_bid = near_puts.loc[near_put_idx[0], 'bid']
                near_put_ask = near_puts.loc[near_put_idx[0], 'ask']
                near_call_mid = (near_call_bid + near_call_ask) / 2 if (near_call_bid is not None and near_call_ask is not None) else None
                near_put_mid = (near_put_bid + near_put_ask) / 2 if (near_put_bid is not None and near_put_ask is not None) else None
                if near_call_mid is not None and near_put_mid is not None:
                    near_straddle = near_call_mid + near_put_mid
            except:
                pass

            try:
                far_call_bid = far_calls.loc[far_call_idx[0], 'bid']
                far_call_ask = far_calls.loc[far_call_idx[0], 'ask']
                far_put_bid = far_puts.loc[far_put_idx[0], 'bid']
                far_put_ask = far_puts.loc[far_put_idx[0], 'ask']
                far_call_mid = (far_call_bid + far_call_ask) / 2 if (far_call_bid is not None and far_call_ask is not None) else None
                far_put_mid = (far_put_bid + far_put_ask) / 2 if (far_put_bid is not None and far_put_ask is not None) else None
                if far_call_mid is not None and far_put_mid is not None:
                    far_straddle = far_call_mid + far_put_mid
            except:
                pass

            if near_straddle is not None and far_straddle is not None:
                expected_move_calendar = str(round(abs(far_straddle - near_straddle) / underlying_price * 100, 2)) + "%"
            else:
                expected_move_calendar = None
        else:
            expected_move_calendar = None

        return {
            'avg_volume_bool': avg_volume >= 1500000,
            'avg_volume': avg_volume,
            'iv30_rv30_bool': iv30_rv30 >= 1.25,
            'iv30_rv30': iv30_rv30,
            'ts_slope_bool': ts_slope_0_45 <= -0.00406,
            'ts_slope_0_45': ts_slope_0_45,
            'expected_move_straddle': expected_move_straddle,
            'straddle_expiry': exp_dates[0],
            'straddle_strike': atm_strike,
            'calendar_short_expiry': near_expiry,
            'calendar_long_expiry': far_expiry,
            'calendar_strike': atm_strike,
            'expected_move_calendar': expected_move_calendar,
        }

    except Exception:
        raise Exception('Error occurred processing')
