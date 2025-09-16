import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="Single Strike Options SuperTrend", layout="wide")

class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'access-token': st.secrets["DHAN_ACCESS_TOKEN"],
            'client-id': st.secrets["DHAN_CLIENT_ID"],
            'Content-Type': 'application/json'
        }
    
    def get_historical_data(self, security_id, exchange_segment, instrument, from_date, to_date, interval="5"):
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "fromDate": from_date,
            "toDate": to_date
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Historical Data API Error: {e}")
            return None
    
    def get_option_chain(self, underlying_scrip, underlying_seg, expiry):
        url = f"{self.base_url}/optionchain"
        payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_expiry_list(self, underlying_scrip, underlying_seg):
        url = f"{self.base_url}/optionchain/expirylist"
        payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None

class TelegramBot:
    def __init__(self):
        self.bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    
    def send_message(self, message):
        if not self.bot_token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            return requests.post(url, data=data, timeout=5).status_code == 200
        except:
            return False

def get_3day_historical_data(api, active_option):
    ist = pytz.timezone('Asia/Kolkata')
    end_date = datetime.now(ist)
    start_date = end_date - timedelta(days=5)
    
    from_date = start_date.strftime('%Y-%m-%d 09:15:00')
    to_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    nifty_data = api.get_historical_data(
        security_id=13,
        exchange_segment="IDX_I",
        instrument="INDEX",
        from_date=from_date,
        to_date=to_date,
        interval="5"
    )
    
    if not nifty_data or 'timestamp' not in nifty_data:
        st.error("Failed to fetch historical data")
        return []
    
    timestamps = nifty_data['timestamp']
    closes = nifty_data['close']
    highs = nifty_data['high']
    lows = nifty_data['low']
    opens_ = nifty_data['open']
    
    dt_timestamps = [datetime.fromtimestamp(ts, ist) for ts in timestamps]
    cutoff_date = max(dt_timestamps) - timedelta(days=3)
    
    filtered_data = []
    for i, dt_ts in enumerate(dt_timestamps):
        if dt_ts < cutoff_date:
            continue
        
        spot_price = closes[i]
        spot_open = opens_[i] if i < len(opens_) else spot_price
        spot_high = highs[i] if i < len(highs) else spot_price
        spot_low = lows[i] if i < len(lows) else spot_price
        
        strike = active_option['strike']
        option_type = active_option['type']
        
        if option_type == 'CE':
            intrinsic = max(0, spot_price - strike)
            moneyness = (spot_price - strike) / strike
        else:
            intrinsic = max(0, strike - spot_price)
            moneyness = (strike - spot_price) / strike
        
        days_to_expiry = max(1, (end_date - dt_ts).days)
        time_value = max(1, abs(moneyness) * 50 * (days_to_expiry / 7))
        
        option_ltp = intrinsic + time_value
        
        underlying_range = (spot_high - spot_low) / spot_price if spot_price > 0 else 0
        option_volatility = underlying_range * 3
        
        option_open = option_ltp * (spot_open / spot_price) if spot_price > 0 else option_ltp
        option_high = option_ltp * (1 + option_volatility)
        option_low = option_ltp * (1 - option_volatility)
        
        option_high = max(option_high, option_open, option_ltp)
        option_low = min(option_low, option_open, option_ltp)
        
        filtered_data.append({
            'timestamp': dt_ts,
            'open': float(option_open),
            'high': float(option_high),
            'low': float(option_low),
            'close': float(option_ltp),
            'time_str': dt_ts.strftime('%m-%d %H:%M'),
            'spot_price': spot_price
        })
    
    return filtered_data

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    if len(high) < period + 1:
        return [], []
    
    tr = []
    for i in range(1, len(high)):
        tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr.append(tr_val)
    
    atr = []
    if len(tr) >= period:
        atr.append(sum(tr[:period]) / period)
        for i in range(period, len(tr)):
            atr.append((atr[-1] * (period - 1) + tr[i]) / period)
    
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    supertrend, trend_direction = [], []
    
    for i in range(len(hl2)):
        if i < len(atr):
            ub = hl2[i] + multiplier * atr[i]
            lb = hl2[i] - multiplier * atr[i]
        else:
            ub = hl2[i] + multiplier * (atr[-1] if atr else 0)
            lb = hl2[i] - multiplier * (atr[-1] if atr else 0)
        
        if i == 0:
            supertrend.append(ub)
            trend_direction.append(1)
        else:
            prev_st = supertrend[-1]
            prev_dir = trend_direction[-1]
            final_ub = ub if ub < prev_st or close[i-1] > prev_st else prev_st
            final_lb = lb if lb > prev_st or close[i-1] < prev_st else prev_st
            
            if prev_dir == 1 and close[i] <= final_lb:
                trend_direction.append(-1)
                supertrend.append(final_ub)
            elif prev_dir == -1 and close[i] >= final_ub:
                trend_direction.append(1)
                supertrend.append(final_lb)
            else:
                trend_direction.append(prev_dir)
                supertrend.append(final_lb if prev_dir == 1 else final_ub)
    
    return supertrend, trend_direction

def main():
    st.title("Single Strike Call & Put Options SuperTrend")
    
    if 'option_data' not in st.session_state:
        st.session_state.option_data = {}
    if 'alert_sent' not in st.session_state:
        st.session_state.alert_sent = set()
    
    api = DhanAPI()
    telegram = TelegramBot()
    
    st.sidebar.header("SuperTrend Settings")
    st_period = st.sidebar.slider("Period", 5, 25, 10)
    st_multiplier = st.sidebar.slider("Multiplier", 1.0, 5.0, 3.0, 0.1)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox("Refresh (seconds)", [5, 10, 15, 30], index=1)
    
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry = st.sidebar.selectbox("Expiry", expiry_data['data'])
    else:
        expiry = "2024-10-31"
    
    # Buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Refresh"):
            st.experimental_rerun()
    with col2:
        if st.button("Clear Data"):
            st.session_state.option_data = {}
            st.success("Cleared!")
            st.experimental_rerun()
    
    option_chain = api.get_option_chain(13, "IDX_I", expiry)
    
    if not (option_chain and 'data' in option_chain):
        st.error("Failed to fetch option chain data")
        return
    
    spot_price = option_chain['data'].get('last_price', 25000)
    atm_strike = round(spot_price / 50) * 50
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nifty Spot", f"₹{spot_price:.2f}")
    with col2:
        st.metric("ATM Strike", atm_strike)
    with col3:
        ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
        st.metric("IST Time", ist_time)
    with col4:
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
        market_status = "OPEN" if market_open <= ist_now <= market_close else "CLOSED"
        st.metric("Market", market_status)
    
    oc_data = option_chain['data'].get('oc', {})
    
    # Prepare one strike data (ATM selected by user)
    strikes_options = []
    strike_key = f"{atm_strike}.000000"
    
    if strike_key not in oc_data:
        st.error(f"No data found for strike {atm_strike}")
        return
    
    strike_data = oc_data[strike_key]
    # Collect CE and PE options data if available and LTP > 0
    if strike_data.get('ce', {}).get('last_price', 0) > 0:
        strikes_options.append({'strike': atm_strike, 'type': 'CE', 'ltp': strike_data['ce']['last_price']})
    if strike_data.get('pe', {}).get('last_price', 0) > 0:
        strikes_options.append({'strike': atm_strike, 'type': 'PE', 'ltp': strike_data['pe']['last_price']})
    
    if not strikes_options:
        st.error("No active CE or PE options found at ATM strike")
        return
    
    with st.spinner("Loading 3 days historical data..."):
        for opt in strikes_options:
            option_key = f"{opt['strike']}_{opt['type']}"
            historical_data = get_3day_historical_data(api, opt)
            if historical_data:
                st.session_state.option_data[option_key] = historical_data
            else:
                st.warning(f"No historical data for {option_key}")
    
    # Plot options chart with SuperTrend for CE only
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=["Call Option (CE)", "Put Option (PE)"])
    colors = ['#2196F3', '#FF9800']
    
    # Plot CE
    ce_data = st.session_state.option_data.get(f"{atm_strike}_CE", [])
    if ce_data:
        df_ce = pd.DataFrame(ce_data)
        fig.add_trace(go.Candlestick(
            x=df_ce['timestamp'],
            open=df_ce['open'], high=df_ce['high'], low=df_ce['low'], close=df_ce['close'],
            name=f"{atm_strike} CE",
            increasing_line_color=colors[0], decreasing_line_color=colors[0],
            showlegend=True), row=1, col=1)
        
        if len(df_ce) >= st_period + 2:
            supertrend, trend_direction = calculate_supertrend(
                df_ce['high'].tolist(), df_ce['low'].tolist(), df_ce['close'].tolist(),
                period=st_period, multiplier=st_multiplier)
            st_up = [st if td == 1 else None for st, td in zip(supertrend, trend_direction)]
            st_down = [st if td == -1 else None for st, td in zip(supertrend, trend_direction)]
            fig.add_trace(go.Scatter(
                x=df_ce['timestamp'], y=st_up, mode='lines',
                name='SuperTrend Up', line=dict(color='#00ff88', width=2), connectgaps=False), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_ce['timestamp'], y=st_down, mode='lines',
                name='SuperTrend Down', line=dict(color='#ff4444', width=2), connectgaps=False), row=1, col=1)
    
    # Plot PE
    pe_data = st.session_state.option_data.get(f"{atm_strike}_PE", [])
    if pe_data:
        df_pe = pd.DataFrame(pe_data)
        fig.add_trace(go.Candlestick(
            x=df_pe['timestamp'],
            open=df_pe['open'], high=df_pe['high'], low=df_pe['low'], close=df_pe['close'],
            name=f"{atm_strike} PE",
            increasing_line_color=colors[1], decreasing_line_color=colors[1],
            showlegend=True), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=800,
        title=f"Strike {atm_strike} Call & Put Options - 3 Days Historical LTP with SuperTrend",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    total_points = sum(len(data) for data in st.session_state.option_data.values())
    all_times = [pt['timestamp'] for data in st.session_state.option_data.values() for pt in data if data]
    if all_times:
        first_time = min(all_times).strftime('%d-%m %H:%M')
        last_time = max(all_times).strftime('%d-%m %H:%M')
        st.info(f"📊 Showing 3-day historical data: {first_time} to {last_time} | Total points: {total_points}")
    else:
        st.info(f"Tracking strike {atm_strike} call & put options | Total data points: {total_points}")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
