import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import numpy as np
import math
from scipy.stats import norm, zscore
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Function to check if it's market hours
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if it's a weekday (Monday to Friday)
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check if current time is between 9:00 AM and 3:45 PM IST
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    
    return market_start <= now <= market_end

# Only run autorefresh during market hours
if is_market_hours():
    st_autorefresh(interval=35000, key="refresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

# API Classes and Helper Functions
class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
    
    def get_intraday_data(self, interval="5", days_back=1):
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "securityId": str(NIFTY_SCRIP),
            "exchangeSegment": NIFTY_SEG,
            "instrument": "INDEX",
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_ltp_data(self):
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None

def get_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_expiry_list():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

# Data Processing Functions
def process_candle_data(data):
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'timestamp': data['timestamp'],
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })
    
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_pivots(df, timeframe="5", length=4):
    if df.empty:
        return []
    
    rule_map = {"3": "3min", "5": "5min", "10": "10min", "15": "15min"}
    rule = rule_map.get(timeframe, "5min")
    
    df_temp = df.set_index('datetime')
    try:
        resampled = df_temp.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(resampled) < length * 2 + 1:
            return []
        
        max_vals = resampled['high'].rolling(window=length*2+1, center=True).max()
        min_vals = resampled['low'].rolling(window=length*2+1, center=True).min()
        
        pivots = []
        for timestamp, value in resampled['high'][resampled['high'] == max_vals].items():
            pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        for timestamp, value in resampled['low'][resampled['low'] == min_vals].items():
            pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        return pivots
    except:
        return []

# Chart Creation Function
def create_chart(df, title):
    if df.empty:
        return go.Figure()
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df)
    
    # Create subplots with 3 rows (price, volume, RSI)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price', 'Volume', 'RSI')
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Volume chart
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    # RSI chart
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['rsi'], name='RSI',
        line=dict(color='#ff9900', width=2)
    ), row=3, col=1)
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            x_start, x_end = df['datetime'].min(), df['datetime'].max()
            
            for pivot in pivots[-5:]:
                fig.add_shape(type="line", x0=x_start, x1=x_end,
                            y0=pivot['value'], y1=pivot['value'],
                            line=dict(color=color, width=1, dash="dash"), row=1, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=800,
                     xaxis_rangeslider_visible=False, showlegend=False)
    
    # Update y-axis for RSI
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig

# Options Analysis Function
def analyze_options(expiry):
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data:
        return None, None
    
    data = option_data['data']
    underlying = data['last_price']
    oc_data = data['oc']
    
    calls, puts = [], []
    for strike, strike_data in oc_data.items():
        if 'ce' in strike_data:
            ce_data = strike_data['ce']
            ce_data['strikePrice'] = float(strike)
            calls.append(ce_data)
        if 'pe' in strike_data:
            pe_data = strike_data['pe']
            pe_data['strikePrice'] = float(strike)
            puts.append(pe_data)
    
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
    
    rename_map = {
        'last_price': 'lastPrice', 'oi': 'openInterest', 'previous_oi': 'previousOpenInterest',
        'top_ask_quantity': 'askQty', 'top_bid_quantity': 'bidQty', 'volume': 'totalTradedVolume'
    }
    for old, new in rename_map.items():
        df.rename(columns={f"{old}_CE": f"{new}_CE", f"{old}_PE": f"{new}_PE"}, inplace=True)
    
    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
    
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 100]
    
    df_filtered['Zone'] = df_filtered['strikePrice'].apply(
        lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM'
    )
    
    bias_results = []
    for _, row in df_filtered.iterrows():
        chg_oi_bias = "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish"
        volume_bias = "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish"
        
        ask_ce = row.get('askQty_CE', 0)
        ask_pe = row.get('askQty_PE', 0)
        bid_ce = row.get('bidQty_CE', 0)
        bid_pe = row.get('bidQty_PE', 0)
        
        ask_bias = "Bearish" if ask_ce > ask_pe else "Bullish"
        bid_bias = "Bullish" if bid_ce > bid_pe else "Bearish"
        
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']
        level = "Support" if pe_oi > 1.12 * ce_oi else "Resistance" if ce_oi > 1.12 * pe_oi else "Neutral"
        
        bias_results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias,
            "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE']
        })
    
    return underlying, pd.DataFrame(bias_results)

# Original Signal Functions (Your existing signals 1-5)
def check_signals(df, option_data, current_price, proximity=5):
    if df.empty or option_data is None or not current_price:
        return
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df)
    current_rsi = df['rsi'].iloc[-1] if not df.empty else None
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    
    ce_chg_oi = abs(row.get('changeinOpenInterest_CE', 0))
    pe_chg_oi = abs(row.get('changeinOpenInterest_PE', 0))
    
    bias_aligned_bullish = (
        row['ChgOI_Bias'] == 'Bullish' and 
        row['Volume_Bias'] == 'Bullish' and
        row['Ask_Bias'] == 'Bullish' and
        row['Bid_Bias'] == 'Bullish'
    )
    
    bias_aligned_bearish = (
        row['ChgOI_Bias'] == 'Bearish' and 
        row['Volume_Bias'] == 'Bearish' and
        row['Ask_Bias'] == 'Bearish' and
        row['Bid_Bias'] == 'Bearish'
    )
    
    # PRIMARY SIGNAL
    pivots = get_pivots(df, "5") + get_pivots(df, "10") + get_pivots(df, "15")
    near_pivot = False
    pivot_level = None
    
    for pivot in pivots:
        if abs(current_price - pivot['value']) <= proximity:
            near_pivot = True
            pivot_level = pivot
            break
    
    if near_pivot:
        primary_bullish_signal = (row['Level'] == 'Support' and bias_aligned_bullish)
        primary_bearish_signal = (row['Level'] == 'Resistance' and bias_aligned_bearish)
        
        if primary_bullish_signal or primary_bearish_signal:
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.2f}

Conditions: {row['Level']}, All Bias Aligned
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success(f"🔔 PRIMARY {signal_type} signal sent!")
    
    # SECONDARY SIGNAL
    put_dominance = pe_chg_oi > 1.3 * ce_chg_oi if ce_chg_oi > 0 else False
    call_dominance = ce_chg_oi > 1.3 * pe_chg_oi if pe_chg_oi > 0 else False
    
    secondary_bullish_signal = (bias_aligned_bullish and put_dominance)
    secondary_bearish_signal = (bias_aligned_bearish and call_dominance)
    
    if secondary_bullish_signal or secondary_bearish_signal:
        signal_type = "CALL" if secondary_bullish_signal else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if secondary_bullish_signal and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        message = f"""
⚡ SECONDARY NIFTY {signal_type} SIGNAL - OI DOMINANCE ⚡

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.2f}

🔥 OI Dominance: {'PUT' if secondary_bullish_signal else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
📊 All Bias Aligned: {row['ChgOI_Bias']}, {row['Volume_Bias']}, {row['Ask_Bias']}, {row['Bid_Bias']}

ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"⚡ SECONDARY {signal_type} signal sent!")
    
    # FOURTH SIGNAL - ALL BIAS ALIGNED
    if bias_aligned_bullish or bias_aligned_bearish:
        signal_type = "CALL" if bias_aligned_bullish else "PUT"
        
        message = f"""
🎯 FOURTH SIGNAL - ALL BIAS ALIGNED {signal_type} 🎯

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.2f}

All ATM Biases Aligned: {signal_type}
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}
PCR: {row['PCR']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"🎯 FOURTH {signal_type} signal sent!")
    
    # RSI EXTREME SIGNALS
    if current_rsi is not None:
        if current_rsi > 70:
            message = f"""
⚠️ RSI OVERBOUGHT SIGNAL ⚠️

📍 Spot: ₹{current_price:.2f}
📊 RSI: {current_rsi:.2f} (Above 70)

RSI indicates overbought conditions. Consider potential reversal or pullback.

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success("⚠️ RSI Overbought signal sent!")
        
        elif current_rsi < 30:
            message = f"""
⚠️ RSI OVERSOLD SIGNAL ⚠️

📍 Spot: ₹{current_price:.2f}
📊 RSI: {current_rsi:.2f} (Below 30)

RSI indicates oversold conditions. Consider potential bounce or reversal.

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success("⚠️ RSI Oversold signal sent!")

# ML Signals Class
class MLSignals:
    def __init__(self):
        self.trend_model = None
        self.historical_oi_data = []  # Store for anomaly detection
        
    def signal_6_trend_prediction(self, df, current_price, option_data):
        """SIGNAL 6: ML Trend Prediction"""
        try:
            if df.empty or len(df) < 30:
                return None
                
            # Calculate features
            df_temp = df.copy()
            df_temp['sma_5'] = df_temp['close'].rolling(5).mean()
            df_temp['sma_20'] = df_temp['close'].rolling(20).mean()
            df_temp['volume_ratio'] = df_temp['volume'] / df_temp['volume'].rolling(10).mean()
            df_temp['price_change'] = df_temp['close'].pct_change()
            df_temp['rsi'] = self._calculate_rsi(df_temp['close'])
            
            # Create target (price direction in next 3 candles)
            df_temp['future_close'] = df_temp['close'].shift(-3)
            df_temp['target'] = (df_temp['future_close'] > df_temp['close']).astype(int)
            
            # Prepare features
            features = ['rsi', 'sma_5', 'sma_20', 'volume_ratio', 'price_change']
            df_clean = df_temp[features + ['target']].dropna()
            
            if len(df_clean) < 20:
                return None
                
            # Train model on recent data
            X = df_clean[features].values
            y = df_clean['target'].values
            
            if len(X) < 10:
                return None
                
            # Use last 50 points for training, predict current
            train_size = min(50, len(X) - 1)
            X_train = X[-train_size-1:-1]
            y_train = y[-train_size-1:-1]
            X_current = X[-1].reshape(1, -1)
            
            # Simple Random Forest
            self.trend_model = RandomForestClassifier(
                n_estimators=20, 
                max_depth=3, 
                random_state=42
            )
            self.trend_model.fit(X_train, y_train)
            
            # Get prediction and confidence
            prediction = self.trend_model.predict(X_current)[0]
            confidence = self.trend_model.predict_proba(X_current)[0].max()
            
            # Only signal if confidence is high
            if confidence > 0.65:
                direction = "CALL" if prediction == 1 else "PUT"
                
                # Get ATM strike for signal
                atm_data = option_data[option_data['Zone'] == 'ATM'] if option_data is not None else None
                atm_strike = atm_data.iloc[0]['Strike'] if atm_data is not None and not atm_data.empty else "N/A"
                
                return {
                    'signal_type': direction,
                    'confidence': round(confidence * 100, 1),
                    'strike': atm_strike,
                    'reason': f"ML Trend Prediction ({confidence*100:.1f}% confidence)"
                }
                
        except Exception as e:
            print(f"Signal 6 error: {e}")
            return None
    
    def signal_7_volume_breakout(self, df, current_price, option_data):
        """SIGNAL 7: Volume Breakout Detection"""
        try:
            if df.empty or len(df) < 20:
                return None
                
            # Calculate volume metrics
            df_temp = df.copy()
            df_temp['volume_sma'] = df_temp['volume'].rolling(20).mean()
            df_temp['volume_ratio'] = df_temp['volume'] / df_temp['volume_sma']
            df_temp['price_change'] = df_temp['close'].pct_change()
            df_temp['high_break'] = df_temp['high'] > df_temp['high'].rolling(10).max().shift(1)
            df_temp['low_break'] = df_temp['low'] < df_temp['low'].rolling(10).min().shift(1)
            
            current_row = df_temp.iloc[-1]
            
            # Volume breakout conditions
            volume_spike = current_row['volume_ratio'] > 1.8  # 80% above average
            significant_move = abs(current_row['price_change']) > 0.003  # 0.3% move
            
            breakout_signal = None
            
            if volume_spike and significant_move:
                if current_row['high_break'] and current_row['price_change'] > 0:
                    breakout_signal = "CALL"
                elif current_row['low_break'] and current_row['price_change'] < 0:
                    breakout_signal = "PUT"
            
            if breakout_signal:
                # Get ATM strike
                atm_data = option_data[option_data['Zone'] == 'ATM'] if option_data is not None else None
                atm_strike = atm_data.iloc[0]['Strike'] if atm_data is not None and not atm_data.empty else "N/A"
                
                return {
                    'signal_type': breakout_signal,
                    'volume_ratio': round(current_row['volume_ratio'], 2),
                    'price_change': round(current_row['price_change'] * 100, 2),
                    'strike': atm_strike,
                    'reason': f"Volume Breakout (Vol: {current_row['volume_ratio']:.1f}x avg)"
                }
                
        except Exception as e:
            print(f"Signal 7 error: {e}")
            return None
    
    def signal_8_options_flow_anomaly(self, option_data, current_price):
        """SIGNAL 8: Unusual Options Flow Detection"""
        try:
            if option_data is None or option_data.empty:
                return None
                
            # Focus on ATM and near ATM strikes
            atm_otm_data = option_data[option_data['Zone'].isin(['ATM', 'OTM'])].copy()
            
            if atm_otm_data.empty:
                return None
            
            # Calculate total change in OI for calls and puts
            total_ce_chg_oi = atm_otm_data['changeinOpenInterest_CE'].sum()
            total_pe_chg_oi = atm_otm_data['changeinOpenInterest_PE'].sum()
            
            total_oi_change = abs(total_ce_chg_oi) + abs(total_pe_chg_oi)
            
            # Store historical data for anomaly detection
            self.historical_oi_data.append({
                'total_oi_change': total_oi_change,
                'ce_chg_oi': total_ce_chg_oi,
                'pe_chg_oi': total_pe_chg_oi
            })
            
            # Keep only last 50 data points
            if len(self.historical_oi_data) > 50:
                self.historical_oi_data = self.historical_oi_data[-50:]
            
            # Need at least 10 data points for anomaly detection
            if len(self.historical_oi_data) < 10:
                return None
            
            # Calculate Z-score for anomaly detection
            recent_totals = [x['total_oi_change'] for x in self.historical_oi_data]
            current_z_score = zscore([total_oi_change] + recent_totals[:-1])[-1]
            
            # Anomaly if Z-score > 2 (2 standard deviations above normal)
            if abs(current_z_score) > 2 and total_oi_change > 30000:
                
                # Determine bias based on which side has more activity
                if abs(total_ce_chg_oi) > abs(total_pe_chg_oi) * 1.5:
                    flow_bias = "PUT"  # Heavy call writing = bearish
                    dominant_flow = "CALL"
                elif abs(total_pe_chg_oi) > abs(total_ce_chg_oi) * 1.5:
                    flow_bias = "CALL"  # Heavy put writing = bullish
                    dominant_flow = "PUT"
                else:
                    return None  # No clear bias
                
                # Get ATM strike
                atm_data = option_data[option_data['Zone'] == 'ATM']
                atm_strike = atm_data.iloc[0]['Strike'] if not atm_data.empty else "N/A"
                
                return {
                    'signal_type': flow_bias,
                    'anomaly_score': round(abs(current_z_score), 2),
                    'total_oi_change': int(total_oi_change),
                    'dominant_flow': dominant_flow,
                    'strike': atm_strike,
                    'reason': f"Unusual {dominant_flow} Flow (Z-score: {abs(current_z_score):.1f})"
                }
                
        except Exception as e:
            print(f"Signal 8 error: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Helper function to calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# ML Signal Integration Function with Debug
def check_ml_signals(df, option_data, current_price, send_telegram_func):
    """
    DEBUG VERSION - Shows ML signal status
    """
    ml_signals = MLSignals()
    
    # Add debug info to main display
    with st.expander("🤖 ML Signals Debug Info", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Signal 6 Status")
            if len(df) < 30:
                st.warning(f"Need 30+ candles (have {len(df)})")
            else:
                signal_6 = ml_signals.signal_6_trend_prediction(df, current_price, option_data)
                if signal_6:
                    st.success(f"🤖 READY: {signal_6['signal_type']} ({signal_6['confidence']}%)")
                    # Send actual signal
                    message = f"""
🤖 SIGNAL 6 - ML TREND PREDICTION 🤖

📍 Spot: ₹{current_price:.2f}
🎯 Strike: {signal_6['strike']}
📊 Direction: {signal_6['signal_type']}
🎯 Confidence: {signal_6['confidence']}%

{signal_6['reason']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
                    send_telegram_func(message)
                    st.success(f"🤖 ML TREND {signal_6['signal_type']} signal sent!")
                else:
                    st.info("Building ML model... (confidence < 65%)")
        
        with col2:
            st.subheader("Signal 7 Status")
            if len(df) < 20:
                st.warning(f"Need 20+ candles (have {len(df)})")
            else:
                # Show current volume ratio
                if not df.empty:
                    current_vol = df['volume'].iloc[-1]
                    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x", "Need 1.8x+")
                
                signal_7 = ml_signals.signal_7_volume_breakout(df, current_price, option_data)
                if signal_7:
                    st.success(f"📈 BREAKOUT: {signal_7['signal_type']}")
                    # Send actual signal
                    message = f"""
📈 SIGNAL 7 - VOLUME BREAKOUT 📈

📍 Spot: ₹{current_price:.2f}
🎯 Strike: {signal_7['strike']}
📊 Direction: {signal_7['signal_type']}
📊 Volume: {signal_7['volume_ratio']}x average
💫 Price Change: {signal_7['price_change']:+.2f}%

{signal_7['reason']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
                    send_telegram_func(message)
                    st.success(f"📈 VOLUME BREAKOUT {signal_7['signal_type']} signal sent!")
                else:
                    st.info("Waiting for volume spike...")
        
        with col3:
            st.subheader("Signal 8 Status")
            data_points = len(ml_signals.historical_oi_data)
            if data_points < 10:
                st.warning(f"Building history... ({data_points}/10)")
            
            if option_data is not None:
                # Show current OI activity
                atm_otm_data = option_data[option_data['Zone'].isin(['ATM', 'OTM'])]
                if not atm_otm_data.empty:
                    total_ce_chg = atm_otm_data['changeinOpenInterest_CE'].sum()
                    total_pe_chg = atm_otm_data['changeinOpenInterest_PE'].sum()
                    total_chg = abs(total_ce_chg) + abs(total_pe_chg)
                    st.metric("Total OI Change", f"{total_chg:,}", "Need 30k+")
            
            signal_8 = ml_signals.signal_8_options_flow_anomaly(option_data, current_price)
            if signal_8:
                st.success(f"🔥 ANOMALY: {signal_8['signal_type']}")
                # Send actual signal
                message = f"""
🔥 SIGNAL 8 - UNUSUAL OPTIONS FLOW 🔥

📍 Spot: ₹{current_price:.2f}
🎯 Strike: {signal_8['strike']}
📊 Direction: {signal_8['signal_type']}
🚨 Anomaly Score: {signal_8['anomaly_score']}
💰 Total OI Change: {signal_8['total_oi_change']:,}
📊 Dominant Flow: {signal_8['dominant_flow']}

{signal_8['reason']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
                send_telegram_func(message)
                st.success(f"🔥 OPTIONS FLOW {signal_8['signal_type']} signal sent!")
            else:
                st.info("No unusual activity detected")

# Main Function
def main():
    st.title("📈 Nifty Trading Analyzer with ML Signals")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")
    
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Original Signals (1-5)", value=True)
    enable_ml_signals = st.sidebar.checkbox("Enable ML Signals (6-8)", value=True)
    
    api = DhanAPI()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart")
        
        data = api.get_intraday_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Calculate RSI if we have data
            current_rsi = None
            if not df.empty:
                df['rsi'] = calculate_rsi(df)
                current_rsi = df['rsi'].iloc[-1]
            
            col1_m, col2_m, col3_m, col4_m = st.columns(4)
            with col1_m:
                st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("High", f"₹{df['high'].max():,.2f}")
            with col3_m:
                st.metric("Low", f"₹{df['low'].min():,.2f}")
            with col4_m:
                st.metric("RSI", f"{current_rsi:.2f}" if current_rsi is not None else "N/A")
        
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("Options")
        
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            selected_expiry = st.selectbox("Expiry", expiry_dates)
            
            underlying_price, option_summary = analyze_options(selected_expiry)
            
            if underlying_price and option_summary is not None:
                st.info(f"Spot: ₹{underlying_price:.2f}")
                st.dataframe(option_summary, use_container_width=True)
                
                # ORIGINAL SIGNALS (1-5)
                if enable_signals and not df.empty and is_market_hours():
                    check_signals(df, option_summary, underlying_price, proximity)
                
                # NEW ML SIGNALS (6-8) WITH DEBUG
                if enable_ml_signals and not df.empty:
                    check_ml_signals(df, option_summary, underlying_price, send_telegram)
                    
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry data unavailable")
    
    # Signal Status Display
    st.sidebar.subheader("📊 Signal Status")
    if enable_signals:
        st.sidebar.success("✅ Original Signals (1-5) Active")
    else:
        st.sidebar.info("⏸️ Original Signals Disabled")
        
    if enable_ml_signals:
        st.sidebar.success("✅ ML Signals (6-8) Active")
    else:
        st.sidebar.info("⏸️ ML Signals Disabled")
    
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    if st.sidebar.button("Test Telegram"):
        send_telegram("🔔 Test message from Nifty Analyzer with ML")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()