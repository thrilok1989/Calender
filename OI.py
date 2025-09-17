import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))

# Primary instrument (for price analysis and options)
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

# Secondary instrument (for volume data) - NIFTY Current Month Futures
NIFTY_FUTURES_SCRIP = 53001  # September 2025 NIFTY Futures - Updated from CSV
NIFTY_FUTURES_SEG = "NSE_FNO"

VP_CONFIG = {"BINS": 20, "TIMEFRAME": "30T", "POC_COLOR": "#FFD700", "VOLUME_COLOR": "#1f77b4"}

def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5: return False
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

# Only run autorefresh during market hours with longer interval to avoid rate limits
if is_market_hours():
    st_autorefresh(interval=120000, key="refresh")  # Increased to 2 minutes

import time

class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
        self.last_call_time = 0
    
    def _rate_limit_delay(self):
        """Ensure minimum 1 second between API calls"""
        current_time = time.time()
        time_diff = current_time - self.last_call_time
        if time_diff < 1.0:
            time.sleep(1.0 - time_diff)
        self.last_call_time = time.time()
    
    def get_intraday_data(self, interval="5", days_back=1, scrip_id=None, segment=None, instrument="INDEX"):
        """Get intraday data for any instrument"""
        self._rate_limit_delay()  # Add delay
        
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        # Use provided scrip/segment or default to NIFTY Index
        if scrip_id is None:
            scrip_id = NIFTY_SCRIP
        if segment is None:
            segment = NIFTY_SEG
        
        payload = {
            "securityId": str(scrip_id),
            "exchangeSegment": segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except: return None
    
    def get_futures_volume_data(self, interval="5", days_back=1):
        """Get volume data from NIFTY Futures"""
        return self.get_intraday_data(
            interval=interval, 
            days_back=days_back,
            scrip_id=NIFTY_FUTURES_SCRIP,
            segment=NIFTY_FUTURES_SEG,
            instrument="FUTIDX"
        )
    
    def test_api_connection(self):
        """Test API connectivity and credentials"""
        self._rate_limit_delay()  # Add delay
        try:
            # Test with a simple LTP call
            url = "https://api.dhan.co/v2/marketfeed/ltp"
            payload = {NIFTY_SEG: [NIFTY_SCRIP]}
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True, "API connection successful"
            elif response.status_code == 401:
                return False, "Authentication failed - Check credentials"
            elif response.status_code == 429:
                return False, "Rate limit exceeded - Wait and retry"
            else:
                return False, f"API error: {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Request timeout - Check network connection"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - Check internet connectivity"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def find_current_nifty_futures(self):
        """Find current month NIFTY futures scrip ID"""
        try:
            # Get instrument list from Dhan API
            url = "https://api.dhan.co/v2/instrument/NSE_FNO"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse CSV data
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    headers = lines[0].split(',')
                    
                    # Find NIFTY futures with nearest expiry
                    nifty_futures = []
                    for line in lines[1:]:
                        fields = line.split(',')
                        if len(fields) > 10:
                            # Look for NIFTY futures (FUTIDX instrument)
                            if 'NIFTY' in fields and 'FUTIDX' in fields:
                                try:
                                    scrip_id = int(fields[0])  # Assuming first column is scrip ID
                                    symbol = fields[2] if len(fields) > 2 else ""
                                    expiry = fields[8] if len(fields) > 8 else ""
                                    nifty_futures.append({
                                        'scrip_id': scrip_id,
                                        'symbol': symbol,
                                        'expiry': expiry
                                    })
                                except (ValueError, IndexError):
                                    continue
                    
                    # Return the first NIFTY futures found (usually current month)
                    if nifty_futures:
                        return nifty_futures[0]
                        
        except Exception as e:
            st.error(f"Error finding futures scrip: {e}")
        
        return None
    
    def get_ltp_data(self):
        self._rate_limit_delay()  # Add delay
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except: return None

def get_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except: return None

def get_expiry_list():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except: return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def process_candle_data(data):
    if not data or 'open' not in data: return pd.DataFrame()
    
    df = pd.DataFrame({
        'timestamp': data['timestamp'], 'open': data['open'], 'high': data['high'],
        'low': data['low'], 'close': data['close'], 'volume': data['volume']
    })
    
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
    return df

def combine_index_futures_data(index_data, futures_data):
    """Combine NIFTY Index price data with Futures volume data"""
    
    # Debug: Check what data we received
    st.write("**Debug - Data Check:**")
    st.write(f"Index data received: {index_data is not None}")
    st.write(f"Futures data received: {futures_data is not None}")
    
    if index_data:
        st.write(f"Index data keys: {list(index_data.keys()) if isinstance(index_data, dict) else 'Not a dict'}")
    if futures_data:
        st.write(f"Futures data keys: {list(futures_data.keys()) if isinstance(futures_data, dict) else 'Not a dict'}")
    
    # Try to process index data first (primary source)
    index_df = pd.DataFrame()
    if index_data:
        try:
            index_df = process_candle_data(index_data)
            st.success(f"Index data processed: {len(index_df)} records")
        except Exception as e:
            st.error(f"Index data processing failed: {e}")
    
    # If index data failed, return empty DataFrame
    if index_df.empty:
        st.error("Primary NIFTY Index data is empty or failed to process")
        return pd.DataFrame()
    
    # Try to get futures volume
    futures_df = pd.DataFrame()
    if futures_data:
        try:
            futures_df = process_candle_data(futures_data)
            st.success(f"Futures data processed: {len(futures_df)} records")
        except Exception as e:
            st.error(f"Futures data processing failed: {e}")
    
    # Create combined DataFrame with Index prices
    combined_df = index_df.copy()
    
    # Replace volume with futures volume if available and valid
    if not futures_df.empty and 'volume' in futures_df.columns and futures_df['volume'].sum() > 0:
        try:
            # Align timestamps and use futures volume
            futures_volume = futures_df.set_index('datetime')['volume']
            combined_df = combined_df.set_index('datetime')
            
            # Match volumes by timestamp (forward fill for missing timestamps)
            combined_df['volume'] = futures_volume.reindex(combined_df.index, method='ffill').fillna(0)
            combined_df = combined_df.reset_index()
            
            # Add metadata
            combined_df.attrs['volume_source'] = 'futures'
            combined_df.attrs['futures_scrip'] = NIFTY_FUTURES_SCRIP
            st.info(f"Using futures volume: {combined_df['volume'].sum():,.0f} total")
        except Exception as e:
            st.warning(f"Failed to combine futures volume: {e}")
            combined_df.attrs['volume_source'] = 'index'
    else:
        combined_df.attrs['volume_source'] = 'index'
        st.warning("Using index volume (may be limited)")
    
    return combined_df

def calculate_volume_profile(df, bins=20):
    if df.empty or len(df) < 10 or 'volume' not in df.columns: return []
    
    df_copy = df[df['volume'].notna() & (df['volume'] > 0)]
    if df_copy.empty: return []
    
    overall_high, overall_low = df_copy['high'].max(), df_copy['low'].min()
    if overall_high == overall_low: return []
    
    price_bins = np.linspace(overall_low, overall_high, bins + 1)
    volume_hist = np.zeros(bins)
    
    for _, row in df_copy.iterrows():
        if pd.isna(row['volume']) or row['volume'] <= 0: continue
        typical_price = (row['high'] + row['low'] + row['close']) / 3
        volume = row['volume']
        bin_idx = np.digitize(typical_price, price_bins) - 1
        bin_idx = max(0, min(bins - 1, bin_idx))
        volume_hist[bin_idx] += volume
    
    if volume_hist.sum() == 0: return []
    
    poc_idx = np.argmax(volume_hist)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    
    total_volume = volume_hist.sum()
    sorted_indices = np.argsort(volume_hist)[::-1]
    cumulative_volume = 0
    value_area_indices = []
    
    for bin_idx in sorted_indices:
        cumulative_volume += volume_hist[bin_idx]
        value_area_indices.append(bin_idx)
        if cumulative_volume >= 0.7 * total_volume: break
    
    if value_area_indices:
        va_high = max([price_bins[i + 1] for i in value_area_indices])
        va_low = min([price_bins[i] for i in value_area_indices])
    else:
        va_high, va_low = overall_high, overall_low
    
    return [{
        "time": df_copy['datetime'].iloc[0], "high": overall_high, "low": overall_low,
        "poc": poc_price, "va_high": va_high, "va_low": va_low,
        "volume_hist": volume_hist, "price_bins": price_bins, "total_volume": total_volume
    }]

def get_volume_profile_insights(profiles, current_price):
    if not profiles or current_price is None: return {}
    
    profile = profiles[0]
    insights = {
        "poc": profile["poc"], "va_high": profile["va_high"], "va_low": profile["va_low"],
        "total_volume": profile["total_volume"], "bias": "NEUTRAL", "strength": "WEAK"
    }
    
    poc_diff = current_price - insights["poc"]
    poc_diff_pct = (poc_diff / insights["poc"]) * 100 if insights["poc"] > 0 else 0
    
    if abs(poc_diff_pct) < 0.1:
        insights["price_vs_poc"] = f"AT POC (±{poc_diff:+.1f})"
        insights["bias"] = "NEUTRAL"
    elif poc_diff > 0:
        insights["price_vs_poc"] = f"ABOVE POC (+{poc_diff:.1f}, +{poc_diff_pct:.2f}%)"
        insights["bias"] = "BULLISH"
    else:
        insights["price_vs_poc"] = f"BELOW POC ({poc_diff:.1f}, {poc_diff_pct:.2f}%)"
        insights["bias"] = "BEARISH"
    
    if current_price > insights["va_high"]:
        insights["price_vs_va"] = f"ABOVE VA (+{current_price - insights['va_high']:.1f})"
        if insights["bias"] == "BULLISH": insights["strength"] = "STRONG"
    elif current_price < insights["va_low"]:
        insights["price_vs_va"] = f"BELOW VA (-{insights['va_low'] - current_price:.1f})"
        if insights["bias"] == "BEARISH": insights["strength"] = "STRONG"
    else:
        insights["price_vs_va"] = "INSIDE VALUE AREA"
        insights["strength"] = "MODERATE" if insights["bias"] != "NEUTRAL" else "WEAK"
    
    return insights

def get_pivots(df, timeframe="5", length=4):
    if df.empty: return []
    
    rule_map = {"3": "3min", "5": "5min", "10": "10min", "15": "15min"}
    rule = rule_map.get(timeframe, "5min")
    
    df_temp = df.set_index('datetime')
    try:
        resampled = df_temp.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(resampled) < length * 2 + 1: return []
        
        max_vals = resampled['high'].rolling(window=length*2+1, center=True).max()
        min_vals = resampled['low'].rolling(window=length*2+1, center=True).min()
        
        pivots = []
        for timestamp, value in resampled['high'][resampled['high'] == max_vals].items():
            pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        for timestamp, value in resampled['low'][resampled['low'] == min_vals].items():
            pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        return pivots
    except: return []

def create_chart(df, title):
    if df.empty: 
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    # Debug: Check data availability
    has_volume = 'volume' in df.columns and df['volume'].sum() > 0
    
    # Create subplots: Main chart + Volume chart + VP histogram
    fig = make_subplots(
        rows=2, cols=2, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        column_widths=[0.75, 0.25],
        row_heights=[0.65, 0.35],
        subplot_titles=('Price Chart', 'Volume Profile', 'Volume', ''),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"colspan": 2, "secondary_y": False}, None]]
    )
    
    # 1. Main Candlestick Chart (row=1, col=1)
    fig.add_trace(go.Candlestick(
        x=df['datetime'], 
        open=df['open'], 
        high=df['high'], 
        low=df['low'], 
        close=df['close'],
        name='Nifty',
        increasing_line_color='#00ff88', 
        decreasing_line_color='#ff4444',
        showlegend=False
    ), row=1, col=1)
    
    # 2. Volume Chart (row=2, col=1 - spans both columns)
    if has_volume:
        volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                        for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['datetime'], 
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        ), row=2, col=1)
    else:
        # Show message if no volume data
        fig.add_annotation(
            text="No Volume Data Available",
            xref="paper", yref="paper",
            x=0.5, y=0.2,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    # 3. Volume Profile Analysis and Visualization
    if has_volume and len(df) > 20:
        try:
            profiles = calculate_volume_profile(df, VP_CONFIG["BINS"])
            
            if profiles and len(profiles) > 0:
                profile = profiles[0]
                
                # Add POC line to main chart
                fig.add_hline(
                    y=profile["poc"], 
                    line_dash="dash", 
                    line_color=VP_CONFIG["POC_COLOR"],
                    line_width=2,
                    annotation_text=f"POC: ₹{profile['poc']:.1f}",
                    annotation_position="top right",
                    row=1, col=1
                )
                
                # Add Value Area rectangle to main chart
                fig.add_hrect(
                    y0=profile["va_low"], 
                    y1=profile["va_high"],
                    fillcolor="rgba(255, 215, 0, 0.1)",
                    line_color="rgba(255, 215, 0, 0.3)",
                    line_width=1,
                    annotation_text=f"VA: ₹{profile['va_low']:.0f}-₹{profile['va_high']:.0f}",
                    annotation_position="bottom right",
                    row=1, col=1
                )
                
                # Add Volume Profile Histogram (row=1, col=2)
                price_centers = (profile["price_bins"][:-1] + profile["price_bins"][1:]) / 2
                volume_hist = profile["volume_hist"]
                
                if len(volume_hist) > 0 and volume_hist.max() > 0:
                    # Normalize volume for better visualization
                    max_vol = volume_hist.max()
                    normalized_vol = (volume_hist / max_vol) * 100
                    
                    # Volume Profile bars (horizontal)
                    fig.add_trace(go.Bar(
                        x=normalized_vol,
                        y=price_centers,
                        orientation='h',
                        name='VP',
                        marker_color=VP_CONFIG["VOLUME_COLOR"],
                        opacity=0.6,
                        showlegend=False,
                        text=[f'{int(v):,}' if v > max_vol*0.3 else '' for v in volume_hist],
                        textposition='middle right',
                        textfont=dict(size=8, color='white')
                    ), row=1, col=2)
                    
                    # Highlight POC in VP histogram
                    poc_idx = np.argmax(volume_hist)
                    if poc_idx < len(normalized_vol):
                        fig.add_trace(go.Bar(
                            x=[normalized_vol[poc_idx]],
                            y=[price_centers[poc_idx]],
                            orientation='h',
                            name='POC',
                            marker_color=VP_CONFIG["POC_COLOR"],
                            opacity=0.9,
                            showlegend=False
                        ), row=1, col=2)
                    
        except Exception as e:
            # Add error message to chart
            fig.add_annotation(
                text=f"VP Error: {str(e)[:50]}...",
                xref="paper", yref="paper",
                x=0.9, y=0.9,
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="rgba(0,0,0,0.7)"
            )
    
    # 4. Add Pivot Lines
    if len(df) > 30:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", "#4444ff"]
        
        for tf, color in zip(timeframes, colors):
            try:
                pivots = get_pivots(df, tf)
                # Show only the most recent 2 pivots to avoid clutter
                for pivot in pivots[-2:]:
                    fig.add_hline(
                        y=pivot['value'],
                        line_dash="dot",
                        line_color=color,
                        line_width=1,
                        opacity=0.6,
                        row=1, col=1
                    )
            except:
                continue
    
    # 5. Update Layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        template='plotly_dark',
        height=700,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=2)
    fig.update_xaxes(title_text="Volume %", row=1, col=2)
    
    return fig

def analyze_options(expiry):
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data: return None, None
    
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
            "Strike": row['strikePrice'], "Zone": row['Zone'], "Level": level,
            "ChgOI_Bias": chg_oi_bias, "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias, "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE']
        })
    
    return underlying, pd.DataFrame(bias_results)

def check_signals(df, option_data, current_price, proximity=5):
    if df.empty or option_data is None or not current_price: return
    
    profiles = calculate_volume_profile(df, VP_CONFIG["BINS"])
    vp_insights = get_volume_profile_insights(profiles, current_price)
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty: return
    
    row = atm_data.iloc[0]
    
    bias_aligned_bullish = (
        row['ChgOI_Bias'] == 'Bullish' and row['Volume_Bias'] == 'Bullish' and
        row['Ask_Bias'] == 'Bullish' and row['Bid_Bias'] == 'Bullish'
    )
    
    bias_aligned_bearish = (
        row['ChgOI_Bias'] == 'Bearish' and row['Volume_Bias'] == 'Bearish' and
        row['Ask_Bias'] == 'Bearish' and row['Bid_Bias'] == 'Bearish'
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
        
        vp_confirmation = ""
        if vp_insights:
            if primary_bullish_signal and vp_insights["bias"] == "BULLISH":
                vp_confirmation = f"✅ VP CONFIRMS: {vp_insights['price_vs_poc']} | {vp_insights['strength']} {vp_insights['bias']}"
            elif primary_bearish_signal and vp_insights["bias"] == "BEARISH":
                vp_confirmation = f"✅ VP CONFIRMS: {vp_insights['price_vs_poc']} | {vp_insights['strength']} {vp_insights['bias']}"
            else:
                vp_confirmation = f"⚠️ VP MIXED: {vp_insights['price_vs_poc']} | {vp_insights['bias']} bias"
        
        if primary_bullish_signal or primary_bearish_signal:
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}

💰 VOLUME PROFILE: {vp_confirmation}
📊 POC: ₹{vp_insights.get('poc', 0):.1f} | VA: ₹{vp_insights.get('va_low', 0):.1f}-₹{vp_insights.get('va_high', 0):.1f}

Conditions: {row['Level']}, All Bias Aligned
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success(f"🔔 PRIMARY {signal_type} signal sent!")
    
    # SECONDARY SIGNAL
    ce_chg_oi = abs(row.get('changeinOpenInterest_CE', 0))
    pe_chg_oi = abs(row.get('changeinOpenInterest_PE', 0))
    
    put_dominance = pe_chg_oi > 1.3 * ce_chg_oi if ce_chg_oi > 0 else False
    call_dominance = ce_chg_oi > 1.3 * pe_chg_oi if pe_chg_oi > 0 else False
    
    secondary_bullish_signal = (bias_aligned_bullish and put_dominance)
    secondary_bearish_signal = (bias_aligned_bearish and call_dominance)
    
    if secondary_bullish_signal or secondary_bearish_signal:
        signal_type = "CALL" if secondary_bullish_signal else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if secondary_bullish_signal and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        vp_info = ""
        if vp_insights:
            vp_info = f"""
💰 VOLUME PROFILE:
📊 {vp_insights.get('price_vs_poc', 'N/A')} | {vp_insights.get('price_vs_va', 'N/A')}
🎯 Bias: {vp_insights.get('strength', 'WEAK')} {vp_insights.get('bias', 'NEUTRAL')}
"""
        
        message = f"""
⚡ SECONDARY NIFTY {signal_type} SIGNAL - OI DOMINANCE ⚡

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

🔥 OI Dominance: {'PUT' if secondary_bullish_signal else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
📊 All Bias Aligned: {row['ChgOI_Bias']}, {row['Volume_Bias']}, {row['Ask_Bias']}, {row['Bid_Bias']}
{vp_info}
ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"⚡ SECONDARY {signal_type} signal sent!")

def debug_data_info(df):
    """Debug function to show data information"""
    if df.empty:
        st.error("DataFrame is empty")
        return
    
    st.subheader("🔍 Data Debug Info")
    
    # Show data source information
    volume_source = df.attrs.get('volume_source', 'unknown')
    if volume_source == 'futures':
        st.info(f"📊 **Data Source**: NIFTY Index prices + Futures volume (Scrip: {df.attrs.get('futures_scrip', 'N/A')})")
    else:
        st.info("📊 **Data Source**: NIFTY Index only")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Columns", len(df.columns))
        st.write("**Columns:**", list(df.columns))
    
    with col2:
        if 'volume' in df.columns:
            valid_volume = df['volume'].notna() & (df['volume'] > 0)
            total_volume = df['volume'].sum()
            st.metric("Valid Volume Records", valid_volume.sum())
            st.metric("Total Volume", f"{total_volume:,.0f}")
            st.metric("Max Volume", f"{df['volume'].max():,.0f}" if df['volume'].max() > 0 else "0")
        else:
            st.error("❌ Volume column missing!")
    
    with col3:
        date_range = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        st.metric("Hours of Data", f"{date_range:.1f}")
        st.metric("Price Range", f"₹{df['low'].min():.1f} - ₹{df['high'].max():.1f}")
        st.metric("Last Price", f"₹{df['close'].iloc[-1]:.2f}")
    
    # Show sample data
    st.subheader("📊 Sample Data")
    sample_df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head(10)
    st.dataframe(sample_df, use_container_width=True)
    
    # Volume statistics
    if 'volume' in df.columns and df['volume'].sum() > 0:
        st.subheader("📈 Volume Statistics")
        vol_stats = df['volume'].describe()
        st.write(vol_stats)
        
        # Check for zero volume records
        zero_vol_count = (df['volume'] == 0).sum()
        if zero_vol_count > 0:
            st.warning(f"⚠️ Found {zero_vol_count} records with zero volume")
        
        # Volume source breakdown
        st.subheader("📊 Volume Data Quality")
        if volume_source == 'futures':
            st.success("✅ Using Futures volume data - should be meaningful for Volume Profile")
        else:
            st.warning("⚠️ Using Index volume data - may be limited")
    else:
        st.error("❌ No valid volume data available for Volume Profile calculation!")

    # Test Volume Profile calculation
    st.subheader("🧪 VP Test")
    try:
        profiles = calculate_volume_profile(df, VP_CONFIG["BINS"])
        if profiles:
            st.success(f"✅ Volume Profile calculated successfully! POC: ₹{profiles[0]['poc']:.1f}")
            st.info(f"📊 VP Insights: Total Volume: {profiles[0]['total_volume']:,.0f}")
        else:
            st.error("❌ Volume Profile calculation failed")
    except Exception as e:
        st.error(f"❌ VP Error: {e}")

def main():
    st.title("📈 Nifty Trading Analyzer with Volume Profile")
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market closed. Time: {current_time.strftime('%H:%M:%S IST')}")
    
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    VP_CONFIG["BINS"] = st.sidebar.slider("VP Bins", 10, 30, 20)
    
    # Volume source option
    st.sidebar.subheader("Volume Data Source")
    use_futures_volume = st.sidebar.checkbox("Use Futures Volume", value=False, 
                                           help="Uses NIFTY Futures volume for Volume Profile calculation")
    
    if use_futures_volume:
        st.sidebar.write(f"Current Futures Scrip ID: **{NIFTY_FUTURES_SCRIP}**")
        
        # Add button to find correct scrip ID
        if st.sidebar.button("Find Current Futures Scrip"):
            with st.sidebar:
                with st.spinner("Searching for current NIFTY futures..."):
                    futures_info = api.find_current_nifty_futures()
                    if futures_info:
                        st.success(f"Found: Scrip {futures_info['scrip_id']}")
                        st.write(f"Symbol: {futures_info['symbol']}")
                        st.write(f"Expiry: {futures_info['expiry']}")
                        st.warning("Update NIFTY_FUTURES_SCRIP in code to use this scrip ID")
                    else:
                        st.error("Could not find current NIFTY futures")
        
        # Manual scrip ID input
        manual_scrip = st.sidebar.number_input(
            "Manual Futures Scrip ID", 
            min_value=1, 
            max_value=99999, 
            value=NIFTY_FUTURES_SCRIP,
            help="Enter correct current month NIFTY futures scrip ID"
        )
        
        if manual_scrip != NIFTY_FUTURES_SCRIP:
            st.sidebar.warning(f"Using manual scrip ID: {manual_scrip}")
            # Temporarily override the global variable
            global NIFTY_FUTURES_SCRIP
            NIFTY_FUTURES_SCRIP = manual_scrip
    
    # Rate limiting option
    st.sidebar.subheader("API Settings")
    manual_refresh = st.sidebar.checkbox("Manual Refresh Only", value=True,
                                       help="Disable auto-refresh to avoid rate limits")
    if manual_refresh:
        refresh_data = st.sidebar.button("🔄 Refresh Data")
    else:
        refresh_data = True
    
    # Add debug option
    show_debug = st.sidebar.checkbox("🔍 Show Debug Info", value=False)
    
    api = DhanAPI()
    
    # Test API connection first
    if show_debug:
        st.subheader("🔌 API Connection Test")
        api_status, api_message = api.test_api_connection()
        if api_status:
            st.success(f"✅ {api_message}")
        else:
            st.error(f"❌ {api_message}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart with Volume Profile")
        
        # Only fetch data if refresh is triggered
        if refresh_data:
            st.write(f"**Refresh triggered - Use Futures Volume: {use_futures_volume}**")
            
            if use_futures_volume:
                # Get both index and futures data
                with st.spinner("Fetching NIFTY Index data..."):
                    index_data = api.get_intraday_data(interval)
                    st.success(f"Index API call completed: {index_data is not None}")
                
                with st.spinner("Waiting 2 seconds to avoid rate limits..."):
                    time.sleep(2)
                
                with st.spinner("Fetching NIFTY Futures volume data..."):
                    st.write(f"Using Futures Scrip ID: {NIFTY_FUTURES_SCRIP}")
                    futures_data = api.get_futures_volume_data(interval)
                    st.success(f"Futures API call completed: {futures_data is not None}")
                    
                    if futures_data:
                        st.write(f"Futures response keys: {list(futures_data.keys())}")
                        if 'volume' in futures_data:
                            vol_sum = sum(futures_data['volume']) if futures_data['volume'] else 0
                            st.write(f"Futures total volume: {vol_sum:,}")
                
                # Combine data with debug output
                df = combine_index_futures_data(index_data, futures_data)
                
                # If combined data failed, fallback to index only
                if df.empty and index_data:
                    st.warning("Combined data failed, falling back to Index data only")
                    df = process_candle_data(index_data)
                    if not df.empty:
                        df.attrs['volume_source'] = 'index_fallback'
                
                # If still empty, try futures only as last resort
                if df.empty and futures_data:
                    st.warning("Index data failed, trying Futures data only")
                    df = process_candle_data(futures_data)
                    if not df.empty:
                        df.attrs['volume_source'] = 'futures_only'
            else:
                # Use index data only
                with st.spinner("Fetching NIFTY Index data only..."):
                    data = api.get_intraday_data(interval)
                    df = process_candle_data(data) if data else pd.DataFrame()
                    if not df.empty:
                        df.attrs['volume_source'] = 'index_only'
            
            # Store data in session state to avoid repeated API calls
            st.session_state.chart_data = df
        else:
            # Use cached data if available
            df = st.session_state.get('chart_data', pd.DataFrame())
            if df.empty:
                st.info("Click 'Refresh Data' button to load data")
                return
        
        # Final check and error message
        if df.empty:
            st.error("❌ **All data sources failed!**")
            st.error("Possible issues:")
            st.error("1. Market is closed")
            st.error("2. Invalid credentials (check DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN)")
            st.error("3. API rate limits exceeded")
            st.error("4. Network connectivity issues")
            st.error("5. Invalid futures scrip ID (needs monthly update)")
            st.info(f"Current futures scrip ID: {NIFTY_FUTURES_SCRIP}")
            st.info("Check https://images.dhan.co/api-data/api-scrip-master.csv for current scrip IDs")
            return  # Exit early if no data
        
        # Show debug info if enabled
        if show_debug:
            debug_data_info(df)
        
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
            
            col1_m, col2_m, col3_m = st.columns(3)
            with col1_m:
                st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("High", f"₹{df['high'].max():,.2f}")
            with col3_m:
                st.metric("Low", f"₹{df['low'].min():,.2f}")
            
            # Volume Profile insights
            if len(df) > 20:
                try:
                    profiles = calculate_volume_profile(df, VP_CONFIG["BINS"])
                    vp_insights = get_volume_profile_insights(profiles, current_price)
                    
                    if vp_insights:
                        st.subheader("💰 Volume Profile Analysis")
                        col_vp1, col_vp2, col_vp3, col_vp4 = st.columns(4)
                        
                        with col_vp1:
                            st.metric("POC", f"₹{vp_insights['poc']:,.1f}")
                        with col_vp2:
                            st.metric("VA High", f"₹{vp_insights['va_high']:,.1f}")
                        with col_vp3:
                            st.metric("VA Low", f"₹{vp_insights['va_low']:,.1f}")
                        with col_vp4:
                            bias_color = "🟢" if vp_insights['bias'] == 'BULLISH' else "🔴" if vp_insights['bias'] == 'BEARISH' else "⚪"
                            st.metric("VP Bias", f"{bias_color} {vp_insights['bias']}")
                        
                        st.info(f"📍 **Position**: {vp_insights['price_vs_poc']} | {vp_insights['price_vs_va']}")
                        st.info(f"💪 **Strength**: {vp_insights['strength']} | **Volume**: {vp_insights['total_volume']:,.0f}")
                except: pass
        
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval}min with Volume Profile")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("Options Analysis")
        
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            selected_expiry = st.selectbox("Expiry", expiry_dates)
            
            underlying_price, option_summary = analyze_options(selected_expiry)
            
            if underlying_price and option_summary is not None:
                st.info(f"Spot: ₹{underlying_price:.2f}")
                st.dataframe(option_summary, use_container_width=True)
                
                if enable_signals and not df.empty and is_market_hours():
                    check_signals(df, option_summary, underlying_price, proximity)
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry data unavailable")
        
        # Volume Profile Trading Levels
        if not df.empty and len(df) > 20:
            st.subheader("📊 VP Trading Levels")
            try:
                profiles = calculate_volume_profile(df, VP_CONFIG["BINS"])
                vp_insights = get_volume_profile_insights(profiles, current_price)
                
                if vp_insights:
                    levels_data = {
                        "Level": ["POC", "VA High", "VA Low"],
                        "Price": [f"₹{vp_insights['poc']:,.1f}", 
                                 f"₹{vp_insights['va_high']:,.1f}", 
                                 f"₹{vp_insights['va_low']:,.1f}"],
                        "Distance": [f"{current_price - vp_insights['poc']:+.1f}", 
                                   f"{current_price - vp_insights['va_high']:+.1f}", 
                                   f"{current_price - vp_insights['va_low']:+.1f}"],
                        "Type": ["Support/Resistance", "Resistance", "Support"]
                    }
                    st.dataframe(pd.DataFrame(levels_data), use_container_width=True)
                    
                    st.subheader("🎯 VP Trading Strategy")
                    if vp_insights['bias'] == 'BULLISH' and vp_insights['strength'] in ['STRONG', 'MODERATE']:
                        st.success("💡 **Bullish Setup**: Price above POC with strong volume support. Look for CALL options on dips to VA Low.")
                    elif vp_insights['bias'] == 'BEARISH' and vp_insights['strength'] in ['STRONG', 'MODERATE']:
                        st.error("💡 **Bearish Setup**: Price below POC with strong volume resistance. Look for PUT options on rallies to VA High.")
                    elif "INSIDE VALUE AREA" in vp_insights['price_vs_va']:
                        st.info("💡 **Range Trading**: Price inside Value Area. Wait for breakout above VA High (bullish) or below VA Low (bearish).")
                    else:
                        st.warning("💡 **Neutral**: No clear VP signal. Wait for price to interact with key VP levels.")
            except Exception as e:
                st.error(f"VP Analysis error: {e}")
    
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    # Enhanced test telegram
    if st.sidebar.button("Test Telegram"):
        if not df.empty and len(df) > 20:
            try:
                profiles = calculate_volume_profile(df, VP_CONFIG["BINS"])
                vp_insights = get_volume_profile_insights(profiles, current_price)
                
                test_message = f"""
🔔 TEST - Nifty Analyzer with Volume Profile

📍 Current Price: ₹{current_price:.2f}
💰 Volume Profile Analysis:
📊 POC: ₹{vp_insights.get('poc', 0):.1f}
📈 VA High: ₹{vp_insights.get('va_high', 0):.1f}
📉 VA Low: ₹{vp_insights.get('va_low', 0):.1f}
🎯 Position: {vp_insights.get('price_vs_poc', 'N/A')}
💪 Bias: {vp_insights.get('strength', 'WEAK')} {vp_insights.get('bias', 'NEUTRAL')}

🕐 {current_time}
"""
            except:
                test_message = "🔔 Test message from Enhanced Nifty Analyzer with Volume Profile"
        else:
            test_message = "🔔 Test message from Enhanced Nifty Analyzer with Volume Profile"
        
        send_telegram(test_message)
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
