import stre  amlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client
import json
import time
import numpy as np

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

class EnhancedNiftyApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"  # Nifty 50 security ID for DhanHQ
        self.vob_zones = []
        
        # Initialize session state
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'sent_rsi_alerts' not in st.session_state:
            st.session_state.sent_rsi_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        if 'app_start_time' not in st.session_state:
            st.session_state.app_start_time = datetime.now()
        
    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
        try:
            self.dhan_token = st.secrets["dhan"]["access_token"]
            self.dhan_client_id = st.secrets["dhan"]["client_id"]
            self.supabase_url = st.secrets["supabase"]["url"]
            self.supabase_key = st.secrets["supabase"]["anon_key"]
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except KeyError as e:
            st.error(f"Missing secret: {e}")
            st.stop()
    
    def setup_supabase(self):
        """Initialize Supabase client"""
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.supabase.table('nifty_data').select("id").limit(1).execute()
        except Exception as e:
            st.warning(f"Supabase connection error: {str(e)}")
            self.supabase = None
    
    def get_dhan_headers(self):
        """Get headers for DhanHQ API calls"""
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.dhan_token,
            'client-id': self.dhan_client_id
        }
    
    def get_nearest_expiry(self):
        """Fetch nearest expiry for Nifty options"""
        payload = {
            "UnderlyingScrip": int(self.nifty_security_id),
            "UnderlyingSeg": "IDX_I"
        }
        
        try:
            response = requests.post(
                "https://api.dhan.co/v2/optionchain/expirylist",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            expiries = response.json().get("data", [])
            return expiries[0] if expiries else None
        except Exception as e:
            st.error(f"Expiry fetch error: {e}")
            return None
    
    def fetch_option_chain(self, expiry):
        """Fetch option chain for Nifty"""
        payload = {
            "UnderlyingScrip": int(self.nifty_security_id),
            "UnderlyingSeg": "IDX_I",
            "Expiry": expiry
        }
        
        try:
            response = requests.post(
                "https://api.dhan.co/v2/optionchain",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            st.error(f"Option chain fetch error: {e}")
            return {}
    
    def analyze_oi_sentiment(self, option_data):
        """Analyze OI sentiment and find ATM strike"""
        if not option_data:
            return None, None, "Neutral"
        
        underlying_price = option_data.get("last_price", 0)
        oc = option_data.get("oc", {})
        
        if not oc:
            return underlying_price, None, "Neutral"
        
        # Find ATM strike
        strikes = list(oc.keys())
        atm_strike = min(strikes, key=lambda x: abs(float(x) - underlying_price))
        
        # Calculate total OI change for all strikes
        total_call_oi_change = 0
        total_put_oi_change = 0
        atm_ce_ltp = 0
        atm_pe_ltp = 0
        
        for strike, data in oc.items():
            ce_data = data.get("ce", {})
            pe_data = data.get("pe", {})
            
            # Calculate OI change (current - previous)
            if ce_data:
                ce_oi_change = ce_data.get("oi", 0) - ce_data.get("previous_oi", 0)
                total_call_oi_change += ce_oi_change
            
            if pe_data:
                pe_oi_change = pe_data.get("oi", 0) - pe_data.get("previous_oi", 0)
                total_put_oi_change += pe_oi_change
            
            # Get ATM LTPs
            if strike == atm_strike:
                atm_ce_ltp = ce_data.get("last_price", 0) if ce_data else 0
                atm_pe_ltp = pe_data.get("last_price", 0) if pe_data else 0
        
        # Determine sentiment
        sentiment = "Neutral"
        if total_put_oi_change >= total_call_oi_change * 1.3:
            sentiment = "Bullish"
        elif total_call_oi_change >= total_put_oi_change * 1.3:
            sentiment = "Bearish"
        
        return {
            'underlying_price': underlying_price,
            'atm_strike': float(atm_strike),
            'atm_ce_ltp': atm_ce_ltp,
            'atm_pe_ltp': atm_pe_ltp,
            'sentiment': sentiment,
            'call_oi_change': total_call_oi_change,
            'put_oi_change': total_put_oi_change
        }
    
    def calculate_ultimate_rsi(self, df, length=14, smooth=14):
        """Calculate Ultimate RSI as per LuxAlgo implementation"""
        if len(df) < length + smooth:
            return pd.Series(index=df.index, dtype=float)
        
        src = df['close']
        
        # Calculate upper and lower using rolling windows
        upper = src.rolling(window=length).max()
        lower = src.rolling(window=length).min()
        
        # Calculate range and difference
        r = upper - lower
        d = src.diff()
        
        # Calculate diff based on conditions
        diff = pd.Series(index=df.index, dtype=float)
        for i in range(1, len(df)):
            if upper.iloc[i] > upper.iloc[i-1]:
                diff.iloc[i] = r.iloc[i]
            elif lower.iloc[i] < lower.iloc[i-1]:
                diff.iloc[i] = -r.iloc[i]
            else:
                diff.iloc[i] = d.iloc[i]
        
        # Calculate RMA (Wilder's moving average)
        def rma(series, period):
            alpha = 1.0 / period
            return series.ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate numerator and denominator
        num = rma(diff, length)
        den = rma(diff.abs(), length)
        
        # Calculate Ultimate RSI
        arsi = (num / den) * 50 + 50
        
        # Calculate signal line
        signal = arsi.ewm(span=smooth, adjust=False).mean()
        
        return arsi, signal
    
    def check_rsi_alerts(self, rsi_value, signal_value, oi_analysis):
        """Check RSI levels and send Telegram alerts"""
        if pd.isna(rsi_value) or pd.isna(signal_value):
            return
        
        current_time = datetime.now(self.ist)
        alert_id = f"rsi_{current_time.strftime('%Y%m%d_%H%M')}"
        
        # Check if we already sent an alert for this time period (avoid spam)
        if alert_id in st.session_state.sent_rsi_alerts:
            return
        
        message = None
        
        if rsi_value >= 80:  # Overbought
            message = f"""🔴 RSI Overbought Alert!
            
📊 Nifty 50 Analysis
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
📈 RSI: {rsi_value:.2f} (Overbought)
📉 Signal: {signal_value:.2f}

💰 Current Price: ₹{oi_analysis['underlying_price']:.2f}
🎯 ATM Strike: {oi_analysis['atm_strike']:.0f}
📊 OI Sentiment: {oi_analysis['sentiment']}

💡 Suggested Trade:
🔻 Consider PE Buy: ₹{oi_analysis['atm_pe_ltp']:.2f}
📊 OI Analysis: Call OI Change: {oi_analysis['call_oi_change']:,} | Put OI Change: {oi_analysis['put_oi_change']:,}"""
            
        elif rsi_value <= 20:  # Oversold
            message = f"""🟢 RSI Oversold Alert!
            
📊 Nifty 50 Analysis
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
📈 RSI: {rsi_value:.2f} (Oversold)
📉 Signal: {signal_value:.2f}

💰 Current Price: ₹{oi_analysis['underlying_price']:.2f}
🎯 ATM Strike: {oi_analysis['atm_strike']:.0f}
📊 OI Sentiment: {oi_analysis['sentiment']}

💡 Suggested Trade:
🔺 Consider CE Buy: ₹{oi_analysis['atm_ce_ltp']:.2f}
📊 OI Analysis: Call OI Change: {oi_analysis['call_oi_change']:,} | Put OI Change: {oi_analysis['put_oi_change']:,}"""
        
        if message and self.send_telegram_message(message):
            st.session_state.sent_rsi_alerts.add(alert_id)
            st.success(f"RSI alert sent at {current_time.strftime('%H:%M:%S')}")
        
        # Clean up old RSI alerts (keep only last 20)
        if len(st.session_state.sent_rsi_alerts) > 20:
            alerts_list = list(st.session_state.sent_rsi_alerts)
            st.session_state.sent_rsi_alerts = set(alerts_list[-10:])
    
    def enhanced_vob_alert(self, zone, oi_analysis):
        """Enhanced VOB alert with OI analysis and trade suggestions"""
        zone_type = zone['type'].title()
        signal_time_str = zone['signal_time'].strftime("%H:%M:%S")
        
        if zone['type'] == 'bullish':
            price_info = f"Base: ₹{zone['base_price']:.2f}\nSupport: ₹{zone['lowest_price']:.2f}"
            suggested_trade = f"🔺 Consider CE Buy: ₹{oi_analysis['atm_ce_ltp']:.2f}"
        else:
            price_info = f"Base: ₹{zone['base_price']:.2f}\nResistance: ₹{zone['highest_price']:.2f}"
            suggested_trade = f"🔻 Consider PE Buy: ₹{oi_analysis['atm_pe_ltp']:.2f}"
        
        message = f"""🚨 New VOB Zone + OI Analysis!

📊 Nifty 50
🔥 VOB Type: {zone_type}
⏰ Time: {signal_time_str} IST
💰 Current Price: ₹{oi_analysis['underlying_price']:.2f}

📈 VOB Levels:
{price_info}

🎯 ATM Strike: {oi_analysis['atm_strike']:.0f}
📊 OI Sentiment: {oi_analysis['sentiment']}
📊 OI Changes: 
   • Call OI: {oi_analysis['call_oi_change']:+,}
   • Put OI: {oi_analysis['put_oi_change']:+,}

💡 Trade Suggestion:
{suggested_trade}

⚠️ Trade at your own risk!"""
        
        return message
    
    def fetch_intraday_data(self, interval="3", days_back=5):
        """Fetch intraday data from DhanHQ API"""
        end_date = datetime.now(self.ist)
        start_date = end_date - timedelta(days=days_back)
        
        from_date = start_date.strftime("%Y-%m-%d 09:15:00")
        to_date = end_date.strftime("%Y-%m-%d 15:30:00")
        
        payload = {
            "securityId": self.nifty_security_id,
            "exchangeSegment": "IDX_I",
            "instrument": "INDEX",
            "interval": interval,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }
        
        try:
            response = requests.post(
                "https://api.dhan.co/v2/charts/intraday",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
    
    def process_data(self, api_data):
        """Process API data into DataFrame"""
        if not api_data or 'open' not in api_data:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': api_data['timestamp'],
            'open': api_data['open'],
            'high': api_data['high'],
            'low': api_data['low'],
            'close': api_data['close'],
            'volume': api_data['volume']
        })
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert(self.ist)
        df = df.set_index('datetime')
        
        return df
    
    def send_telegram_message(self, message):
        """Send message to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Telegram error: {e}")
            return False
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df, period=200):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3
    
    def detect_vob_zones(self, df, length1=5):
        """Detect VOB zones based on Pine Script logic"""
        if len(df) < length1 + 13:
            return []
        
        ema1 = self.calculate_ema(df['close'], length1)
        ema2 = self.calculate_ema(df['close'], length1 + 13)
        atr = self.calculate_atr(df)
        
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        vob_zones = []
        
        for i in df.index:
            if cross_up.loc[i]:
                start_idx = max(0, df.index.get_loc(i) - (length1 + 13))
                lookback_data = df.iloc[start_idx:df.index.get_loc(i)+1]
                
                if not lookback_data.empty:
                    lowest_idx = lookback_data['low'].idxmin()
                    lowest_price = lookback_data.loc[lowest_idx, 'low']
                    base_price = min(lookback_data.loc[lowest_idx, 'open'], 
                                   lookback_data.loc[lowest_idx, 'close'])
                    
                    if pd.notna(atr.loc[i]) and (base_price - lowest_price) < atr.loc[i] * 0.5:
                        base_price = lowest_price + atr.loc[i] * 0.5
                    
                    vob_zones.append({
                        'type': 'bullish',
                        'start_time': lowest_idx,
                        'end_time': i,
                        'base_price': base_price,
                        'lowest_price': lowest_price,
                        'signal_time': i
                    })
            
            elif cross_down.loc[i]:
                start_idx = max(0, df.index.get_loc(i) - (length1 + 13))
                lookback_data = df.iloc[start_idx:df.index.get_loc(i)+1]
                
                if not lookback_data.empty:
                    highest_idx = lookback_data['high'].idxmax()
                    highest_price = lookback_data.loc[highest_idx, 'high']
                    base_price = max(lookback_data.loc[highest_idx, 'open'], 
                                   lookback_data.loc[highest_idx, 'close'])
                    
                    if pd.notna(atr.loc[i]) and (highest_price - base_price) < atr.loc[i] * 0.5:
                        base_price = highest_price - atr.loc[i] * 0.5
                    
                    vob_zones.append({
                        'type': 'bearish',
                        'start_time': highest_idx,
                        'end_time': i,
                        'base_price': base_price,
                        'highest_price': highest_price,
                        'signal_time': i
                    })
        
        return vob_zones
    
    def check_new_vob_zones(self, current_zones, oi_analysis):
        """Check for new VOB zones and send enhanced Telegram alerts"""
        if not current_zones or not oi_analysis:
            return
        
        new_alerts_sent = 0
        
        for zone in current_zones:
            zone_id = f"{zone['type']}_{zone['signal_time'].isoformat()}_{zone['base_price']:.2f}"
            
            if zone_id not in st.session_state.sent_vob_alerts:
                zone_age_minutes = (datetime.now(self.ist) - zone['signal_time']).total_seconds() / 60
                
                if zone_age_minutes <= 5:
                    message = self.enhanced_vob_alert(zone, oi_analysis)
                    
                    if self.send_telegram_message(message):
                        st.success(f"Enhanced VOB alert sent for {zone['type']} at {zone['signal_time'].strftime('%H:%M:%S')}")
                        st.session_state.sent_vob_alerts.add(zone_id)
                        new_alerts_sent += 1
                else:
                    st.session_state.sent_vob_alerts.add(zone_id)
        
        # Clean up old alerts
        if len(st.session_state.sent_vob_alerts) > 100:
            alerts_list = list(st.session_state.sent_vob_alerts)
            st.session_state.sent_vob_alerts = set(alerts_list[-50:])
        
        if new_alerts_sent > 0:
            st.info(f"Sent {new_alerts_sent} enhanced VOB alert(s)")
    
    def save_to_supabase(self, df, interval):
        """Save data to Supabase"""
        if df.empty or not self.supabase:
            return
        
        try:
            records = []
            for idx, row in df.iterrows():
                records.append({
                    'datetime': idx.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'interval': interval,
                    'symbol': 'NIFTY50'
                })
            
            self.supabase.table('nifty_data').upsert(records).execute()
            
        except Exception as e:
            st.warning(f"Database save error: {e}")
    
    def load_from_supabase(self, interval, hours_back=24):
        """Load data from Supabase"""
        if not self.supabase:
            return pd.DataFrame()
            
        try:
            cutoff_time = (datetime.now(self.ist) - timedelta(hours=hours_back)).isoformat()
            
            response = self.supabase.table('nifty_data')\
                .select("*")\
                .eq('interval', str(interval))\
                .eq('symbol', 'NIFTY50')\
                .gte('datetime', cutoff_time)\
                .order('datetime')\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            st.warning(f"Database load error: {str(e)}")
        
        return pd.DataFrame()
    
    def create_enhanced_chart(self, df, interval, vob_zones=None, rsi_data=None):
        """Create enhanced chart with VOB zones and RSI"""
        if df.empty:
            return None
        
        # Create subplots: Price, Volume, RSI
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Nifty 50 Price Action with VOB Zones', 'Volume', 'Ultimate RSI'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Nifty 50',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add VOB zones
        if vob_zones:
            for zone in vob_zones[-10:]:
                color = '#26ba9f' if zone['type'] == 'bullish' else '#ba2646'
                
                fig.add_shape(
                    type="line",
                    x0=zone['start_time'], y0=zone['base_price'],
                    x1=zone['end_time'], y1=zone['base_price'],
                    line=dict(color=color, width=2),
                    row=1, col=1
                )
                
                if zone['type'] == 'bullish':
                    support_price = zone['lowest_price']
                    fig.add_shape(
                        type="line",
                        x0=zone['start_time'], y0=support_price,
                        x1=zone['end_time'], y1=support_price,
                        line=dict(color=color, width=2),
                        row=1, col=1
                    )
                    fig.add_shape(
                        type="rect",
                        x0=zone['start_time'], y0=support_price,
                        x1=zone['end_time'], y1=zone['base_price'],
                        fillcolor=color,
                        opacity=0.1,
                        line_width=0,
                        row=1, col=1
                    )
                else:
                    resistance_price = zone['highest_price']
                    fig.add_shape(
                        type="line",
                        x0=zone['start_time'], y0=resistance_price,
                        x1=zone['end_time'], y1=resistance_price,
                        line=dict(color=color, width=2),
                        row=1, col=1
                    )
                    fig.add_shape(
                        type="rect",
                        x0=zone['start_time'], y0=zone['base_price'],
                        x1=zone['end_time'], y1=resistance_price,
                        fillcolor=color,
                        opacity=0.1,
                        line_width=0,
                        row=1, col=1
                    )
        
        # Volume bars
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # RSI subplot
        if rsi_data is not None:
            rsi, signal = rsi_data
            
            # RSI line
            fig.add_trace(
                go.Scatter(
                    x=rsi.index,
                    y=rsi,
                    mode='lines',
                    name='Ultimate RSI',
                    line=dict(color='#ffffff', width=2)
                ),
                row=3, col=1
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=signal.index,
                    y=signal,
                    mode='lines',
                    name='Signal',
                    line=dict(color='#ff5d00', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=80, line_dash="dash", line_color="#089981", row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="#f23645", row=3, col=1)
            
            # Fill overbought/oversold areas
            fig.add_shape(
                type="rect",
                x0=rsi.index[0], y0=80, x1=rsi.index[-1], y1=100,
                fillcolor="#089981", opacity=0.1, line_width=0,
                row=3, col=1
            )
            fig.add_shape(
                type="rect",
                x0=rsi.index[0], y0=0, x1=rsi.index[-1], y1=20,
                fillcolor="#f23645", opacity=0.1, line_width=0,
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Enhanced Nifty 50 Analysis - {interval} Min",
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=900,
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', side="right")
        
        # Set RSI y-axis range
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        return fig
    
    def run_for_25_seconds(self):
        """Run the app for exactly 25 seconds then stop"""
        start_time = time.time()
        end_time = start_time + 25
        
        # Create a placeholder for dynamic updates
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = 25 - elapsed
            
            # Update status
            with status_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"⏱️ Running: {elapsed:.1f}s / 25s")
                with col2:
                    st.info(f"🔄 Iteration: {iteration}")
                with col3:
                    st.info(f"⏰ Remaining: {remaining:.1f}s")
            
            try:
                # Fetch fresh data each iteration
                api_data = self.fetch_intraday_data(interval=st.session_state.get('timeframe', '3'))
                if api_data:
                    df = self.process_data(api_data)
                    
                    if not df.empty:
                        # Calculate indicators
                        vob_zones = []
                        oi_analysis = None
                        rsi_data = None
                        
                        # VOB calculation
                        if st.session_state.get('vob_enabled', True) and len(df) >= 18:
                            vob_zones = self.detect_vob_zones(df, length1=st.session_state.get('vob_sensitivity', 5))
                        
                        # OI Analysis
                        if st.session_state.get('oi_enabled', True):
                            try:
                                expiry = self.get_nearest_expiry()
                                if expiry:
                                    option_data = self.fetch_option_chain(expiry)
                                    if option_data:
                                        oi_analysis = self.analyze_oi_sentiment(option_data)
                            except:
                                pass
                        
                        # RSI calculation
                        rsi_length = st.session_state.get('rsi_length', 14)
                        rsi_smooth = st.session_state.get('rsi_smooth', 14)
                        if st.session_state.get('rsi_enabled', True) and len(df) >= rsi_length + rsi_smooth:
                            rsi_data = self.calculate_ultimate_rsi(df, rsi_length, rsi_smooth)
                        
                        # Update metrics
                        with metrics_placeholder.container():
                            self.display_metrics(df, rsi_data, oi_analysis, vob_zones)
                        
                        # Update chart
                        with chart_placeholder.container():
                            chart = self.create_enhanced_chart(df, st.session_state.get('timeframe', '3'), vob_zones, rsi_data)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                        
                        # Check for alerts
                        if st.session_state.get('telegram_enabled', False):
                            if vob_zones and oi_analysis:
                                self.check_new_vob_zones(vob_zones, oi_analysis)
                            if rsi_data and rsi_data[0] is not None and oi_analysis:
                                latest_rsi = rsi_data[0].iloc[-1]
                                latest_signal = rsi_data[1].iloc[-1]
                                self.check_rsi_alerts(latest_rsi, latest_signal, oi_analysis)
            
            except Exception as e:
                st.error(f"Error in iteration {iteration}: {str(e)}")
            
            # Wait 3 seconds before next iteration (to avoid too frequent API calls)
            time.sleep(3)
        
        # Final completion message
        st.success("✅ 25-second analysis completed! App has stopped to avoid API rate limits.")
        st.info("🔄 Refresh the page to run another 25-second session.")
    
    def display_metrics(self, df, rsi_data, oi_analysis, vob_zones):
        """Display key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        with col1:
            change = latest['close'] - prev['close']
            change_pct = (change / prev['close']) * 100
            st.metric(
                "Nifty Price", 
                f"₹{latest['close']:.2f}",
                f"{change:+.2f} ({change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("Day High", f"₹{df['high'].max():.2f}")
        
        with col3:
            st.metric("Day Low", f"₹{df['low'].min():.2f}")
        
        with col4:
            if rsi_data and rsi_data[0] is not None:
                latest_rsi = rsi_data[0].iloc[-1]
                rsi_status = "Overbought" if latest_rsi >= 80 else "Oversold" if latest_rsi <= 20 else "Normal"
                st.metric("Ultimate RSI", f"{latest_rsi:.2f}", rsi_status)
            else:
                st.metric("Volume", f"{df['volume'].sum():,}")
        
        with col5:
            if oi_analysis:
                sentiment_color = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
                st.metric("OI Sentiment", 
                         f"{sentiment_color.get(oi_analysis['sentiment'], '🟡')} {oi_analysis['sentiment']}")
            else:
                st.metric("VOB Zones", len(vob_zones) if vob_zones else 0)
    
    def run(self):
        """Main application"""
        st.title("📈 Enhanced Nifty Trading Dashboard")
        st.markdown("*With VOB Zones, Ultimate RSI & OI Analysis - 25 Second Auto Run*")
        
        # Sidebar controls
        with st.sidebar:
            st.header("📊 Settings")
            
            timeframe = st.selectbox(
                "Timeframe",
                options=['1', '3', '5', '15'],
                index=1,
                format_func=lambda x: f"{x} Min"
            )
            st.session_state.timeframe = timeframe
            
            # VOB Settings
            st.subheader("VOB Indicator")
            vob_enabled = st.checkbox("Enable VOB Zones", value=True)
            vob_sensitivity = st.slider("VOB Sensitivity", 3, 10, 5)
            st.session_state.vob_enabled = vob_enabled
            st.session_state.vob_sensitivity = vob_sensitivity
            
            # RSI Settings - Updated ranges from 5 to 20
            st.subheader("Ultimate RSI")
            rsi_enabled = st.checkbox("Enable Ultimate RSI", value=True)
            rsi_length = st.slider("RSI Length", 5, 20, 14)
            rsi_smooth = st.slider("RSI Smoothing", 5, 20, 14)
            st.session_state.rsi_enabled = rsi_enabled
            st.session_state.rsi_length = rsi_length
            st.session_state.rsi_smooth = rsi_smooth
            
            # OI Analysis Settings
            st.subheader("Options Analysis")
            oi_enabled = st.checkbox("Enable OI Analysis", value=True)
            st.session_state.oi_enabled = oi_enabled
            
            # Telegram Settings
            st.subheader("Telegram Alerts")
            telegram_enabled = st.checkbox("Enable Telegram Alerts", 
                                         value=bool(self.telegram_bot_token))
            st.session_state.telegram_enabled = telegram_enabled
            
            if telegram_enabled:
                st.info(f"VOB Alerts: {len(st.session_state.sent_vob_alerts)}")
                st.info(f"RSI Alerts: {len(st.session_state.sent_rsi_alerts)}")
                if st.button("Clear Alert History"):
                    st.session_state.sent_vob_alerts.clear()
                    st.session_state.sent_rsi_alerts.clear()
                    st.success("Alert history cleared!")
            
            # Auto-run control
            st.subheader("🚀 Auto Run Control")
            st.info("📌 App will auto-run for 25 seconds")
            
            if st.button("🔄 Start New 25s Session"):
                st.rerun()
        
        # Auto-run the 25-second session
        st.info("🚀 Starting 25-second auto-analysis session...")
        self.run_for_25_seconds()

# Initialize and run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
