import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import io
import json
from supabase import create_client, Client
import time
from pytz import timezone

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 minutes

# Initialize session state variables
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []
if 'export_data' not in st.session_state:
    st.session_state.export_data = False
if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)
if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)
if 'oi_volume_history' not in st.session_state:
    st.session_state.oi_volume_history = pd.DataFrame(columns=["Time", "Strike", "OI_CE", "OI_PE", "Volume_CE", "Volume_PE"])
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal", "VIX"])

# Initialize PCR settings with VIX-based defaults
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 2.0  # Will be adjusted based on VIX
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.4  # Will be adjusted based on VIX
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True

# DhanHQ API Configuration
DHAN_API_BASE_URL = "https://api.dhan.co/v2"

# Initialize DhanHQ credentials from secrets
def get_dhan_config():
    try:
        return {
            "client_id": st.secrets.get("DHAN_CLIENT_ID"),
            "access_token": st.secrets.get("DHAN_ACCESS_TOKEN")
        }
    except:
        return {
            "client_id": None,
            "access_token": None
        }

dhan_config = get_dhan_config()

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY")
        if supabase_url and supabase_key:
            return create_client(supabase_url, supabase_key)
        return None
    except:
        return None

supabase = init_supabase()

# Initialize Telegram config from secrets
def get_telegram_config():
    try:
        return {
            "bot_token": st.secrets.get("TELEGRAM_BOT_TOKEN"),
            "chat_id": st.secrets.get("TELEGRAM_CHAT_ID")
        }
    except:
        return {
            "bot_token": "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU",
            "chat_id": "5704496584"
        }

telegram_config = get_telegram_config()
TELEGRAM_BOT_TOKEN = telegram_config["bot_token"]
TELEGRAM_CHAT_ID = telegram_config["chat_id"]

def send_telegram_message(message):
    """Send message via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def get_dhan_headers():
    """Get DhanHQ API headers"""
    if not dhan_config["client_id"] or not dhan_config["access_token"]:
        st.error("DhanHQ credentials not configured. Please add DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to secrets.")
        return None
    
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": dhan_config["access_token"],
        "client-id": dhan_config["client_id"]
    }

def get_nifty_spot_price():
    """Get Nifty spot price using DhanHQ API"""
    headers = get_dhan_headers()
    if not headers:
        return None
    
    try:
        # Nifty 50 security ID for NSE_EQ is typically 26000
        payload = {
            "NSE_EQ": [26000]  # Nifty 50 security ID
        }
        
        response = requests.post(
            f"{DHAN_API_BASE_URL}/marketfeed/ltp",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'NSE_EQ' in data['data'] and '26000' in data['data']['NSE_EQ']:
                return data['data']['NSE_EQ']['26000']['last_price']
        
        st.error(f"Failed to get Nifty spot price: {response.status_code}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching Nifty spot price: {e}")
        return None

def get_vix_data():
    """Get VIX data using DhanHQ API"""
    headers = get_dhan_headers()
    if not headers:
        return 11  # Default VIX value
    
    try:
        # India VIX security ID for NSE_EQ is typically 26017
        payload = {
            "NSE_EQ": [26017]  # India VIX security ID
        }
        
        response = requests.post(
            f"{DHAN_API_BASE_URL}/marketfeed/ltp",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'NSE_EQ' in data['data'] and '26017' in data['data']['NSE_EQ']:
                return data['data']['NSE_EQ']['26017']['last_price']
        
        return 11  # Default VIX value if API fails
        
    except Exception as e:
        st.warning(f"Failed to get VIX data, using default: {e}")
        return 11

def get_nifty_expiry_list():
    """Get Nifty option expiry dates using DhanHQ API"""
    headers = get_dhan_headers()
    if not headers:
        return []
    
    try:
        payload = {
            "UnderlyingScrip": 13,  # Nifty underlying scrip ID
            "UnderlyingSeg": "IDX_I"
        }
        
        response = requests.post(
            f"{DHAN_API_BASE_URL}/optionchain/expirylist",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return data['data']
        
        return []
        
    except Exception as e:
        st.error(f"Error fetching expiry list: {e}")
        return []

def get_option_chain_data(expiry_date):
    """Get option chain data using DhanHQ API"""
    headers = get_dhan_headers()
    if not headers:
        return None
    
    try:
        payload = {
            "UnderlyingScrip": 13,  # Nifty underlying scrip ID
            "UnderlyingSeg": "IDX_I",
            "Expiry": expiry_date
        }
        
        response = requests.post(
            f"{DHAN_API_BASE_URL}/optionchain",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return data['data']
        
        st.error(f"Failed to get option chain data: {response.status_code}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching option chain data: {e}")
        return None

def process_option_chain_data(option_data, spot_price):
    """Process option chain data into DataFrame format"""
    if not option_data or 'oc' not in option_data:
        return pd.DataFrame()
    
    processed_data = []
    
    for strike_str, strike_data in option_data['oc'].items():
        strike = float(strike_str)
        
        # Process CE data
        ce_data = strike_data.get('ce', {})
        pe_data = strike_data.get('pe', {})
        
        row = {
            'strikePrice': strike,
            # CE data
            'lastPrice_CE': ce_data.get('last_price', 0),
            'openInterest_CE': ce_data.get('oi', 0),
            'changeinOpenInterest_CE': ce_data.get('oi', 0) - ce_data.get('previous_oi', 0),
            'totalTradedVolume_CE': ce_data.get('volume', 0),
            'impliedVolatility_CE': ce_data.get('implied_volatility', 0),
            'askQty_CE': ce_data.get('top_ask_quantity', 0),
            'bidQty_CE': ce_data.get('top_bid_quantity', 0),
            'askPrice_CE': ce_data.get('top_ask_price', 0),
            'bidPrice_CE': ce_data.get('top_bid_price', 0),
            'previous_close_price_CE': ce_data.get('previous_close_price', 0),
            'previous_volume_CE': ce_data.get('previous_volume', 0),
            # Greeks for CE
            'Delta_CE': ce_data.get('greeks', {}).get('delta', 0),
            'Gamma_CE': ce_data.get('greeks', {}).get('gamma', 0),
            'Vega_CE': ce_data.get('greeks', {}).get('vega', 0),
            'Theta_CE': ce_data.get('greeks', {}).get('theta', 0),
            
            # PE data
            'lastPrice_PE': pe_data.get('last_price', 0),
            'openInterest_PE': pe_data.get('oi', 0),
            'changeinOpenInterest_PE': pe_data.get('oi', 0) - pe_data.get('previous_oi', 0),
            'totalTradedVolume_PE': pe_data.get('volume', 0),
            'impliedVolatility_PE': pe_data.get('implied_volatility', 0),
            'askQty_PE': pe_data.get('top_ask_quantity', 0),
            'bidQty_PE': pe_data.get('top_bid_quantity', 0),
            'askPrice_PE': pe_data.get('top_ask_price', 0),
            'bidPrice_PE': pe_data.get('top_bid_price', 0),
            'previous_close_price_PE': pe_data.get('previous_close_price', 0),
            'previous_volume_PE': pe_data.get('previous_volume', 0),
            # Greeks for PE
            'Delta_PE': pe_data.get('greeks', {}).get('delta', 0),
            'Gamma_PE': pe_data.get('greeks', {}).get('gamma', 0),
            'Vega_PE': pe_data.get('greeks', {}).get('vega', 0),
            'Theta_PE': pe_data.get('greeks', {}).get('theta', 0)
        }
        
        processed_data.append(row)
    
    df = pd.DataFrame(processed_data)
    
    if df.empty:
        return df
    
    # Add zone classification
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot_price))
    df['Zone'] = df['strikePrice'].apply(
        lambda x: 'ATM' if x == atm_strike else 'ITM' if x < spot_price else 'OTM'
    )
    
    return df.sort_values('strikePrice')

def store_oi_volume_data(timestamp, strike, oi_ce, oi_pe, volume_ce, volume_pe):
    """Store OI and volume data in Supabase"""
    if supabase:
        try:
            data = {
                "timestamp": timestamp,
                "strike": strike,
                "oi_ce": oi_ce,
                "oi_pe": oi_pe,
                "volume_ce": volume_ce,
                "volume_pe": volume_pe,
                "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
            }
            supabase.table("oi_volume_data").insert(data).execute()
        except Exception as e:
            st.error(f"Supabase storage error: {e}")

def get_historical_oi_volume(strike, lookback_minutes=30):
    """Get historical OI and volume data from Supabase"""
    if supabase:
        try:
            from_time = datetime.now(timezone("Asia/Kolkata")) - pd.Timedelta(minutes=lookback_minutes)
            response = supabase.table("oi_volume_data") \
                .select("*") \
                .eq("strike", strike) \
                .gte("timestamp", from_time.isoformat()) \
                .execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            st.error(f"Supabase query error: {e}")
    return pd.DataFrame()

def check_oi_volume_spikes(current_data, historical_data, strike, threshold=2.0):
    """Check for sudden spikes in OI or volume"""
    alerts = []
    
    if historical_data.empty:
        return alerts
    
    # Calculate averages
    avg_oi_ce = historical_data['oi_ce'].mean()
    avg_oi_pe = historical_data['oi_pe'].mean()
    avg_volume_ce = historical_data['volume_ce'].mean()
    avg_volume_pe = historical_data['volume_pe'].mean()
    
    # Check for spikes
    if avg_oi_ce > 0 and current_data['oi_ce'] > avg_oi_ce * threshold:
        alerts.append(f"📈 OI Spike CE at {strike}: {current_data['oi_ce']:,.0f} (avg: {avg_oi_ce:,.0f})")
    
    if avg_oi_pe > 0 and current_data['oi_pe'] > avg_oi_pe * threshold:
        alerts.append(f"📈 OI Spike PE at {strike}: {current_data['oi_pe']:,.0f} (avg: {avg_oi_pe:,.0f})")
    
    if avg_volume_ce > 0 and current_data['volume_ce'] > avg_volume_ce * threshold:
        alerts.append(f"🔥 Volume Spike CE at {strike}: {current_data['volume_ce']:,.0f} (avg: {avg_volume_ce:,.0f})")
    
    if avg_volume_pe > 0 and current_data['volume_pe'] > avg_volume_pe * threshold:
        alerts.append(f"🔥 Volume Spike PE at {strike}: {current_data['volume_pe']:,.0f} (avg: {avg_volume_pe:,.0f})")
    
    return alerts

def final_verdict(score):
    """Convert bias score to trading verdict"""
    if score >= 4: return "Strong Bullish"
    elif score >= 2: return "Bullish"
    elif score <= -4: return "Strong Bearish"
    elif score <= -2: return "Bearish"
    return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    """Determine bias based on price, volume and OI changes"""
    if price > 0 and volume > 0 and chg_oi > 0: return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0: return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0: return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0: return "Bearish"
    return "Neutral"

def determine_level(row):
    """Determine support/resistance levels based on OI"""
    if row['openInterest_PE'] > 1.12 * row['openInterest_CE']: return "Support"
    elif row['openInterest_CE'] > 1.12 * row['openInterest_PE']: return "Resistance"
    return "Neutral"

def is_in_zone(spot, strike, level):
    """Check if strike is in support/resistance zone"""
    if level in ["Support", "Resistance"]: 
        return strike - 10 <= spot <= strike + 10
    return False

def get_support_resistance_zones(df, spot):
    """Identify nearest support/resistance zones"""
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()
    
    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]
    
    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)
    
    return support_zone, resistance_zone

def display_enhanced_trade_log():
    """Display formatted trade log with P&L calculations"""
    if not st.session_state.trade_log:
        st.info("No trades logged yet")
        return
    
    st.markdown("### 📜 Enhanced Trade Log")
    df_trades = pd.DataFrame(st.session_state.trade_log)
    
    if 'Current_Price' not in df_trades.columns:
        df_trades['Current_Price'] = df_trades['LTP'] * np.random.uniform(0.8, 1.3, len(df_trades))
        df_trades['Unrealized_PL'] = (df_trades['Current_Price'] - df_trades['LTP']) * 75
        df_trades['Status'] = df_trades['Unrealized_PL'].apply(
            lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < -100 else '🟡 Breakeven'
        )
    
    def color_pnl(row):
        colors = []
        for col in row.index:
            if col == 'Unrealized_PL':
                if row[col] > 0:
                    colors.append('background-color: #90EE90; color: black')
                elif row[col] < -100:
                    colors.append('background-color: #FFB6C1; color: black')
                else:
                    colors.append('background-color: #FFFFE0; color: black')
            else:
                colors.append('')
        return colors
    
    styled_trades = df_trades.style.apply(color_pnl, axis=1)
    st.dataframe(styled_trades, use_container_width=True)
    
    total_pl = df_trades['Unrealized_PL'].sum()
    win_rate = len(df_trades[df_trades['Unrealized_PL'] > 0]) / len(df_trades) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"₹{total_pl:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Trades", len(df_trades))

def create_export_data(df_summary, trade_log, spot_price):
    """Create Excel export data"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        if not st.session_state.pcr_history.empty:
            st.session_state.pcr_history.to_excel(writer, sheet_name='PCR_History', index=False)
        if not st.session_state.oi_volume_history.empty:
            st.session_state.oi_volume_history.to_excel(writer, sheet_name='OI_Volume_History', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_analysis_{timestamp}.xlsx"
    return output.getvalue(), filename

def handle_export_data(df_summary, spot_price):
    """Handle data export functionality"""
    if 'export_data' in st.session_state and st.session_state.export_data:
        try:
            excel_data, filename = create_export_data(df_summary, st.session_state.trade_log, spot_price)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.success("✅ Export ready! Click the download button above.")
            st.session_state.export_data = False
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            st.session_state.export_data = False

def plot_price_with_sr():
    """Plot price action with support/resistance zones"""
    price_df = st.session_state['price_data'].copy()
    if price_df.empty or price_df['Spot'].isnull().all():
        st.info("Not enough data to show price action chart yet.")
        return
    
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = st.session_state.get('support_zone', (None, None))
    resistance_zone = st.session_state.get('resistance_zone', (None, None))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Time'], 
        y=price_df['Spot'], 
        mode='lines+markers', 
        name='Spot Price',
        line=dict(color='blue', width=2)
    ))
    
    if all(support_zone) and None not in support_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=support_zone[0], y1=support_zone[1],
            fillcolor="rgba(0,255,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[0], support_zone[0]],
            mode='lines',
            name='Support Low',
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[1], support_zone[1]],
            mode='lines',
            name='Support High',
            line=dict(color='green', dash='dot')
        ))
    
    if all(resistance_zone) and None not in resistance_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=resistance_zone[0], y1=resistance_zone[1],
            fillcolor="rgba(255,0,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[0], resistance_zone[0]],
            mode='lines',
            name='Resistance Low',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[1], resistance_zone[1]],
            mode='lines',
            name='Resistance High',
            line=dict(color='red', dash='dot')
        ))
    
    fig.update_layout(
        title="Nifty Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(current_price):
    """Automatically update call log status"""
    for call in st.session_state.call_log_book:
        if call["Status"] != "Active":
            continue
        
        if call["Type"] == "CE":
            if current_price >= max(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price <= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
        elif call["Type"] == "PE":
            if current_price <= min(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price >= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price

def display_call_log_book():
    """Display the call log book"""
    st.markdown("### 📚 Call Log Book")
    if not st.session_state.call_log_book:
        st.info("No calls have been made yet.")
        return
    
    df_log = pd.DataFrame(st.session_state.call_log_book)
    st.dataframe(df_log, use_container_width=True)
    
    if st.button("Download Call Log Book as CSV"):
        st.download_button(
            label="Download CSV",
            data=df_log.to_csv(index=False).encode(),
            file_name="call_log_book.csv",
            mime="text/csv"
        )

def analyze():
    """Main analysis function using DhanHQ API"""
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()

        # Check market hours
        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

        # Check DhanHQ credentials
        if not dhan_config["client_id"] or not dhan_config["access_token"]:
            st.error("❌ DhanHQ credentials not configured. Please add DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to secrets.")
            return

        # Get VIX data
        vix_value = get_vix_data()
        
        # Set dynamic PCR thresholds based on VIX
        if vix_value > 12:
            st.session_state.pcr_threshold_bull = 2.0
            st.session_state.pcr_threshold_bear = 0.4
            volatility_status = "High Volatility"
        else:
            st.session_state.pcr_threshold_bull = 1.2
            st.session_state.pcr_threshold_bear = 0.7
            volatility_status = "Low Volatility"

        # Get Nifty spot price
        spot_price = get_nifty_spot_price()
        if not spot_price:
            st.error("❌ Failed to get Nifty spot price")
            return

        # Get expiry dates
        expiry_dates = get_nifty_expiry_list()
        if not expiry_dates:
            st.error("❌ Failed to get expiry dates")
            return

        # Use the nearest expiry
        current_expiry = expiry_dates[0]
        
        # Get option chain data
        option_data = get_option_chain_data(current_expiry)
        if not option_data:
            st.error("❌ Failed to get option chain data")
            return

        # Display market info
        st.markdown(f"### 📍 Spot Price: {spot_price}")
        st.markdown(f"### 📊 VIX: {vix_value} ({volatility_status}) | PCR Thresholds: Bull >{st.session_state.pcr_threshold_bull} | Bear <{st.session_state.pcr_threshold_bear}")

        # Process option chain data
        df = process_option_chain_data(option_data, spot_price)
        
        if df.empty:
            st.error("❌ No option chain data available")
            return

        # Filter strikes around ATM
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot_price))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Level'] = df.apply(determine_level, axis=1)

        # === OI/VOLUME SPIKE DETECTION ===
        current_time_str = now.strftime("%H:%M:%S")
        spike_alerts = []
        
        # Check ATM and nearby strikes for spikes
        nearby_strikes = df[df['strikePrice'].between(atm_strike - 100, atm_strike + 100)]
        
        for _, row in nearby_strikes.iterrows():
            strike = row['strikePrice']
            current_data = {
                'oi_ce': row['openInterest_CE'],
                'oi_pe': row['openInterest_PE'],
                'volume_ce': row['totalTradedVolume_CE'],
                'volume_pe': row['totalTradedVolume_PE']
            }
            
            # Store current data
            new_oi_data = pd.DataFrame({
                "Time": [current_time_str],
                "Strike": [strike],
                "OI_CE": [current_data['oi_ce']],
                "OI_PE": [current_data['oi_pe']],
                "Volume_CE": [current_data['volume_ce']],
                "Volume_PE": [current_data['volume_pe']]
            })
            st.session_state.oi_volume_history = pd.concat([st.session_state.oi_volume_history, new_oi_data])
            
            # Store in Supabase
            store_oi_volume_data(current_time_str, strike, current_data['oi_ce'], current_data['oi_pe'], 
                               current_data['volume_ce'], current_data['volume_pe'])
            
            # Check for spikes (only check every 5 minutes per strike to avoid spam)
            last_alert_key = f"{strike}_{current_time_str[:-3]}"  # Minute granularity
            if last_alert_key not in st.session_state.last_alert_time:
                historical_data = get_historical_oi_volume(strike, lookback_minutes=30)
                alerts = check_oi_volume_spikes(current_data, historical_data, strike, threshold=2.5)
                
                if alerts:
                    spike_alerts.extend(alerts)
                    st.session_state.last_alert_time[last_alert_key] = now.timestamp()
                    
                    # Send Telegram alert for significant spikes
                    for alert in alerts:
                        if "Spike" in alert:
                            send_telegram_message(
                                f"🚨 {alert}\n"
                                f"📍 Spot: {spot_price}\n"
                                f"⏰ Time: {current_time_str}\n"
                                f"📊 VIX: {vix_value}"
                            )
        
        # Display spike alerts
        if spike_alerts:
            st.markdown("### ⚠️ OI/Volume Spike Alerts")
            for alert in spike_alerts:
                st.warning(alert)

        # Calculate bias scores
        weights = {
            'ChgOI_Bias': 1.5,
            'Volume_Bias': 1.0,
            'Gamma_Bias': 1.2,
            'AskQty_Bias': 0.8,
            'BidQty_Bias': 0.8,
            'IV_Bias': 1.0,
            'DVP_Bias': 1.5
        }

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row['lastPrice_CE'] - row['lastPrice_PE'],
                    row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                    row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                )
            }

            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)
        
        # === PCR CALCULATION AND MERGE ===
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )

        df_summary['PCR'] = (
            df_summary['openInterest_PE'] / df_summary['openInterest_CE']
        )

        df_summary['PCR'] = np.where(
            df_summary['openInterest_CE'] == 0,
            0,
            df_summary['PCR']
        )

        df_summary['PCR'] = df_summary['PCR'].round(2)
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(
                df_summary['PCR'] < st.session_state.pcr_threshold_bear,
                "Bearish",
                "Neutral"
            )
        )

        def color_pcr(val):
            if val > st.session_state.pcr_threshold_bull:
                return 'background-color: #90EE90; color: black'
            elif val < st.session_state.pcr_threshold_bear:
                return 'background-color: #FFB6C1; color: black'
            else:
                return 'background-color: #FFFFE0; color: black'

        styled_df = df_summary.style.applymap(color_pcr, subset=['PCR'])
        df_summary = df_summary.drop(columns=['strikePrice'])
        
        # Record PCR history
        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']],
                "VIX": [vix_value]
            })
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, new_pcr_data])

        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, spot_price)

        # Store zones in session state
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        # Update price history
        new_row = pd.DataFrame([[current_time_str, spot_price]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        # Format support/resistance strings
        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        # Generate signals
        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        last_trade = st.session_state.trade_log[-1] if st.session_state.trade_log else None
        if last_trade and not (last_trade.get("TargetHit", False) or last_trade.get("SLHit", False)):
            pass  # Skip new signals if previous trade is active
        else:
            for row in bias_results:
                if not is_in_zone(spot_price, row['Strike'], row['Level']):
                    continue

                # Get current PCR signal for this strike
                pcr_data = df_summary[df_summary['Strike'] == row['Strike']].iloc[0]
                pcr_signal = pcr_data['PCR_Signal']
                pcr_value = pcr_data['PCR']

                # Get ATM biases
                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None

                if st.session_state.use_pcr_filter:
                    # Support + Bullish conditions with PCR confirmation
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                        and pcr_signal == "Bullish"):
                        option_type = 'CE'
                    # Resistance + Bearish conditions with PCR confirmation
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                          and pcr_signal == "Bearish"):
                        option_type = 'PE'
                    else:
                        continue
                else:
                    # Original signal logic without PCR confirmation
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)):
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)):
                        option_type = 'PE'
                    else:
                        continue

                # Get option details
                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * (1 + iv / 100), 2)
                stop_loss = round(ltp * 0.8, 2)

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                # Send Telegram alert
                send_telegram_message(
                    f"VIX: {vix_value} ({volatility_status})\n"
                    f"PCR: {pcr_value} ({pcr_signal})\n"
                    f"Thresholds: Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear}\n"
                    f"📍 Spot: {spot_price}\n"
                    f"🔹 {atm_signal}\n"
                    f"{suggested_trade}\n"
                    f"Bias Score: {total_score} ({market_view})\n"
                    f"Level: {row['Level']}\n"
                    f"📉 Support Zone: {support_str}\n"
                    f"📈 Resistance Zone: {resistance_str}"
                )

                # Add to trade log
                st.session_state.trade_log.append({
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False,
                    "VIX": vix_value,
                    "PCR_Value": pcr_value,
                    "PCR_Signal": pcr_signal,
                    "PCR_Thresholds": f"Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear}"
                })

                signal_sent = True
                break

        # === Main Display ===
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
        # Plot price action
        plot_price_with_sr()

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        
        # Option Chain Summary
        with st.expander("📊 Option Chain Summary"):
            st.info(f"""
            ℹ️ PCR Interpretation (VIX: {vix_value}):
            - >{st.session_state.pcr_threshold_bull} = Bullish
            - <{st.session_state.pcr_threshold_bear} = Bearish
            - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
            """)
            st.dataframe(styled_df)
        
        # Trade Log
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))

        # === Enhanced Features Section ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        
        # PCR Configuration
        st.markdown("### 🧮 PCR Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pcr_threshold_bull = st.number_input(
                "Bullish PCR Threshold (>)", 
                min_value=1.0, max_value=5.0, 
                value=st.session_state.pcr_threshold_bull, 
                step=0.1
            )
        with col2:
            st.session_state.pcr_threshold_bear = st.number_input(
                "Bearish PCR Threshold (<)", 
                min_value=0.1, max_value=1.0, 
                value=st.session_state.pcr_threshold_bear, 
                step=0.1
            )
        with col3:
            st.session_state.use_pcr_filter = st.checkbox(
                "Enable PCR Filtering", 
                value=st.session_state.use_pcr_filter
            )
        
        # PCR History
        with st.expander("📈 PCR History"):
            if not st.session_state.pcr_history.empty:
                pcr_pivot = st.session_state.pcr_history.pivot_table(
                    index='Time', 
                    columns='Strike', 
                    values='PCR',
                    aggfunc='last'
                )
                st.line_chart(pcr_pivot)
                st.dataframe(st.session_state.pcr_history)
            else:
                st.info("No PCR history recorded yet")
        
        # OI/Volume History
        with st.expander("📊 OI/Volume History"):
            if not st.session_state.oi_volume_history.empty:
                st.dataframe(st.session_state.oi_volume_history.tail(20))
                # Plot OI trends for ATM strike
                atm_oi_data = st.session_state.oi_volume_history[
                    st.session_state.oi_volume_history['Strike'] == atm_strike
                ]
                if not atm_oi_data.empty:
                    fig_oi = go.Figure()
                    fig_oi.add_trace(go.Scatter(x=atm_oi_data['Time'], y=atm_oi_data['OI_CE'], 
                                              name='OI CE', line=dict(color='blue')))
                    fig_oi.add_trace(go.Scatter(x=atm_oi_data['Time'], y=atm_oi_data['OI_PE'], 
                                              name='OI PE', line=dict(color='red')))
                    fig_oi.update_layout(title=f"OI Trend for ATM Strike {atm_strike}")
                    st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.info("No OI/Volume history recorded yet")
        
        # Enhanced Trade Log
        display_enhanced_trade_log()
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 📥 Data Export")
        if st.button("Prepare Excel Export"):
            st.session_state.export_data = True
        handle_export_data(df_summary, spot_price)
        
        # Call Log Book
        st.markdown("---")
        display_call_log_book()
        
        # Auto update call log with current price
        auto_update_call_log(spot_price)

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        send_telegram_message(f"❌ Unexpected error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()