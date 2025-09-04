import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import io
import os
import json

# === Dhan API Configuration ===
try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
except Exception:
    DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

# === Supabase Configuration ===
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "") 
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
except Exception:
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Initialize Supabase client only if credentials are provided
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Connected to Supabase")
    except Exception as e:
        st.warning(f"⚠️ Supabase connection failed: {e}")
        supabase_client = None
else:
    st.info("ℹ️ Supabase not configured. Add SUPABASE_URL and SUPABASE_KEY to secrets.toml or environment variables to enable data storage.")

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

# Initialize session state for price data
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])

# Initialize session state for enhanced features
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

# Initialize PCR-related session state
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 1.2
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.7
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# Initialize market logic session state
if 'market_bias' not in st.session_state:
    st.session_state.market_bias = "Neutral"
if 'price_history' not in st.session_state:
    st.session_state.price_history = pd.DataFrame(columns=["timestamp", "price"])
if 'option_chain_history' not in st.session_state:
    st.session_state.option_chain_history = pd.DataFrame(columns=["timestamp", "strike", "call_oi", "put_oi", "call_pcr", "put_pcr"])
if 'use_market_logic' not in st.session_state:
    st.session_state.use_market_logic = False

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

# === Dhan API Functions ===
def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/optionchain"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan option chain: {e}")
        return None

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str):
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan expiry list: {e}")
        return None

def store_price_data(price):
    if not supabase_client:
        return
    try:
        data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "price": price,
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        supabase_client.table("price_history").insert(data).execute()
    except Exception as e:
        st.error(f"Error storing price data: {e}")

def get_trade_log():
    if not supabase_client:
        return []
    try:
        response = supabase_client.table("trade_log").select("*").order("timestamp", desc=True).execute()
        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        st.error(f"Error retrieving trade log: {e}")
        return []

def check_target_sl_hits(current_price):
    if not supabase_client:
        return
    try:
        response = supabase_client.table("trade_log").select("*").eq("target_hit", False).eq("sl_hit", False).execute()
        if response.data:
            for trade in response.data:
                strike = trade['strike']
                option_type = trade['option_type']
                entry_price = trade['entry_price']
                target_price = trade['target_price']
                stop_loss = trade['stop_loss']
                target_hit = False
                sl_hit = False
                
                if option_type == 'CE':
                    if current_price >= target_price:
                        target_hit = True
                    elif current_price <= stop_loss:
                        sl_hit = True
                elif option_type == 'PE':
                    if current_price <= target_price:
                        target_hit = True
                    elif current_price >= stop_loss:
                        sl_hit = True
                
                if target_hit or sl_hit:
                    update_data = {
                        "target_hit": target_hit,
                        "sl_hit": sl_hit,
                        "exit_price": current_price,
                        "exit_time": datetime.now(timezone("Asia/Kolkata")).isoformat()
                    }
                    supabase_client.table("trade_log").update(update_data).eq("id", trade['id']).execute()
                    
                    message = f"🎯 {'Target' if target_hit else 'Stop Loss'} Hit!\nStrike: {strike} {option_type}\nEntry: ₹{entry_price}\nExit: ₹{current_price}\nP&L: ₹{(current_price - entry_price) * 75}"
                    send_telegram_message(message)
    except Exception as e:
        st.error(f"Error checking target/SL hits: {e}")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
        return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
    except:
        return 0, 0, 0, 0, 0

def final_verdict(score):
    if score >= 4:
        return "Strong Bullish"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bearish"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0:
        return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0:
        return "Bearish"
    else:
        return "Neutral"

def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    return pressure, bias

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
}

def determine_level(row):
    ce_oi = row.get('openInterest_CE', 0)
    pe_oi = row.get('openInterest_PE', 0)
    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level in ["Support", "Resistance"]:
        return strike - 8 <= spot <= strike + 8
    return False

def get_support_resistance_zones(df, spot):
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()

    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]

    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)

    return support_zone, resistance_zone

def store_option_chain_data(option_chain_data):
    if not supabase_client:
        return
    try:
        timestamp = datetime.now(timezone("Asia/Kolkata")).isoformat()
        data_to_insert = []
        for _, row in option_chain_data.iterrows():
            data_to_insert.append({
                "timestamp": timestamp,
                "strike": row['Strike'],
                "call_oi": row.get('openInterest_CE', 0),
                "put_oi": row.get('openInterest_PE', 0),
                "call_pcr": row.get('PCR_CE', 0),
                "put_pcr": row.get('PCR_PE', 0),
                "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
            })
        batch_size = 50
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            supabase_client.table("option_chain_history").insert(batch).execute()
    except Exception as e:
        st.error(f"Error storing option chain data: {e}")

def get_historical_data(minutes=10):
    if not supabase_client:
        return pd.DataFrame(), pd.DataFrame()
    try:
        from datetime import timedelta
        time_threshold = (datetime.now(timezone("Asia/Kolkata")) - timedelta(minutes=minutes)).isoformat()
        price_response = supabase_client.table("price_history").select("*").gte("timestamp", time_threshold).order("timestamp", desc=True).execute()
        option_chain_response = supabase_client.table("option_chain_history").select("*").gte("timestamp", time_threshold).order("timestamp", desc=True).execute()
        price_df = pd.DataFrame(price_response.data) if price_response.data else pd.DataFrame()
        option_chain_df = pd.DataFrame(option_chain_response.data) if option_chain_response.data else pd.DataFrame()
        return price_df, option_chain_df
    except Exception as e:
        st.error(f"Error retrieving historical data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_price_trend(price_df):
    if len(price_df) < 2:
        return "Neutral"
    recent_prices = price_df.sort_values('timestamp', ascending=False).head(5)
    if len(recent_prices) < 2:
        return "Neutral"
    prices = recent_prices['price'].values
    if prices[0] > prices[-1]:
        return "Rising"
    elif prices[0] < prices[-1]:
        return "Falling"
    else:
        return "Neutral"

def calculate_pcr_trend(option_chain_df):
    if len(option_chain_df) < 2:
        return "Neutral", 0, 0
    recent_data = option_chain_df.sort_values('timestamp', ascending=False).head(10)
    if len(recent_data) < 2:
        return "Neutral", 0, 0
    call_pcr_avg = recent_data['call_pcr'].mean() if 'call_pcr' in recent_data.columns else 0
    put_pcr_avg = recent_data['put_pcr'].mean() if 'put_pcr' in recent_data.columns else 0
    if put_pcr_avg > call_pcr_avg:
        return "Put PCR Dominant", call_pcr_avg, put_pcr_avg
    elif call_pcr_avg > put_pcr_avg:
        return "Call PCR Dominant", call_pcr_avg, put_pcr_avg
    else:
        return "Neutral", call_pcr_avg, put_pcr_avg

def apply_market_logic(price_df, option_chain_df):
    price_trend = calculate_price_trend(price_df)
    pcr_trend, call_pcr, put_pcr = calculate_pcr_trend(option_chain_df)
    if pcr_trend == "Put PCR Dominant" and price_trend == "Falling":
        return "Bearish", f"Put PCR ({put_pcr:.2f}) > Call PCR ({call_pcr:.2f}) + Price Falling"
    elif pcr_trend == "Put PCR Dominant" and price_trend == "Rising":
        return "Bullish", f"Put PCR ({put_pcr:.2f}) > Call PCR ({call_pcr:.2f}) + Price Rising"
    elif pcr_trend == "Call PCR Dominant" and price_trend == "Rising":
        return "Bullish", f"Call PCR ({call_pcr:.2f}) > Put PCR ({put_pcr:.2f}) + Price Rising"
    elif pcr_trend == "Call PCR Dominant" and price_trend == "Falling":
        return "Bearish", f"Call PCR ({call_pcr:.2f}) > Put PCR ({put_pcr:.2f}) + Price Falling"
    else:
        return "Neutral", "No clear signal from market logic"

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("00:00", "%H:%M").time()
        market_end = datetime.strptime("23:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

        expiry_data = get_dhan_expiry_list(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
        if not expiry_data or 'data' not in expiry_data:
            st.error("Failed to get expiry list from Dhan API")
            return

        expiry_dates = expiry_data['data']
        if not expiry_dates:
            st.error("No expiry dates available")
            return

        expiry = expiry_dates[0]

        option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
        if not option_chain_data or 'data' not in option_chain_data:
            st.error("Failed to get option chain from Dhan API")
            return

        data = option_chain_data['data']
        underlying = data['last_price']

        store_price_data(underlying)
        check_target_sl_hits(underlying)

        oc_data = data['oc']

        calls, puts = [], []
        for strike, strike_data in oc_data.items():
            if 'ce' in strike_data:
                ce_data = strike_data['ce']
                ce_data['strikePrice'] = float(strike)
                ce_data['expiryDate'] = expiry
                calls.append(ce_data)
            if 'pe' in strike_data:
                pe_data = strike_data['pe']
                pe_data['strikePrice'] = float(strike)
                pe_data['expiryDate'] = expiry
                puts.append(pe_data)

        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)

        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        column_mapping = {
            'last_price': 'lastPrice',
            'oi': 'openInterest',
            'previous_close_price': 'previousClose',
            'previous_oi': 'previousOpenInterest',
            'previous_volume': 'previousVolume',
            'top_ask_price': 'askPrice',
            'top_ask_quantity': 'askQty',
            'top_bid_price': 'bidPrice',
            'top_bid_quantity': 'bidQty',
            'volume': 'totalTradedVolume'
        }

        for old_col, new_col in column_mapping.items():
            if f"{old_col}_CE" in df.columns:
                df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
            if f"{old_col}_PE" in df.columns:
                df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)

        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']

        for col in ['impliedVolatility_CE', 'impliedVolatility_PE']:
            if col not in df.columns:
                df[col] = 0

        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone("Asia/Kolkata"))
        T = max((expiry_date - now).days, 1) / 365
        r = 0.06

        for idx, row in df.iterrows():
            strike = row['strikePrice']
            try:
                if 'impliedVolatility_CE' in row and row['impliedVolatility_CE'] > 0:
                    greeks = calculate_greeks('CE', underlying, strike, T, r, row['impliedVolatility_CE'] / 100)
                else:
                    greeks = calculate_greeks('CE', underlying, strike, T, r, 0.15)
            except:
                greeks = (0, 0, 0, 0, 0)
            df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks
            try:
                if 'impliedVolatility_PE' in row and row['impliedVolatility_PE'] > 0:
                    greeks = calculate_greeks('PE', underlying, strike, T, r, row['impliedVolatility_PE'] / 100)
                else:
                    greeks = calculate_greeks('PE', underlying, strike, T, r, 0.15)
            except:
                greeks = (0, 0, 0, 0, 0)
            df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks

        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000
        total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000

        st.markdown("## 📊 Open Interest Change (in Lakhs)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 CALL ΔOI", f"{total_ce_change:+.1f}L", delta_color="inverse")
        with col2:
            st.metric("📈 PUT ΔOI", f"{total_pe_change:+.1f}L", delta_color="normal")

        if total_ce_change > total_pe_change:
            st.error(f"🚨 Call OI Dominance (Difference: {abs(total_ce_change - total_pe_change):.1f}L)")
        elif total_pe_change > total_ce_change:
            st.success(f"🚀 Put OI Dominance (Difference: {abs(total_pe_change - total_ce_change):.1f}L)")
        else:
            st.info("⚖️ OI Changes Balanced")

        # Bias scoring and summarizing
        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue
            bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
                row.get('bidQty_CE', 0), 
                row.get('askQty_CE', 0),                                 
                row.get('bidQty_PE', 0), 
                row.get('askQty_PE', 0)
            )
            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
                "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
                "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) < row.get('Gamma_PE', 0) else "Bearish",
                "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
                "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
                "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) > row.get('impliedVolatility_PE', 0) else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                    row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                    row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
                ),
                "BidAskPressure": bid_ask_pressure,
                "PressureBias": pressure_bias
            }
            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
                elif k == "PressureBias":
                    score += weights.get("PressureBias", 1) if bias == "Bullish" else -weights.get("PressureBias", 1)
            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)

        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )

        df_summary['PCR'] = (
            df_summary['openInterest_PE'] / df_summary['openInterest_CE']
        )
        df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, df_summary['PCR'])
        df_summary['PCR'] = df_summary['PCR'].round(2)
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(df_summary['PCR'] < st.session_state.pcr_threshold_bear, "Bearish", "Neutral")
        )

        store_option_chain_data(df_summary)

        market_bias, market_reason = "Neutral", "Market logic not enabled"
        if st.session_state.use_market_logic:
            price_history, option_chain_history = get_historical_data()
            market_bias, market_reason = apply_market_logic(price_history, option_chain_history)
        st.session_state.market_bias = market_bias

        styled_df = df_summary.style.applymap(lambda v: 'background-color: #90EE90; color: black' if v > st.session_state.pcr_threshold_bull else ('background-color: #FFB6C1; color: black' if v < st.session_state.pcr_threshold_bear else 'background-color: #FFFFE0; color: black'), subset=['PCR']).applymap(lambda v: 'background-color: #90EE90; color: black' if v > 500 else ('background-color: #FFB6C1; color: black' if v < -500 else 'background-color: #FFFFE0; color: black'), subset=['BidAskPressure'])
        df_summary = df_summary.drop(columns=['strikePrice'])

        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']]
            })
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, new_pcr_data])

        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        trade_data = get_trade_log()
        last_trade = trade_data[0] if trade_data else None

        if last_trade and not (last_trade.get("target_hit", False) or last_trade.get("sl_hit", False)):
            pass
        else:
            for row in bias_results:
                if not is_in_zone(underlying, row['Strike'], row['Level']):
                    continue

                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None
                pcr_signal = df_summary[df_summary['Strike'] == row['Strike']]['PCR_Signal'].values[0]

                if st.session_state.use_pcr_filter:
                    if (row['Level'] == "Support" and total_score >= 4 and "Bullish" in market_view and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None) and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None) and pcr_signal == "Bullish" and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bullish", "Neutral"])):
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 and "Bearish" in market_view and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None) and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None) and pcr_signal == "Bearish" and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bearish", "Neutral"])):
                        option_type = 'PE'
                    else:
                        continue
                else:
                    if (row['Level'] == "Support" and total_score >= 4 and "Bullish" in market_view and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None) and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None) and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bullish", "Neutral"])):
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 and "Bearish" in market_view and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None) and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None) and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bearish", "Neutral"])):
                        option_type = 'PE'
                    else:
                        continue

                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * (1 + iv / 100), 2)
                stop_loss = round(ltp * 0.8, 2)

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                send_telegram_message(
                    f"⚙️ PCR Config: Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear} "
                    f"(Filter {'ON' if st.session_state.use_pcr_filter else 'OFF'})\n"
                    f"📍 Spot: {underlying}\n"
                    f"🔹 {atm_signal}\n"
                    f"{suggested_trade}\n"
                    f"PCR: {df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0]} ({pcr_signal})\n"
                    f"Bias Score: {total_score} ({market_view})\n"
                    f"Market Bias: {st.session_state.market_bias} ({market_reason})\n"
                    f"Level: {row['Level']}\n"
                    f"📉 Support Zone: {support_str}\n"
                    f"📈 Resistance Zone: {resistance_str}"
                )

                trade_data = {
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False,
                    "PCR": df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0],
                    "PCR_Signal": pcr_signal,
                    "Market_Bias": st.session_state.market_bias
                }

                store_trade_log(trade_data)

                signal_sent = True
                break

        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")

        if st.session_state.use_market_logic:
            st.info(f"📈 Market Bias: **{st.session_state.market_bias}** - {market_reason}")

        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")

        with st.expander("📊 Option Chain Summary"):
            st.info(f"""
            ℹ️ PCR Interpretation:
            - >{st.session_state.pcr_threshold_bull} = Strong Put Activity (Bullish)
            - <{st.session_state.pcr_threshold_bear} = Strong Call Activity (Bearish)
            - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
            """)

            if st.session_state.use_market_logic:
                st.info("""
            ℹ️ Market Logic:
            - Put PCR > Call PCR + Price Falling -> Bearish
            - Put PCR > Call PCR + Price Rising -> Bullish
            - Call PCR > Put PCR + Price Rising -> Bullish
            - Call PCR > Put PCR + Price Falling -> Bearish
            """)

            st.dataframe(styled_df)

        trade_data = get_trade_log()
        if trade_data:
            st.markdown("### 📜 Trade Log")
            df_trades = pd.DataFrame(trade_data)
            df_trades.rename(columns={
                'option_type': 'Type',
                'strike': 'Strike',
                'entry_price': 'LTP',
                'target_price': 'Target',
                'stop_loss': 'SL',
                'pcr': 'PCR',
                'pcr_signal': 'PCR_Signal',
                'market_bias': 'Market_Bias',
                'target_hit': 'TargetHit',
                'sl_hit': 'SLHit',
                'exit_price': 'Exit_Price',
                'exit_time': 'Exit_Time'
            }, inplace=True)
            st.dataframe(df_trades)

        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")

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

        st.markdown("### 🔄 Market Logic Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.use_market_logic = st.checkbox(
                "Enable Market Logic",
                value=st.session_state.use_market_logic,
                help="Uses historical data to determine market bias based on PCR and price trends"
            )

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

        display_enhanced_trade_log()

        st.markdown("---")
        st.markdown("### 📥 Data Export")
        if st.button("Prepare Excel Export"):
            st.session_state.export_data = True
        handle_export_data(df_summary, underlying)

        auto_update_call_log(underlying)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()
