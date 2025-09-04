import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone

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
        st.success("Connected to Supabase")
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        supabase_client = None

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

# Initialize minimal session state
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

# === Instrument Mapping ===
NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"
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
    # === Basic Supabase Functions ===
def store_trade_log(trade_data):
    """Store trade log entry in Supabase"""
    if not supabase_client:
        return
        
    try:
        if 'Time' not in trade_data:
            trade_data['Time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S")
        
        supabase_trade_data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "strike": trade_data.get("Strike", 0),
            "option_type": trade_data.get("Type", ""),
            "entry_price": trade_data.get("LTP", 0),
            "target_price": trade_data.get("Target", 0),
            "stop_loss": trade_data.get("SL", 0),
            "target_hit": trade_data.get("TargetHit", False),
            "sl_hit": trade_data.get("SLHit", False),
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        
        supabase_client.table("trade_log").insert(supabase_trade_data).execute()
    except Exception as e:
        st.error(f"Error storing trade log: {e}")

def get_trade_log():
    """Get trade log from Supabase"""
    if not supabase_client:
        return []
        
    try:
        response = supabase_client.table("trade_log") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(10) \
            .execute()
        
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error retrieving trade log: {e}")
        return []

def check_target_sl_hits(current_price):
    """Check if any active trades have hit target or stop loss"""
    if not supabase_client:
        return
        
    try:
        response = supabase_client.table("trade_log") \
            .select("*") \
            .eq("target_hit", False) \
            .eq("sl_hit", False) \
            .execute()
        
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
                    
                    supabase_client.table("trade_log") \
                        .update(update_data) \
                        .eq("id", trade['id']) \
                        .execute()
                    
                    message = f"{'Target' if target_hit else 'Stop Loss'} Hit!\n"
                    message += f"Strike: {strike} {option_type}\n"
                    message += f"Entry: {entry_price}\n"
                    message += f"Exit: {current_price}\n"
                    message += f"P&L: {(current_price - entry_price) * 75}"
                    
                    send_telegram_message(message)
    except Exception as e:
        st.error(f"Error checking target/SL hits: {e}")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("Telegram message failed.")
    except Exception as e:
        st.error(f"Telegram error: {e}")    
# === Core Calculation Functions ===
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

# Weights for bias scoring
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
    if level == "Support":
        return strike - 8 <= spot <= strike + 8
    elif level == "Resistance":
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
    # === Main Analysis Function Part A ===
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
            st.warning("Market Closed (Mon-Fri 9:00-15:40)")
            return

        # Get expiry list from Dhan API
        expiry_data = get_dhan_expiry_list(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
        if not expiry_data or 'data' not in expiry_data:
            st.error("Failed to get expiry list from Dhan API")
            return
        
        expiry_dates = expiry_data['data']
        if not expiry_dates:
            st.error("No expiry dates available")
            return
        
        expiry = expiry_dates[0]  # Use nearest expiry
        
        # Get option chain from Dhan API
        option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
        if not option_chain_data or 'data' not in option_chain_data:
            st.error("Failed to get option chain from Dhan API")
            return
        
        data = option_chain_data['data']
        underlying = data['last_price']
        
        # Check for target/SL hits
        check_target_sl_hits(underlying)

        # Process option chain data
        oc_data = data['oc']
        
        # Convert to DataFrame format
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
        
        # Merge call and put data
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        
        # Rename columns to match NSE format
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
        
        # Calculate change in open interest
        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
        
        # Add missing columns with default values
        for col in ['impliedVolatility_CE', 'impliedVolatility_PE']:
            if col not in df.columns:
                df[col] = 0
        
        # Calculate time to expiry
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone("Asia/Kolkata"))
        T = max((expiry_date - now).days, 1) / 365
        r = 0.06

        # Calculate Greeks with error handling
        for idx, row in df.iterrows():
            strike = row['strikePrice']
            
            # Calculate Greeks for CE
            try:
                if 'impliedVolatility_CE' in row and row['impliedVolatility_CE'] > 0:
                    greeks = calculate_greeks('CE', underlying, strike, T, r, row['impliedVolatility_CE'] / 100)
                else:
                    greeks = calculate_greeks('CE', underlying, strike, T, r, 0.15)
            except:
                greeks = (0, 0, 0, 0, 0)
            
            df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks
            
            # Calculate Greeks for PE
            try:
                if 'impliedVolatility_PE' in row and row['impliedVolatility_PE'] > 0:
                    greeks = calculate_greeks('PE', underlying, strike, T, r, row['impliedVolatility_PE'] / 100)
                else:
                    greeks = calculate_greeks('PE', underlying, strike, T, r, 0.15)
            except:
                greeks = (0, 0, 0, 0, 0)
            
            df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks

        return df, underlying, now
        
    except Exception as e:
        st.error(f"Error: {e}")
        send_telegram_message(f"Error: {str(e)}")
        return None, None, None
        # === Main Analysis Function Part B ===
def process_analysis(df, underlying, now):
    try:
        # Continue with analysis logic
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # Open Interest Change Comparison
        total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000
        total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000
        
        st.markdown("## Open Interest Change (in Lakhs)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CALL ΔOI", 
                     f"{total_ce_change:+.1f}L",
                     delta_color="inverse")
            
        with col2:
            st.metric("PUT ΔOI", 
                     f"{total_pe_change:+.1f}L",
                     delta_color="normal")
        
        if total_ce_change > total_pe_change:
            st.error(f"Call OI Dominance (Difference: {abs(total_ce_change - total_pe_change):.1f}L)")
        elif total_pe_change > total_ce_change:
            st.success(f"Put OI Dominance (Difference: {abs(total_pe_change - total_ce_change):.1f}L)")
        else:
            st.info("OI Changes Balanced")

        # Bias calculation and scoring
        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            # Add bid/ask pressure calculation
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

            # Calculate score based on bias
            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)
        
        # Calculate market view and zones
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        return df_summary, bias_results, atm_row, market_view, support_zone, resistance_zone, support_str, resistance_str, total_score, df
        
    except Exception as e:
        st.error(f"Error in analysis processing: {e}")
        return None, None, None, None, None, None, None, None, None, None
        # === Signal Generation and Display ===
def generate_signals_and_display(df_summary, bias_results, atm_row, market_view, support_zone, resistance_zone, support_str, resistance_str, total_score, df, underlying, now):
    try:
        # Signal generation logic
        atm_signal, suggested_trade = "No Signal", ""
        
        # Get the latest trade to check if we have an active position
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

                # Simplified signal logic without PCR filtering
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

                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * 1.25, 2)  # Simple 25% target
                stop_loss = round(ltp * 0.8, 2)  # 20% stop loss

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ {ltp} | Target: {target} | SL: {stop_loss}"

                send_telegram_message(
                    f"Spot: {underlying}\n"
                    f"{atm_signal}\n"
                    f"{suggested_trade}\n"
                    f"Bias Score: {total_score} ({market_view})\n"
                    f"Level: {row['Level']}\n"
                    f"Support Zone: {support_str}\n"
                    f"Resistance Zone: {resistance_str}"
                )

                trade_data = {
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False
                }

                # Store trade in Supabase
                store_trade_log(trade_data)
                break

        # Main Display
        st.markdown(f"### Spot Price: {underlying}")
        st.success(f"Market View: **{market_view}** | Bias Score: {total_score}")
        
        st.markdown(f"### Support Zone: `{support_str}`")
        st.markdown(f"### Resistance Zone: `{resistance_str}`")

        if suggested_trade:
            st.info(f"{atm_signal}\n{suggested_trade}")
        
        # Display option chain summary
        with st.expander("Option Chain Summary"):
            st.dataframe(df_summary)
        
        # Display recent trade log
        trade_data = get_trade_log()
        if trade_data:
            st.markdown("### Recent Trades")
            df_trades = pd.DataFrame(trade_data)
            df_trades.rename(columns={
                'option_type': 'Type',
                'strike': 'Strike',
                'entry_price': 'Entry',
                'target_price': 'Target',
                'stop_loss': 'SL',
                'target_hit': 'Target Hit',
                'sl_hit': 'SL Hit'
            }, inplace=True)
            st.dataframe(df_trades[['Type', 'Strike', 'Entry', 'Target', 'SL', 'Target Hit', 'SL Hit']])

    except Exception as e:
        st.error(f"Error in signal generation: {e}")

# Complete analyze function
def analyze_complete():
    result = analyze()
    if result[0] is None:
        return
    
    df, underlying, now = result
    
    analysis_result = process_analysis(df, underlying, now)
    if analysis_result[0] is None:
        return
    
    df_summary, bias_results, atm_row, market_view, support_zone, resistance_zone, support_str, resistance_str, total_score, df = analysis_result
    
    generate_signals_and_display(df_summary, bias_results, atm_row, market_view, support_zone, resistance_zone, support_str, resistance_str, total_score, df, underlying, now)
    # === Main Function Call ===
if __name__ == "__main__":
    analyze_complete()
