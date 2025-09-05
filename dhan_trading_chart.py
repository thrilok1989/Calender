import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pytz import timezone
import json
from supabase import create_client, Client

# Streamlit Configuration
st.set_page_config(
    page_title="DhanHQ Trading Dashboard with Enhanced POC",
    layout="wide",
    initial_sidebar_state="expanded"
)

def is_market_hours():
    """Check if current time is within market hours"""
    now = datetime.now(timezone('Asia/Kolkata'))
    
    if now.weekday() >= 5:
        return False
    
    market_start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= now <= market_end

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

# Initialize session state
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'current_poc' not in st.session_state:
    st.session_state.current_poc = None
if 'spot_price' not in st.session_state:
    st.session_state.spot_price = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = pd.DataFrame()

DHAN_API_BASE_URL = "https://api.dhan.co/v2"

def get_dhan_config():
    try:
        return {
            "client_id": st.secrets.get("DHAN_CLIENT_ID"),
            "access_token": st.secrets.get("DHAN_ACCESS_TOKEN")
        }
    except:
        return {"client_id": None, "access_token": None}

dhan_config = get_dhan_config()

def get_dhan_headers():
    if not dhan_config["client_id"] or not dhan_config["access_token"]:
        return None
    
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": dhan_config["access_token"],
        "client-id": dhan_config["client_id"]
    }

def send_telegram_notification(message):
    try:
        bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=data, timeout=5)
        return response.status_code == 200
    except:
        return False

def store_poc_data(poc_data):
    if not supabase:
        return False
    
    try:
        supabase.table("poc_history").insert(poc_data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to store POC data: {e}")
        return False

def store_proximity_alert(alert_data):
    if not supabase:
        return False
    
    try:
        supabase.table("proximity_alerts").insert(alert_data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to store proximity alert: {e}")
        return False

def get_poc_history():
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table("poc_history").select("*").order("created_at", desc=True).limit(50).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Failed to retrieve POC history: {e}")
        return pd.DataFrame()

def get_proximity_alerts():
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table("proximity_alerts").select("*").order("created_at", desc=True).limit(20).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Failed to retrieve proximity alerts: {e}")
        return pd.DataFrame()

def delete_all_history():
    if not supabase:
        return False
    
    try:
        supabase.table("poc_history").delete().neq("id", 0).execute()
        supabase.table("proximity_alerts").delete().neq("id", 0).execute()
        return True
    except Exception as e:
        st.error(f"Failed to delete history: {e}")
        return False

def make_api_request(url, headers, payload, timeout=15):
    try:
        st.session_state.api_call_count += 1
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 429:
            st.warning("Rate limit reached. Waiting 3 seconds...")
            time.sleep(3)
            return None
        elif response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def get_live_data(security_id, exchange_segment):
    headers = get_dhan_headers()
    if not headers:
        return None
    
    try:
        payload = {exchange_segment: [int(security_id)]}
        
        data = make_api_request(f"{DHAN_API_BASE_URL}/marketfeed/quote", headers, payload)
        
        if data and 'data' in data and exchange_segment in data['data']:
            security_data = data['data'][exchange_segment].get(str(security_id), {})
            
            if security_data:
                return {
                    'last_price': security_data.get('last_price', 0),
                    'volume': security_data.get('volume', 0),
                    'high': security_data.get('ohlc', {}).get('high', 0),
                    'low': security_data.get('ohlc', {}).get('low', 0),
                    'open': security_data.get('ohlc', {}).get('open', 0),
                    'close': security_data.get('ohlc', {}).get('close', 0),
                    'timestamp': datetime.now(timezone('Asia/Kolkata'))
                }
        return None
    except Exception as e:
        st.error(f"Live data error: {str(e)}")
        return None

def calculate_improved_poc(price_data):
    if not price_data or len(price_data) == 0:
        return None
    
    try:
        df = pd.DataFrame(price_data)
        
        if df.empty or 'volume' not in df.columns or df['volume'].sum() == 0:
            return None
        
        # Calculate typical price (HLC/3) for each candle
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP as POC proxy
        df['volume_price'] = df['typical_price'] * df['volume']
        total_volume_price = df['volume_price'].sum()
        total_volume = df['volume'].sum()
        
        poc_price = total_volume_price / total_volume if total_volume > 0 else 0
        
        # Analyze POC strength and characteristics
        poc_analysis = analyze_poc_strength(df, poc_price)
        
        return {
            'poc_price': poc_price,
            'total_volume': total_volume,
            'strength': poc_analysis['strength'],
            'strength_score': poc_analysis['strength_score'],
            'role': poc_analysis['role'],
            'confidence': poc_analysis['confidence']
        }
        
    except Exception as e:
        st.error(f"POC calculation error: {e}")
        return None

def analyze_poc_strength(df, poc_price):
    try:
        # Calculate distance of each candle from POC
        df['poc_distance'] = abs(df['typical_price'] - poc_price)
        df['poc_percentage'] = df['poc_distance'] / poc_price * 100
        
        # Volume concentration analysis
        close_to_poc = df[df['poc_percentage'] <= 0.5]  # Within 0.5% of POC
        volume_at_poc = close_to_poc['volume'].sum()
        total_volume = df['volume'].sum()
        
        volume_concentration = volume_at_poc / total_volume if total_volume > 0 else 0
        
        # Price action analysis around POC
        above_poc = df[df['typical_price'] > poc_price]
        below_poc = df[df['typical_price'] <= poc_price]
        
        volume_above = above_poc['volume'].sum()
        volume_below = below_poc['volume'].sum()
        
        # Analyze recent price action around POC
        recent_candles = df.tail(10)
        recent_closes = recent_candles['close']
        recent_lows = recent_candles['low']
        recent_highs = recent_candles['high']
        
        # CORRECTED: Support means price is ABOVE POC and POC acts as support level
        # When price approaches POC from above, it should bounce UP (support behavior)
        support_tests = len(recent_candles[
            (recent_lows <= poc_price * 1.005) & (recent_closes >= poc_price)
        ])  # Price dipped to POC but closed at or above it
        
        # CORRECTED: Resistance means price is BELOW POC and POC acts as resistance level  
        # When price approaches POC from below, it should get rejected DOWN (resistance behavior)
        resistance_tests = len(recent_candles[
            (recent_highs >= poc_price * 0.995) & (recent_closes <= poc_price)
        ])  # Price reached POC but closed at or below it
        
        # Determine the average position of recent closes relative to POC
        avg_recent_close = recent_closes.mean()
        price_above_poc = avg_recent_close > poc_price
        
        # CORRECTED LOGIC:
        # If price is generally trading ABOVE POC → POC is acting as SUPPORT
        # If price is generally trading BELOW POC → POC is acting as RESISTANCE
        
        if price_above_poc and support_tests >= resistance_tests:
            role = "Support"  # Price above POC, POC supporting from below
        elif not price_above_poc and resistance_tests >= support_tests:
            role = "Resistance"  # Price below POC, POC resisting from above
        elif price_above_poc:
            role = "Support"  # Default: if price above POC, it's likely support
        elif not price_above_poc:
            role = "Resistance"  # Default: if price below POC, it's likely resistance
        else:
            role = "Neutral"
        
        # Calculate strength score (0-100)
        total_tests = support_tests + resistance_tests
        
        strength_factors = {
            'volume_concentration': volume_concentration * 40,  # 40% weight
            'poc_tests': min(total_tests, 5) * 8,  # 40% weight (max 5 tests)
            'volume_balance': (1 - abs(volume_above - volume_below) / total_volume) * 20  # 20% weight
        }
        
        strength_score = sum(strength_factors.values())
        
        # Classify strength
        if strength_score >= 70:
            strength = "Very Strong"
        elif strength_score >= 50:
            strength = "Strong"
        elif strength_score >= 30:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Confidence level
        confidence = min(100, volume_concentration * 100 + total_tests * 10)
        
        return {
            'strength': strength,
            'strength_score': round(strength_score, 2),
            'role': role,
            'confidence': round(confidence, 2),
            'support_tests': support_tests,
            'resistance_tests': resistance_tests,
            'volume_concentration': round(volume_concentration * 100, 2),
            'price_above_poc': price_above_poc
        }
        
    except Exception as e:
        st.error(f"POC strength analysis error: {e}")
        return {
            'strength': 'Unknown',
            'strength_score': 0,
            'role': 'Neutral',
            'confidence': 0,
            'support_tests': 0,
            'resistance_tests': 0,
            'volume_concentration': 0,
            'price_above_poc': False
        }

def check_poc_change(new_poc_data, current_poc, threshold=0.1):
    if not new_poc_data or current_poc is None:
        return True
    
    new_price = new_poc_data['poc_price']
    price_change_pct = abs(new_price - current_poc) / current_poc * 100
    return price_change_pct > threshold

def check_spot_poc_proximity(spot_price, poc_price, proximity_threshold=0.2):
    if not spot_price or not poc_price:
        return False
    
    distance_pct = abs(spot_price - poc_price) / poc_price * 100
    return distance_pct <= proximity_threshold

def update_poc_analysis(price_data, current_spot_price):
    poc_data = calculate_improved_poc(price_data)
    
    if not poc_data:
        return None
    
    current_time = datetime.now(timezone('Asia/Kolkata'))
    
    # Check for POC change
    if check_poc_change(poc_data, st.session_state.current_poc):
        old_poc = st.session_state.current_poc
        st.session_state.current_poc = poc_data['poc_price']
        
        # Prepare data for Supabase storage
        poc_entry = {
            'timestamp': current_time.strftime('%H:%M:%S'),
            'old_poc': old_poc,
            'new_poc': poc_data['poc_price'],
            'strength': poc_data['strength'],
            'strength_score': poc_data['strength_score'],
            'role': poc_data['role'],
            'confidence': poc_data['confidence'],
            'total_volume': poc_data['total_volume'],
            'change_pct': ((poc_data['poc_price'] - old_poc) / old_poc * 100) if old_poc else 0,
            'created_at': current_time.isoformat()
        }
        
        # Store in Supabase
        store_poc_data(poc_entry)
        
        # Send notification
        if old_poc:
            change_direction = "UP" if poc_data['poc_price'] > old_poc else "DOWN"
            change_pct = abs(poc_entry['change_pct'])
            
            message = f"""POC CHANGED {change_direction}
Old POC: {old_poc:.2f}
New POC: {poc_data['poc_price']:.2f}
Change: {change_pct:.2f}%
Strength: {poc_data['strength']} ({poc_data['strength_score']:.1f}/100)
Role: {poc_data['role']}
Confidence: {poc_data['confidence']:.1f}%
Time: {current_time.strftime('%H:%M:%S')}"""
        else:
            message = f"""POC CALCULATED
POC: {poc_data['poc_price']:.2f}
Strength: {poc_data['strength']} ({poc_data['strength_score']:.1f}/100)
Role: {poc_data['role']}
Confidence: {poc_data['confidence']:.1f}%
Time: {current_time.strftime('%H:%M:%S')}"""
        
        send_telegram_notification(message)
    
    # Check spot price proximity to POC
    if current_spot_price and check_spot_poc_proximity(current_spot_price, poc_data['poc_price']):
        last_alert = getattr(st.session_state, 'last_proximity_alert', None)
        if not last_alert or (current_time - last_alert).seconds > 60:
            
            proximity_entry = {
                'timestamp': current_time.strftime('%H:%M:%S'),
                'spot_price': current_spot_price,
                'poc_price': poc_data['poc_price'],
                'distance': abs(current_spot_price - poc_data['poc_price']),
                'distance_pct': abs(current_spot_price - poc_data['poc_price']) / poc_data['poc_price'] * 100,
                'poc_strength': poc_data['strength'],
                'poc_role': poc_data['role'],
                'created_at': current_time.isoformat()
            }
            
            # Store in Supabase
            store_proximity_alert(proximity_entry)
            
            message = f"""SPOT NEAR POC ({poc_data['role']})
Spot: {current_spot_price:.2f}
POC: {poc_data['poc_price']:.2f} ({poc_data['strength']})
Distance: {proximity_entry['distance']:.2f} ({proximity_entry['distance_pct']:.2f}%)
Time: {current_time.strftime('%H:%M:%S')}"""
            
            send_telegram_notification(message)
            st.session_state.last_proximity_alert = current_time
    
    return poc_data

class TradingDashboard:
    def __init__(self):
        self.instruments = {
            "NIFTY 50": {"id": 26000, "segment": "NSE_EQ", "type": "EQUITY"},
            "BANK NIFTY": {"id": 26001, "segment": "NSE_EQ", "type": "EQUITY"},
            "NIFTY INDEX": {"id": 13, "segment": "IDX_I", "type": "INDEX"}
        }
    
    def display_configuration(self):
        st.sidebar.markdown("## Configuration")
        
        market_open = is_market_hours()
        status_color = "🟢" if market_open else "🔴"
        st.sidebar.markdown(f"{status_color} **Market Status:** {'Open' if market_open else 'Closed'}")
        
        selected_instrument = st.sidebar.selectbox(
            "Select Instrument",
            list(self.instruments.keys()),
            index=2
        )
        
        data_mode = st.sidebar.radio(
            "Select Data Mode",
            ["Live Only", "Historical Only", "Historical + Live"],
            index=0
        )
        
        if market_open:
            st.sidebar.info("Auto-updating every 25 seconds during market hours")
            update_interval = 25
        else:
            update_interval = st.sidebar.slider(
                "Manual Update Interval (seconds)",
                min_value=10,
                max_value=120,
                value=30
            )
        
        poc_change_threshold = st.sidebar.slider(
            "POC Change Threshold (%)",
            min_value=0.05,
            max_value=1.0,
            value=0.1,
            step=0.05
        )
        
        poc_proximity_threshold = st.sidebar.slider(
            "POC Proximity Alert (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.05
        )
        
        return {
            "instrument": selected_instrument,
            "update_interval": update_interval,
            "data_mode": data_mode,
            "market_open": market_open,
            "poc_change_threshold": poc_change_threshold,
            "poc_proximity_threshold": poc_proximity_threshold
        }
    
    def display_status(self):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Calls", st.session_state.api_call_count)
        
        with col2:
            last_update = st.session_state.last_update
            if last_update:
                time_diff = datetime.now(timezone('Asia/Kolkata')) - last_update
                st.metric("Last Update", f"{time_diff.seconds}s ago")
            else:
                st.metric("Last Update", "Never")
        
        with col3:
            if st.session_state.current_poc:
                st.metric("Current POC", f"{st.session_state.current_poc:.2f}")
            else:
                st.metric("Current POC", "Calculating...")
        
        with col4:
            if st.session_state.spot_price:
                st.metric("Spot Price", f"{st.session_state.spot_price:.2f}")
            else:
                st.metric("Spot Price", "Fetching...")
    
    def display_poc_analysis(self):
        st.markdown("## Point of Control (POC) Analysis")
        
        poc_history_df = get_poc_history()
        proximity_alerts_df = get_proximity_alerts()
        
        if not poc_history_df.empty:
            latest_poc = poc_history_df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.info(f"**POC:** {latest_poc.get('new_poc', 0):.2f}")
            
            with col2:
                strength = latest_poc.get('strength', 'Unknown')
                strength_score = latest_poc.get('strength_score', 0)
                if strength_score > 60:
                    st.success(f"**Strength:** {strength} ({strength_score:.1f}/100)")
                elif strength_score > 30:
                    st.warning(f"**Strength:** {strength} ({strength_score:.1f}/100)")
                else:
                    st.error(f"**Strength:** {strength} ({strength_score:.1f}/100)")
            
            with col3:
                role = latest_poc.get('role', 'Neutral')
                if role == "Support":
                    st.info(f"**Role:** {role}")
                elif role == "Resistance":
                    st.warning(f"**Role:** {role}")
                else:
                    st.write(f"**Role:** {role}")
            
            with col4:
                confidence = latest_poc.get('confidence', 0)
                if confidence > 70:
                    st.success(f"**Confidence:** {confidence:.1f}%")
                elif confidence > 40:
                    st.warning(f"**Confidence:** {confidence:.1f}%")
                else:
                    st.error(f"**Confidence:** {confidence:.1f}%")
            
            if st.session_state.spot_price and latest_poc.get('new_poc'):
                distance = abs(st.session_state.spot_price - latest_poc['new_poc'])
                distance_pct = (distance / latest_poc['new_poc']) * 100
                
                st.markdown(f"""
                **Distance Analysis:**
                - Absolute Distance: {distance:.2f} points
                - Percentage Distance: {distance_pct:.2f}%
                - Status: {'🟢 Near POC' if distance_pct < 0.5 else '🟡 Moderate Distance' if distance_pct < 1.0 else '🔴 Far from POC'}
                """)
        
        if not poc_history_df.empty:
            st.markdown("### POC Change History")
            
            display_columns = ['timestamp', 'new_poc', 'strength', 'strength_score', 'role', 'confidence', 'change_pct']
            available_columns = [col for col in display_columns if col in poc_history_df.columns]
            
            if available_columns:
                recent_changes = poc_history_df[available_columns].head(10)
                recent_changes.columns = ['Time', 'POC', 'Strength', 'Score', 'Role', 'Confidence%', 'Change%']
                st.dataframe(recent_changes, use_container_width=True)
        
        if not proximity_alerts_df.empty:
            st.markdown("### POC Proximity Alerts")
            
            display_columns = ['timestamp', 'spot_price', 'poc_price', 'distance_pct', 'poc_strength', 'poc_role']
            available_columns = [col for col in display_columns if col in proximity_alerts_df.columns]
            
            if available_columns:
                recent_alerts = proximity_alerts_df[available_columns].head(5)
                recent_alerts.columns = ['Time', 'Spot', 'POC', 'Distance%', 'Strength', 'Role']
                st.dataframe(recent_alerts, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ Delete All History", type="secondary"):
                if st.session_state.get('confirm_delete', False):
                    with st.spinner("Deleting all history from database..."):
                        if delete_all_history():
                            st.success("All history deleted successfully!")
                            st.session_state.confirm_delete = False
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Failed to delete history")
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Click again to confirm deletion of ALL history")
        
        with col2:
            if st.session_state.get('confirm_delete', False):
                if st.button("Cancel", type="primary"):
                    st.session_state.confirm_delete = False
                    st.rerun()
    
    def fetch_and_update_data(self, config):
        instrument_info = self.instruments[config["instrument"]]
        
        if config["data_mode"] in ["Live Only", "Historical + Live"]:
            live_data = get_live_data(
                instrument_info["id"],
                instrument_info["segment"]
            )
            
            if live_data:
                live_df = pd.DataFrame([{
                    'datetime': live_data['timestamp'],
                    'open': live_data['open'],
                    'high': live_data['high'],
                    'low': live_data['low'],
                    'close': live_data['last_price'],
                    'volume': live_data['volume']
                }])
                
                st.session_state.live_data = live_df
                st.session_state.last_update = datetime.now(timezone('Asia/Kolkata'))

def main():
    st.title("DhanHQ Trading Dashboard with Enhanced POC Analysis")
    
    # Check DhanHQ credentials
    if not dhan_config["client_id"] or not dhan_config["access_token"]:
        st.error("DhanHQ credentials not configured!")
        st.markdown("""
        ### Setup Instructions:
        Add your credentials to Streamlit secrets:
        ```
        DHAN_CLIENT_ID = "your_client_id"
        DHAN_ACCESS_TOKEN = "your_access_token"
        SUPABASE_URL = "your_supabase_url"
        SUPABASE_KEY = "your_supabase_anon_key"
        ```
        """)
        return
    
    # Check Supabase connection
    if not supabase:
        st.error("Supabase connection failed. Check your credentials.")
        return
    
    current_time = datetime.now(timezone('Asia/Kolkata'))
    st.markdown(f"**Current IST Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    market_open = is_market_hours()
    
    dashboard = TradingDashboard()
    config = dashboard.display_configuration()
    dashboard.display_status()
    
    should_fetch_data = False
    
    if market_open and config["data_mode"] in ["Live Only", "Historical + Live"]:
        if (st.session_state.last_update is None or 
            (current_time - st.session_state.last_update).seconds >= 25):
            should_fetch_data = True
        
        if st.session_state.last_update:
            seconds_since_update = (current_time - st.session_state.last_update).seconds
            seconds_until_next = 25 - (seconds_since_update % 25)
            st.info(f"Next auto-update in: {seconds_until_next} seconds")
    
    manual_fetch = st.sidebar.button("Fetch Data Now")
    
    if should_fetch_data or manual_fetch:
        with st.spinner("Fetching live data and updating POC analysis..."):
            dashboard.fetch_and_update_data(config)
            
            if not st.session_state.live_data.empty:
                latest_price = st.session_state.live_data['close'].iloc[-1]
                st.session_state.spot_price = latest_price
                
                price_data_list = st.session_state.live_data.to_dict('records')
                poc_result = update_poc_analysis(price_data_list, st.session_state.spot_price)
                
                if poc_result:
                    strength_indicator = {
                        "Very Strong": "🟢",
                        "Strong": "🟡", 
                        "Moderate": "🟠",
                        "Weak": "🔴"
                    }.get(poc_result['strength'], "⚪")
                    
                    role_indicator = {
                        "Support": "🛡️",
                        "Resistance": "🚧", 
                        "Neutral": "⚖️"
                    }.get(poc_result['role'], "⚖️")
                    
                    st.success(f"{strength_indicator} POC: {poc_result['poc_price']:.2f} | Strength: {poc_result['strength']} ({poc_result['strength_score']:.1f}/100) | {role_indicator} Role: {poc_result['role']}")
        
        if market_open and not manual_fetch:
            time.sleep(1)
            st.rerun()
    
    dashboard.display_poc_analysis()
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Export")
    
    if st.sidebar.button("Export Data"):
        poc_history_df = get_poc_history()
        proximity_alerts_df = get_proximity_alerts()
        
        export_data = {
            'export_timestamp': current_time.isoformat(),
            'current_poc': st.session_state.current_poc,
            'current_spot': st.session_state.spot_price,
            'live_data': st.session_state.live_data.to_dict('records') if not st.session_state.live_data.empty else [],
            'poc_history': poc_history_df.to_dict('records') if not poc_history_df.empty else [],
            'proximity_alerts': proximity_alerts_df.to_dict('records') if not proximity_alerts_df.empty else []
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        st.sidebar.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"trading_data_{current_time.strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# Auto-refresh during market hours
if is_market_hours():
    time.sleep(0.5)
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    st.session_state.refresh_counter += 1
    
    if st.session_state.refresh_counter % 50 == 0:
        st.rerun()

if __name__ == "__main__":
    main()
