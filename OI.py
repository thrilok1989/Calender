# streamlit_app.py - Enhanced Part 1: Imports and Configuration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
from supabase import create_client, Client
import os
from typing import Dict, List, Optional
import json
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SR Scanner Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Enhanced Part 2: Telegram Notifier Class
class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """Initialize Telegram notifier"""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}" if bot_token else None
        
    def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        if not self.bot_token or not self.chat_id:
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        if not self.bot_token:
            return False
            
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def format_alert_message(self, alert_type: str, underlying: str, data: Dict) -> str:
        """Format alert message for Telegram"""
        strike = data['strike']
        option_type = data['type']
        
        if alert_type == "FAST_TRANSITION":
            emoji = "🚨"
            title = "FAST TRANSITION ALERT"
            message = f"""
{emoji} <b>{title}</b> {emoji}

<b>Underlying:</b> {underlying}
<b>Strike:</b> {strike} {option_type}
<b>Direction:</b> {data['transition_direction']}
<b>Speed:</b> {data['transition_speed']}
<b>SR Score:</b> {data['sr_score']:.1f}
<b>Current State:</b> {data['current_state']}
<b>Velocity:</b> {data['score_velocity']:.2f}

<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>
"""
        elif alert_type == "HIGH_VELOCITY":
            emoji = "⚡"
            title = "HIGH VELOCITY ALERT"
            message = f"""
{emoji} <b>{title}</b> {emoji}

<b>Underlying:</b> {underlying}
<b>Strike:</b> {strike} {option_type}
<b>Velocity:</b> {data['score_velocity']:.2f}
<b>Acceleration:</b> {data['score_acceleration']:.2f}
<b>SR Score:</b> {data['sr_score']:.1f}
<b>Direction:</b> {data['transition_direction']}

<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>
"""
        elif alert_type == "STRONG_LEVEL":
            emoji = "🎯"
            level_type = "SUPPORT" if data['sr_score'] > 0 else "RESISTANCE"
            title = f"STRONG {level_type} LEVEL"
            message = f"""
{emoji} <b>{title}</b> {emoji}

<b>Underlying:</b> {underlying}
<b>Strike:</b> {strike} {option_type}
<b>SR Score:</b> {data['sr_score']:.1f}
<b>State:</b> {data['current_state']}
<b>OI Change:</b> {data['oi_change_pct']:.2f}%
<b>Bid/Ask Ratio:</b> {data['bid_ask_ratio']:.2f}

<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>
"""
        elif alert_type == "BREAKOUT":
            emoji = "💥"
            title = "POTENTIAL BREAKOUT"
            message = f"""
{emoji} <b>{title}</b> {emoji}

<b>Underlying:</b> {underlying}
<b>Strike:</b> {strike} {option_type}
<b>Previous State:</b> Strong Level
<b>Current State:</b> {data['current_state']}
<b>Speed:</b> {data['transition_speed']}
<b>Score Change:</b> {data['sr_score']:.1f}

<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>
"""
        elif alert_type == "STATE_CHANGE":
            emoji = "🔄"
            title = "STATE CHANGE ALERT"
            message = f"""
{emoji} <b>{title}</b> {emoji}

<b>Underlying:</b> {underlying}
<b>Strike:</b> {strike} {option_type}
<b>New State:</b> {data['current_state']}
<b>Direction:</b> {data['transition_direction']}
<b>Speed:</b> {data['transition_speed']}
<b>SR Score:</b> {data['sr_score']:.1f}

<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>
"""
        else:
            message = f"Alert: {underlying} {strike} {option_type} - {alert_type}"
        
        return message
    
    def send_batch_alerts(self, alerts: List[Dict]) -> bool:
        """Send multiple alerts in batch"""
        if not alerts:
            return True
            
        # Group alerts by type
        alert_summary = {}
        for alert in alerts:
            alert_type = alert['type']
            if alert_type not in alert_summary:
                alert_summary[alert_type] = []
            alert_summary[alert_type].append(alert)
        
        # Send summary message first
        summary_message = f"""
📊 <b>MARKET ALERT SUMMARY</b> 📊
<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>

"""
        
        for alert_type, alert_list in alert_summary.items():
            summary_message += f"<b>{alert_type}:</b> {len(alert_list)} alerts\n"
        
        summary_message += f"\n<i>Total Alerts: {len(alerts)}</i>"
        
        success = self.send_message(summary_message)
        
        # Send individual alerts for critical ones
        critical_alerts = [a for a in alerts if a['type'] in ['FAST_TRANSITION', 'BREAKOUT', 'HIGH_VELOCITY']]
        for alert in critical_alerts[:5]:  # Limit to 5 detailed alerts to avoid spam
            detailed_message = self.format_alert_message(
                alert['type'], alert['underlying'], alert['data']
            )
            self.send_message(detailed_message)
            time.sleep(0.5)  # Small delay between messages
        
        return success
    
    def send_startup_message(self, underlying: str) -> bool:
        """Send startup notification"""
        message = f"""
🤖 <b>SR Scanner Started</b> 🤖

<b>Underlying:</b> {underlying}
<b>Status:</b> Monitoring Active
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Ready to send alerts for:
• Fast Transitions
• High Velocity Changes  
• Strong S/R Levels
• Potential Breakouts
"""
        return self.send_message(message)
# Enhanced Part 3: SupabaseManager Class
class SupabaseManager:
    def __init__(self):
        """Initialize Supabase client"""
        try:
            self.url = st.secrets["SUPABASE_URL"]
            self.key = st.secrets["SUPABASE_ANON_KEY"]
            self.supabase: Client = create_client(self.url, self.key)
            self.create_tables()
        except Exception as e:
            st.error(f"Supabase connection failed: {e}")
            self.supabase = None
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.supabase:
            return
        
        try:
            # Check if table exists
            result = self.supabase.table('market_data').select("id").limit(1).execute()
        except:
            # Table doesn't exist, show instructions
            st.sidebar.warning("""
            Database tables not found. 
            Please run the SQL setup script in your Supabase dashboard.
            """)
    
    def save_market_data(self, data: List[Dict]):
        """Save market data to Supabase"""
        if not self.supabase or not data:
            return False
        
        try:
            result = self.supabase.table('market_data').insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def get_historical_data(self, underlying: str, strike: float, option_type: str, 
                           hours_back: int = 24) -> pd.DataFrame:
        """Get historical data for analysis"""
        if not self.supabase:
            return pd.DataFrame()
        
        try:
            since = datetime.now() - timedelta(hours=hours_back)
            
            result = self.supabase.table('market_data')\
                .select("*")\
                .eq('underlying', underlying)\
                .eq('strike', strike)\
                .eq('option_type', option_type)\
                .gte('timestamp', since.isoformat())\
                .order('timestamp')\
                .execute()
            
            return pd.DataFrame(result.data)
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_latest_scan_data(self, underlying: str, minutes_back: int = 10) -> pd.DataFrame:
        """Get latest scan data"""
        if not self.supabase:
            return pd.DataFrame()
        
        try:
            since = datetime.now() - timedelta(minutes=minutes_back)
            
            result = self.supabase.table('market_data')\
                .select("*")\
                .eq('underlying', underlying)\
                .gte('timestamp', since.isoformat())\
                .order('timestamp', desc=True)\
                .execute()
            
            df = pd.DataFrame(result.data)
            if not df.empty:
                # Get latest record for each strike-type combination
                df = df.groupby(['strike', 'option_type']).first().reset_index()
            return df
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return pd.DataFrame()
    
    def get_previous_states(self, underlying: str, strikes: List[float], 
                           minutes_back: int = 30) -> Dict:
        """Get previous states for comparison"""
        if not self.supabase:
            return {}
        
        try:
            since = datetime.now() - timedelta(minutes=minutes_back)
            until = datetime.now() - timedelta(minutes=5)
            
            result = self.supabase.table('market_data')\
                .select("strike,option_type,current_state,sr_score")\
                .eq('underlying', underlying)\
                .in_('strike', strikes)\
                .gte('timestamp', since.isoformat())\
                .lt('timestamp', until.isoformat())\
                .order('timestamp', desc=True)\
                .execute()
            
            df = pd.DataFrame(result.data)
            if not df.empty:
                # Get most recent state for each strike-type
                df = df.groupby(['strike', 'option_type']).first().reset_index()
                
                previous_states = {}
                for _, row in df.iterrows():
                    key = f"{row['strike']}_{row['option_type']}"
                    previous_states[key] = {
                        'state': row['current_state'],
                        'score': row['sr_score']
                    }
                return previous_states
            
            return {}
        except Exception as e:
            logger.error(f"Error fetching previous states: {e}")
            return {}
    
    def save_alert_log(self, alert_data: Dict):
        """Save alert to database for tracking"""
        if not self.supabase:
            return False
        
        try:
            # Create alerts table if it doesn't exist (run this in Supabase SQL editor first)
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'underlying': alert_data['underlying'],
                'strike': alert_data['data']['strike'],
                'option_type': alert_data['data']['type'],
                'alert_type': alert_data['type'],
                'sr_score': alert_data['data']['sr_score'],
                'transition_direction': alert_data['data']['transition_direction'],
                'transition_speed': alert_data['data']['transition_speed'],
                'score_velocity': alert_data['data']['score_velocity']
            }
            
            # Note: You'll need to create this table in Supabase
            # result = self.supabase.table('alert_logs').insert(alert_record).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving alert log: {e}")
            return False
# Enhanced Part 4: DhanSRScanner Class - Initialization and API Methods
class DhanSRScanner:
    def __init__(self, access_token: str, client_id: str, db_manager: SupabaseManager):
        """Initialize scanner with database integration"""
        self.access_token = access_token
        self.client_id = client_id
        self.db_manager = db_manager
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            "access-token": access_token,
            "client-id": client_id,
            "Content-Type": "application/json"
        }
        
        # Underlying mapping for Dhan API
        self.underlying_map = {
            "NIFTY": {"scrip": 13, "segment": "IDX_I"},
            "BANKNIFTY": {"scrip": 25, "segment": "IDX_I"},
            "FINNIFTY": {"scrip": 27, "segment": "IDX_I"}
        }
        
        # Store previous scan data for comparison
        self.previous_scan_data = {}
    
    def get_expiry_list(self, underlying: str) -> List[str]:
        """Get expiry list for underlying"""
        try:
            url = f"{self.base_url}/optionchain/expirylist"
            data = {
                "UnderlyingScrip": self.underlying_map[underlying]["scrip"],
                "UnderlyingSeg": self.underlying_map[underlying]["segment"]
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"Error fetching expiry list: {e}")
            return []
    
    def get_option_chain(self, underlying: str, expiry: str) -> Dict:
        """Fetch option chain data from Dhan API"""
        try:
            url = f"{self.base_url}/optionchain"
            data = {
                "UnderlyingScrip": self.underlying_map[underlying]["scrip"],
                "UnderlyingSeg": self.underlying_map[underlying]["segment"],
                "Expiry": expiry
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                return result.get("data", {})
            return {}
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return {}
    
    def get_market_quotes(self, security_ids: List[str], exchange_segment: str = "NSE_FNO") -> Dict:
        """Fetch market quotes for multiple securities"""
        try:
            url = f"{self.base_url}/marketfeed/quote"
            data = {exchange_segment: security_ids}
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                return result.get("data", {}).get(exchange_segment, {})
            return {}
        except Exception as e:
            logger.error(f"Error fetching market quotes: {e}")
            return {}
    
    def find_atm_strikes(self, spot_price: float, option_chain: Dict) -> List[float]:
        """Find ATM and ATM±1 strike prices"""
        if not option_chain or 'oc' not in option_chain:
            return []
        
        strikes = [float(strike) for strike in option_chain['oc'].keys()]
        strikes.sort()
        
        # Find closest strike to spot price (ATM)
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        atm_index = strikes.index(atm_strike)
        
        result_strikes = []
        # Get ATM±1 strikes
        if atm_index > 0:
            result_strikes.append(strikes[atm_index - 1])  # ATM-1
        result_strikes.append(atm_strike)  # ATM
        if atm_index < len(strikes) - 1:
            result_strikes.append(strikes[atm_index + 1])  # ATM+1
        
        return result_strikes
# Enhanced Part 5: SR Score Calculation and Analysis Methods
    def calculate_sr_score_with_history(self, strike: float, option_type: str, 
                                       option_data: Dict, spot_price: float) -> Dict:
        """Calculate SR score using current data and database history"""
        
        # Get historical data for this strike
        historical_df = self.db_manager.get_historical_data(
            "NIFTY", strike, option_type.lower(), hours_back=6
        )
        
        score = 0
        
        # Current data from option chain
        current_oi = option_data.get('oi', 0)
        prev_oi = option_data.get('previous_oi', current_oi)
        current_volume = option_data.get('volume', 0)
        prev_volume = option_data.get('previous_volume', 0)
        ltp = option_data.get('last_price', 0)
        prev_close = option_data.get('previous_close_price', ltp)
        
        # Calculate changes
        oi_change = current_oi - prev_oi
        oi_change_pct = (oi_change / prev_oi * 100) if prev_oi > 0 else 0
        price_change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0
        volume_ratio = current_volume / prev_volume if prev_volume > 0 else 1
        
        # OI Analysis (40% weight)
        if option_type.upper() == 'PE':
            # For PE: Increasing OI = stronger support
            score += min(oi_change_pct * 2, 40) if oi_change_pct > 0 else max(oi_change_pct * 2, -40)
        else:
            # For CE: Increasing OI = stronger resistance
            score -= min(oi_change_pct * 2, 40) if oi_change_pct > 0 else max(oi_change_pct * 2, -40)
        
        # Order flow analysis (30% weight) - using bid/ask data
        bid_qty = option_data.get('top_bid_quantity', 0)
        ask_qty = option_data.get('top_ask_quantity', 0)
        total_qty = bid_qty + ask_qty
        
        if total_qty > 0:
            bid_ask_ratio = bid_qty / ask_qty if ask_qty > 0 else 10
            
            # Normalize ratio impact
            if bid_ask_ratio > 1:
                flow_score = min(np.log(bid_ask_ratio) * 15, 30)
            else:
                flow_score = max(-np.log(1/bid_ask_ratio) * 15, -30)
            
            if option_type.upper() == 'PE':
                score += flow_score  # Strong buying in PE = support
            else:
                score -= flow_score  # Strong buying in CE = resistance weakening
        
        # Price action analysis (20% weight)
        distance_from_spot = abs(strike - spot_price) / spot_price
        proximity_weight = max(0.5, 1 - distance_from_spot * 10)
        
        if option_type.upper() == 'PE':
            # PE losing value = support weakening
            score -= price_change_pct * proximity_weight
        else:
            # CE losing value = resistance holding
            score -= price_change_pct * proximity_weight
        
        # Volume analysis (10% weight)
        if current_volume > 0:
            volume_score = min(np.log(current_volume + 1) * 2, 10)
            if option_type.upper() == 'PE':
                score += volume_score * 0.5
            else:
                score -= volume_score * 0.5
        
        # Historical velocity and acceleration from database
        velocity = 0
        acceleration = 0
        if len(historical_df) >= 3:
            scores = historical_df['sr_score'].tail(5).values
            if len(scores) >= 2:
                velocity = (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0
            if len(scores) >= 4:
                old_velocity = scores[-3] - scores[-4]
                new_velocity = scores[-1] - scores[-2]
                acceleration = new_velocity - old_velocity
        
        return {
            'sr_score': max(-100, min(100, score)),
            'oi_change': oi_change,
            'oi_change_pct': oi_change_pct,
            'price_change_pct': price_change_pct,
            'velocity': velocity,
            'acceleration': acceleration,
            'historical_points': len(historical_df),
            'bid_ask_ratio': bid_qty / ask_qty if ask_qty > 0 else 10
        }
    
    def determine_transition_state(self, score_data: Dict, option_type: str) -> Dict:
        """Determine transition state and speed"""
        score = score_data['sr_score']
        velocity = score_data['velocity']
        acceleration = score_data['acceleration']
        
        # State determination
        if option_type.upper() == 'PE':
            if score >= 60: state = "STRONG_SUPPORT"
            elif score >= 30: state = "MODERATE_SUPPORT"
            elif score >= 10: state = "WEAK_SUPPORT"
            elif score >= -10: state = "NEUTRAL"
            elif score >= -30: state = "WEAK_RESISTANCE"
            elif score >= -60: state = "MODERATE_RESISTANCE"
            else: state = "STRONG_RESISTANCE"
        else:
            if score <= -60: state = "STRONG_RESISTANCE"
            elif score <= -30: state = "MODERATE_RESISTANCE"
            elif score <= -10: state = "WEAK_RESISTANCE"
            elif score <= 10: state = "NEUTRAL"
            elif score <= 30: state = "WEAK_SUPPORT"
            elif score <= 60: state = "MODERATE_SUPPORT"
            else: state = "STRONG_SUPPORT"
        
        # Transition direction
        if velocity > 5:
            direction = 'SUPPORT_STRENGTHENING' if 'SUPPORT' in state else 'RESISTANCE_WEAKENING'
        elif velocity < -5:
            direction = 'RESISTANCE_STRENGTHENING' if 'RESISTANCE' in state else 'SUPPORT_WEAKENING'
        else:
            direction = 'STABLE'
        
        # Speed calculation
        speed_magnitude = abs(velocity)
        if speed_magnitude > 15:
            speed = 'VERY_FAST' if abs(acceleration) > 3 else 'FAST'
        elif speed_magnitude > 8:
            speed = 'FAST' if abs(acceleration) > 2 else 'MODERATE'
        elif speed_magnitude > 3:
            speed = 'SLOW'
        else:
            speed = 'VERY_SLOW'
        
        return {
            'current_state': state,
            'transition_direction': direction,
            'transition_speed': speed,
            'score_velocity': velocity,
            'score_acceleration': acceleration
        }
    
    def check_and_send_alerts(self, df: pd.DataFrame, underlying: str, 
                             telegram_notifier: TelegramNotifier, 
                             alert_settings: Dict) -> List[Dict]:
        """Check for alert conditions and send Telegram notifications"""
        if df.empty or not telegram_notifier or not telegram_notifier.bot_token:
            return []
        
        alerts = []
        
        # Check for fast transitions
        if alert_settings.get('fast_transitions', True):
            fast_transitions = df[df['transition_speed'].isin(['FAST', 'VERY_FAST'])]
            for _, row in fast_transitions.iterrows():
                alerts.append({
                    'type': 'FAST_TRANSITION',
                    'underlying': underlying,
                    'data': row.to_dict()
                })
        
        # Check for high velocity
        if alert_settings.get('high_velocity', True):
            high_velocity = df[abs(df['score_velocity']) > 10]
            for _, row in high_velocity.iterrows():
                alerts.append({
                    'type': 'HIGH_VELOCITY', 
                    'underlying': underlying,
                    'data': row.to_dict()
                })
        
        # Check for strong levels
        if alert_settings.get('strong_levels', True):
            strong_levels = df[abs(df['sr_score']) > 60]
            for _, row in strong_levels.iterrows():
                alerts.append({
                    'type': 'STRONG_LEVEL',
                    'underlying': underlying,
                    'data': row.to_dict()
                })
        
        # Check for potential breakouts
        if alert_settings.get('breakouts', True):
            potential_breakouts = df[
                (abs(df['sr_score']) < 20) & 
                (df['transition_speed'].isin(['FAST', 'VERY_FAST'])) &
                (abs(df['score_velocity']) > 8)
            ]
            for _, row in potential_breakouts.iterrows():
                alerts.append({
                    'type': 'BREAKOUT',
                    'underlying': underlying,
                    'data': row.to_dict()
                })
        
        # Check for state changes (if we have previous data)
        if alert_settings.get('state_changes', True) and hasattr(self, 'previous_scan_data'):
            strikes = df['strike'].unique()
            previous_states = self.db_manager.get_previous_states(underlying, strikes.tolist())
            
            for _, row in df.iterrows():
                key = f"{row['strike']}_{row['type'].lower()}"
                if key in previous_states:
                    prev_state = previous_states[key]['state']
                    if prev_state != row['current_state'] and abs(row['sr_score']) > 30:
                        alerts.append({
                            'type': 'STATE_CHANGE',
                            'underlying': underlying,
                            'data': row.to_dict()
                        })
        
        # Send alerts if any found
        if alerts:
            success = telegram_notifier.send_batch_alerts(alerts)
            if success:
                # Save alert logs to database
                for alert in alerts:
                    self.db_manager.save_alert_log(alert)
                logger.info(f"Sent {len(alerts)} alerts to Telegram")
            return alerts
        
        return []
# Enhanced Part 6: Main Scanning Function
    def scan_and_save(self, underlying: str = "NIFTY", expiry: str = None, 
                     telegram_notifier: TelegramNotifier = None, 
                     alert_settings: Dict = None) -> pd.DataFrame:
        """Main scan function with database integration and Telegram alerts"""
        
        # Default alert settings
        if alert_settings is None:
            alert_settings = {
                'fast_transitions': True,
                'high_velocity': True,
                'strong_levels': True,
                'breakouts': True,
                'state_changes': True
            }
        
        # Get expiry list if not provided
        if not expiry:
            expiry_list = self.get_expiry_list(underlying)
            if not expiry_list:
                st.error("Could not fetch expiry list")
                return pd.DataFrame()
            expiry = expiry_list[0]  # Use nearest expiry
        
        # Get option chain
        option_chain = self.get_option_chain(underlying, expiry)
        if not option_chain:
            st.error("Could not fetch option chain")
            return pd.DataFrame()
        
        spot_price = option_chain.get('last_price', 0)
        
        # Find ATM strikes
        strikes = self.find_atm_strikes(spot_price, option_chain)
        if not strikes:
            st.error("Could not determine strikes")
            return pd.DataFrame()
        
        results = []
        db_records = []
        
        for strike in strikes:
            strike_key = f"{strike:.6f}"
            if strike_key in option_chain.get('oc', {}):
                strike_data = option_chain['oc'][strike_key]
                
                for option_type in ['ce', 'pe']:
                    if option_type in strike_data:
                        option_data = strike_data[option_type]
                        
                        # Calculate SR analysis with database history
                        score_data = self.calculate_sr_score_with_history(
                            strike, option_type, option_data, spot_price
                        )
                        
                        # Determine transition state
                        transition_data = self.determine_transition_state(score_data, option_type)
                        
                        # Prepare result
                        result = {
                            'timestamp': datetime.now(),
                            'strike': strike,
                            'type': option_type.upper(),
                            'spot_price': spot_price,
                            'distance_pct': round(((strike - spot_price) / spot_price) * 100, 2),
                            'ltp': option_data.get('last_price', 0),
                            'change_pct': score_data['price_change_pct'],
                            'volume': option_data.get('volume', 0),
                            'oi': option_data.get('oi', 0),
                            'oi_change': score_data['oi_change'],
                            'oi_change_pct': round(score_data['oi_change_pct'], 2),
                            'bid_qty': option_data.get('top_bid_quantity', 0),
                            'ask_qty': option_data.get('top_ask_quantity', 0),
                            'bid_ask_ratio': round(score_data['bid_ask_ratio'], 2),
                            'sr_score': round(score_data['sr_score'], 1),
                            'iv': round(option_data.get('implied_volatility', 0), 2),
                            'delta': round(option_data.get('greeks', {}).get('delta', 0), 4),
                            'theta': round(option_data.get('greeks', {}).get('theta', 0), 2),
                            'gamma': round(option_data.get('greeks', {}).get('gamma', 0), 6),
                            'vega': round(option_data.get('greeks', {}).get('vega', 0), 2),
                            **transition_data
                        }
                        
                        results.append(result)
                        
                        # Prepare database record
                        db_record = {
                            'underlying': underlying,
                            'strike': float(strike),
                            'option_type': option_type.lower(),
                            'expiry': expiry,
                            'spot_price': float(spot_price),
                            'ltp': float(option_data.get('last_price', 0)),
                            'change_pct': float(score_data['price_change_pct']),
                            'volume': int(option_data.get('volume', 0)),
                            'open_interest': int(option_data.get('oi', 0)),
                            'oi_change': int(score_data['oi_change']),
                            'oi_change_pct': float(score_data['oi_change_pct']),
                            'bid_qty': int(option_data.get('top_bid_quantity', 0)),
                            'ask_qty': int(option_data.get('top_ask_quantity', 0)),
                            'bid_ask_ratio': float(score_data['bid_ask_ratio']),
                            'sr_score': float(score_data['sr_score']),
                            'current_state': transition_data['current_state'],
                            'transition_direction': transition_data['transition_direction'],
                            'transition_speed': transition_data['transition_speed'],
                            'score_velocity': float(transition_data['score_velocity']),
                            'score_acceleration': float(transition_data['score_acceleration']),
                            'trend_strength': 0.0
                        }
                        
                        db_records.append(db_record)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to database
        if db_records:
            success = self.db_manager.save_market_data(db_records)
            if success:
                st.success(f"Saved {len(db_records)} records to database")
            else:
                st.warning("Failed to save data to database")
        
        # Check for alerts and send notifications
        if telegram_notifier and not df.empty:
            alerts = self.check_and_send_alerts(df, underlying, telegram_notifier, alert_settings)
            if alerts:
                st.info(f"Sent {len(alerts)} alerts to Telegram")
        
        # Store current scan data for next comparison
        self.previous_scan_data = df.copy()
        
        return df
# Enhanced Part 7: Chart Creation and Helper Functions
def create_charts(df: pd.DataFrame, db_manager: SupabaseManager):
    """Create interactive charts"""
    if df.empty:
        return None, None, None
    
    # SR Score Chart
    fig_score = go.Figure()
    
    for option_type in ['CE', 'PE']:
        data = df[df['type'] == option_type]
        if not data.empty:
            colors = ['red' if x < 0 else 'green' for x in data['sr_score']]
            fig_score.add_trace(go.Scatter(
                x=data['strike'],
                y=data['sr_score'],
                mode='markers+lines',
                name=f'{option_type} Score',
                text=[f"State: {state}<br>Direction: {direction}<br>Speed: {speed}<br>Velocity: {velocity:.2f}" 
                      for state, direction, speed, velocity in zip(data['current_state'], 
                                                        data['transition_direction'],
                                                        data['transition_speed'],
                                                        data['score_velocity'])],
                hovertemplate='<b>%{x}</b><br>Score: %{y}<br>%{text}<extra></extra>',
                marker=dict(
                    size=abs(data['sr_score']).values * 0.3 + 8,
                    color=colors,
                    line=dict(width=2, color='white')
                )
            ))
    
    fig_score.update_layout(
        title="Support Resistance Score by Strike",
        xaxis_title="Strike Price",
        yaxis_title="SR Score (-100 to +100)",
        hovermode='closest',
        height=500
    )
    
    # Add horizontal lines for score levels
    fig_score.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_score.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5)
    fig_score.add_hline(y=-30, line_dash="dot", line_color="red", opacity=0.5)
    fig_score.add_hline(y=60, line_dash="dot", line_color="darkgreen", opacity=0.5)
    fig_score.add_hline(y=-60, line_dash="dot", line_color="darkred", opacity=0.5)
    
    # Historical Chart for ATM strike
    fig_historical = None
    if not df.empty:
        atm_data = df[abs(df['distance_pct']) == abs(df['distance_pct']).min()].iloc[0]
        historical_df = db_manager.get_historical_data(
            "NIFTY", atm_data['strike'], atm_data['type'].lower(), hours_back=24
        )
        
        if not historical_df.empty and len(historical_df) > 1:
            fig_historical = go.Figure()
            fig_historical.add_trace(go.Scatter(
                x=pd.to_datetime(historical_df['timestamp']),
                y=historical_df['sr_score'],
                mode='lines+markers',
                name='SR Score',
                line=dict(color='blue', width=2)
            ))
            
            # Add state change markers
            fig_historical.add_trace(go.Scatter(
                x=pd.to_datetime(historical_df['timestamp']),
                y=historical_df['sr_score'],
                mode='markers',
                name='State Changes',
                marker=dict(
                    size=8,
                    color=historical_df['sr_score'],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                text=historical_df['current_state'],
                hovertemplate='<b>%{x}</b><br>Score: %{y}<br>State: %{text}<extra></extra>'
            ))
            
            fig_historical.update_layout(
                title=f"Historical SR Score - {atm_data['strike']} {atm_data['type']}",
                xaxis_title="Time",
                yaxis_title="SR Score",
                height=400
            )
    
    # Velocity Chart
    fig_velocity = go.Figure()
    
    for option_type in ['CE', 'PE']:
        data = df[df['type'] == option_type]
        if not data.empty:
            fig_velocity.add_trace(go.Bar(
                x=data['strike'],
                y=data['score_velocity'],
                name=f'{option_type} Velocity',
                text=[f"{speed}<br>{vel:.2f}" for speed, vel in zip(data['transition_speed'], data['score_velocity'])],
                textposition='auto',
                marker_color='green' if option_type == 'PE' else 'red',
                opacity=0.7
            ))
    
    fig_velocity.update_layout(
        title="Score Velocity by Strike",
        xaxis_title="Strike Price", 
        yaxis_title="Velocity",
        height=400,
        barmode='group'
    )
    
    # Add velocity threshold lines
    fig_velocity.add_hline(y=10, line_dash="dash", line_color="orange", 
                          annotation_text="High Velocity Threshold")
    fig_velocity.add_hline(y=-10, line_dash="dash", line_color="orange")
    
    return fig_score, fig_historical, fig_velocity

def display_setup_instructions():
    """Display setup instructions for new users"""
    with st.expander("Setup Instructions", expanded=False):
        st.markdown("""
        ### Database Setup (One-time)
        1. Create a Supabase account and project
        2. Go to SQL Editor in your Supabase dashboard
        3. Run this SQL script:
        
        ```sql
        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            underlying VARCHAR(20),
            strike DECIMAL(10,2),
            option_type VARCHAR(2),
            expiry DATE,
            spot_price DECIMAL(10,2),
            ltp DECIMAL(10,4),
            change_pct DECIMAL(8,4),
            volume BIGINT,
            open_interest BIGINT,
            oi_change BIGINT,
            oi_change_pct DECIMAL(8,4),
            bid_qty BIGINT,
            ask_qty BIGINT,
            bid_ask_ratio DECIMAL(10,4),
            sr_score DECIMAL(8,2),
            current_state VARCHAR(50),
            transition_direction VARCHAR(50),
            transition_speed VARCHAR(20),
            score_velocity DECIMAL(8,4),
            score_acceleration DECIMAL(8,4),
            trend_strength DECIMAL(8,2)
        );
        
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
        CREATE INDEX IF NOT EXISTS idx_market_data_strike_type ON market_data(strike, option_type);
        ```
        
        ### Streamlit Secrets Setup
        Create `.streamlit/secrets.toml` file:
        ```toml
        SUPABASE_URL = "your-supabase-url"
        SUPABASE_ANON_KEY = "your-supabase-anon-key"
        ```
        """)

def display_telegram_setup():
    """Display Telegram bot setup instructions"""
    with st.expander("Telegram Bot Setup", expanded=False):
        st.markdown("""
        ### Create Telegram Bot
        1. Open Telegram and search for @BotFather
        2. Send `/newbot` command
        3. Follow instructions to create your bot
        4. Copy the Bot Token provided
        
        ### Get Chat ID
        1. Send a message to your bot
        2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
        3. Look for "chat":{"id": YOUR_CHAT_ID}
        4. Copy the Chat ID
        
        ### Test Setup
        Send a test message to verify configuration works.
        
        ### Alert Types
        - **Fast Transitions**: FAST/VERY_FAST speed changes
        - **High Velocity**: Score velocity > 10
        - **Strong Levels**: |SR Score| > 60
        - **Breakouts**: Score near zero with high speed
        - **State Changes**: Transitions between support/resistance states
        """)

def get_transition_emoji(direction: str, speed: str) -> str:
    """Get visual indicators for transitions"""
    direction_map = {
        'SUPPORT_STRENGTHENING': '⬆️',
        'SUPPORT_WEAKENING': '⬇️', 
        'RESISTANCE_STRENGTHENING': '⬆️',
        'RESISTANCE_WEAKENING': '⬇️',
        'STABLE': '➡️'
    }
    
    speed_map = {
        'VERY_FAST': '⚡⚡',
        'FAST': '⚡',
        'MODERATE': '🔄',
        'SLOW': '🐌',
        'VERY_SLOW': '🐌🐌'
    }
    
    return f"{direction_map.get(direction, '❓')} {speed_map.get(speed, '❓')}"

def format_alert_summary(alerts: List[Dict]) -> str:
    """Format alert summary for display"""
    if not alerts:
        return "No alerts generated"
    
    summary = {}
    for alert in alerts:
        alert_type = alert['type']
        if alert_type not in summary:
            summary[alert_type] = 0
        summary[alert_type] += 1
    
    result = "Alert Summary:\n"
    for alert_type, count in summary.items():
        emoji = "🚨" if "FAST" in alert_type else "⚡" if "VELOCITY" in alert_type else "🎯"
        result += f"{emoji} {alert_type}: {count}\n"
    
    return result
# Enhanced Part 8: Streamlit UI with Telegram Integration
def main():
    """Main Streamlit app with Telegram notifications"""
    st.title("Support Resistance Scanner Pro")
    st.markdown("*Real-time Options Flow Analysis with Cloud Database & Telegram Alerts*")
    
    # Setup instructions
    display_setup_instructions()
    display_telegram_setup()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API credentials
    access_token = st.sidebar.text_input("Dhan Access Token", type="password")
    client_id = st.sidebar.text_input("Dhan Client ID")
    
    # Scan settings
    underlying = st.sidebar.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    
    # Telegram configuration
    st.sidebar.header("Telegram Notifications")
    telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts")
    
    telegram_bot_token = None
    telegram_chat_id = None
    alert_settings = {}
    
    if telegram_enabled:
        telegram_bot_token = st.sidebar.text_input("Bot Token", type="password")
        telegram_chat_id = st.sidebar.text_input("Chat ID")
        
        # Test connection button
        if telegram_bot_token and telegram_chat_id:
            if st.sidebar.button("Test Telegram Connection"):
                test_notifier = TelegramNotifier(telegram_bot_token, telegram_chat_id)
                if test_notifier.test_connection():
                    test_msg = test_notifier.send_message("Test message from SR Scanner!")
                    if test_msg:
                        st.sidebar.success("Telegram connection successful!")
                    else:
                        st.sidebar.error("Failed to send test message")
                else:
                    st.sidebar.error("Telegram connection failed")
        
        # Alert settings
        st.sidebar.subheader("Alert Settings")
        alert_settings = {
            'fast_transitions': st.sidebar.checkbox("Fast Transitions", value=True),
            'high_velocity': st.sidebar.checkbox("High Velocity Changes", value=True),
            'strong_levels': st.sidebar.checkbox("Strong S/R Levels", value=True),
            'breakouts': st.sidebar.checkbox("Potential Breakouts", value=True),
            'state_changes': st.sidebar.checkbox("State Changes", value=False)
        }
        
        # Alert thresholds
        st.sidebar.subheader("Alert Thresholds")
        velocity_threshold = st.sidebar.slider("Velocity Threshold", 5, 20, 10)
        score_threshold = st.sidebar.slider("Strong Level Threshold", 30, 80, 60)
    
    # Initialize database
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = SupabaseManager()
    
    db_manager = st.session_state.db_manager
    
    if not access_token or not client_id:
        st.warning("Please enter your Dhan API credentials in the sidebar.")
        st.info("Get your credentials from: https://dhanhq.co/api")
        return
    
    # Initialize scanner
    scanner = DhanSRScanner(access_token, client_id, db_manager)
    
    # Initialize Telegram notifier
    telegram_notifier = None
    if telegram_enabled and telegram_bot_token and telegram_chat_id:
        telegram_notifier = TelegramNotifier(telegram_bot_token, telegram_chat_id)
    
    # Control buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("Scan Now", type="primary"):
            with st.spinner("Scanning market data..."):
                try:
                    df = scanner.scan_and_save(
                        underlying, None, telegram_notifier, alert_settings
                    )
                    st.session_state.current_data = df
                    st.session_state.last_scan_time = datetime.now()
                    
                    # Send startup notification if first scan
                    if telegram_notifier and 'first_scan' not in st.session_state:
                        telegram_notifier.send_startup_message(underlying)
                        st.session_state.first_scan = True
                        
                except Exception as e:
                    st.error(f"Scan failed: {e}")
    
    with col2:
        if st.button("Load Latest"):
            try:
                latest_df = db_manager.get_latest_scan_data(underlying)
                if not latest_df.empty:
                    st.session_state.current_data = latest_df
                    st.success(f"Loaded {len(latest_df)} records from database")
                else:
                    st.warning("No recent data found")
            except Exception as e:
                st.error(f"Load failed: {e}")
    
    with col3:
        auto_refresh_enabled = st.session_state.get('auto_refresh', False)
        if st.button(f"Auto Refresh {'ON' if auto_refresh_enabled else 'OFF'}"):
            st.session_state.auto_refresh = not auto_refresh_enabled
            if st.session_state.auto_refresh:
                st.success("Auto refresh enabled (60s interval)")
            else:
                st.info("Auto refresh disabled")
    
    with col4:
        if st.button("Clear Cache"):
            keys_to_clear = ['current_data', 'last_scan_time', 'auto_refresh', 'first_scan']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Cache cleared")
    
    with col5:
        if telegram_notifier and st.button("Send Test Alert"):
            test_alert = {
                'type': 'FAST_TRANSITION',
                'underlying': underlying,
                'data': {
                    'strike': 24000,
                    'type': 'PE',
                    'sr_score': 45.2,
                    'transition_direction': 'SUPPORT_STRENGTHENING',
                    'transition_speed': 'FAST',
                    'score_velocity': 12.5,
                    'current_state': 'MODERATE_SUPPORT'
                }
            }
            telegram_notifier.send_batch_alerts([test_alert])
            st.success("Test alert sent!")
    
    # Auto refresh logic
    if st.session_state.get('auto_refresh', False):
        if 'last_scan_time' not in st.session_state or \
           (datetime.now() - st.session_state.last_scan_time).seconds > 60:
            time.sleep(1)
            st.rerun()
    
    # Display data
    if 'current_data' in st.session_state and not st.session_state.current_data.empty:
        df = st.session_state.current_data
        
        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            fast_transitions = len(df[df['transition_speed'].isin(['FAST', 'VERY_FAST'])])
            st.metric("Fast Transitions", fast_transitions)
        
        with col2:
            high_velocity = len(df[abs(df['score_velocity']) > 10])
            st.metric("High Velocity", high_velocity)
        
        with col3:
            strong_levels = len(df[abs(df['sr_score']) > 50])
            st.metric("Strong S/R Levels", strong_levels)
        
        with col4:
            avg_score = df['sr_score'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with col5:
            if 'last_scan_time' in st.session_state:
                time_ago = (datetime.now() - st.session_state.last_scan_time).seconds
                st.metric("Last Scan", f"{time_ago}s ago")
        
        with col6:
            telegram_status = "Connected" if telegram_notifier else "Disabled"
            st.metric("Telegram", telegram_status)
        
        # Alert summary if alerts were sent
        if hasattr(scanner, 'previous_scan_data') and telegram_notifier:
            alerts_sent = st.session_state.get('alerts_sent', 0)
            if alerts_sent > 0:
                st.info(f"Sent {alerts_sent} alerts this session")
        
        # Charts
        st.subheader("Visual Analysis")
        fig_score, fig_historical, fig_velocity = create_charts(df, db_manager)
        
        if fig_score:
            st.plotly_chart(fig_score, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if fig_historical:
                st.plotly_chart(fig_historical, use_container_width=True)
        
        with col2:
            if fig_velocity:
                st.plotly_chart(fig_velocity, use_container_width=True)
# Enhanced Part 9: Data Display and Final Components
        # Data table
        st.subheader("Detailed Analysis")
        
        # Filter options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            state_filter = st.multiselect(
                "Filter by State",
                options=df['current_state'].unique(),
                default=df['current_state'].unique()
            )
        
        with col2:
            speed_filter = st.multiselect(
                "Filter by Speed", 
                options=df['transition_speed'].unique(),
                default=df['transition_speed'].unique()
            )
        
        with col3:
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=df['transition_direction'].unique(),
                default=df['transition_direction'].unique()
            )
        
        with col4:
            score_range = st.slider(
                "SR Score Range",
                min_value=int(df['sr_score'].min()),
                max_value=int(df['sr_score'].max()),
                value=(int(df['sr_score'].min()), int(df['sr_score'].max()))
            )
        
        # Apply filters
        filtered_df = df[
            (df['current_state'].isin(state_filter)) &
            (df['transition_speed'].isin(speed_filter)) &
            (df['transition_direction'].isin(direction_filter)) &
            (df['sr_score'] >= score_range[0]) &
            (df['sr_score'] <= score_range[1])
        ]
        
        # Display table with enhanced formatting
        if not filtered_df.empty:
            # Add visual indicators
            filtered_df['visual'] = filtered_df.apply(
                lambda row: get_transition_emoji(row['transition_direction'], row['transition_speed']), 
                axis=1
            )
            
            display_columns = [
                'visual', 'strike', 'type', 'distance_pct', 'current_state', 'sr_score',
                'transition_direction', 'transition_speed', 'score_velocity',
                'oi_change_pct', 'bid_ask_ratio', 'iv', 'delta'
            ]
            
            # Color-code rows based on SR score and alerts
            def color_rows(row):
                if row['sr_score'] > 60:
                    return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
                elif row['sr_score'] < -60:
                    return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                elif row['transition_speed'] in ['FAST', 'VERY_FAST']:
                    return ['background-color: rgba(255, 255, 0, 0.2)'] * len(row)
                elif abs(row['score_velocity']) > 10:
                    return ['background-color: rgba(255, 165, 0, 0.2)'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = filtered_df[display_columns].style.format({
                'distance_pct': '{:.2f}%',
                'sr_score': '{:.1f}',
                'score_velocity': '{:.2f}',
                'oi_change_pct': '{:.2f}%',
                'bid_ask_ratio': '{:.2f}',
                'iv': '{:.2f}%',
                'delta': '{:.4f}'
            }).apply(color_rows, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Export data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"sr_analysis_{underlying}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
            
            # Alerts section
            st.subheader("Alert Conditions")
            
            alert_cols = st.columns(4)
            
            with alert_cols[0]:
                fast_df = filtered_df[filtered_df['transition_speed'].isin(['FAST', 'VERY_FAST'])]
                if not fast_df.empty:
                    st.warning(f"Fast Transitions Detected ({len(fast_df)})")
                    for _, row in fast_df.iterrows():
                        st.write(f"• {row['strike']} {row['type']}: {row['transition_direction']} at {row['transition_speed']} speed")
            
            with alert_cols[1]:
                high_vel_df = filtered_df[abs(filtered_df['score_velocity']) > 10]
                if not high_vel_df.empty:
                    st.error(f"High Velocity Changes ({len(high_vel_df)})")
                    for _, row in high_vel_df.iterrows():
                        st.write(f"• {row['strike']} {row['type']}: Velocity {row['score_velocity']:.2f}")
            
            with alert_cols[2]:
                strong_df = filtered_df[abs(filtered_df['sr_score']) > 60]
                if not strong_df.empty:
                    st.success(f"Strong Support/Resistance Levels ({len(strong_df)})")
                    for _, row in strong_df.iterrows():
                        level_type = "Support" if row['sr_score'] > 0 else "Resistance"
                        st.write(f"• {row['strike']} {row['type']}: Strong {level_type} (Score: {row['sr_score']:.1f})")
            
            with alert_cols[3]:
                breakout_df = filtered_df[
                    (abs(filtered_df['sr_score']) < 20) & 
                    (filtered_df['transition_speed'].isin(['FAST', 'VERY_FAST']))
                ]
                if not breakout_df.empty:
                    st.info(f"Potential Breakouts ({len(breakout_df)})")
                    for _, row in breakout_df.iterrows():
                        st.write(f"• {row['strike']} {row['type']}: {row['current_state']} at {row['transition_speed']} speed")
        
        else:
            st.warning("No data matches the selected filters")
    
    else:
        st.info("Click 'Scan Now' to start analysis or 'Load Latest' to view recent data.")
        
        # Show sample data structure
        with st.expander("Expected Data Structure"):
            sample_data = {
                'Strike': [24000, 24050, 24100],
                'Type': ['PE', 'CE', 'PE'],
                'SR Score': [45.2, -22.1, 12.5],
                'State': ['MODERATE_SUPPORT', 'WEAK_RESISTANCE', 'WEAK_SUPPORT'],
                'Direction': ['SUPPORT_STRENGTHENING', 'RESISTANCE_WEAKENING', 'STABLE'],
                'Speed': ['FAST', 'MODERATE', 'SLOW'],
                'Velocity': [8.3, -5.1, 1.2]
            }
            st.dataframe(pd.DataFrame(sample_data))
    
    # Telegram status in sidebar
    if telegram_enabled:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Telegram Status")
        if telegram_notifier:
            if telegram_notifier.test_connection():
                st.sidebar.success("Connected")
            else:
                st.sidebar.error("Connection Failed")
        else:
            st.sidebar.warning("Not Configured")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Dhan API, Streamlit, Supabase & Telegram*")

if __name__ == "__main__":
    main()
