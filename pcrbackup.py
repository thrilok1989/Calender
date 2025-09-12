import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from pytz import timezone
import json
import math
from scipy.stats import norm

# === Configuration ===
DHAN_CLIENT_ID = "YOUR_DHAN_CLIENT_ID"
DHAN_ACCESS_TOKEN = "YOUR_DHAN_ACCESS_TOKEN"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"
STOP_LOSS_PERCENTAGE = 20  # 20% stop loss

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DhanOptionsBot:
    def __init__(self):
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes cooldown between signals
        
    def send_telegram_message(self, message):
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def get_dhan_option_chain(self, underlying_scrip, underlying_seg, expiry):
        """Fetch option chain from Dhan API"""
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
            logger.error(f"Error fetching Dhan option chain: {e}")
            return None

    def get_dhan_expiry_list(self, underlying_scrip, underlying_seg):
        """Fetch expiry list from Dhan API"""
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
            logger.error(f"Error fetching Dhan expiry list: {e}")
            return None

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:15", "%H:%M").time()
        market_end = datetime.strptime("15:30", "%H:%M").time()
        
        return current_day < 5 and market_start <= current_time <= market_end

    def calculate_bid_ask_pressure(self, call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
        """Calculate bid-ask pressure"""
        pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
        if pressure > 500:
            bias = "Bullish"
        elif pressure < -500:
            bias = "Bearish"
        else:
            bias = "Neutral"
        return pressure, bias

    def determine_level(self, ce_oi, pe_oi):
        """Determine if strike is support or resistance"""
        if pe_oi > 1.12 * ce_oi:
            return "Support"
        elif ce_oi > 1.12 * pe_oi:
            return "Resistance"
        else:
            return "Neutral"

    def analyze_atm_conditions(self, df, underlying_price):
        """Analyze ATM conditions for trading signals"""
        # Find ATM strike
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying_price))
        atm_row = df[df['strikePrice'] == atm_strike].iloc[0]
        
        # Extract ATM data
        ce_oi = atm_row.get('openInterest_CE', 0)
        pe_oi = atm_row.get('openInterest_PE', 0)
        ce_oi_change = atm_row.get('changeinOpenInterest_CE', 0)
        pe_oi_change = atm_row.get('changeinOpenInterest_PE', 0)
        ce_volume = atm_row.get('totalTradedVolume_CE', 0)
        pe_volume = atm_row.get('totalTradedVolume_PE', 0)
        ce_bid_qty = atm_row.get('bidQty_CE', 0)
        pe_bid_qty = atm_row.get('bidQty_PE', 0)
        ce_ask_qty = atm_row.get('askQty_CE', 0)
        pe_ask_qty = atm_row.get('askQty_PE', 0)
        ce_ltp = atm_row.get('lastPrice_CE', 0)
        pe_ltp = atm_row.get('lastPrice_PE', 0)
        
        # Determine level
        level = self.determine_level(ce_oi, pe_oi)
        
        # Calculate bid-ask pressure
        pressure, pressure_bias = self.calculate_bid_ask_pressure(ce_bid_qty, ce_ask_qty, pe_bid_qty, pe_ask_qty)
        
        # Bullish conditions check
        bullish_conditions = {
            'oi_change': pe_oi_change > ce_oi_change,  # More PUT OI buildup
            'volume': pe_volume > ce_volume,  # More PUT volume
            'bid_qty': pe_bid_qty > ce_bid_qty,  # More PUT bids
            'ask_qty': ce_ask_qty > pe_ask_qty,  # More CALL asks
            'level': level == "Support",
            'pressure': pressure_bias == "Bullish"
        }
        
        # Bearish conditions check
        bearish_conditions = {
            'oi_change': ce_oi_change > pe_oi_change,  # More CALL OI buildup
            'volume': ce_volume > pe_volume,  # More CALL volume
            'bid_qty': ce_bid_qty > pe_bid_qty,  # More CALL bids
            'ask_qty': pe_ask_qty > ce_ask_qty,  # More PUT asks
            'level': level == "Resistance",
            'pressure': pressure_bias == "Bearish"
        }
        
        # Count bullish and bearish signals
        bullish_score = sum(bullish_conditions.values())
        bearish_score = sum(bearish_conditions.values())
        
        signal_data = {
            'atm_strike': atm_strike,
            'underlying_price': underlying_price,
            'level': level,
            'ce_ltp': ce_ltp,
            'pe_ltp': pe_ltp,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'bullish_conditions': bullish_conditions,
            'bearish_conditions': bearish_conditions,
            'pressure': pressure,
            'pressure_bias': pressure_bias,
            'ce_oi_change': ce_oi_change,
            'pe_oi_change': pe_oi_change,
            'ce_volume': ce_volume,
            'pe_volume': pe_volume
        }
        
        return signal_data

    def generate_trading_signal(self, signal_data):
        """Generate trading signal based on conditions"""
        current_time = time.time()
        
        # Check for BULLISH signal (minimum 4 out of 6 conditions)
        if (signal_data['bullish_score'] >= 4 and 
            signal_data['level'] == "Support" and
            current_time - self.last_signal_time.get('bullish', 0) > self.signal_cooldown):
            
            atm_strike = signal_data['atm_strike']
            ce_ltp = signal_data['ce_ltp']
            stop_loss = ce_ltp * (100 - STOP_LOSS_PERCENTAGE) / 100
            
            message = f"""
🟢 <b>BULLISH SIGNAL - BUY CALL</b>

📊 <b>ATM Details:</b>
• Strike: {atm_strike}
• Level: {signal_data['level']}
• NIFTY Spot: {signal_data['underlying_price']:.2f}

💰 <b>Trade Setup:</b>
• BUY {atm_strike} CALL
• Entry Price: ₹{ce_ltp:.2f}
• Stop Loss: ₹{stop_loss:.2f} ({STOP_LOSS_PERCENTAGE}%)

📈 <b>Signal Strength:</b> {signal_data['bullish_score']}/6

📊 <b>Supporting Data:</b>
• PUT OI Change: {signal_data['pe_oi_change']:,}
• CALL OI Change: {signal_data['ce_oi_change']:,}
• PUT Volume: {signal_data['pe_volume']:,}
• CALL Volume: {signal_data['ce_volume']:,}
• Bid-Ask Pressure: {signal_data['pressure']:,} ({signal_data['pressure_bias']})

⏰ Time: {datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            self.send_telegram_message(message.strip())
            self.last_signal_time['bullish'] = current_time
            logger.info(f"BULLISH signal sent for ATM {atm_strike}")
            return "BULLISH"
        
        # Check for BEARISH signal (minimum 4 out of 6 conditions)
        elif (signal_data['bearish_score'] >= 4 and 
              signal_data['level'] == "Resistance" and
              current_time - self.last_signal_time.get('bearish', 0) > self.signal_cooldown):
            
            atm_strike = signal_data['atm_strike']
            pe_ltp = signal_data['pe_ltp']
            stop_loss = pe_ltp * (100 - STOP_LOSS_PERCENTAGE) / 100
            
            message = f"""
🔴 <b>BEARISH SIGNAL - BUY PUT</b>

📊 <b>ATM Details:</b>
• Strike: {atm_strike}
• Level: {signal_data['level']}
• NIFTY Spot: {signal_data['underlying_price']:.2f}

💰 <b>Trade Setup:</b>
• BUY {atm_strike} PUT
• Entry Price: ₹{pe_ltp:.2f}
• Stop Loss: ₹{stop_loss:.2f} ({STOP_LOSS_PERCENTAGE}%)

📉 <b>Signal Strength:</b> {signal_data['bearish_score']}/6

📊 <b>Supporting Data:</b>
• CALL OI Change: {signal_data['ce_oi_change']:,}
• PUT OI Change: {signal_data['pe_oi_change']:,}
• CALL Volume: {signal_data['ce_volume']:,}
• PUT Volume: {signal_data['pe_volume']:,}
• Bid-Ask Pressure: {signal_data['pressure']:,} ({signal_data['pressure_bias']})

⏰ Time: {datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            self.send_telegram_message(message.strip())
            self.last_signal_time['bearish'] = current_time
            logger.info(f"BEARISH signal sent for ATM {atm_strike}")
            return "BEARISH"
        
        return "NO_SIGNAL"

    def process_option_chain_data(self, option_chain_data):
        """Process option chain data and create DataFrame"""
        if not option_chain_data or 'data' not in option_chain_data:
            return None, None
        
        data = option_chain_data['data']
        underlying = data['last_price']
        
        # Flatten option chain data
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
        
        if not calls or not puts:
            return None, underlying
        
        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        
        # Rename columns to match expected format
        column_mapping = {
            'last_price': 'lastPrice',
            'oi': 'openInterest',
            'previous_oi': 'previousOpenInterest',
            'top_ask_quantity': 'askQty',
            'top_bid_quantity': 'bidQty',
            'volume': 'totalTradedVolume'
        }
        
        for old_col, new_col in column_mapping.items():
            if f"{old_col}_CE" in df.columns:
                df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
            if f"{old_col}_PE" in df.columns:
                df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)
        
        # Calculate change in OI
        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
        
        return df, underlying

    def run_monitoring(self):
        """Main monitoring loop"""
        logger.info("Starting Dhan Options Trading Bot...")
        
        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market is closed. Waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Get expiry list
                expiry_data = self.get_dhan_expiry_list(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
                if not expiry_data or 'data' not in expiry_data:
                    logger.error("Failed to get expiry list")
                    time.sleep(30)
                    continue
                
                expiry_dates = expiry_data['data']
                if not expiry_dates:
                    logger.error("No expiry dates available")
                    time.sleep(30)
                    continue
                
                expiry = expiry_dates[0]  # Use nearest expiry
                logger.info(f"Using expiry: {expiry}")
                
                # Get option chain data
                option_chain_data = self.get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
                
                df, underlying_price = self.process_option_chain_data(option_chain_data)
                if df is None:
                    logger.error("Failed to process option chain data")
                    time.sleep(30)
                    continue
                
                logger.info(f"NIFTY Spot: {underlying_price:.2f}")
                
                # Analyze ATM conditions
                signal_data = self.analyze_atm_conditions(df, underlying_price)
                
                # Generate trading signal
                signal = self.generate_trading_signal(signal_data)
                
                if signal == "NO_SIGNAL":
                    logger.info(f"ATM {signal_data['atm_strike']}: Level={signal_data['level']}, "
                              f"Bull Score={signal_data['bullish_score']}, Bear Score={signal_data['bearish_score']}")
                
                # Wait before next iteration (API rate limit consideration)
                time.sleep(10)  # 10 seconds between checks
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)

def main():
    # Initialize bot
    bot = DhanOptionsBot()
    
    # Send startup message
    startup_msg = """
🤖 <b>Dhan Options Trading Bot Started</b>

📊 Monitoring ATM NIFTY Options for:
• Support/Resistance levels
• OI changes
• Volume patterns
• Bid/Ask quantities

⚡ Will alert on strong signals (4/6 conditions)
🛑 Stop Loss: 20%

<i>Bot is now monitoring market conditions...</i>
    """
    bot.send_telegram_message(startup_msg.strip())
    
    # Start monitoring
    bot.run_monitoring()

if __name__ == "__main__":
    main()
