import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict
import warnings
from supabase import create_client, Client
import asyncio
import asyncpg
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

class SupabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
    def create_tables(self):
        """Create necessary tables for storing market data"""
        # Market data table
        try:
            self.supabase.table('market_data').select('*').limit(1).execute()
        except:
            print("Creating market_data table...")
            # Table should be created in Supabase dashboard with this structure:
            # market_data (id, timestamp, security_id, exchange_segment, open, high, low, close, volume, buy_quantity, sell_quantity, created_at)
        
        # Volume footprint table
        try:
            self.supabase.table('volume_footprint').select('*').limit(1).execute()
        except:
            print("Creating volume_footprint table...")
            # volume_footprint (id, timestamp, security_id, price_level, volume, is_poc, session_high, session_low, created_at)
        
        # Alerts table
        try:
            self.supabase.table('alerts').select('*').limit(1).execute()
        except:
            print("Creating alerts table...")
            # alerts (id, timestamp, alert_type, message, security_id, price, volume, is_sent, created_at)
    
    def insert_market_data(self, data: Dict):
        """Insert market data into Supabase"""
        try:
            result = self.supabase.table('market_data').insert({
                'timestamp': data['timestamp'].isoformat(),
                'security_id': data.get('security_id', '11536'),
                'exchange_segment': data.get('exchange_segment', 'NSE_EQ'),
                'open': float(data['open']),
                'high': float(data['high']),
                'low': float(data['low']),
                'close': float(data['close']),
                'volume': int(data['volume']),
                'buy_quantity': int(data.get('buy_quantity', 0)),
                'sell_quantity': int(data.get('sell_quantity', 0))
            }).execute()
            return result.data
        except Exception as e:
            print(f"Error inserting market data: {e}")
            return None
    
    def insert_volume_footprint(self, security_id: str, footprint_data: Dict, session_high: float, session_low: float):
        """Insert volume footprint data"""
        try:
            records = []
            poc_price = max(footprint_data, key=footprint_data.get) if footprint_data else None
            
            for price_level, volume in footprint_data.items():
                records.append({
                    'timestamp': datetime.now().isoformat(),
                    'security_id': security_id,
                    'price_level': float(price_level),
                    'volume': float(volume),
                    'is_poc': price_level == poc_price,
                    'session_high': float(session_high),
                    'session_low': float(session_low)
                })
            
            if records:
                result = self.supabase.table('volume_footprint').insert(records).execute()
                return result.data
        except Exception as e:
            print(f"Error inserting volume footprint: {e}")
            return None
    
    def insert_alert(self, alert_type: str, message: str, security_id: str, price: float = None, volume: int = None):
        """Insert alert into database"""
        try:
            result = self.supabase.table('alerts').insert({
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'message': message,
                'security_id': security_id,
                'price': float(price) if price else None,
                'volume': int(volume) if volume else None,
                'is_sent': False
            }).execute()
            return result.data
        except Exception as e:
            print(f"Error inserting alert: {e}")
            return None
    
    def get_recent_market_data(self, security_id: str, limit: int = 100):
        """Get recent market data from database"""
        try:
            result = self.supabase.table('market_data')\
                .select('*')\
                .eq('security_id', security_id)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return []
    
    def get_unsent_alerts(self):
        """Get alerts that haven't been sent yet"""
        try:
            result = self.supabase.table('alerts')\
                .select('*')\
                .eq('is_sent', False)\
                .order('timestamp', desc=False)\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []
    
    def mark_alert_sent(self, alert_id: int):
        """Mark alert as sent"""
        try:
            result = self.supabase.table('alerts')\
                .update({'is_sent': True})\
                .eq('id', alert_id)\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error marking alert as sent: {e}")
            return None

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send message via Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error sending Telegram message: {response.text}")
                return None
        except Exception as e:
            print(f"Exception sending Telegram message: {e}")
            return None
    
    def send_chart_alert(self, alert_type: str, security_id: str, price: float, volume: int, additional_info: str = ""):
        """Send formatted trading alert"""
        message = f"""
🚨 <b>{alert_type.upper()} ALERT</b> 🚨

📊 <b>Security:</b> {security_id}
💰 <b>Price:</b> ₹{price:.2f}
📈 <b>Volume:</b> {volume:,}
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

{additional_info}

#TradingAlert #{security_id}
        """.strip()
        
        return self.send_message(message)

class DhanTradingChart:
    def __init__(self, access_token: str, client_id: str, supabase_url: str, supabase_key: str, 
                 telegram_bot_token: str, telegram_chat_id: str):
        self.access_token = access_token
        self.client_id = client_id
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': access_token,
            'client-id': client_id
        }
        
        # Initialize database and notifications
        self.db = SupabaseManager(supabase_url, supabase_key)
        self.telegram = TelegramNotifier(telegram_bot_token, telegram_chat_id)
        
        # Chart data storage
        self.chart_data = pd.DataFrame()
        self.volume_footprint_data = {}
        
        # Volume Footprint settings (converted from Pine Script)
        self.bins = 20
        self.timeframe = '1D'
        self.poc_color = '#298ada'
        self.bg_color = 'rgba(120, 123, 134, 0.9)'
        
        # Current session data
        self.current_high = None
        self.current_low = None
        self.volume_profile = defaultdict(float)
        self.session_start_time = None
        self.previous_poc = None
        
        # Alert thresholds
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.price_change_threshold = 0.02  # 2% price change
        
        # Chart figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                                     gridspec_kw={'height_ratios': [3, 1]})
        self.fig.patch.set_facecolor('black')
        
        # Initialize database tables
        self.db.create_tables()
        
        # Start alert monitoring thread
        self.start_alert_monitoring()
    
    def fetch_market_data(self, security_id="11536", exchange_segment="NSE_EQ"):
        """Fetch real-time market data using DhanHQ API"""
        try:
            quote_url = f"{self.base_url}/marketfeed/quote"
            quote_payload = {exchange_segment: [int(security_id)]}
            
            response = requests.post(quote_url, headers=self.headers, json=quote_payload)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and exchange_segment in data['data']:
                    market_data = data['data'][exchange_segment][security_id]
                    processed_data = self.process_market_data(market_data)
                    
                    # Add metadata
                    if processed_data:
                        processed_data['security_id'] = security_id
                        processed_data['exchange_segment'] = exchange_segment
                        
                        # Store in database
                        self.db.insert_market_data(processed_data)
                    
                    return processed_data
            else:
                print(f"Error fetching data: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"Exception in fetch_market_data: {e}")
            return None
    
    def process_market_data(self, raw_data):
        """Process raw market data into structured format"""
        try:
            current_time = datetime.now()
            
            processed_data = {
                'timestamp': current_time,
                'open': raw_data['ohlc']['open'],
                'high': raw_data['ohlc']['high'],
                'low': raw_data['ohlc']['low'],
                'close': raw_data['last_price'],
                'volume': raw_data['volume'],
                'buy_quantity': raw_data['buy_quantity'],
                'sell_quantity': raw_data['sell_quantity'],
                'depth': raw_data['depth']
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing market data: {e}")
            return None
    
    def check_alerts(self, market_data, security_id):
        """Check for various trading alerts"""
        if not market_data or len(self.chart_data) < 10:
            return
        
        current_price = market_data['close']
        current_volume = market_data['volume']
        
        # Calculate average volume (last 20 periods)
        recent_volumes = self.chart_data.tail(20)['volume'].values
        avg_volume = np.mean(recent_volumes) if len(recent_volumes) > 0 else current_volume
        
        # Volume spike alert
        if current_volume > avg_volume * self.volume_spike_threshold and avg_volume > 0:
            alert_msg = f"Volume spike detected! Current: {current_volume:,}, Average: {avg_volume:,.0f}"
            self.db.insert_alert("VOLUME_SPIKE", alert_msg, security_id, current_price, current_volume)
        
        # Price change alert
        if len(self.chart_data) > 1:
            previous_price = self.chart_data.iloc[-2]['close']
            price_change_pct = abs((current_price - previous_price) / previous_price)
            
            if price_change_pct > self.price_change_threshold:
                direction = "UP" if current_price > previous_price else "DOWN"
                alert_msg = f"Significant price movement {direction}: {price_change_pct*100:.1f}% change"
                self.db.insert_alert("PRICE_CHANGE", alert_msg, security_id, current_price, current_volume)
        
        # POC shift alert
        current_poc = self.calculate_poc()
        if current_poc and self.previous_poc:
            poc_change_pct = abs((current_poc['price'] - self.previous_poc['price']) / self.previous_poc['price'])
            if poc_change_pct > 0.01:  # 1% POC shift
                alert_msg = f"Point of Control shifted: {self.previous_poc['price']:.2f} → {current_poc['price']:.2f}"
                self.db.insert_alert("POC_SHIFT", alert_msg, security_id, current_poc['price'], current_poc['volume'])
        
        if current_poc:
            self.previous_poc = current_poc
    
    def start_alert_monitoring(self):
        """Start monitoring and sending alerts"""
        def alert_loop():
            while True:
                try:
                    # Get unsent alerts
                    alerts = self.db.get_unsent_alerts()
                    
                    for alert in alerts:
                        # Send Telegram notification
                        additional_info = f"Alert ID: {alert['id']}"
                        
                        result = self.telegram.send_chart_alert(
                            alert['alert_type'],
                            alert['security_id'],
                            alert['price'] or 0,
                            alert['volume'] or 0,
                            additional_info
                        )
                        
                        if result:
                            # Mark as sent
                            self.db.mark_alert_sent(alert['id'])
                            print(f"Sent alert: {alert['alert_type']} for {alert['security_id']}")
                        
                        # Small delay between messages
                        time.sleep(1)
                    
                    # Check every 30 seconds for new alerts
                    time.sleep(30)
                    
                except Exception as e:
                    print(f"Error in alert monitoring: {e}")
                    time.sleep(30)
        
        alert_thread = threading.Thread(target=alert_loop, daemon=True)
        alert_thread.start()
    
    def update_volume_footprint(self, market_data):
        """Update volume footprint data (converted from Pine Script logic)"""
        if not market_data:
            return
            
        current_price = market_data['close']
        volume = market_data['volume']
        high = market_data['high']
        low = market_data['low']
        
        # Initialize session if needed
        if self.current_high is None or self.current_low is None:
            self.current_high = high
            self.current_low = low
            self.session_start_time = market_data['timestamp']
        else:
            # Update session high/low
            if high > self.current_high:
                self.current_high = high
            if low < self.current_low:
                self.current_low = low
        
        # Calculate volume footprint
        if self.current_high > self.current_low:
            price_range = self.current_high - self.current_low
            step = price_range / self.bins if self.bins > 0 else 1
            
            # Volume weighted calculation (similar to Pine Script)
            recent_data = self.get_recent_data()
            volumes = [d.get('volume', 0) for d in recent_data]
            volume_stdev = np.std(volumes) if len(volumes) > 1 else 1
            volume_val = volume / volume_stdev if volume_stdev > 0 else volume
            
            # Find which bin this price falls into
            if step > 0:
                bin_index = int((current_price - self.current_low) / step)
                bin_index = max(0, min(bin_index, self.bins - 1))
                
                # Update volume profile
                bin_price = self.current_low + (bin_index * step)
                self.volume_profile[bin_price] += volume_val
        
        # Store volume footprint in database
        if market_data.get('security_id'):
            self.db.insert_volume_footprint(
                market_data['security_id'],
                dict(self.volume_profile),
                self.current_high,
                self.current_low
            )
    
    def get_recent_data(self, periods=200):
        """Get recent data for calculations"""
        if len(self.chart_data) > periods:
            return self.chart_data.tail(periods).to_dict('records')
        return self.chart_data.to_dict('records')
    
    def calculate_poc(self):
        """Calculate Point of Control (highest volume price level)"""
        if not self.volume_profile:
            return None
            
        poc_price = max(self.volume_profile, key=self.volume_profile.get)
        poc_volume = self.volume_profile[poc_price]
        
        return {'price': poc_price, 'volume': poc_volume}
    
    def draw_candlestick_chart(self):
        """Draw candlestick chart with volume footprint"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Set dark theme
        self.ax1.set_facecolor('black')
        self.ax2.set_facecolor('black')
        
        if len(self.chart_data) == 0:
            return
        
        # Get recent data for display
        display_data = self.chart_data.tail(50) if len(self.chart_data) > 50 else self.chart_data
        
        # Draw candlesticks
        for i, (idx, row) in enumerate(display_data.iterrows()):
            x = i
            open_price = row['open']
            high_price = row['high'] 
            low_price = row['low']
            close_price = row['close']
            
            # Determine candle color
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw high-low line
            self.ax1.plot([x, x], [low_price, high_price], color='white', linewidth=1)
            
            # Draw candle body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = patches.Rectangle((x-0.4, body_bottom), 0.8, body_height, 
                                   facecolor=color, edgecolor='white', alpha=0.8)
            self.ax1.add_patch(rect)
        
        # Draw Volume Footprint
        self.draw_volume_footprint()
        
        # Draw volume bars
        volumes = display_data['volume'].values
        volume_colors = ['green' if display_data.iloc[i]['close'] >= display_data.iloc[i]['open'] 
                        else 'red' for i in range(len(display_data))]
        
        self.ax2.bar(range(len(volumes)), volumes, color=volume_colors, alpha=0.7)
        
        # Style the charts
        self.ax1.set_title('Price Action with Volume Footprint (Supabase + Telegram)', color='white', fontsize=14)
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Volume', color='white', fontsize=12)
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.3)
        
        # Set labels
        if len(display_data) > 0:
            timestamps = [t.strftime('%H:%M') for t in display_data['timestamp']]
            tick_positions = range(0, len(timestamps), max(1, len(timestamps)//10))
            self.ax1.set_xticks(tick_positions)
            self.ax1.set_xticklabels([timestamps[i] for i in tick_positions], rotation=45)
            self.ax2.set_xticks(tick_positions)
            self.ax2.set_xticklabels([timestamps[i] for i in tick_positions], rotation=45)
    
    def draw_volume_footprint(self):
        """Draw volume footprint boxes (converted from Pine Script)"""
        if not self.volume_profile or not self.current_high or not self.current_low:
            return
            
        # Calculate POC
        poc_data = self.calculate_poc()
        if not poc_data:
            return
            
        max_volume = max(self.volume_profile.values())
        price_range = self.current_high - self.current_low
        
        if price_range <= 0:
            return
            
        step = price_range / self.bins
        
        # Get current chart x-axis limits
        xlims = self.ax1.get_xlim()
        chart_width = xlims[1] - xlims[0]
        
        # Draw volume footprint boxes
        for i in range(self.bins):
            lower_price = self.current_low + (i * step)
            upper_price = lower_price + step
            
            # Find volume for this price level
            volume_in_bin = 0
            for price, volume in self.volume_profile.items():
                if lower_price <= price < upper_price:
                    volume_in_bin += volume
            
            if volume_in_bin > 0:
                # Calculate box dimensions
                box_width = (volume_in_bin / max_volume) * (chart_width * 0.1)
                x_start = xlims[1] - (chart_width * 0.15)
                
                # Determine color
                is_poc = (poc_data['price'] >= lower_price and poc_data['price'] < upper_price)
                color = self.poc_color if is_poc else 'gray'
                alpha = 0.8 if is_poc else 0.5
                
                # Draw box
                rect = patches.Rectangle((x_start, lower_price), box_width, step,
                                       facecolor=color, alpha=alpha, edgecolor='white')
                self.ax1.add_patch(rect)
                
                # Add volume text for POC
                if is_poc:
                    self.ax1.text(x_start + box_width/2, (lower_price + upper_price)/2, 
                                f'{volume_in_bin:.1f}', ha='center', va='center', 
                                color='white', fontsize=8, weight='bold')
        
        # Draw POC line
        if poc_data:
            self.ax1.axhline(y=poc_data['price'], color=self.poc_color, 
                           linestyle='--', alpha=0.8, linewidth=2)
            
            # Add POC label
            self.ax1.text(xlims[1], poc_data['price'], f' POC: {poc_data["price"]:.2f}', 
                         va='center', ha='left', color=self.poc_color, fontsize=10, weight='bold')
        
        # Add high/low labels
        self.ax1.text(xlims[1], self.current_high, f' High: {self.current_high:.2f}', 
                     va='bottom', ha='left', color='white', fontsize=10)
        self.ax1.text(xlims[1], self.current_low, f' Low: {self.current_low:.2f}', 
                     va='top', ha='left', color='white', fontsize=10)
    
    def add_market_info(self):
        """Add market information display"""
        if len(self.chart_data) == 0:
            return
            
        latest = self.chart_data.iloc[-1]
        alert_count = len(self.db.get_unsent_alerts())
        
        info_text = f"""
Timeframe: {self.timeframe} | DB: Connected | Alerts: {alert_count}
Current Price: {latest['close']:.2f}
High: {self.current_high:.2f} | Low: {self.current_low:.2f}
Volume: {latest['volume']:,}
Buy: {latest.get('buy_quantity', 0):,} | Sell: {latest.get('sell_quantity', 0):,}
Last Update: {latest['timestamp'].strftime('%H:%M:%S')}
        """.strip()
        
        # Add info box
        self.ax1.text(0.02, 0.98, info_text, transform=self.ax1.transAxes, 
                     va='top', ha='left', bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor='black', alpha=0.8), color='white', fontsize=9)
    
    def update_chart_data(self, security_id="11536", exchange_segment="NSE_EQ"):
        """Update chart data and redraw"""
        market_data = self.fetch_market_data(security_id, exchange_segment)
        
        if market_data:
            # Add to chart data
            new_row = pd.DataFrame([market_data])
            self.chart_data = pd.concat([self.chart_data, new_row], ignore_index=True)
            
            # Keep only recent data (last 1000 points)
            if len(self.chart_data) > 1000:
                self.chart_data = self.chart_data.tail(1000).reset_index(drop=True)
            
            # Update volume footprint
            self.update_volume_footprint(market_data)
            
            # Check for alerts
            self.check_alerts(market_data, security_id)
            
            # Redraw chart
            self.draw_candlestick_chart()
            self.add_market_info()
            
            print(f"Updated at {market_data['timestamp'].strftime('%H:%M:%S')} - "
                  f"Price: {market_data['close']:.2f}, Volume: {market_data['volume']:,}")
    
    def start_real_time_updates(self, security_id="11536", exchange_segment="NSE_EQ", interval=23):
        """Start real-time updates every 23 seconds"""
        def update_loop():
            while True:
                try:
                    self.update_chart_data(security_id, exchange_segment)
                    plt.pause(0.1)  # Small pause to update display
                    time.sleep(interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in update loop: {e}")
                    time.sleep(interval)
        
        # Start update thread
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        return update_thread
    
    def show_chart(self, security_id="11536", exchange_segment="NSE_EQ"):
        """Display the chart with real-time updates"""
        plt.ion()  # Turn on interactive mode
        
        # Send startup notification
        self.telegram.send_message(f"🚀 <b>Trading Chart Started</b>\n\n📊 Security: {security_id}\n💱 Exchange: {exchange_segment}\n⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Initial data fetch
        self.update_chart_data(security_id, exchange_segment)
        
        # Start real-time updates
        update_thread = self.start_real_time_updates(security_id, exchange_segment, 23)
        
        try:
            plt.show()
            
            # Keep the main thread alive
            while True:
                plt.pause(1)
                
        except KeyboardInterrupt:
            self.telegram.send_message("🛑 <b>Trading Chart Stopped</b>\n\nChart monitoring has been stopped by user.")
            print("Chart closed by user")
        finally:
            plt.ioff()
            plt.close()

def main():
    """Main function to run the trading chart"""
    # Configuration - Replace with your actual credentials
    CONFIG = {
        'DHAN_ACCESS_TOKEN': 'your_dhan_access_token_here',
        'DHAN_CLIENT_ID': 'your_dhan_client_id_here',
        'SUPABASE_URL': 'your_supabase_project_url_here',
        'SUPABASE_KEY': 'your_supabase_anon_key_here',
        'TELEGRAM_BOT_TOKEN': 'your_telegram_bot_token_here',
        'TELEGRAM_CHAT_ID': 'your_telegram_chat_id_here'
    }
    
    # Validate all credentials
    missing_configs = [key for key, value in CONFIG.items() if value.startswith('your_')]
    if missing_configs:
        print("Please update the following configuration values:")
        for config in missing_configs:
            print(f"  - {config}")
        return
    
    # Create and start the trading chart
    chart = DhanTradingChart(
        access_token=CONFIG['DHAN_ACCESS_TOKEN'],
        client_id=CONFIG['DHAN_CLIENT_ID'],
        supabase_url=CONFIG['SUPABASE_URL'],
        supabase_key=CONFIG['SUPABASE_KEY'],
        telegram_bot_token=CONFIG['TELEGRAM_BOT_TOKEN'],
        telegram_chat_id=CONFIG['TELEGRAM_CHAT_ID']
    )
    
    print("Starting DhanHQ Trading Chart with Supabase & Telegram...")
    print("Features:")
    print("- Real-time chart updates every 23 seconds")
    print("- Volume Footprint indicator")
    print("- Data storage in Supabase database")
    print("- Telegram notifications for alerts")
    print("- Volume spike, price change, and POC shift alerts")
    print("\nPress Ctrl+C to stop")
    
    try:
        # You can customize the security and exchange here
        chart.show_chart(security_id="11536", exchange_segment="NSE_EQ")
    except Exception as e:
        print(f"Error running chart: {e}")

# Database Schema for Supabase Tables
"""
Create these tables in your Supabase dashboard:

1. market_data table:
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    security_id TEXT NOT NULL,
    exchange_segment TEXT NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    buy_quantity BIGINT DEFAULT 0,
    sell_quantity BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX idx_market_data_security ON market_data(security_id, timestamp DESC);

2. volume_footprint table:
CREATE TABLE volume_footprint (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    security_id TEXT NOT NULL,
    price_level DECIMAL(10,2) NOT NULL,
    volume DECIMAL(15,4) NOT NULL,
    is_poc BOOLEAN DEFAULT FALSE,
    session_high DECIMAL(10,2),
    session_low DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_volume_footprint_timestamp ON volume_footprint(timestamp DESC);
CREATE INDEX idx_volume_footprint_security ON volume_footprint(security_id, timestamp DESC);

3. alerts table:
CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    security_id TEXT NOT NULL,
    price DECIMAL(10,2),
    volume BIGINT,
    is_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_alerts_unsent ON alerts(is_sent, timestamp) WHERE is_sent = FALSE;
CREATE INDEX idx_alerts_security ON alerts(security_id, timestamp DESC);

4. Row Level Security (RLS) - Enable if needed:
ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE volume_footprint ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- Create policies as needed for your use case
"""

# Installation Requirements
"""
Install required packages:
pip install supabase pandas numpy matplotlib requests python-telegram-bot

Package versions:
- supabase>=2.0.0
- pandas>=1.5.0
- numpy>=1.24.0
- matplotlib>=3.6.0
- requests>=2.28.0
"""

# Telegram Bot Setup Instructions
"""
1. Create a Telegram Bot:
   - Message @BotFather on Telegram
   - Send /newbot command
   - Choose a name and username for your bot
   - Copy the bot token

2. Get your Chat ID:
   - Start a chat with your bot
   - Send a message to your bot
   - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   - Look for "chat":{"id": YOUR_CHAT_ID}
   - Use this chat ID in the configuration

3. Test your bot:
   - Send a test message using the bot token and chat ID
"""

# Supabase Setup Instructions
"""
1. Create a Supabase project:
   - Go to https://supabase.com
   - Create a new project
   - Copy your project URL and anon key

2. Create tables:
   - Go to your Supabase dashboard
   - Navigate to SQL Editor
   - Run the SQL commands provided in the database schema above

3. Configure API access:
   - Ensure your anon key has access to insert/select on the tables
   - Configure Row Level Security if needed
"""

# Configuration Template
"""
Replace the following in CONFIG dictionary:

DHAN_ACCESS_TOKEN: Your DhanHQ API access token
DHAN_CLIENT_ID: Your DhanHQ client ID
SUPABASE_URL: https://your-project.supabase.co
SUPABASE_KEY: Your Supabase anon key
TELEGRAM_BOT_TOKEN: Your Telegram bot token (from BotFather)
TELEGRAM_CHAT_ID: Your Telegram chat ID (numeric)
"""

if __name__ == "__main__":
    main()