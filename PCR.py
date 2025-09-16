import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
import pytz
import json

# Page config
st.set_page_config(page_title="Debug Nifty Options Data", layout="wide")

class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'access-token': st.secrets["DHAN_ACCESS_TOKEN"],
            'client-id': st.secrets["DHAN_CLIENT_ID"],
            'Content-Type': 'application/json'
        }
    
    def get_option_chain(self, underlying_scrip, underlying_seg, expiry):
        url = f"{self.base_url}/optionchain"
        payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"API Exception: {e}")
            return None
    
    def get_expiry_list(self, underlying_scrip, underlying_seg):
        url = f"{self.base_url}/optionchain/expirylist"
        payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Expiry API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Expiry API Exception: {e}")
            return None

def main():
    st.title("Debug Nifty Options Data")
    
    api = DhanAPI()
    
    # Get expiry dates
    st.subheader("1. Available Expiry Dates")
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry_dates = expiry_data['data']
        st.write(f"Found {len(expiry_dates)} expiries:")
        for exp in expiry_dates[:5]:  # Show first 5
            st.write(f"• {exp}")
        
        selected_expiry = st.selectbox("Select Expiry for Analysis", expiry_dates)
    else:
        st.error("Failed to get expiry dates")
        selected_expiry = "2024-10-31"
    
    if st.button("Analyze Option Chain"):
        st.subheader("2. Option Chain Analysis")
        
        option_chain = api.get_option_chain(13, "IDX_I", selected_expiry)
        
        if option_chain and 'data' in option_chain:
            # Show basic info
            spot_price = option_chain['data'].get('last_price', 0)
            st.metric("Nifty Spot Price", f"₹{spot_price:.2f}")
            
            oc_data = option_chain['data'].get('oc', {})
            st.write(f"Total strikes available: {len(oc_data)}")
            
            # Analyze all available strikes
            strikes_info = []
            
            for strike_key, strike_data in oc_data.items():
                try:
                    strike_price = float(strike_key.split('.')[0])
                    
                    # CE data
                    ce_data = strike_data.get('ce', {})
                    ce_ltp = ce_data.get('last_price', 0)
                    ce_volume = ce_data.get('volume', 0)
                    ce_oi = ce_data.get('oi', 0)
                    
                    # PE data
                    pe_data = strike_data.get('pe', {})
                    pe_ltp = pe_data.get('last_price', 0)
                    pe_volume = pe_data.get('volume', 0)
                    pe_oi = pe_data.get('oi', 0)
                    
                    strikes_info.append({
                        'Strike': int(strike_price),
                        'Strike_Key': strike_key,
                        'CE_LTP': ce_ltp,
                        'CE_Volume': ce_volume,
                        'CE_OI': ce_oi,
                        'PE_LTP': pe_ltp,
                        'PE_Volume': pe_volume,
                        'PE_OI': pe_oi,
                        'CE_Active': ce_ltp > 0,
                        'PE_Active': pe_ltp > 0
                    })
                except:
                    continue
            
            # Sort by strike price
            strikes_info.sort(key=lambda x: x['Strike'])
            
            # Convert to DataFrame for display
            df = pd.DataFrame(strikes_info)
            
            st.subheader("3. All Available Strikes")
            st.dataframe(df, use_container_width=True)
            
            # Find ATM and nearby strikes
            atm_strike = round(spot_price / 50) * 50
            st.write(f"Calculated ATM Strike: {atm_strike}")
            
            # Filter active strikes (with LTP > 0)
            active_ce = df[df['CE_Active'] == True]
            active_pe = df[df['PE_Active'] == True]
            
            st.subheader("4. Active Strikes (LTP > 0)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Active Call Options (CE):**")
                if not active_ce.empty:
                    for _, row in active_ce.head(10).iterrows():
                        st.write(f"• {row['Strike']} CE: ₹{row['CE_LTP']:.2f} (Vol: {row['CE_Volume']}, OI: {row['CE_OI']})")
                else:
                    st.write("No active CE options found")
            
            with col2:
                st.write("**Active Put Options (PE):**")
                if not active_pe.empty:
                    for _, row in active_pe.head(10).iterrows():
                        st.write(f"• {row['Strike']} PE: ₹{row['PE_LTP']:.2f} (Vol: {row['PE_Volume']}, OI: {row['PE_OI']})")
                else:
                    st.write("No active PE options found")
            
            # Check specific strike (25150)
            st.subheader("5. Specific Strike Analysis (25150)")
            
            specific_strike = df[df['Strike'] == 25150]
            if not specific_strike.empty:
                row = specific_strike.iloc[0]
                st.write(f"Strike Key in API: `{row['Strike_Key']}`")
                st.write(f"25150 CE - LTP: ₹{row['CE_LTP']}, Volume: {row['CE_Volume']}, OI: {row['CE_OI']}")
                st.write(f"25150 PE - LTP: ₹{row['PE_LTP']}, Volume: {row['PE_Volume']}, OI: {row['PE_OI']}")
                
                if row['CE_LTP'] == 0:
                    st.error("25150 CE has zero LTP - no trading activity")
                else:
                    st.success(f"25150 CE is active with LTP ₹{row['CE_LTP']}")
            else:
                st.error("25150 strike not found in option chain")
                
                # Find closest strikes
                st.write("**Closest available strikes:**")
                df['Distance'] = abs(df['Strike'] - 25150)
                closest = df.nsmallest(5, 'Distance')
                
                for _, row in closest.iterrows():
                    st.write(f"• {row['Strike']} - CE: ₹{row['CE_LTP']}, PE: ₹{row['PE_LTP']}")
            
            # Market timing check
            st.subheader("6. Market Status")
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if market_open <= ist_now <= market_close:
                st.success(f"Market is OPEN - Current time: {ist_now.strftime('%H:%M:%S')} IST")
            else:
                st.warning(f"Market is CLOSED - Current time: {ist_now.strftime('%H:%M:%S')} IST")
                st.write("Market hours: 9:15 AM - 3:30 PM IST")
            
            # Raw API response (for debugging)
            if st.checkbox("Show Raw API Response"):
                st.subheader("7. Raw API Response")
                st.json(option_chain)
                
        else:
            st.error("Failed to get option chain data")

if __name__ == "__main__":
    main()