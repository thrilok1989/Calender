import streamlit as st
import pandas as pd
import requests
import time
import datetime
from supabase import create_client
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Streamlit Config ---
st.set_page_config(
    page_title="Nifty Option Screener – Intraday PCR", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Market Hours Check ---
def is_market_open():
    now_utc = datetime.datetime.utcnow()
    ist_offset = datetime.timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset
    
    # Check for weekends
    if now_ist.weekday() > 4:
        return False, "Weekend - Market Closed"
    
    # Check for market hours
    market_open = datetime.time(9, 15)  # More accurate market open time
    market_close = datetime.time(15, 30)  # More accurate market close time
    pre_market = datetime.time(9, 0)
    
    current_time = now_ist.time()
    
    if current_time < pre_market:
        return False, "Pre-market Hours"
    elif pre_market <= current_time < market_open:
        return False, "Pre-market Session"
    elif market_open <= current_time <= market_close:
        return True, "Market Open"
    else:
        return False, "After Hours"

market_status, status_message = is_market_open()

# Enhanced status display
if not market_status:
    if "Market Open" not in status_message:
        st.warning(f"⛔ {status_message}")
        if st.checkbox("Continue anyway (for testing/demo)"):
            pass
        else:
            st.stop()

# --- Enhanced Auto Refresh ---
def auto_refresh(interval_sec=60):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    
    elapsed = time.time() - st.session_state["last_refresh"]
    if elapsed > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()
    
    # Display countdown
    remaining = max(0, interval_sec - elapsed)
    st.sidebar.write(f"⏱️ Next refresh in: {int(remaining)}s")

# --- Enhanced Telegram Setup ---
def send_telegram_message(message, parse_mode="HTML"):
    try:
        bot_token = st.secrets["TELEGRAM"]["BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM"]["CHAT_ID"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id, 
            "text": message, 
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to send Telegram message: {e}")
        return False

# --- Enhanced Supabase Setup ---
@st.cache_resource
def init_supabase():
    try:
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Supabase initialization failed: {e}")
        return None

supabase = init_supabase()

# --- Enhanced PCR Logging ---
def log_pcr(expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi, additional_metrics=None):
    if not supabase:
        return False
    
    try:
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "expiry": expiry,
            "underlying": float(underlying),
            "atm_strike": float(atm_strike),
            "pcr": float(pcr),
            "total_ce_oi": int(total_ce_oi),
            "total_pe_oi": int(total_pe_oi)
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            data.update(additional_metrics)
            
        result = supabase.table("nifty_pcr_logs").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Supabase logging failed: {e}")
        return False

# --- Enhanced Option Chain Fetch ---
@st.cache_data(ttl=120)  # Reduced TTL for more frequent updates
def fetch_option_chain(retries=3):
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }
    
    for attempt in range(retries):
        try:
            s = requests.Session()
            # Get cookies first
            s.get("https://www.nseindia.com", headers=headers, timeout=10)
            time.sleep(1)  # Brief pause
            
            r = s.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                st.warning(f"Attempt {attempt + 1}: HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

# --- Enhanced Data Processing ---
def process_option_data(data, selected_expiry):
    if not data or 'records' not in data:
        return None, None, None, None
    
    raw_data = data['records']['data']
    underlying = data['records'].get('underlyingValue', 0)
    
    ce_list, pe_list = [], []
    
    for item in raw_data:
        if item.get("expiryDate") != selected_expiry:
            continue
            
        strike = item.get("strikePrice", 0)
        
        if "CE" in item:
            ce = item["CE"]
            ce_list.append({
                "strikePrice": strike,
                "OI_CE": ce.get("openInterest", 0),
                "Chg_OI_CE": ce.get("changeinOpenInterest", 0),
                "Vol_CE": ce.get("totalTradedVolume", 0),
                "LTP_CE": ce.get("lastPrice", 0),
                "IV_CE": ce.get("impliedVolatility", 0),
            })
        
        if "PE" in item:
            pe = item["PE"]
            pe_list.append({
                "strikePrice": strike,
                "OI_PE": pe.get("openInterest", 0),
                "Chg_OI_PE": pe.get("changeinOpenInterest", 0),
                "Vol_PE": pe.get("totalTradedVolume", 0),
                "LTP_PE": pe.get("lastPrice", 0),
                "IV_PE": pe.get("impliedVolatility", 0),
            })
    
    return pd.DataFrame(ce_list), pd.DataFrame(pe_list), underlying, raw_data

# --- Main Application ---
st.title("📊 NIFTY Option Screener – Intraday PCR")
st.sidebar.title("⚙️ Settings")

# Fetch and process data
with st.spinner("🔄 Fetching option chain data..."):
    data = fetch_option_chain()

if data is None:
    st.error("❌ Could not fetch data from NSE. Please try again later.")
    st.stop()

# Expiry selection
expiry_list = data['records']['expiryDates']
selected_expiry = st.sidebar.selectbox("📅 Select Expiry Date", expiry_list)

# Process data
df_ce, df_pe, underlying, raw_data = process_option_data(data, selected_expiry)

if df_ce is None or df_pe is None:
    st.error("❌ Error processing option data.")
    st.stop()

# Strike range selection
strike_range = st.sidebar.slider("📏 Strike Range (±points)", 200, 1000, 500, 50)

# ATM Strike calculation
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))

# Filter data
df_ce_filtered = df_ce[
    (df_ce["strikePrice"] >= atm_strike - strike_range) & 
    (df_ce["strikePrice"] <= atm_strike + strike_range)
].copy()

df_pe_filtered = df_pe[
    (df_pe["strikePrice"] >= atm_strike - strike_range) & 
    (df_pe["strikePrice"] <= atm_strike + strike_range)
].copy()

# Enhanced PCR Calculations
total_ce_oi = df_ce_filtered["OI_CE"].sum()
total_pe_oi = df_pe_filtered["OI_PE"].sum()
total_ce_vol = df_ce_filtered["Vol_CE"].sum()
total_pe_vol = df_pe_filtered["Vol_PE"].sum()

pcr_oi = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0
pcr_vol = round(total_pe_vol / total_ce_vol, 3) if total_ce_vol > 0 else 0

# Track PCR changes
if "previous_pcr_oi" not in st.session_state:
    st.session_state.previous_pcr_oi = pcr_oi
if "previous_pcr_vol" not in st.session_state:
    st.session_state.previous_pcr_vol = pcr_vol

pcr_oi_change = pcr_oi - st.session_state.previous_pcr_oi
pcr_vol_change = pcr_vol - st.session_state.previous_pcr_vol

st.session_state.previous_pcr_oi = pcr_oi
st.session_state.previous_pcr_vol = pcr_vol

# Log to Supabase
additional_metrics = {
    "pcr_volume": pcr_vol,
    "total_ce_volume": int(total_ce_vol),
    "total_pe_volume": int(total_pe_vol),
    "strike_range": strike_range
}
log_pcr(selected_expiry, underlying, atm_strike, pcr_oi, total_ce_oi, total_pe_oi, additional_metrics)

# --- Enhanced Dashboard ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📈 NIFTY Spot", f"{underlying:.2f}")

with col2:
    st.metric("📉 PCR (OI)", pcr_oi, delta=f"{pcr_oi_change:.3f}")

with col3:
    st.metric("📊 PCR (Volume)", pcr_vol, delta=f"{pcr_vol_change:.3f}")

with col4:
    st.metric("🎯 ATM Strike", f"{atm_strike}")

# Enhanced Sentiment Analysis
def get_sentiment(pcr_oi, pcr_vol):
    if pcr_oi > 1.2 and pcr_vol > 1.2:
        return "🟢 Strong Bullish", "success"
    elif pcr_oi > 1.0 and pcr_vol > 1.0:
        return "🟢 Bullish", "success"
    elif pcr_oi < 0.8 and pcr_vol < 0.8:
        return "🔴 Strong Bearish", "error"
    elif pcr_oi < 1.0 and pcr_vol < 1.0:
        return "🔴 Bearish", "error"
    else:
        return "🟡 Neutral", "warning"

sentiment, sentiment_type = get_sentiment(pcr_oi, pcr_vol)

col1, col2 = st.columns(2)
with col1:
    if sentiment_type == "success":
        st.success(f"Market Sentiment: {sentiment}")
    elif sentiment_type == "error":
        st.error(f"Market Sentiment: {sentiment}")
    else:
        st.warning(f"Market Sentiment: {sentiment}")

with col2:
    if st.button("📨 Send Telegram Alert"):
        msg = f"""📊 <b>NIFTY PCR Alert</b>
        
💹 <b>Spot:</b> {underlying:.2f}
🎯 <b>ATM Strike:</b> {atm_strike}
📉 <b>PCR (OI):</b> {pcr_oi}
📊 <b>PCR (Vol):</b> {pcr_vol}
🎭 <b>Sentiment:</b> {sentiment}
⏰ <b>Time:</b> {datetime.datetime.now().strftime('%H:%M:%S')}"""
        
        if send_telegram_message(msg):
            st.success("✅ Alert sent successfully!")
        else:
            st.error("❌ Failed to send alert")

# Auto-refresh
auto_refresh(60)

# --- Enhanced Option Chain Display ---
if not df_ce_filtered.empty and not df_pe_filtered.empty:
    merged_df = pd.merge(df_ce_filtered, df_pe_filtered, on="strikePrice", how="outer").sort_values("strikePrice")
    
    # Add price tags based on actual IV
    def get_price_tag(iv):
        if pd.isna(iv) or iv == 0:
            return "❓ No Data"
        elif iv < 12:
            return "🟢 Cheap"
        elif iv > 20:
            return "🔴 Expensive" 
        else:
            return "🟡 Fair"
    
    merged_df["Price_Tag_CE"] = merged_df["IV_CE"].apply(get_price_tag)
    merged_df["Price_Tag_PE"] = merged_df["IV_PE"].apply(get_price_tag)
    
    # Highlight ATM row
    def highlight_atm(row):
        if row["strikePrice"] == atm_strike:
            return ['background-color: yellow'] * len(row)
        return [''] * len(row)
    
    st.markdown("### 🧾 Option Chain Analysis")
    styled_df = merged_df.style.apply(highlight_atm, axis=1)
    st.dataframe(styled_df, use_container_width=True)

# --- Enhanced Intraday Analysis ---
def fetch_today_logs():
    if not supabase:
        return pd.DataFrame()
    
    try:
        today = datetime.datetime.now().date().isoformat()
        res = supabase.table("nifty_pcr_logs").select("*").gte("timestamp", today).order("timestamp").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
        return pd.DataFrame()

logs_df = fetch_today_logs()

if not logs_df.empty:
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    
    def get_session(time):
        if datetime.time(9, 15) <= time < datetime.time(11, 30):
            return "Morning"
        elif datetime.time(11, 30) <= time < datetime.time(13, 30):
            return "Midday"
        elif datetime.time(13, 30) <= time <= datetime.time(15, 30):
            return "Closing"
        else:
            return "Off Hours"
    
    logs_df["session"] = logs_df["timestamp"].dt.time.apply(get_session)
    logs_df = logs_df[logs_df["session"] != "Off Hours"]
    
    if not logs_df.empty:
        st.markdown("### 📈 Intraday PCR Trend")
        
        # Enhanced chart with dual PCR
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('PCR Trend', 'NIFTY Spot Price'),
            vertical_spacing=0.1
        )
        
        # PCR OI line
        fig.add_trace(
            go.Scatter(
                x=logs_df["timestamp"], 
                y=logs_df["pcr"], 
                mode='lines+markers',
                name='PCR (OI)',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # PCR Volume line if available
        if 'pcr_volume' in logs_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=logs_df["timestamp"], 
                    y=logs_df["pcr_volume"], 
                    mode='lines+markers',
                    name='PCR (Volume)',
                    line=dict(color='orange', width=2, dash='dash'),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # NIFTY Spot line
        fig.add_trace(
            go.Scatter(
                x=logs_df["timestamp"], 
                y=logs_df["underlying"], 
                mode='lines+markers',
                name='NIFTY Spot',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Add horizontal lines for PCR levels
        fig.add_hline(y=1.2, line_dash="dash", line_color="green", 
                     annotation_text="Bullish Level", row=1, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="Bearish Level", row=1, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Session Analysis
        if len(logs_df["session"].unique()) > 1:
            session_avg = logs_df.groupby("session").agg({
                "pcr": "mean",
                "pcr_volume": "mean" if "pcr_volume" in logs_df.columns else lambda x: 0,
                "underlying": "mean"
            }).round(3)
            
            st.markdown("### 📊 Session-wise Analysis")
            col1, col2, col3 = st.columns(3)
            
            sessions = ["Morning", "Midday", "Closing"]
            cols = [col1, col2, col3]
            
            for i, session in enumerate(sessions):
                if session in session_avg.index:
                    with cols[i]:
                        st.metric(
                            f"🕐 {session} PCR (OI)", 
                            session_avg.loc[session, "pcr"]
                        )
                        if "pcr_volume" in session_avg.columns:
                            st.metric(
                                f"📊 {session} PCR (Vol)", 
                                session_avg.loc[session, "pcr_volume"]
                            )
            
            # Enhanced Sentiment Flip Detection
            st.markdown("### 🔄 Sentiment Flip Analysis")
            
            flips = []
            sessions_order = ["Morning", "Midday", "Closing"]
            
            if "sentiment_flips_alerted" not in st.session_state:
                st.session_state.sentiment_flips_alerted = []
            
            for i in range(len(sessions_order) - 1):
                s1, s2 = sessions_order[i], sessions_order[i + 1]
                
                if s1 in session_avg.index and s2 in session_avg.index:
                    p1, p2 = session_avg.loc[s1, "pcr"], session_avg.loc[s2, "pcr"]
                    flip_text = None
                    
                    if p1 > 1.1 and p2 < 0.9:
                        flip_text = f"⚠️ Bullish ({s1}) → Bearish ({s2})"
                    elif p1 < 0.9 and p2 > 1.1:
                        flip_text = f"✅ Bearish ({s1}) → Bullish ({s2})"
                    elif abs(p1 - p2) > 0.3:
                        direction = "📈" if p2 > p1 else "📉"
                        flip_text = f"{direction} Significant change: {s1} ({p1:.2f}) → {s2} ({p2:.2f})"
                    
                    if flip_text:
                        flips.append(flip_text)
                        
                        # Auto-alert for major flips
                        if flip_text not in st.session_state.sentiment_flips_alerted and ("Bullish → Bearish" in flip_text or "Bearish → Bullish" in flip_text):
                            message = f"""🚨 <b>NIFTY PCR Flip Alert</b>
                            
{flip_text}
💹 <b>Current Spot:</b> {underlying:.2f}
⏰ <b>Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                            
                            if send_telegram_message(message):
                                st.session_state.sentiment_flips_alerted.append(flip_text)
                                st.success(f"📱 Auto-alert sent: {flip_text}")
            
            for flip in flips:
                if "Bullish → Bearish" in flip or "Bearish → Bullish" in flip:
                    st.warning(flip)
                else:
                    st.info(flip)
            
            if not flips:
                st.success("✅ No significant sentiment flips detected today.")

else:
    st.info("📊 No intraday data available yet. Data will appear as the day progresses.")

# --- Sidebar Information ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Key Levels")
st.sidebar.markdown("""
- **PCR > 1.2**: Bullish (More Puts)
- **PCR < 0.8**: Bearish (More Calls)
- **PCR 0.8-1.2**: Neutral Range
""")

st.sidebar.markdown("### ⏰ Market Status")
st.sidebar.write(f"Status: {status_message}")
st.sidebar.write(f"Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown("*Data sourced from NSE India. This is for educational purposes only.*")