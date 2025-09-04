import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests, pandas as pd, numpy as np
from datetime import datetime
from scipy.stats import norm
from pytz import timezone
from supabase import create_client

st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="refresh")

core_cols = ["Time", "Strike", "OI_CE", "OI_PE", "Volume_CE", "Volume_PE"]
defaults = {
    "oi_volume_history": pd.DataFrame(columns=core_cols),
    "last_alert_time": {},
    "pcr_history": pd.DataFrame(columns=["Time","Strike","PCR","Signal","VIX"]),
    "pcr_threshold_bull": 1.2,
    "pcr_threshold_bear": 0.7,
    "use_pcr_filter": True
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

@st.cache_resource
def init_supabase():
    try:
        s = st.secrets; url, key = s.get("SUPABASE_URL"), s.get("SUPABASE_KEY")
        return create_client(url, key) if url and key else None
    except: return None

supabase = init_supabase()
def tg_cfg(): 
    s=st.secrets
    return {"bot_token": s.get("TELEGRAM_BOT_TOKEN",""),"chat_id": s.get("TELEGRAM_CHAT_ID","")}
TGC=tg_cfg()

def send_telegram(msg):
    if TGC["bot_token"] and TGC["chat_id"]:
        url=f"https://api.telegram.org/bot{TGC['bot_token']}/sendMessage"
        try: requests.post(url,data={"chat_id":TGC["chat_id"],"text":msg})
        except: pass

def supabase_store(ts, strike, oi_ce, oi_pe, vol_ce, vol_pe):
    if supabase:
        data = {"timestamp": ts, "strike": strike, "oi_ce": oi_ce, "oi_pe": oi_pe, "volume_ce": vol_ce, "volume_pe": vol_pe, "created_at":datetime.now(timezone("Asia/Kolkata")).isoformat()}
        try: supabase.table("oi_volume_data").insert(data).execute()
        except: pass

def supabase_hist(strike, lookback=30):
    if supabase:
        from_time = datetime.now(timezone("Asia/Kolkata")) - pd.Timedelta(minutes=lookback)
        try:
            res = supabase.table("oi_volume_data").select("*").eq("strike", strike).gte("timestamp", from_time.isoformat()).execute()
            return pd.DataFrame(res.data)
        except: pass
    return pd.DataFrame()

def oi_vol_spikes(cur, hist, strike, threshold=2.0):
    alerts = []; 
    if not hist.empty:
        avg = {k: hist[k].mean() for k in ['oi_ce','oi_pe','volume_ce','volume_pe']}
        if avg['oi_ce']>0 and cur['oi_ce']>avg['oi_ce']*threshold: alerts.append(f"📈 OI Spike CE {strike}: {cur['oi_ce']:.0f} (avg:{avg['oi_ce']:.0f})")
        if avg['oi_pe']>0 and cur['oi_pe']>avg['oi_pe']*threshold: alerts.append(f"📈 OI Spike PE {strike}: {cur['oi_pe']:.0f} (avg:{avg['oi_pe']:.0f})")
        if avg['volume_ce']>0 and cur['volume_ce']>avg['volume_ce']*threshold: alerts.append(f"🔥 Vol Spike CE {strike}: {cur['volume_ce']:.0f} (avg:{avg['volume_ce']:.0f})")
        if avg['volume_pe']>0 and cur['volume_pe']>avg['volume_pe']*threshold: alerts.append(f"🔥 Vol Spike PE {strike}: {cur['volume_pe']:.0f} (avg:{avg['volume_pe']:.0f})")
    return alerts

def greeks(opt, S, K, T, r, sigma):
    import math
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt=="CE":
        delta,theta,rho= norm.cdf(d1),(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365,(K*T*np.exp(-r*T)*norm.cdf(d2))/100
    else:
        delta,theta,rho= -norm.cdf(-d1),(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365,(-K*T*np.exp(-r*T)*norm.cdf(-d2))/100
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T)); vega = S*norm.pdf(d1)*np.sqrt(T)/100
    return round(delta,4),round(gamma,4),round(vega,4),round(theta,4),round(rho,4)

def delta_vol_bias(price, vol, chg_oi):
    if price>0 and vol>0 and chg_oi>0: return "Bullish"
    elif price<0 and vol>0 and chg_oi>0: return "Bearish"
    elif price>0 and vol>0 and chg_oi<0: return "Bullish"
    elif price<0 and vol>0 and chg_oi<0: return "Bearish"
    return "Neutral"

def verdict(score):
    if score >= 4: return "Strong Bullish"
    elif score >= 2: return "Bullish"
    elif score <= -4: return "Strong Bearish"
    elif score <= -2: return "Bearish"
    return "Neutral"

def pcr_levels(df_summary):
    s=df_summary[df_summary.PCR>1.8]["Strike"].tolist()
    r=df_summary[df_summary.PCR<0.6]["Strike"].tolist()
    return s,r

def analyze():
    try:
        now=datetime.now(timezone("Asia/Kolkata"))
        day=now.weekday(); tm=now.time()
        if day>=5 or not(datetime.strptime("09:00","%H:%M").time() <= tm <= datetime.strptime("19:40","%H:%M").time()):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)"); return
        sess=requests.Session(); sess.headers.update({"User-Agent":"Mozilla/5.0"})
        try: sess.get("https://www.nseindia.com",timeout=5)
        except: st.error("❌ NSE session fail"); return
        try: vix=sess.get("https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX",timeout=10).json()["data"][0]["lastPrice"]
        except: vix=11
        st.session_state.pcr_threshold_bull=2.0 if vix>12 else 1.2
        st.session_state.pcr_threshold_bear=0.4 if vix>12 else 0.7
        url="https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        try: data=sess.get(url,timeout=10).json()
        except: st.error("❌ Option chain fetch fail"); return
        if not data or 'records' not in data: st.error("❌ Empty NSE resp"); return
        rec=data['records']['data']
        expiry=data['records']['expiryDates'][0]
        underlying=data['records']['underlyingValue']
        st.markdown(f"### 📍 Spot: {underlying}")
        st.markdown(f"### 📊 VIX: {vix} | PCR Thresholds: Bull >{st.session_state.pcr_threshold_bull} | Bear <{st.session_state.pcr_threshold_bear}")
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        T = max((expiry_date-datetime.now(timezone("Asia/Kolkata"))).days,1)/365; r=0.06
        calls,puts=[],[]
        for it in rec:
            if 'CE' in it and it['CE']['expiryDate']==expiry:
                ce=it['CE']
                if ce['impliedVolatility']>0: ce.update(dict(zip(['Delta','Gamma','Vega','Theta','Rho'],greeks('CE',underlying,ce['strikePrice'],T,r,ce['impliedVolatility']/100))))
                calls.append(ce)
            if 'PE' in it and it['PE']['expiryDate']==expiry:
                pe=it['PE']
                if pe['impliedVolatility']>0: pe.update(dict(zip(['Delta','Gamma','Vega','Theta','Rho'],greeks('PE',underlying,pe['strikePrice'],T,r,pe['impliedVolatility']/100))))
                puts.append(pe)
        df_ce=pd.DataFrame(calls); df_pe=pd.DataFrame(puts)
        df=pd.merge(df_ce,df_pe,on='strikePrice',suffixes=('_CE','_PE')).sort_values('strikePrice')
        atm=min(df['strikePrice'], key=lambda x: abs(x-underlying))
        df=df[df['strikePrice'].between(atm-200,atm+200)]
        df['Zone']=df['strikePrice'].apply(lambda x: 'ATM' if x==atm else 'ITM' if x<underlying else 'OTM')
        now_str=now.strftime("%H:%M:%S")
        spike_alerts=[]
        for _,row in df[df['strikePrice'].between(atm-100,atm+100)].iterrows():
            s=row['strikePrice']
            d={'oi_ce':row['openInterest_CE'],'oi_pe':row['openInterest_PE'],'volume_ce':row['totalTradedVolume_CE'],'volume_pe':row['totalTradedVolume_PE']}
            n=pd.DataFrame({"Time":[now_str],"Strike":[s],"OI_CE":[d['oi_ce']],"OI_PE":[d['oi_pe']],"Volume_CE":[d['volume_ce']],"Volume_PE":[d['volume_pe']]})
            st.session_state.oi_volume_history=pd.concat([st.session_state.oi_volume_history,n])
            supabase_store(now_str,s,d['oi_ce'],d['oi_pe'],d['volume_ce'],d['volume_pe'])
            las=f"{s}_{now_str[:-3]}"
            if las not in st.session_state.last_alert_time:
                hist=supabase_hist(s,30)
                alerts=oi_vol_spikes(d,hist,s,2.5)
                if alerts:
                    spike_alerts+=alerts; st.session_state.last_alert_time[las]=now.timestamp()
                    for a in alerts:
                        if "Spike" in a: send_telegram(f"🚨 {a}\n📍 Spot:{underlying}\n⏰ {now_str}\n📊 VIX:{vix}")
        if spike_alerts: 
            st.markdown("### ⚠️ OI/Volume Spike Alerts"); [st.warning(a) for a in spike_alerts]
        weights={'ChgOI_Bias':1.5,'Volume_Bias':1,'Gamma_Bias':1.2,'AskQty_Bias':0.8,'BidQty_Bias':0.8,'IV_Bias':1,'DVP_Bias':1.5}
        bias_results=[]; total_score=0
        for _,row in df.iterrows():
            if abs(row['strikePrice']-atm)>100: continue
            score=0
            rdat={
                "Strike":row['strikePrice'],
                "Zone":row['Zone'],
                "ChgOI_Bias":"Bullish" if row['changeinOpenInterest_CE']<row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias":"Bullish" if row['totalTradedVolume_CE']<row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias":"Bullish" if row['Gamma_CE']<row['Gamma_PE'] else "Bearish",
                "AskQty_Bias":"Bullish" if row['askQty_PE']>row['askQty_CE'] else "Bearish",
                "BidQty_Bias":"Bearish" if row['bidQty_PE']>row['bidQty_CE'] else "Bullish",
                "IV_Bias":"Bullish" if row['impliedVolatility_CE']>row['impliedVolatility_PE'] else "Bearish",
                "DVP_Bias": delta_vol_bias(row['lastPrice_CE']-row['lastPrice_PE'], row['totalTradedVolume_CE']-row['totalTradedVolume_PE'], row['changeinOpenInterest_CE']-row['changeinOpenInterest_PE'])
            }
            for k in [k for k in rdat if "_Bias" in k]:
                score += weights[k] if rdat[k]=="Bullish" else -weights[k]
            rdat["BiasScore"]=score; rdat["Verdict"]=verdict(score); total_score+=score; bias_results.append(rdat)
        df_sum=pd.DataFrame(bias_results)
        df_sum=pd.merge(df_sum,df[['strikePrice','openInterest_CE','openInterest_PE']],left_on='Strike',right_on='strikePrice',how='left')
        df_sum['PCR']=np.where(df_sum['openInterest_CE']==0,0,(df_sum['openInterest_PE']/df_sum['openInterest_CE']).round(2))
        df_sum['PCR_Signal']=np.where(df_sum['PCR']>st.session_state.pcr_threshold_bull,"Bullish",np.where(df_sum['PCR']<st.session_state.pcr_threshold_bear,"Bearish","Neutral"))
        df_sum=df_sum.drop(columns=['strikePrice'])

        def color_pcr(val):
            if val>st.session_state.pcr_threshold_bull: return 'background-color: #90EE90; color: black'
            elif val<st.session_state.pcr_threshold_bear: return 'background-color: #FFB6C1; color: black'
            else: return 'background-color: #FFFFE0; color: black'
        styled_df = df_sum.style.applymap(color_pcr, subset=['PCR'])

        for _,row in df_sum.iterrows():
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, pd.DataFrame({"Time":[now_str],"Strike":[row['Strike']],"PCR":[row['PCR']],"Signal":[row['PCR_Signal']],"VIX":[vix]})])

        atm_row = df_sum[df_sum["Zone"]=="ATM"].iloc[0] if not df_sum[df_sum["Zone"]=="ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        sup, res = pcr_levels(df_sum)
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ PCR-Based Support: `{', '.join(map(str,sup)) if sup else 'N/A'}`")
        st.markdown(f"### 🚧 PCR-Based Resistance: `{', '.join(map(str,res)) if res else 'N/A'}`")
        with st.expander("📊 Option Chain Summary"):
            st.info(f"ℹ️ PCR: >1.8=Support, <0.6=Resistance | Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'} | VIX:{vix}")
            st.dataframe(styled_df,use_container_width=True)
        st.markdown("### 🧮 PCR Configuration")
        col1,col2,col3=st.columns(3)
        with col1:
            st.session_state.pcr_threshold_bull=st.number_input("Bullish PCR (>)",1.0,5.0,st.session_state.pcr_threshold_bull,0.1)
        with col2:
            st.session_state.pcr_threshold_bear=st.number_input("Bearish PCR (<)",0.1,1.0,st.session_state.pcr_threshold_bear,0.1)
        with col3:
            st.session_state.use_pcr_filter=st.checkbox("Enable PCR Filtering",st.session_state.use_pcr_filter)
        with st.expander("📈 PCR History"):
            if not st.session_state.pcr_history.empty:
                piv=st.session_state.pcr_history.pivot_table(index='Time',columns='Strike',values='PCR',aggfunc='last')
                st.line_chart(piv)
                st.dataframe(st.session_state.pcr_history)
            else: st.info("No PCR history recorded yet")
        with st.expander("📊 OI/Volume History"):
            if not st.session_state.oi_volume_history.empty:
                st.dataframe(st.session_state.oi_volume_history.tail(20))
                atm_oi = st.session_state.oi_volume_history[st.session_state.oi_volume_history['Strike']==atm]
                if not atm_oi.empty:
                    import plotly.graph_objects as go
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=atm_oi['Time'],y=atm_oi['OI_CE'],name='OI CE',line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=atm_oi['Time'],y=atm_oi['OI_PE'],name='OI PE',line=dict(color='red')))
                    fig.update_layout(title=f"OI Trend ATM {atm}")
                    st.plotly_chart(fig,use_container_width=True)
            else: st.info("No OI/Volume history recorded yet")
    except Exception as e: st.error(f"❌ Unexpected error: {e}")

if __name__=="__main__": analyze()
