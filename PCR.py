import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from scipy.stats import norm
from pytz import timezone
import io
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
from scipy import stats

# Deep Learning imports (optional - install with: pip install tensorflow)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not installed. Deep learning features disabled. Install with: pip install tensorflow")

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

# Initialize Supabase client
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("Connected to Supabase")
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        supabase_client = None
else:
    st.info("Supabase not configured. Add SUPABASE_URL and SUPABASE_KEY to secrets.toml or environment variables to enable data storage.")

# === Streamlit Config ===
st.set_page_config(page_title="AI-Enhanced Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=2200000, key="datarefresh")

# Initialize session state - Basic
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

if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 1.2
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.7
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# ML-specific session state
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'feature_scaler' not in st.session_state:
    st.session_state.feature_scaler = StandardScaler()
if 'ml_features_history' not in st.session_state:
    st.session_state.ml_features_history = pd.DataFrame()
if 'ml_predictions' not in st.session_state:
    st.session_state.ml_predictions = []

# AI-specific session state
if 'ai_models' not in st.session_state:
    st.session_state.ai_models = {}
if 'neural_network' not in st.session_state:
    st.session_state.neural_network = None
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = None
if 'price_lstm' not in st.session_state:
    st.session_state.price_lstm = None
if 'clustering_model' not in st.session_state:
    st.session_state.clustering_model = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = []
if 'pattern_recognition' not in st.session_state:
    st.session_state.pattern_recognition = {}
if 'volatility_predictor' not in st.session_state:
    st.session_state.volatility_predictor = None

# === Telegram Config ===
try:
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
except Exception:
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# === Instrument Mapping ===
NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"
# === Advanced AI Functions ===

def create_neural_network(input_shape):
    """Create a deep neural network for market prediction"""
    if not TF_AVAILABLE:
        return None
    
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes: Bearish, Neutral, Bullish
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(sequence_length, features):
    """Create LSTM model for price prediction"""
    if not TF_AVAILABLE:
        return None
    
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(25),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def detect_market_anomalies(features_df):
    """Detect unusual market conditions using Isolation Forest"""
    if len(features_df) < 20:
        return {}
    
    try:
        if st.session_state.anomaly_detector is None:
            st.session_state.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Fit on historical data
            features_scaled = StandardScaler().fit_transform(features_df.fillna(0))
            st.session_state.anomaly_detector.fit(features_scaled)
        
        # Detect anomalies in recent data
        recent_features = features_df.tail(1).fillna(0)
        anomaly_score = st.session_state.anomaly_detector.decision_function(recent_features)
        is_anomaly = st.session_state.anomaly_detector.predict(recent_features)[0] == -1
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score[0],
            'interpretation': 'Unusual market condition detected!' if is_anomaly else 'Normal market condition'
        }
        
    except Exception as e:
        st.warning(f"Anomaly detection error: {e}")
        return {}

def cluster_market_regimes(features_df):
    """Identify market regimes using clustering"""
    if len(features_df) < 30:
        return None
    
    try:
        if st.session_state.clustering_model is None:
            # Prepare features for clustering
            cluster_features = ['volatility', 'rsi', 'momentum_5', 'avg_pcr', 'total_bias_score']
            available_features = [f for f in cluster_features if f in features_df.columns]
            
            if len(available_features) < 3:
                return None
            
            X = features_df[available_features].fillna(0)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            st.session_state.clustering_model = {
                'model': kmeans,
                'scaler': scaler,
                'features': available_features
            }
            
            # Define cluster characteristics
            cluster_names = {
                0: "📈 Trending Bull Market",
                1: "📉 Trending Bear Market", 
                2: "🌊 Volatile Sideways",
                3: "😴 Low Volatility Range"
            }
            
            return {
                'current_regime': cluster_names.get(clusters[-1], "Unknown"),
                'regime_id': clusters[-1],
                'regime_history': clusters
            }
        
        else:
            # Predict current regime
            model_data = st.session_state.clustering_model
            recent_features = features_df[model_data['features']].tail(1).fillna(0)
            recent_scaled = model_data['scaler'].transform(recent_features)
            current_regime = model_data['model'].predict(recent_scaled)[0]
            
            cluster_names = {
                0: "📈 Trending Bull Market",
                1: "📉 Trending Bear Market",
                2: "🌊 Volatile Sideways", 
                3: "😴 Low Volatility Range"
            }
            
            return {
                'current_regime': cluster_names.get(current_regime, "Unknown"),
                'regime_id': current_regime
            }
            
    except Exception as e:
        st.warning(f"Clustering error: {e}")
        return None

def advanced_pattern_recognition(price_data, df_summary):
    """Advanced pattern recognition using multiple AI techniques"""
    patterns = {}
    
    if len(price_data) < 20:
        return patterns
    
    try:
        prices = price_data['Spot'].values
        
        # 1. Price Pattern Detection
        # Head and Shoulders
        if len(prices) >= 10:
            recent_prices = prices[-10:]
            local_maxima = []
            for i in range(1, len(recent_prices)-1):
                if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                    local_maxima.append((i, recent_prices[i]))
            
            if len(local_maxima) >= 3:
                # Check for head and shoulders pattern
                if (local_maxima[1][1] > local_maxima[0][1] and 
                    local_maxima[1][1] > local_maxima[2][1]):
                    patterns['head_shoulders'] = {
                        'detected': True,
                        'signal': 'Bearish',
                        'confidence': 0.7
                    }
        
        # 2. Support/Resistance AI Detection
        def find_support_resistance(prices, window=5):
            supports = []
            resistances = []
            
            for i in range(window, len(prices) - window):
                # Support: local minimum
                if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
                   all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                    supports.append(prices[i])
                
                # Resistance: local maximum  
                if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
                   all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                    resistances.append(prices[i])
            
            return supports, resistances
        
        supports, resistances = find_support_resistance(prices)
        current_price = prices[-1]
        
        # Find nearest support/resistance
        if supports:
            nearest_support = max([s for s in supports if s < current_price], default=None)
            if nearest_support:
                support_distance = (current_price - nearest_support) / current_price
                patterns['support_analysis'] = {
                    'nearest_support': nearest_support,
                    'distance_pct': support_distance * 100,
                    'strength': 'Strong' if support_distance < 0.02 else 'Moderate'
                }
        
        if resistances:
            nearest_resistance = min([r for r in resistances if r > current_price], default=None)
            if nearest_resistance:
                resistance_distance = (nearest_resistance - current_price) / current_price
                patterns['resistance_analysis'] = {
                    'nearest_resistance': nearest_resistance,
                    'distance_pct': resistance_distance * 100,
                    'strength': 'Strong' if resistance_distance < 0.02 else 'Moderate'
                }
        
        # 3. Options Flow Patterns
        if not df_summary.empty:
            # Unusual options activity detection
            avg_oi_ce = df_summary['openInterest_CE'].mean()
            avg_oi_pe = df_summary['openInterest_PE'].mean()
            max_oi_ce = df_summary['openInterest_CE'].max()
            max_oi_pe = df_summary['openInterest_PE'].max()
            
            # Detect unusual activity
            if max_oi_ce > 3 * avg_oi_ce:
                patterns['unusual_call_activity'] = {
                    'detected': True,
                    'intensity': 'High',
                    'strike': df_summary.loc[df_summary['openInterest_CE'].idxmax(), 'Strike'],
                    'signal': 'Potential bullish breakout'
                }
            
            if max_oi_pe > 3 * avg_oi_pe:
                patterns['unusual_put_activity'] = {
                    'detected': True,
                    'intensity': 'High', 
                    'strike': df_summary.loc[df_summary['openInterest_PE'].idxmax(), 'Strike'],
                    'signal': 'Potential bearish breakdown'
                }
        
        # 4. Momentum Divergence Detection
        if len(prices) >= 15:
            # Price momentum vs options momentum
            price_momentum = (prices[-1] - prices[-5]) / prices[-5]
            
            if not df_summary.empty:
                pcr_momentum = df_summary['PCR'].mean() - 1.0  # Deviation from neutral
                
                # Detect divergence
                if price_momentum > 0.01 and pcr_momentum > 0.2:  # Price up, high PCR (bearish options)
                    patterns['momentum_divergence'] = {
                        'type': 'Bearish Divergence',
                        'description': 'Price rising but options show bearish sentiment',
                        'signal': 'Potential reversal down'
                    }
                elif price_momentum < -0.01 and pcr_momentum < -0.2:  # Price down, low PCR (bullish options)
                    patterns['momentum_divergence'] = {
                        'type': 'Bullish Divergence',
                        'description': 'Price falling but options show bullish sentiment',
                        'signal': 'Potential reversal up'
                    }
        
        return patterns
        
    except Exception as e:
        st.warning(f"Pattern recognition error: {e}")
        return {}

def predict_volatility(features_df):
    """Predict future volatility using AI"""
    if len(features_df) < 30:
        return None
    
    try:
        if 'volatility' not in features_df.columns:
            return None
        
        # Prepare volatility prediction data
        volatility_data = features_df['volatility'].dropna()
        
        if len(volatility_data) < 20:
            return None
        
        # Simple LSTM for volatility prediction
        def create_volatility_sequences(data, seq_length=10):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        
        if st.session_state.volatility_predictor is None and len(volatility_data) >= 50:
            # Create and train volatility model
            X, y = create_volatility_sequences(volatility_data.values)
            
            if len(X) > 20 and TF_AVAILABLE:
                model = keras.Sequential([
                    layers.LSTM(20, input_shape=(X.shape[1], 1)),
                    layers.Dense(10),
                    layers.Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse')
                
                X = X.reshape(X.shape[0], X.shape[1], 1)
                model.fit(X, y, epochs=50, batch_size=5, verbose=0)
                
                st.session_state.volatility_predictor = model
        
        # Make prediction
        if st.session_state.volatility_predictor is not None and TF_AVAILABLE:
            recent_vol = volatility_data.tail(10).values.reshape(1, 10, 1)
            predicted_vol = st.session_state.volatility_predictor.predict(recent_vol, verbose=0)[0][0]
            current_vol = volatility_data.iloc[-1]
            
            vol_change = (predicted_vol - current_vol) / current_vol
            
            return {
                'current_volatility': current_vol,
                'predicted_volatility': predicted_vol,
                'volatility_change_pct': vol_change * 100,
                'volatility_signal': 'Increasing' if vol_change > 0.1 else 'Decreasing' if vol_change < -0.1 else 'Stable'
            }
        
        return None
        
    except Exception as e:
        st.warning(f"Volatility prediction error: {e}")
        return None

def generate_ai_insights(features_dict, patterns, anomaly_info, regime_info, vol_prediction):
    """Generate human-readable AI insights"""
    insights = []
    
    try:
        # Market Regime Insights
        if regime_info:
            insights.append(f"🎯 **Market Regime**: {regime_info.get('current_regime', 'Unknown')}")
        
        # Anomaly Insights
        if anomaly_info and anomaly_info.get('is_anomaly'):
            insights.append(f"⚠️ **Anomaly Alert**: {anomaly_info.get('interpretation', 'Unusual market activity')}")
        
        # Volatility Insights
        if vol_prediction:
            vol_signal = vol_prediction.get('volatility_signal', 'Stable')
            vol_change = vol_prediction.get('volatility_change_pct', 0)
            insights.append(f"📊 **Volatility Forecast**: {vol_signal} ({vol_change:+.1f}%)")
        
        # Pattern Insights
        if patterns:
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.get('detected') or pattern_name in ['support_analysis', 'resistance_analysis']:
                    if pattern_name == 'head_shoulders':
                        insights.append(f"📉 **Head & Shoulders**: Bearish pattern detected")
                    elif pattern_name == 'support_analysis':
                        distance = pattern_data.get('distance_pct', 0)
                        insights.append(f"🛡️ **Support**: {distance:.1f}% away, {pattern_data.get('strength', 'Moderate')} level")
                    elif pattern_name == 'resistance_analysis':
                        distance = pattern_data.get('distance_pct', 0)
                        insights.append(f"🚧 **Resistance**: {distance:.1f}% away, {pattern_data.get('strength', 'Moderate')} level")
                    elif pattern_name == 'unusual_call_activity':
                        strike = pattern_data.get('strike', 0)
                        insights.append(f"🚀 **Unusual Calls**: Heavy activity at {strike} strike")
                    elif pattern_name == 'unusual_put_activity':
                        strike = pattern_data.get('strike', 0)
                        insights.append(f"🔻 **Unusual Puts**: Heavy activity at {strike} strike")
                    elif pattern_name == 'momentum_divergence':
                        insights.append(f"⚡ **Divergence**: {pattern_data.get('type', 'Unknown')} - {pattern_data.get('signal', '')}")
        
        # Technical Insights
        if 'rsi' in features_dict:
            rsi = features_dict['rsi']
            if rsi > 70:
                insights.append(f"📈 **RSI Alert**: Overbought at {rsi:.1f}")
            elif rsi < 30:
                insights.append(f"📉 **RSI Alert**: Oversold at {rsi:.1f}")
        
        # Options Flow Insights
        if 'atm_pcr' in features_dict:
            pcr = features_dict['atm_pcr']
            if pcr > 1.5:
                insights.append(f"🐻 **Heavy Put Buying**: PCR at {pcr:.2f}")
            elif pcr < 0.6:
                insights.append(f"🐂 **Heavy Call Buying**: PCR at {pcr:.2f}")
        
        return insights[:6]  # Limit to top 6 insights
        
    except Exception as e:
        st.warning(f"AI insights error: {e}")
        return ["⚠️ AI insights temporarily unavailable"]

# === ML Feature Engineering Functions ===
def calculate_technical_indicators(price_data):
    """Calculate technical indicators for ML features"""
    if len(price_data) < 20:
        return {}
    
    prices = price_data['Spot'].values
    
    # Moving Averages
    sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
    sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
    sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
    
    # RSI
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(prices)
    
    # Bollinger Bands
    bb_middle = sma_20
    bb_std = np.std(prices[-20:]) if len(prices) >= 20 else 0
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    # Volatility
    volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
    
    # Price momentum
    momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
    momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
    
    return {
        'sma_5': sma_5,
        'sma_10': sma_10,
        'sma_20': sma_20,
        'rsi': rsi,
        'bb_position': bb_position,
        'volatility': volatility,
        'momentum_5': momentum_5,
        'momentum_10': momentum_10,
        'price_above_sma5': 1 if prices[-1] > sma_5 else 0,
        'price_above_sma10': 1 if prices[-1] > sma_10 else 0,
        'price_above_sma20': 1 if prices[-1] > sma_20 else 0
    }
    def extract_options_features(df_summary, underlying):
        """Extract ML features from options data"""
        features = {}
    
        # ATM features
        atm_data = df_summary[df_summary['Zone'] == 'ATM'] if 'Zone' in df_summary.columns else pd.DataFrame()
            if not atm_data.empty:
            atm_row = atm_data.iloc[0]
            features.update({
            'atm_pcr': atm_row.get('PCR', 1),
            'atm_bias_score': atm_row.get('BiasScore', 0),
            'atm_oi_ce': atm_row.get('openInterest_CE', 0),
            'atm_oi_pe': atm_row.get('openInterest_PE', 0),
            'atm_chg_oi_ce': atm_row.get('changeinOpenInterest_CE', 0),
            'atm_chg_oi_pe': atm_row.get('changeinOpenInterest_PE', 0),
            'atm_pressure': atm_row.get('BidAskPressure', 0)
        })
    else:
        # Default values if no ATM data
        features.update({
            'atm_pcr': 1.0,
            'atm_bias_score': 0,
            'atm_oi_ce': 0,
            'atm_oi_pe': 0,
            'atm_chg_oi_ce': 0,
            'atm_chg_oi_pe': 0,
            'atm_pressure': 0
        })
    
    # Aggregate features
    features.update({
        'total_ce_oi': df_summary.get('openInterest_CE', pd.Series()).sum() if 'openInterest_CE' in df_summary.columns else 0,
        'total_pe_oi': df_summary.get('openInterest_PE', pd.Series()).sum() if 'openInterest_PE' in df_summary.columns else 0,
        'total_ce_chg_oi': df_summary.get('changeinOpenInterest_CE', pd.Series()).sum() if 'changeinOpenInterest_CE' in df_summary.columns else 0,
        'total_pe_chg_oi': df_summary.get('changeinOpenInterest_PE', pd.Series()).sum() if 'changeinOpenInterest_PE' in df_summary.columns else 0,
        'avg_pcr': df_summary.get('PCR', pd.Series()).mean() if 'PCR' in df_summary.columns else 1.0,
        'total_bias_score': df_summary.get('BiasScore', pd.Series()).sum() if 'BiasScore' in df_summary.columns else 0,
        'bullish_strikes': len(df_summary[df_summary.get('Verdict', '') == 'Bullish']) if 'Verdict' in df_summary.columns else 0,
        'bearish_strikes': len(df_summary[df_summary.get('Verdict', '') == 'Bearish']) if 'Verdict' in df_summary.columns else 0,
        'max_oi_strike_distance': 0
    })
    
    # Max OI analysis
    try:
        if 'openInterest_CE' in df_summary.columns and 'openInterest_PE' in df_summary.columns:
            df_summary_copy = df_summary.copy()
            df_summary_copy['total_oi'] = df_summary_copy['openInterest_CE'] + df_summary_copy['openInterest_PE']
            if not df_summary_copy['total_oi'].empty:
                max_oi_idx = df_summary_copy['total_oi'].idxmax()
                if not pd.isna(max_oi_idx) and 'Strike' in df_summary_copy.columns:
                    max_oi_strike = df_summary_copy.loc[max_oi_idx, 'Strike']
                    features['max_oi_strike_distance'] = abs(max_oi_strike - underlying) / underlying
    except Exception as e:
        # If any error in max OI calculation, set to 0
        features['max_oi_strike_distance'] = 0
    
    # Time-based features
    now = datetime.now(timezone("Asia/Kolkata"))
    features.update({
        'hour_of_day': now.hour,
        'minute_of_hour': now.minute,
        'is_opening_hour': 1 if 9 <= now.hour <= 10 else 0,
        'is_closing_hour': 1 if 14 <= now.hour <= 15 else 0,
        'time_to_close': max(0, (15.5 - (now.hour + now.minute/60)) / 6.5)  # Normalized time to market close
    })
    
    return features
    def safe_extract_options_features(df_summary, underlying):
    """Safe wrapper for extract_options_features with error handling"""
    try:
        return extract_options_features(df_summary, underlying)
    except Exception as e:
        st.warning(f"Error extracting options features: {e}")
        # Return default features
        now = datetime.now(timezone("Asia/Kolkata"))
        return {
            'atm_pcr': 1.0,
            'atm_bias_score': 0,
            'atm_oi_ce': 0,
            'atm_oi_pe': 0,
            'atm_chg_oi_ce': 0,
            'atm_chg_oi_pe': 0,
            'atm_pressure': 0,
            'total_ce_oi': 0,
            'total_pe_oi': 0,
            'total_ce_chg_oi': 0,
            'total_pe_chg_oi': 0,
            'avg_pcr': 1.0,
            'total_bias_score': 0,
            'bullish_strikes': 0,
            'bearish_strikes': 0,
            'max_oi_strike_distance': 0,
            'hour_of_day': now.hour,
            'minute_of_hour': now.minute,
            'is_opening_hour': 1 if 9 <= now.hour <= 10 else 0,
            'is_closing_hour': 1 if 14 <= now.hour <= 15 else 0,
            'time_to_close': max(0, (15.5 - (now.hour + now.minute/60)) / 6.5)
        }

def create_target_variable(price_history, future_minutes=30):
    """Create target variable for ML training"""
    if len(price_history) < future_minutes + 1:
        return 0
    
    current_price = price_history.iloc[-future_minutes-1]['Spot']
    future_price = price_history.iloc[-1]['Spot']
    
    price_change_pct = (future_price - current_price) / current_price * 100
    
    # Classification: 0=Bearish, 1=Neutral, 2=Bullish
    if price_change_pct > 0.3:
        return 2  # Bullish
    elif price_change_pct < -0.3:
        return 0  # Bearish
    else:
        return 1  # Neutral

def prepare_ml_dataset():
    """Prepare dataset for ML training from historical data"""
    if 'ml_features_history' not in st.session_state or st.session_state.ml_features_history.empty:
        return None, None
    
    df = st.session_state.ml_features_history.copy()
    
    if len(df) < 50:
        return None, None
    
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'spot_price']]
    X = df[feature_columns].fillna(0)
    y = df['target'].fillna(1)
    
    return X, y

def train_ml_models():
    """Train multiple ML models for market direction prediction"""
    X, y = prepare_ml_dataset()
    
    if X is None or len(X) < 50:
        return None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        model_performance = {}
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        models['RandomForest'] = rf_model
        model_performance['RandomForest'] = accuracy_score(y_test, rf_pred)
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        models['GradientBoosting'] = gb_model
        model_performance['GradientBoosting'] = accuracy_score(y_test, gb_pred)
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=6)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        models['XGBoost'] = xgb_model
        model_performance['XGBoost'] = accuracy_score(y_test, xgb_pred)
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        models['LogisticRegression'] = lr_model
        model_performance['LogisticRegression'] = accuracy_score(y_test, lr_pred)
        
        st.session_state.ml_models = models
        st.session_state.feature_scaler = scaler
        
        return model_performance
        
    except Exception as e:
        st.error(f"Error training ML models: {e}")
        return None

def get_ml_predictions(features_dict):
    """Get predictions from trained ML models"""
    if not st.session_state.ml_models or not features_dict:
        return {}, {}
    
    try:
        features_df = pd.DataFrame([features_dict])
        
        if hasattr(st.session_state.feature_scaler, 'feature_names_in_'):
            feature_cols = st.session_state.feature_scaler.feature_names_in_
            missing_cols = set(feature_cols) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
            features_df = features_df[feature_cols]
        
        features_scaled = st.session_state.feature_scaler.transform(features_df)
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in st.session_state.ml_models.items():
            try:
                pred = model.predict(features_scaled)[0]
                pred_proba = model.predict_proba(features_scaled)[0]
                
                predictions[model_name] = pred
                probabilities[model_name] = {
                    'Bearish': pred_proba[0],
                    'Neutral': pred_proba[1],
                    'Bullish': pred_proba[2]
                }
            except Exception as e:
                st.warning(f"Error with {model_name}: {e}")
                continue
        
        return predictions, probabilities
        
    except Exception as e:
        st.error(f"Error getting ML predictions: {e}")
        return {}, {}

def ensemble_prediction(predictions, probabilities):
    """Create ensemble prediction from multiple models"""
    if not predictions:
        return 1, "Neutral", {}
    
    pred_counts = {0: 0, 1: 0, 2: 0}
    for pred in predictions.values():
        pred_counts[pred] += 1
    
    ensemble_pred = max(pred_counts, key=pred_counts.get)
    
    ensemble_proba = {'Bearish': 0, 'Neutral': 0, 'Bullish': 0}
    if probabilities:
        for model_proba in probabilities.values():
            for signal, prob in model_proba.items():
                ensemble_proba[signal] += prob
        
        num_models = len(probabilities)
        for signal in ensemble_proba:
            ensemble_proba[signal] /= num_models
    
    signal_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    ensemble_signal = signal_map[ensemble_pred]
    
    return ensemble_pred, ensemble_signal, ensemble_proba

# === Dhan API Functions ===
def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
    """Get option chain data from Dhan API"""
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
    """Get expiry list from Dhan API"""
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
    """Store price data in Supabase"""
    if not supabase_client:
        return
        
    try:
        data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "price": price,
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        supabase_client.table("nifty_price_history").insert(data).execute()
    except Exception as e:
        st.error(f"Error storing price data: {e}")

def store_ml_features(features_dict, spot_price, target=None):
    """Store ML features for future training"""
    if not supabase_client:
        new_row = features_dict.copy()
        new_row.update({
            'timestamp': datetime.now(timezone("Asia/Kolkata")).isoformat(),
            'spot_price': spot_price,
            'target': target
        })
        
        if st.session_state.ml_features_history.empty:
            st.session_state.ml_features_history = pd.DataFrame([new_row])
        else:
            st.session_state.ml_features_history = pd.concat([
                st.session_state.ml_features_history, 
                pd.DataFrame([new_row])
            ], ignore_index=True)
        
        if len(st.session_state.ml_features_history) > 1000:
            st.session_state.ml_features_history = st.session_state.ml_features_history.tail(1000)
        return
        
    try:
        data = features_dict.copy()
        data.update({
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "spot_price": spot_price,
            "target": target,
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        })
        supabase_client.table("nifty_ml_features").insert(data).execute()
    except Exception as e:
        st.error(f"Error storing ML features: {e}")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception:
        pass

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
    """Calculate bid/ask pressure"""
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    
    return pressure, bias

def determine_level(row):
    ce_oi = row.get('openInterest_CE', 0)
    pe_oi = row.get('openInterest_PE', 0)

    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def color_pressure(val):
    if val > 500:
        return 'background-color: #90EE90; color: black'
    elif val < -500:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_pcr(val):
    if val > st.session_state.pcr_threshold_bull:
        return 'background-color: #90EE90; color: black'
    elif val < st.session_state.pcr_threshold_bear:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'
      # === Main Enhanced Analysis Function ===
def analyze():
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("00:00", "%H:%M").time()
        market_end = datetime.strptime("18:40", "%H:%M").time()

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
        
        expiry = expiry_dates[0]
        
        # Get option chain from Dhan API
        option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
        if not option_chain_data or 'data' not in option_chain_data:
            st.error("Failed to get option chain from Dhan API")
            return
        
        data = option_chain_data['data']
        underlying = data['last_price']
        
        # Store price data
        store_price_data(underlying)
        
        # Update price history in session state
        new_price_row = pd.DataFrame({
            "Time": [now.strftime("%H:%M:%S")],
            "Spot": [underlying]
        })
        
        if st.session_state.price_data.empty:
            st.session_state.price_data = new_price_row
        else:
            st.session_state.price_data = pd.concat([st.session_state.price_data, new_price_row], ignore_index=True)
        
        # Keep only last 100 price points
        if len(st.session_state.price_data) > 100:
            st.session_state.price_data = st.session_state.price_data.tail(100)

        # Process option chain data
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
        
        # Merge call and put data
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        
        # Rename columns to match standard format
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

        # Calculate Greeks for calls and puts
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

        # Analysis logic
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # Open Interest Change Comparison
        total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000
        total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000

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
        
        # PCR CALCULATION
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )

        # Calculate PCR
        df_summary['PCR'] = np.where(
            df_summary['openInterest_CE'] == 0,
            0,
            df_summary['openInterest_PE'] / df_summary['openInterest_CE']
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

        # ===== ADVANCED AI SECTION =====
        
        # Calculate technical indicators
        tech_indicators = calculate_technical_indicators(st.session_state.price_data)
        
        # Extract options features
        options_features = extract_options_features(df_summary, underlying)
        
        # Combine all features
        all_features = {**tech_indicators, **options_features}
        
        # Store features for future training
        target = None
        if len(st.session_state.price_data) >= 30:
            target = create_target_variable(st.session_state.price_data)
        
        store_ml_features(all_features, underlying, target)
        
        # Advanced AI Analysis
        anomaly_info = detect_market_anomalies(st.session_state.ml_features_history)
        regime_info = cluster_market_regimes(st.session_state.ml_features_history)
        patterns = advanced_pattern_recognition(st.session_state.price_data, df_summary)
        vol_prediction = predict_volatility(st.session_state.ml_features_history)
        
        # Generate AI insights
        ai_insights = generate_ai_insights(all_features, patterns, anomaly_info, regime_info, vol_prediction)
        st.session_state.ai_insights = ai_insights
        
        # Get ML predictions
        ml_predictions = {}
        ml_probabilities = {}
        ensemble_pred = 1
        ensemble_signal = "Neutral"
        ensemble_proba = {}
        
        if st.session_state.ml_models:
            ml_predictions, ml_probabilities = get_ml_predictions(all_features)
            if ml_predictions:
                ensemble_pred, ensemble_signal, ensemble_proba = ensemble_prediction(ml_predictions, ml_probabilities)

        # Style the dataframe
        styled_df = df_summary.style.applymap(color_pcr, subset=['PCR']).applymap(color_pressure, subset=['BidAskPressure'])
        df_summary_display = df_summary.drop(columns=['strikePrice'], errors='ignore')

        # Calculate market view
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        traditional_view = atm_row['Verdict'] if atm_row is not None else "Neutral"

        # Continue to Part 4B for UI Display...
            # ===== DISPLAY SECTION =====
        
        st.markdown("# 🧠 Advanced AI Nifty Options Analyzer")
        
        # Enhanced AI Status Bar
        ai_status_col1, ai_status_col2, ai_status_col3, ai_status_col4 = st.columns(4)
        
        with ai_status_col1:
            regime_text = regime_info.get('current_regime', '🔄 Analyzing...') if regime_info else '🔄 Learning...'
            st.metric("🎯 Market Regime", regime_text)
        
        with ai_status_col2:
            anomaly_status = "🚨 Anomaly!" if anomaly_info and anomaly_info.get('is_anomaly') else "✅ Normal"
            st.metric("🔍 Market State", anomaly_status)
        
        with ai_status_col3:
            vol_trend = vol_prediction.get('volatility_signal', 'Stable') if vol_prediction else 'Analyzing...'
            st.metric("📊 Volatility Trend", vol_trend)
        
        with ai_status_col4:
            pattern_count = len([p for p in patterns.values() if p.get('detected', False)])
            st.metric("🔍 Patterns Found", f"{pattern_count} detected")
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nifty Spot", f"{underlying:,.2f}")
        
        with col2:
            st.metric("Traditional View", traditional_view, f"Score: {total_score}")
            
        with col3:
            confidence_color = "🟢" if max(ensemble_proba.values()) > 0.6 else "🟡" if max(ensemble_proba.values()) > 0.4 else "🔴"
            confidence_text = f"{max(ensemble_proba.values()):.1%}" if ensemble_proba else "Training..."
            st.metric("🤖 AI Prediction", f"{confidence_color} {ensemble_signal}", f"Confidence: {confidence_text}")
        
        with col4:
            signal_agreement = "✅ Aligned" if traditional_view.replace("Strong ", "") == ensemble_signal else "⚠️ Divergent"
            st.metric("Signal Agreement", signal_agreement)

        # AI Insights Panel
        st.markdown("---")
        st.markdown("## 🧠 AI Market Intelligence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💡 Real-time AI Insights")
            if ai_insights:
                for insight in ai_insights:
                    st.info(insight)
            else:
                st.info("🔄 AI is analyzing market patterns...")
        
        with col2:
            st.markdown("### 🎯 Pattern Detection")
            if patterns:
                for pattern_name, pattern_data in patterns.items():
                    if pattern_data.get('detected') or 'analysis' in pattern_name:
                        if pattern_name == 'head_shoulders':
                            st.warning("📉 Head & Shoulders Pattern")
                        elif pattern_name == 'support_analysis':
                            distance = pattern_data.get('distance_pct', 0)
                            st.success(f"🛡️ Support: {distance:.1f}% away")
                        elif pattern_name == 'resistance_analysis':
                            distance = pattern_data.get('distance_pct', 0)
                            st.error(f"🚧 Resistance: {distance:.1f}% away")
                        elif 'unusual' in pattern_name:
                            st.warning(f"⚡ Unusual Activity Detected")
            else:
                st.info("🔍 Scanning for patterns...")

        # ML Dashboard
        st.markdown("---")
        st.markdown("## 🧠 Machine Learning Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if ml_probabilities:
                st.markdown("### Model Predictions")
                
                # Create prediction summary
                pred_df = pd.DataFrame({
                    'Model': list(ml_probabilities.keys()),
                    'Prediction': [list(ml_predictions.values())[i] for i in range(len(ml_predictions))],
                    'Signal': [['Bearish', 'Neutral', 'Bullish'][pred] for pred in ml_predictions.values()],
                    'Bearish_Prob': [proba['Bearish'] for proba in ml_probabilities.values()],
                    'Neutral_Prob': [proba['Neutral'] for proba in ml_probabilities.values()],
                    'Bullish_Prob': [proba['Bullish'] for proba in ml_probabilities.values()]
                })
                
                # Add ensemble prediction
                ensemble_row = pd.DataFrame({
                    'Model': ['🎯 ENSEMBLE'],
                    'Prediction': [ensemble_pred],
                    'Signal': [ensemble_signal],
                    'Bearish_Prob': [ensemble_proba.get('Bearish', 0)],
                    'Neutral_Prob': [ensemble_proba.get('Neutral', 0)],
                    'Bullish_Prob': [ensemble_proba.get('Bullish', 0)]
                })
                
                pred_df = pd.concat([pred_df, ensemble_row], ignore_index=True)
                
                # Style prediction dataframe
                def color_prediction(val):
                    if val == 'Bullish':
                        return 'background-color: #90EE90; color: black'
                    elif val == 'Bearish':
                        return 'background-color: #FFB6C1; color: black'
                    else:
                        return 'background-color: #FFFFE0; color: black'
                
                styled_pred_df = pred_df.style.applymap(color_prediction, subset=['Signal'])
                st.dataframe(styled_pred_df, use_container_width=True)
            else:
                st.info("🔄 Training ML models... Need more data points for predictions.")
        
        with col2:
            st.markdown("### Feature Importance")
            if all_features:
                # Show top 5 most important features
                feature_importance = {
                    'RSI': tech_indicators.get('rsi', 50),
                    'ATM PCR': options_features.get('atm_pcr', 1),
                    'Volatility': tech_indicators.get('volatility', 0) * 100,
                    'Momentum': tech_indicators.get('momentum_5', 0) * 100,
                    'Bias Score': options_features.get('total_bias_score', 0)
                }
                
                for feature, value in feature_importance.items():
                    if feature == 'RSI':
                        color = "🟢" if 30 <= value <= 70 else "🔴"
                    elif feature == 'ATM PCR':
                        color = "🟢" if value > 1.2 else "🔴" if value < 0.7 else "🟡"
                    else:
                        color = "🟡"
                    
                    st.metric(f"{color} {feature}", f"{value:.2f}")

        # Open Interest Change Display
        st.markdown("---")
        st.markdown("## 📊 Open Interest Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CALL ΔOI", 
                     f"{total_ce_change:+.1f}L",
                     delta_color="inverse")
            
        with col2:
            st.metric("PUT ΔOI", 
                     f"{total_pe_change:+.1f}L",
                     delta_color="normal")

        # Options Chain Summary
        confidence_text = f"{max(ensemble_proba.values()):.1%}" if ensemble_proba else 'Training...'
        pcr_filter_status = 'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'

        st.info(f"""
        **Traditional Analysis:**
        - ATM Strike: {atm_strike}
        - Market View: {traditional_view}
        - Total Bias Score: {total_score}

        **AI Enhancement:**
        - ML Prediction: {ensemble_signal}
        - Confidence: {confidence_text}
        - Features Used: {len(all_features)} indicators

        **PCR Thresholds:**
        - Bullish: >{st.session_state.pcr_threshold_bull}
        - Bearish: <{st.session_state.pcr_threshold_bear}
        - Filter: {pcr_filter_status}
        """)
            
        st.dataframe(styled_df, use_container_width=True)

        # Technical Indicators Display
        if tech_indicators:
            st.markdown("---")
            st.markdown("## 📈 Technical Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi = tech_indicators.get('rsi', 50)
                rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_signal)
            
            with col2:
                volatility = tech_indicators.get('volatility', 0) * 100
                st.metric("Volatility", f"{volatility:.2f}%")
            
            with col3:
                momentum = tech_indicators.get('momentum_5', 0) * 100
                st.metric("5-Min Momentum", f"{momentum:+.2f}%")
            
            with col4:
                bb_pos = tech_indicators.get('bb_position', 0.5) * 100
                bb_signal = "Upper Band" if bb_pos > 80 else "Lower Band" if bb_pos < 20 else "Middle"
                st.metric("Bollinger Position", f"{bb_pos:.1f}%", bb_signal)

        # Continue to Part 4C for Advanced Models and Configuration...
            # Advanced AI Models Section
        st.markdown("---")
        st.markdown("## 🚀 Advanced AI Models")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🧠 Neural Networks", "🔍 Anomaly Detection", "🎯 Pattern Recognition", "📊 Volatility AI"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Train Deep Neural Network", type="primary"):
                    if len(st.session_state.ml_features_history) >= 100:
                        with st.spinner("Training deep neural network..."):
                            try:
                                X, y = prepare_ml_dataset()
                                if X is not None and len(X) >= 100 and TF_AVAILABLE:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                    
                                    scaler = StandardScaler()
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_test_scaled = scaler.transform(X_test)
                                    
                                    nn_model = create_neural_network(X_train_scaled.shape[1])
                                    if nn_model:
                                        history = nn_model.fit(
                                            X_train_scaled, y_train,
                                            epochs=100,
                                            batch_size=32,
                                            validation_split=0.2,
                                            verbose=0
                                        )
                                        
                                        test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
                                        
                                        st.session_state.neural_network = {
                                            'model': nn_model,
                                            'scaler': scaler,
                                            'accuracy': test_acc,
                                            'history': history.history
                                        }
                                        
                                        st.success(f"✅ Neural Network trained! Accuracy: {test_acc:.3f}")
                                        send_telegram_message(f"🧠 Nifty AI: Neural Network trained with {test_acc:.3f} accuracy")
                                    else:
                                        st.error("TensorFlow not available for neural network")
                                else:
                                    st.warning("Need more training data or TensorFlow not available")
                            except Exception as e:
                                st.error(f"Neural network training failed: {e}")
                    else:
                        st.info("Need 100+ data points for neural network training")
                
                if st.session_state.neural_network:
                    nn_acc = st.session_state.neural_network['accuracy']
                    st.metric("🧠 Neural Network Status", f"✅ Active", f"Accuracy: {nn_acc:.1%}")
                    
                    # Neural network prediction
                    try:
                        nn_data = st.session_state.neural_network
                        features_df = pd.DataFrame([all_features])
                        features_scaled = nn_data['scaler'].transform(features_df.fillna(0))
                        nn_pred_proba = nn_data['model'].predict(features_scaled, verbose=0)[0]
                        nn_prediction = np.argmax(nn_pred_proba)
                        
                        signal_map = {0: "🔴 Bearish", 1: "🟡 Neutral", 2: "🟢 Bullish"}
                        confidence = nn_pred_proba[nn_prediction]
                        
                        st.metric("🎯 Neural Prediction", signal_map[nn_prediction], f"{confidence:.1%} confidence")
                    except Exception as e:
                        st.error(f"Neural prediction error: {e}")
            
            with col2:
                if st.button("🧠 Train LSTM Price Model"):
                    if len(st.session_state.price_data) >= 60 and TF_AVAILABLE:
                        with st.spinner("Training LSTM for price prediction..."):
                            try:
                                prices = st.session_state.price_data['Spot'].values
                                
                                def create_sequences(data, seq_length=10):
                                    X, y = [], []
                                    for i in range(seq_length, len(data)):
                                        X.append(data[i-seq_length:i])
                                        y.append(data[i])
                                    return np.array(X), np.array(y)
                                
                                X, y = create_sequences(prices)
                                if len(X) >= 20:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                    
                                    # Reshape for LSTM
                                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                                    
                                    # Scale data
                                    scaler = MinMaxScaler()
                                    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
                                    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
                                    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                                    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
                                    
                                    # Create LSTM model
                                    lstm_model = create_lstm_model(X_train.shape[1], 1)
                                    if lstm_model:
                                        lstm_model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=5, verbose=0)
                                        
                                        # Evaluate
                                        lstm_loss = lstm_model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
                                        
                                        st.session_state.price_lstm = {
                                            'model': lstm_model,
                                            'scaler': scaler,
                                            'loss': lstm_loss
                                        }
                                        
                                        st.success(f"✅ LSTM trained! Loss: {lstm_loss:.4f}")
                                        
                                        # Make next price prediction
                                        recent_prices = prices[-10:].reshape(1, 10, 1)
                                        recent_scaled = scaler.transform(recent_prices.reshape(-1, 1)).reshape(recent_prices.shape)
                                        next_price_scaled = lstm_model.predict(recent_scaled, verbose=0)
                                        next_price = scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0]
                                        
                                        price_change = (next_price - underlying) / underlying * 100
                                        st.metric("📈 Next Price Forecast", f"{next_price:.2f}", f"{price_change:+.2f}%")
                                    else:
                                        st.error("TensorFlow not available for LSTM")
                                        
                            except Exception as e:
                                st.error(f"LSTM training failed: {e}")
                    else:
                        st.info("Need 60+ price points and TensorFlow for LSTM training")
        
        with tab2:
            st.markdown("### 🕵️ Market Anomaly Detection")
            
            if anomaly_info:
                col1, col2 = st.columns(2)
                with col1:
                    anomaly_status = "🚨 ANOMALY DETECTED!" if anomaly_info.get('is_anomaly') else "✅ Normal Market"
                    st.metric("Market State", anomaly_status)
                
                with col2:
                    anomaly_score = anomaly_info.get('anomaly_score', 0)
                    st.metric("Anomaly Score", f"{anomaly_score:.3f}", "Lower = More Unusual")
                
                if anomaly_info.get('is_anomaly'):
                    st.warning(f"⚠️ {anomaly_info.get('interpretation', 'Unusual market conditions detected')}")
                    st.info("🔍 **Possible Causes**: Major news, earnings, policy changes, or technical breakouts")
            else:
                st.info("🔄 Anomaly detection initializing... Need more data points")
            
            if st.button("🔄 Retrain Anomaly Detector"):
                st.session_state.anomaly_detector = None
                st.success("Anomaly detector reset for retraining")
        
        with tab3:
            st.markdown("### 🎯 Advanced Pattern Recognition")
            
            if patterns:
                # Support/Resistance Analysis
                if 'support_analysis' in patterns:
                    support_info = patterns['support_analysis']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🛡️ Nearest Support", f"{support_info['nearest_support']:.2f}")
                    with col2:
                        st.metric("📏 Distance", f"{support_info['distance_pct']:.2f}%", f"{support_info['strength']} Level")
                
                if 'resistance_analysis' in patterns:
                    resistance_info = patterns['resistance_analysis']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🚧 Nearest Resistance", f"{resistance_info['nearest_resistance']:.2f}")
                    with col2:
                        st.metric("📏 Distance", f"{resistance_info['distance_pct']:.2f}%", f"{resistance_info['strength']} Level")
                
                # Chart Patterns
                if 'head_shoulders' in patterns and patterns['head_shoulders'].get('detected'):
                    st.error("📉 **Head & Shoulders Pattern Detected** - Bearish Reversal Signal")
                
                # Options Flow Patterns
                unusual_activities = []
                if 'unusual_call_activity' in patterns:
                    call_info = patterns['unusual_call_activity']
                    unusual_activities.append(f"🚀 Heavy Call Activity at {call_info['strike']} strike")
                
                if 'unusual_put_activity' in patterns:
                    put_info = patterns['unusual_put_activity']
                    unusual_activities.append(f"🔻 Heavy Put Activity at {put_info['strike']} strike")
                
                if unusual_activities:
                    st.markdown("**🔥 Unusual Options Activity:**")
                    for activity in unusual_activities:
                        st.warning(activity)
                
                # Momentum Divergence
                if 'momentum_divergence' in patterns:
                    div_info = patterns['momentum_divergence']
                    st.info(f"⚡ **{div_info['type']}**: {div_info['description']} → {div_info['signal']}")
            
            else:
                st.info("🔍 No significant patterns detected in current market conditions")
        
        with tab4:
            st.markdown("### 📊 AI Volatility Prediction")
            
            if vol_prediction:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_vol = vol_prediction['current_volatility'] * 100
                    st.metric("📊 Current Volatility", f"{current_vol:.2f}%")
                
                with col2:
                    predicted_vol = vol_prediction['predicted_volatility'] * 100
                    st.metric("🔮 Predicted Volatility", f"{predicted_vol:.2f}%")
                
                with col3:
                    vol_change = vol_prediction['volatility_change_pct']
                    vol_signal = vol_prediction['volatility_signal']
                    st.metric("📈 Volatility Trend", vol_signal, f"{vol_change:+.1f}%")
                
                # Volatility interpretation
                if vol_signal == "Increasing":
                    st.warning("⚠️ **Rising Volatility Expected** - Consider protective strategies")
                elif vol_signal == "Decreasing":
                    st.success("✅ **Calming Markets Expected** - Trend strategies may work better")
                else:
                    st.info("📊 **Stable Volatility Expected** - Current strategies should continue")
            
            else:
                st.info("🔄 Volatility AI training... Need more historical data")
            
            if st.button("🎯 Train Volatility Predictor"):
                st.session_state.volatility_predictor = None
                st.success("Volatility predictor reset for retraining")

        # Model Training Section
        st.markdown("### 🎯 Model Training & Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_points = len(st.session_state.ml_features_history) if not st.session_state.ml_features_history.empty else 0
            st.metric("Training Data Points", data_points, "Need 50+ for training")
        
        with col2:
            if st.button("🚀 Train ML Models", type="primary"):
                with st.spinner("Training ML models..."):
                    performance = train_ml_models()
                    if performance:
                        st.success("Models trained successfully!")
                        for model, acc in performance.items():
                            st.text(f"{model}: {acc:.3f} accuracy")
                        send_telegram_message(f"Nifty ML: Models retrained. Best accuracy: {max(performance.values()):.3f}")
                    else:
                        st.warning("Need more training data or training failed.")
        
        with col3:
            if st.session_state.ml_models:
                st.success(f"✅ {len(st.session_state.ml_models)} models loaded")
            else:
                st.warning("⚠️ No trained models")

        # Enhanced Features
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")
        
        # PCR Configuration
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

        # Data Management Section
        st.markdown("### 🗄️ Data Management")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Export ML Data", use_container_width=True):
                if not st.session_state.ml_features_history.empty:
                    csv = st.session_state.ml_features_history.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"nifty_ml_features_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No ML data available for export")
        
        with col2:
            if st.button("🧹 Clear ML Data", use_container_width=True):
                st.session_state.ml_features_history = pd.DataFrame()
                st.session_state.ml_models = {}
                st.success("ML data cleared!")
        
        with col3:
            if st.button("🗑️ Delete All History", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_delete', False):
                    # Clear all data
                    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])
                    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])
                    st.session_state.trade_log = []
                    st.session_state.call_log_book = []
                    st.session_state.ml_features_history = pd.DataFrame()
                    st.session_state.ml_models = {}
                    st.session_state.neural_network = None
                    st.session_state.anomaly_detector = None
                    st.session_state.price_lstm = None
                    st.session_state.clustering_model = None
                    st.session_state.volatility_predictor = None
                    st.session_state.confirm_delete = False
                    st.success("All history deleted successfully!")
                    send_telegram_message("Nifty AI: All historical data deleted")
                    st.rerun()
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Click again to confirm deletion")

        # Continue to Final Part for Performance Dashboard and Error Handling...
# AI Performance Metrics
        st.markdown("---")
        st.markdown("## 📊 AI Performance Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ai_models_count = sum([
                1 if st.session_state.ml_models else 0,
                1 if st.session_state.neural_network else 0,
                1 if st.session_state.anomaly_detector else 0,
                1 if st.session_state.price_lstm else 0,
                1 if st.session_state.clustering_model else 0,
                1 if st.session_state.volatility_predictor else 0
            ])
            st.metric("🤖 Active AI Models", f"{ai_models_count}/6", "Fully Operational" if ai_models_count >= 4 else "Training...")
        
        with col2:
            insights_count = len(st.session_state.ai_insights)
            st.metric("💡 AI Insights", f"{insights_count} active", "Real-time Analysis")
        
        with col3:
            if ensemble_proba:
                max_confidence = max(ensemble_proba.values())
                confidence_level = "High" if max_confidence > 0.7 else "Medium" if max_confidence > 0.5 else "Low"
                st.metric("🎯 Prediction Confidence", f"{max_confidence:.1%}", confidence_level)
            else:
                st.metric("🎯 Prediction Confidence", "Training...", "Collecting Data")
        
        with col4:
            pattern_strength = len([p for p in patterns.values() if p.get('detected', False)])
            pattern_level = "Strong" if pattern_strength >= 3 else "Moderate" if pattern_strength >= 1 else "Weak"
            st.metric("🔍 Pattern Strength", f"{pattern_strength} patterns", pattern_level)

        # Advanced Analytics Section
        st.markdown("---")
        st.markdown("## 📈 Advanced Analytics")
        
        # Create tabs for different analytics views
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["📊 Model Performance", "🎯 Feature Analysis", "📈 Historical Trends"])
        
        with analytics_tab1:
            st.markdown("### 🏆 Model Performance Metrics")
            
            if st.session_state.ml_models:
                # Model accuracy comparison
                model_names = list(st.session_state.ml_models.keys())
                st.info(f"**Active Models**: {', '.join(model_names)}")
                
                # Performance summary
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Prediction Statistics:**")
                    if ensemble_proba:
                        for signal, prob in ensemble_proba.items():
                            confidence_bar = "█" * int(prob * 10) + "░" * (10 - int(prob * 10))
                            st.text(f"{signal}: {confidence_bar} {prob:.1%}")
                
                with col2:
                    st.markdown("**📊 Model Health:**")
                    health_score = ai_models_count / 6 * 100
                    health_status = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Fair"
                    st.metric("System Health", f"{health_score:.0f}%", health_status)
            else:
                st.info("No trained models available. Train models to see performance metrics.")
        
        with analytics_tab2:
            st.markdown("### 🔍 Feature Analysis")
            
            if all_features:
                # Feature correlation analysis
                feature_df = pd.DataFrame([all_features])
                numeric_features = feature_df.select_dtypes(include=[np.number])
                
                if len(numeric_features.columns) > 1:
                    # Calculate feature correlations
                    correlations = numeric_features.corr()
                    
                    # Show top correlated features
                    st.markdown("**📊 Feature Correlations:**")
                    
                    # Find highly correlated pairs
                    high_corr_pairs = []
                    for i in range(len(correlations.columns)):
                        for j in range(i+1, len(correlations.columns)):
                            corr_val = correlations.iloc[i, j]
                            if abs(corr_val) > 0.7:  # High correlation threshold
                                high_corr_pairs.append((
                                    correlations.columns[i], 
                                    correlations.columns[j], 
                                    corr_val
                                ))
                    
                    if high_corr_pairs:
                        for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
                            correlation_strength = "Strong" if abs(corr) > 0.8 else "Moderate"
                            st.text(f"• {feat1} ↔ {feat2}: {corr:.3f} ({correlation_strength})")
                    else:
                        st.info("No highly correlated features detected")
                
                # Feature importance estimation
                st.markdown("**⭐ Key Features Impact:**")
                important_features = {
                    'Technical': ['rsi', 'volatility', 'momentum_5'],
                    'Options': ['atm_pcr', 'total_bias_score', 'atm_pressure'],
                    'Market': ['hour_of_day', 'time_to_close']
                }
                
                for category, features in important_features.items():
                    available_features = [f for f in features if f in all_features]
                    if available_features:
                        st.text(f"📈 {category}: {len(available_features)}/{len(features)} active")
            else:
                st.info("No feature data available for analysis")
        
        with analytics_tab3:
            st.markdown("### 📈 Historical Trends")
            
            if not st.session_state.ml_features_history.empty:
                # Show data collection progress
                total_points = len(st.session_state.ml_features_history)
                st.metric("📊 Data Points Collected", total_points, "Historical Records")
                
                # Show recent trends
                if total_points >= 10:
                    recent_data = st.session_state.ml_features_history.tail(10)
                    
                    # Price trend
                    if 'spot_price' in recent_data.columns:
                        price_trend = recent_data['spot_price'].iloc[-1] - recent_data['spot_price'].iloc[0]
                        trend_direction = "📈 Upward" if price_trend > 0 else "📉 Downward" if price_trend < 0 else "➡️ Sideways"
                        st.metric("Recent Price Trend", trend_direction, f"{price_trend:+.2f} points")
                    
                    # Volatility trend
                    if 'volatility' in recent_data.columns:
                        vol_trend = recent_data['volatility'].mean()
                        vol_level = "High" if vol_trend > 0.02 else "Low" if vol_trend < 0.01 else "Normal"
                        st.metric("Average Volatility", f"{vol_trend*100:.2f}%", vol_level)
            else:
                st.info("Collecting historical data... Check back after a few refresh cycles")

        # System Status and Alerts
        st.markdown("---")
        st.markdown("## 🚨 System Status & Alerts")
        
        # Create alert panels
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.markdown("### ⚠️ Active Alerts")
            
            alerts = []
            
            # High confidence predictions
            if ensemble_proba and max(ensemble_proba.values()) > 0.8:
                alerts.append(f"🎯 High confidence {ensemble_signal} prediction ({max(ensemble_proba.values()):.1%})")
            
            # Anomaly alerts
            if anomaly_info and anomaly_info.get('is_anomaly'):
                alerts.append("🚨 Market anomaly detected - Unusual conditions")
            
            # Pattern alerts
            if patterns:
                pattern_alerts = []
                for pattern_name, pattern_data in patterns.items():
                    if pattern_data.get('detected'):
                        if pattern_name == 'head_shoulders':
                            pattern_alerts.append("📉 Head & Shoulders pattern - Bearish signal")
                        elif 'unusual' in pattern_name:
                            pattern_alerts.append("⚡ Unusual options activity detected")
                
                alerts.extend(pattern_alerts)
            
            # Volatility alerts
            if vol_prediction and vol_prediction.get('volatility_signal') != 'Stable':
                vol_change = vol_prediction.get('volatility_change_pct', 0)
                alerts.append(f"📊 Volatility {vol_prediction['volatility_signal'].lower()} ({vol_change:+.1f}%)")
            
            if alerts:
                for alert in alerts[:5]:  # Show max 5 alerts
                    st.warning(alert)
            else:
                st.success("✅ No active alerts - Market conditions normal")
        
        with alert_col2:
            st.markdown("### 📊 System Health")
            
            # System health checks
            health_checks = []
            
            # Data availability
            if not st.session_state.ml_features_history.empty:
                data_health = len(st.session_state.ml_features_history)
                if data_health >= 100:
                    health_checks.append(("📊 Data Collection", "✅ Excellent", f"{data_health} points"))
                elif data_health >= 50:
                    health_checks.append(("📊 Data Collection", "🟡 Good", f"{data_health} points"))
                else:
                    health_checks.append(("📊 Data Collection", "🔴 Limited", f"{data_health} points"))
            else:
                health_checks.append(("📊 Data Collection", "🔴 Starting", "0 points"))
            
            # Model status
            if st.session_state.ml_models:
                model_count = len(st.session_state.ml_models)
                health_checks.append(("🤖 ML Models", "✅ Active", f"{model_count} models"))
            else:
                health_checks.append(("🤖 ML Models", "🟡 Training", "Need data"))
            
            # AI features
            ai_features_active = sum([
                1 if st.session_state.neural_network else 0,
                1 if st.session_state.anomaly_detector else 0,
                1 if st.session_state.clustering_model else 0
            ])
            
            if ai_features_active >= 2:
                health_checks.append(("🧠 AI Features", "✅ Active", f"{ai_features_active}/3"))
            elif ai_features_active >= 1:
                health_checks.append(("🧠 AI Features", "🟡 Partial", f"{ai_features_active}/3"))
            else:
                health_checks.append(("🧠 AI Features", "🔴 Inactive", "0/3"))
            
            # TensorFlow availability
            if TF_AVAILABLE:
                health_checks.append(("🔧 TensorFlow", "✅ Available", "Deep learning ready"))
            else:
                health_checks.append(("🔧 TensorFlow", "🟡 Missing", "Limited AI features"))
            
            # Display health status
            for component, status, detail in health_checks:
                st.text(f"{component}: {status} ({detail})")

        # Telegram Integration Status
        st.markdown("---")
        st.markdown("## 📱 Telegram Integration")
        
        telegram_col1, telegram_col2 = st.columns(2)
        
        with telegram_col1:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                st.success("✅ Telegram alerts configured")
                
                # Test telegram
                if st.button("📤 Send Test Alert"):
                    test_message = f"🧠 Nifty AI Test: System operational at {datetime.now().strftime('%H:%M:%S')}"
                    send_telegram_message(test_message)
                    st.info("Test message sent!")
            else:
                st.warning("⚠️ Telegram not configured")
                st.info("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable alerts")
        
        with telegram_col2:
            st.markdown("**📋 Alert Triggers:**")
            st.text("• High confidence predictions (>70%)")
            st.text("• Market anomalies detected")
            st.text("• Model training completion")
            st.text("• System errors and warnings")

        # Send ML prediction to Telegram if significant
        if ensemble_signal != "Neutral" and ensemble_proba and max(ensemble_proba.values()) > 0.7:
            confidence = max(ensemble_proba.values())
            message = f"🤖 Nifty AI Alert: {ensemble_signal} signal with {confidence:.1%} confidence at {underlying:,.2f}"
            send_telegram_message(message)
        
        # Send anomaly alerts
        if anomaly_info and anomaly_info.get('is_anomaly'):
            anomaly_message = f"🚨 Nifty Anomaly Alert: Unusual market conditions detected at {underlying:,.2f}"
            send_telegram_message(anomaly_message)

    except Exception as e:
        st.error(f"Error in analysis: {e}")
        send_telegram_message(f"Nifty AI Error: {str(e)}")
        
        # Error recovery
        st.markdown("---")
        st.markdown("## 🔧 Error Recovery")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset AI Models"):
                st.session_state.ml_models = {}
                st.session_state.neural_network = None
                st.session_state.anomaly_detector = None
                st.session_state.price_lstm = None
                st.session_state.clustering_model = None
                st.session_state.volatility_predictor = None
                st.success("AI models reset. They will retrain automatically.")
        
        with col2:
            if st.button("🧹 Clear All Data"):
                for key in list(st.session_state.keys()):
                    if key.startswith(('ml_', 'ai_', 'neural_', 'anomaly_', 'price_', 'clustering_', 'volatility_')):
                        del st.session_state[key]
                st.success("All AI data cleared. Starting fresh.")

# Main Function Call
if __name__ == "__main__":
    analyze()
