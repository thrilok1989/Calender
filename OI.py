import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

class VolumeProfileAnalyzer:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://api.dhan.co/v2"
    
    def get_candle_data(self, security_id, exchange_segment, instrument, 
                       interval, from_date, to_date):
        """Fetch intraday candle data from DhanHQ API"""
        url = f"{self.base_url}/charts/intraday"
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.access_token
        }
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_candle_data(data)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None
    
    def _format_candle_data(self, raw_data):
        """Convert raw API response to DataFrame"""
        if not raw_data or 'open' not in raw_data:
            return None
        
        df = pd.DataFrame({
            'timestamp': raw_data['timestamp'],
            'open': raw_data['open'],
            'high': raw_data['high'],
            'low': raw_data['low'],
            'close': raw_data['close'],
            'volume': raw_data['volume']
        })
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def create_volume_profile(self, df, price_levels=50, volume_distribution='uniform'):
        """
        Create approximate volume profile from OHLCV candle data
        
        Args:
            df: DataFrame with OHLCV data
            price_levels: Number of price levels for the profile
            volume_distribution: 'uniform', 'weighted', or 'vwap_weighted'
        """
        if df is None or df.empty or df['volume'].sum() == 0:
            st.error("No volume data available for volume profile")
            return None
        
        # Define price range
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_step = (max_price - min_price) / price_levels
        
        # Create price bins
        price_bins = np.arange(min_price, max_price + price_step, price_step)
        volume_at_price = np.zeros(len(price_bins) - 1)
        
        # Distribute volume across price levels for each candle
        for idx, row in df.iterrows():
            if row['volume'] == 0:
                continue
                
            candle_low = row['low']
            candle_high = row['high']
            candle_volume = row['volume']
            candle_close = row['close']
            candle_open = row['open']
            
            # Find which price bins this candle overlaps
            start_bin = max(0, int((candle_low - min_price) / price_step))
            end_bin = min(len(price_bins) - 2, int((candle_high - min_price) / price_step))
            
            if start_bin == end_bin:
                # Volume concentrated in one price level
                volume_at_price[start_bin] += candle_volume
            else:
                # Distribute volume across price range
                if volume_distribution == 'uniform':
                    # Uniform distribution across price range
                    volume_per_level = candle_volume / (end_bin - start_bin + 1)
                    volume_at_price[start_bin:end_bin + 1] += volume_per_level
                
                elif volume_distribution == 'weighted':
                    # Weight distribution based on candle body vs wicks
                    body_size = abs(candle_close - candle_open)
                    total_range = candle_high - candle_low
                    
                    if total_range > 0:
                        body_weight = 0.7  # 70% of volume in body
                        wick_weight = 0.3  # 30% in wicks
                        
                        # Distribute body volume
                        body_low = min(candle_open, candle_close)
                        body_high = max(candle_open, candle_close)
                        body_start_bin = max(start_bin, int((body_low - min_price) / price_step))
                        body_end_bin = min(end_bin, int((body_high - min_price) / price_step))
                        
                        if body_start_bin <= body_end_bin:
                            body_volume_per_level = (candle_volume * body_weight) / (body_end_bin - body_start_bin + 1)
                            volume_at_price[body_start_bin:body_end_bin + 1] += body_volume_per_level
                        
                        # Distribute wick volume
                        remaining_bins = list(range(start_bin, end_bin + 1))
                        body_bins = list(range(body_start_bin, body_end_bin + 1))
                        wick_bins = [b for b in remaining_bins if b not in body_bins]
                        
                        if wick_bins:
                            wick_volume_per_level = (candle_volume * wick_weight) / len(wick_bins)
                            for bin_idx in wick_bins:
                                volume_at_price[bin_idx] += wick_volume_per_level
                
                elif volume_distribution == 'vwap_weighted':
                    # Weight toward VWAP (approximated as (H+L+C)/3)
                    vwap_price = (row['high'] + row['low'] + row['close']) / 3
                    vwap_bin = int((vwap_price - min_price) / price_step)
                    vwap_bin = max(start_bin, min(end_bin, vwap_bin))
                    
                    # 50% of volume at VWAP level, rest distributed uniformly
                    volume_at_price[vwap_bin] += candle_volume * 0.5
                    remaining_volume = candle_volume * 0.5
                    remaining_volume_per_level = remaining_volume / (end_bin - start_bin + 1)
                    volume_at_price[start_bin:end_bin + 1] += remaining_volume_per_level
        
        # Create result DataFrame
        profile_df = pd.DataFrame({
            'price': price_bins[:-1] + price_step/2,  # Mid-point of each bin
            'volume': volume_at_price
        })
        
        # Calculate additional metrics
        total_volume = profile_df['volume'].sum()
        profile_df['volume_pct'] = (profile_df['volume'] / total_volume) * 100
        
        # Find Point of Control (POC) - price level with highest volume
        poc_idx = profile_df['volume'].idxmax()
        poc_price = profile_df.loc[poc_idx, 'price']
        poc_volume = profile_df.loc[poc_idx, 'volume']
        
        # Calculate Value Area (approximately 70% of volume)
        sorted_profile = profile_df.sort_values('volume', ascending=False)
        cumulative_volume = 0
        value_area_prices = []
        
        for idx, row in sorted_profile.iterrows():
            cumulative_volume += row['volume']
            value_area_prices.append(row['price'])
            if cumulative_volume >= total_volume * 0.7:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        return {
            'profile_df': profile_df,
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'total_volume': total_volume
        }
    
    def plot_volume_profile(self, df, profile_data, symbol):
        """Create volume profile visualization"""
        if profile_data is None:
            return None
        
        profile_df = profile_data['profile_df']
        
        # Create subplot: candlestick + volume profile
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            shared_yaxes=True,
            subplot_titles=(f'{symbol} Price Chart', 'Volume Profile'),
            horizontal_spacing=0.02
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add POC line
        fig.add_hline(
            y=profile_data['poc_price'],
            line=dict(color='blue', width=2, dash='solid'),
            annotation_text=f"POC: {profile_data['poc_price']:.2f}",
            annotation_position="left",
            row=1, col=1
        )
        
        # Add Value Area
        fig.add_hrect(
            y0=profile_data['value_area_low'],
            y1=profile_data['value_area_high'],
            fillcolor='lightblue',
            opacity=0.2,
            line_width=0,
            annotation_text="Value Area",
            annotation_position="left",
            row=1, col=1
        )
        
        # Volume Profile (horizontal bars)
        fig.add_trace(
            go.Bar(
                y=profile_df['price'],
                x=profile_df['volume'],
                orientation='h',
                name='Volume Profile',
                marker_color='rgba(66, 165, 245, 0.7)',
                hovertemplate='Price: %{y:.2f}<br>Volume: %{x:,.0f}<br>%{customdata:.1f}% of total<extra></extra>',
                customdata=profile_df['volume_pct']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{symbol} Volume Profile Analysis',
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Volume", row=1, col=2)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        return fig

def main():
    st.title("Volume Profile Analyzer")
    st.markdown("Approximate volume profile analysis using DhanHQ candle data")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Get access token
    try:
        access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    except KeyError:
        access_token = st.sidebar.text_input("DhanHQ Access Token", type="password")
        if not access_token:
            st.warning("Please enter your DhanHQ access token")
            return
    
    analyzer = VolumeProfileAnalyzer(access_token)
    
    # Symbol and timeframe selection
    symbol = st.sidebar.text_input("Symbol", value="RELIANCE").upper()
    
    interval = st.sidebar.selectbox("Interval", ["1", "5", "15", "25", "60"])
    
    # Date selection
    from_date = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=5))
    to_date = st.sidebar.date_input("To Date", value=datetime.now())
    
    # Volume profile settings
    st.sidebar.subheader("Volume Profile Settings")
    price_levels = st.sidebar.slider("Price Levels", 20, 100, 50)
    
    distribution_method = st.sidebar.selectbox(
        "Volume Distribution",
        ["uniform", "weighted", "vwap_weighted"],
        help="uniform: Equal distribution across price range\n"
             "weighted: More volume in candle body\n"
             "vwap_weighted: Concentrated around VWAP"
    )
    
    if st.sidebar.button("Generate Volume Profile", type="primary"):
        with st.spinner("Fetching data and generating volume profile..."):
            
            # Fetch candle data
            df = analyzer.get_candle_data(
                security_id="1333",  # RELIANCE - you'd need to look up actual security IDs
                exchange_segment="NSE_EQ",
                instrument="EQUITY",
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )
            
            if df is not None and not df.empty:
                # Generate volume profile
                profile_data = analyzer.create_volume_profile(
                    df, 
                    price_levels=price_levels,
                    volume_distribution=distribution_method
                )
                
                if profile_data:
                    # Display metrics
                    st.subheader("Volume Profile Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("POC Price", f"₹{profile_data['poc_price']:.2f}")
                    with col2:
                        st.metric("Value Area High", f"₹{profile_data['value_area_high']:.2f}")
                    with col3:
                        st.metric("Value Area Low", f"₹{profile_data['value_area_low']:.2f}")
                    with col4:
                        st.metric("Total Volume", f"{profile_data['total_volume']:,.0f}")
                    
                    # Plot volume profile
                    fig = analyzer.plot_volume_profile(df, profile_data, symbol)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display top volume levels
                    st.subheader("Top Volume Levels")
                    top_levels = profile_data['profile_df'].nlargest(10, 'volume')
                    st.dataframe(
                        top_levels[['price', 'volume', 'volume_pct']].round(2),
                        use_container_width=True
                    )
                else:
                    st.error("Could not generate volume profile")
            else:
                st.error("No data received")
    
    # Instructions
    with st.expander("How Volume Profile Works"):
        st.markdown("""
        **Volume Profile Analysis:**
        - Distributes volume across price levels based on OHLC candle data
        - **POC (Point of Control):** Price level with highest volume
        - **Value Area:** Price range containing ~70% of total volume
        - **Distribution Methods:**
          - *Uniform:* Equal volume distribution across price range
          - *Weighted:* More volume concentrated in candle body
          - *VWAP Weighted:* Volume concentrated around typical price
        
        **Limitations:**
        - Approximate analysis based on candle data, not tick data
        - Less accurate than true tick-by-tick volume profile
        - Best used with higher frequency data (1-5 minute candles)
        """)

if __name__ == "__main__":
    main()