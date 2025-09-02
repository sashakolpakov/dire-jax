#!/usr/bin/env python3

"""
Prototype: Using DIRE to embed and visualize tick-level financial data from Polygon.io
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import deque
import os

# Polygon API client
try:
    from polygon import RESTClient
except ImportError:
    print("Installing polygon-api-client...")
    import subprocess
    subprocess.check_call(["pip", "install", "polygon-api-client"])
    from polygon import RESTClient

# Our DIRE implementation
from dire_jax.dire_pytorch import DiRePyTorch

# Set API key (you can also set as environment variable)
API_KEY = "BBRlVSAOiNqeJ77yjnveFBqZZTOFv2gN"


class TickDataEmbedder:
    """
    Fetches tick data from Polygon and creates embeddings using DIRE.
    """
    
    def __init__(self, api_key=API_KEY):
        self.client = RESTClient(api_key)
        self.embedder = DiRePyTorch(
            n_components=2,
            max_iter_layout=32,
            verbose=True
        )
    
    def fetch_aggregates(self, ticker, multiplier=1, timespan="minute", from_date=None, to_date=None):
        """
        Fetch aggregate bars (easier to start with than raw ticks).
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching {timespan} data for {ticker} from {from_date} to {to_date}...")
        
        # Fetch aggregate bars
        aggs = []
        for a in self.client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            limit=50000
        ):
            aggs.append({
                'timestamp': a.timestamp,
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume,
                'vwap': a.vwap if hasattr(a, 'vwap') else (a.open + a.close) / 2,
                'transactions': a.transactions if hasattr(a, 'transactions') else 0
            })
        
        df = pd.DataFrame(aggs)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
        
        print(f"Fetched {len(df)} bars")
        return df
    
    def fetch_trades(self, ticker, date_str, limit=50000):
        """
        Fetch actual trade ticks for a specific date.
        """
        print(f"Fetching trades for {ticker} on {date_str}...")
        
        trades = []
        for trade in self.client.list_trades(
            ticker=ticker,
            timestamp_gte=date_str + "T09:30:00Z",
            timestamp_lt=date_str + "T16:00:00Z",
            limit=limit
        ):
            trades.append({
                'timestamp': trade.participant_timestamp if hasattr(trade, 'participant_timestamp') else trade.sip_timestamp,
                'price': trade.price,
                'size': trade.size,
                'conditions': trade.conditions if hasattr(trade, 'conditions') else []
            })
        
        df = pd.DataFrame(trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df = df.set_index('timestamp').sort_index()
        
        print(f"Fetched {len(df)} trades")
        return df
    
    def compute_microstructure_features(self, data, window_size='1min'):
        """
        Compute features suitable for embedding from OHLCV or tick data.
        """
        features_list = []
        
        if 'close' in data.columns:  # Aggregate bar data
            # Resample if needed
            if window_size and window_size != 'raw':
                data_resampled = data.resample(window_size).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'vwap': 'mean'
                }).dropna()
            else:
                data_resampled = data
            
            # Compute features
            for i in range(20, len(data_resampled)):  # Need some history
                window = data_resampled.iloc[i-20:i]
                current = data_resampled.iloc[i]
                
                features = [
                    # Price features
                    np.log(current['close'] / window['close'].iloc[0]),  # log return over window
                    (current['high'] - current['low']) / current['open'],  # range
                    (current['close'] - current['open']) / current['open'],  # period return
                    
                    # Volume features  
                    np.log(current['volume'] + 1),  # log volume
                    current['volume'] / window['volume'].mean(),  # relative volume
                    
                    # Volatility features
                    window['close'].pct_change().std(),  # realized volatility
                    ((current['high'] - current['low']) / current['open']),  # Parkinson volatility proxy
                    
                    # Microstructure
                    (current['vwap'] - current['close']) / current['close'],  # VWAP deviation
                    window['volume'].std() / (window['volume'].mean() + 1e-8),  # volume dispersion
                    
                    # Momentum
                    (current['close'] - window['close'].mean()) / (window['close'].std() + 1e-8),  # z-score
                    window['close'].pct_change().mean() * 100,  # average return
                ]
                
                features_list.append(features)
        
        elif 'price' in data.columns:  # Tick data
            # Aggregate ticks into windows
            resampled = data.resample(window_size).agg({
                'price': ['first', 'last', 'min', 'max', 'std', 'count'],
                'size': ['sum', 'mean', 'std', 'max']
            })
            resampled.columns = ['_'.join(col).strip() for col in resampled.columns]
            
            for i in range(5, len(resampled)):
                window = resampled.iloc[i-5:i]
                current = resampled.iloc[i]
                
                features = [
                    # Price movement
                    np.log(current['price_last'] / current['price_first']),
                    (current['price_max'] - current['price_min']) / current['price_first'],
                    current['price_std'] / current['price_first'],
                    
                    # Volume/liquidity
                    np.log(current['size_sum'] + 1),
                    current['size_mean'] / (window['size_mean'].mean() + 1e-8),
                    current['size_std'] / (current['size_mean'] + 1e-8),
                    
                    # Trade intensity
                    current['price_count'],  # number of trades
                    current['size_max'] / (current['size_mean'] + 1e-8),  # large trade indicator
                ]
                
                features_list.append(features)
        
        return np.array(features_list)
    
    def create_embedding(self, features):
        """
        Create 2D embedding from features using DIRE.
        """
        print(f"Creating embedding for {len(features)} data points...")
        
        # Normalize features
        features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Create embedding
        embedding = self.embedder.fit_transform(features_norm)
        
        return embedding
    
    def analyze_market_session(self, ticker="SPY", date_str=None):
        """
        Complete pipeline: fetch data, compute features, create embedding.
        """
        if not date_str:
            # Use last trading day
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            if datetime.now().weekday() == 0:  # Monday
                date_str = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        
        # Fetch minute bars for the day
        df = self.fetch_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_date=date_str,
            to_date=date_str
        )
        
        if df.empty:
            print("No data fetched!")
            return None, None, None
        
        # Compute features
        features = self.compute_microstructure_features(df, window_size=None)
        
        if len(features) == 0:
            print("No features computed!")
            return df, None, None
        
        # Create embedding
        embedding = self.create_embedding(features)
        
        # Add embedding back to dataframe for analysis
        embed_df = pd.DataFrame(
            embedding, 
            columns=['x', 'y'],
            index=df.index[20:20+len(embedding)]  # Align with feature computation
        )
        
        # Add time-based coloring
        embed_df['hour'] = embed_df.index.hour
        embed_df['minute'] = embed_df.index.minute
        embed_df['time_of_day'] = embed_df['hour'] + embed_df['minute'] / 60
        
        # Add volume from original data
        embed_df['volume'] = df.loc[embed_df.index, 'volume'].values
        embed_df['close'] = df.loc[embed_df.index, 'close'].values
        embed_df['returns'] = df.loc[embed_df.index, 'close'].pct_change().values
        
        return df, features, embed_df
    
    def visualize_embedding(self, embed_df, color_by='time_of_day'):
        """
        Create interactive visualization of the embedding.
        """
        import plotly.express as px
        
        fig = px.scatter(
            embed_df,
            x='x',
            y='y',
            color=color_by,
            hover_data=['close', 'volume', 'returns'],
            title=f'Market Microstructure Embedding (colored by {color_by})',
            labels={'x': 'DIRE Dimension 1', 'y': 'DIRE Dimension 2'}
        )
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(width=1000, height=700)
        fig.show()
        
        return fig


def main():
    """
    Demo: Embed and visualize market microstructure.
    """
    print("="*60)
    print("Financial Market Microstructure Embedding with DIRE")
    print("="*60)
    
    embedder = TickDataEmbedder()
    
    # Analyze SPY (S&P 500 ETF) - most liquid equity
    ticker = "SPY"
    
    # Get data and create embedding
    raw_data, features, embedding_df = embedder.analyze_market_session(ticker)
    
    if embedding_df is not None:
        print(f"\nEmbedding created successfully!")
        print(f"Shape: {embedding_df.shape}")
        print(f"Time range: {embedding_df.index[0]} to {embedding_df.index[-1]}")
        
        # Visualize colored by different features
        print("\nCreating visualizations...")
        
        # 1. Color by time of day (see market open/close effects)
        embedder.visualize_embedding(embedding_df, color_by='time_of_day')
        
        # 2. Color by volume (see high volume clusters)
        embedder.visualize_embedding(embedding_df, color_by='volume')
        
        # 3. Color by returns (see volatility clusters)
        embedder.visualize_embedding(embedding_df, color_by='returns')
        
        # Analysis
        print("\nInteresting patterns to look for:")
        print("1. Market open (9:30-10:00) should cluster separately")
        print("2. Lunch time (12:00-13:00) might show different structure")
        print("3. Market close (15:30-16:00) often has distinct patterns")
        print("4. High volume periods might form their own clusters")
        print("5. Volatility events should be visible as outliers")
        
        return embedding_df
    
    return None


if __name__ == "__main__":
    embedding = main()